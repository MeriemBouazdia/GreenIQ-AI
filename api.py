import os
import hashlib
import logging
from contextlib import asynccontextmanager

import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

GEMINI_MODEL = "gemini-3.5-flash-lite"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("greeniq")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client = genai.Client(api_key=api_key)

DB_CONNINFO = (
    f"host={os.getenv('DB_HOST', 'localhost')} "
    f"port={os.getenv('DB_PORT', '5432')} "
    f"dbname={os.getenv('DB_NAME', 'Greenhouseapp')} "
    f"user={os.getenv('DB_USER', 'postgres')} "
    f"password={os.getenv('DB_PASSWORD', '')}"
)

# Opened explicitly during the FastAPI lifespan startup below, so a DB
# outage at import time doesn't crash the process before it can log anything.
pool = ConnectionPool(conninfo=DB_CONNINFO, min_size=1, max_size=10, open=False)


def get_connection():
    """Returns a pooled connection as a context manager (same call shape as before)."""
    return pool.connection()


# ---------------------------------------------------------------------------
# Schema migration (idempotent, safe to run on every startup)
# ---------------------------------------------------------------------------

def run_migrations():
    """
    Adds the columns needed for auto-learning without touching existing data:
    - content_hash: for atomic, race-free de-duplication on insert
    - created_at:   so entries can be aged out / audited later
    - source_url:   provenance of auto-learned knowledge
    - verified:     lets curated vs. auto-learned content be trusted differently
    """
    statements = [
        "ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS content_hash TEXT;",
        "ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();",
        "ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS source_url TEXT;",
        "ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS verified BOOLEAN DEFAULT false;",
    ]

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)

                # Backfill content_hash for any rows that pre-date this column.
                cur.execute(
                    "SELECT id, content FROM knowledge_documents WHERE content_hash IS NULL;"
                )
                rows = cur.fetchall()
                for row_id, content in rows:
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    cur.execute(
                        "UPDATE knowledge_documents SET content_hash = %s WHERE id = %s;",
                        (content_hash, row_id),
                    )

                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        knowledge_documents_content_hash_idx
                    ON knowledge_documents (content_hash);
                    """
                )
            conn.commit()
        logger.info("Schema migration check complete (%d row(s) backfilled).", len(rows))
    except Exception:
        logger.exception(
            "Schema migration failed - check DB permissions and the "
            "knowledge_documents table schema manually."
        )
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    run_migrations()
    yield
    pool.close()


app = FastAPI(
    title="GreenIQ AI API",
    description="RAG API for GreenIQ Smart Greenhouse",
    version="1.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Knowledge base search
# ---------------------------------------------------------------------------

def search_knowledge(question, limit=5):
    try:
        question_embedding = model.encode(question).tolist()
    except Exception:
        logger.exception("Failed to embed the question")
        return []

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        source,
                        content,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM knowledge_documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (question_embedding, question_embedding, limit),
                )
                return cur.fetchall()
    except Exception:
        logger.exception("Knowledge base search failed")
        return []


def create_embedding(text):
    return model.encode(text).tolist()


def generate_answer_from_knowledge(question, results):
    context = ""

    for index, row in enumerate(results, start=1):
        source, content, similarity = row

        context += f"""
SOURCE {index}: {source}
SIMILARITY: {similarity:.4f}

{content}

"""

    prompt = f"""
You are an AI assistant for GreenIQ, a smart greenhouse system.

Answer the user's question using ONLY the provided knowledge.

If the provided knowledge does not contain enough information,
return exactly:

KNOWLEDGE_NOT_FOUND

Do not invent facts.

KNOWLEDGE:
{context}

USER QUESTION:
{question}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return (response.text or "KNOWLEDGE_NOT_FOUND").strip()
    except Exception:
        logger.exception("Gemini call failed while answering from existing knowledge")
        raise


# ---------------------------------------------------------------------------
# Web-grounded knowledge acquisition
# ---------------------------------------------------------------------------

def _extract_first_grounding_url(response):
    """
    Best-effort extraction of the first Google Search grounding source URL.
    The exact response shape can vary across SDK versions, so this fails
    quietly and just returns None rather than breaking the answer flow.
    """
    try:
        candidate = response.candidates[0]
        chunks = candidate.grounding_metadata.grounding_chunks
        for chunk in chunks:
            uri = getattr(getattr(chunk, "web", None), "uri", None)
            if uri:
                return uri
    except Exception:
        pass
    return None


def search_new_knowledge(question):
    """
    Searches the web (via Gemini + Google Search grounding) when the internal
    knowledge base doesn't have enough information.
    Returns (knowledge_text, source_url). knowledge_text is "" on failure.
    """
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    prompt = f"""
You are the knowledge acquisition component of GreenIQ.

The GreenIQ internal knowledge base does not contain enough
information to answer this question.

Search the web using reliable and relevant sources.

Focus on agricultural, greenhouse, crop, irrigation, plant
disease, environmental or IoT information when appropriate.

Question:
{question}

Return a concise factual knowledge entry that can be stored
in a knowledge base.

The answer must:
- contain factual information
- avoid speculation
- avoid unsupported claims
- be useful for future GreenIQ questions
- mention important conditions or limitations when relevant
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
    except Exception:
        logger.exception("Web knowledge search via Gemini failed")
        return "", None

    text = (response.text or "").strip()
    source_url = _extract_first_grounding_url(response)
    return text, source_url


def save_knowledge(source, content, source_url=None):
    content = content.strip()
    if not content:
        return False

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    try:
        embedding = model.encode(content).tolist()
    except Exception:
        logger.exception("Failed to embed new knowledge before saving")
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_documents
                        (source, content, embedding, content_hash, source_url, verified, created_at)
                    VALUES (%s, %s, %s::vector, %s, %s, false, now())
                    ON CONFLICT (content_hash) DO NOTHING
                    RETURNING id;
                    """,
                    (source, content, embedding, content_hash, source_url),
                )
                inserted = cur.fetchone()
            conn.commit()

        if inserted is None:
            logger.info("Knowledge already existed (content_hash match) - skipped insert.")
            return False

        logger.info("Saved new knowledge entry id=%s from source=%s", inserted[0], source)
        return True
    except Exception:
        logger.exception("Failed to save new knowledge to the database")
        return False


def generate_answer_with_new_knowledge(question, new_knowledge):
    prompt = f"""
You are GreenIQ, an AI assistant for smart greenhouse management.

Answer the user's question using the verified information
obtained from the web.

NEW VERIFIED KNOWLEDGE:
{new_knowledge}

USER QUESTION:
{question}

Give a clear, practical and concise answer.

Do not mention internal database operations.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return (response.text or "").strip()
    except Exception:
        logger.exception("Gemini call failed while answering from newly acquired knowledge")
        raise


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "message": "GreenIQ AI API is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        return {"answer": "Please provide a question."}

    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long (max {MAX_QUESTION_LENGTH} characters).",
        )

    try:
        results = search_knowledge(question)

        if results:
            best_similarity = float(results[0][2])

            if best_similarity >= SIMILARITY_THRESHOLD:
                answer = generate_answer_from_knowledge(question, results)

                if answer != "KNOWLEDGE_NOT_FOUND":
                    return {"answer": answer}

        new_knowledge, source_url = search_new_knowledge(question)

        if not new_knowledge:
            return {
                "answer": "I could not find reliable information to answer this question."
            }

        save_knowledge(
            source="Google Search + Gemini",
            content=new_knowledge,
            source_url=source_url,
        )

        answer = generate_answer_with_new_knowledge(question, new_knowledge)
        return {"answer": answer}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error while answering a chat question")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again.",
        )