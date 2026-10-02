import os

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Greenhouse RAG API",
    description="API for the Smart Greenhouse AI Assistant",
    version="1.0.0"
)


# =========================
# LOAD AI MODELS
# =========================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model loaded successfully!")


# =========================
# GEMINI CLIENT
# =========================

client = genai.Client(api_key=api_key)


# =========================
# DATABASE CONNECTION
# =========================

def get_connection():

    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "Greenhouseapp"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD")
    )


# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    question: str


# =========================
# RAG SEARCH
# =========================

def search_knowledge(question, limit=3):

    question_embedding = model.encode(question).tolist()

    conn = get_connection()

    try:

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
                (
                    question_embedding,
                    question_embedding,
                    limit
                )
            )

            results = cur.fetchall()

            return results

    finally:

        conn.close()


# =========================
# GENERATE ANSWER
# =========================

def generate_answer(question, results):

    context = ""

    for index, row in enumerate(results, start=1):

        source, content, similarity = row

        context += f"""

SOURCE {index}: {source}
SIMILARITY: {similarity:.4f}

{content}

"""

    prompt = f"""
You are an AI assistant for a smart greenhouse.

Answer the user's question using ONLY the information provided
in the context below.

If the answer is not contained in the context, say clearly that
the knowledge base does not contain enough information.

Give practical and clear answers suitable for greenhouse management.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# =========================
# HOME ENDPOINT
# =========================

@app.get("/")
def home():

    return {
        "message": "Greenhouse RAG API is running!"
    }


# =========================
# CHAT ENDPOINT
# =========================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        results = search_knowledge(
            request.question
        )

        answer = generate_answer(
            request.question,
            results
        )

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )