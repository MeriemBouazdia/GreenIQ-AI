import os
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client = genai.Client(api_key=api_key)

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

def search_knowledge(question, limit=3):
    question_embedding = model.encode(question).tolist()

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

        return cur.fetchall()


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


print("\n================================")
print("GREENHOUSE RAG CHATBOT")
print("Type 'exit' to stop.")
print("================================\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    results = search_knowledge(question)

    answer = generate_answer(question, results)

    print("\nChatbot:")
    print(answer)
    print("\n--------------------------------\n")

conn.close()