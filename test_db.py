import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

cur.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        metadata JSONB,
        embedding VECTOR(768)
    )
""")

conn.commit()

print("PostgreSQL connected successfully!")
print("pgvector extension enabled!")
print("knowledge_documents table created!")

cur.close()
conn.close()