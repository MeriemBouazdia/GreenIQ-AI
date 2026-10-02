import pickle
import psycopg

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
print("PostgreSQL connected successfully!")

with open("greenhouse_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} embeddings from file.")

with conn.cursor() as cur:
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)

conn.commit()

print("pgvector extension is ready!")

with conn.cursor() as cur:
    cur.execute("""
        DROP TABLE IF EXISTS knowledge_documents;
    """)

    cur.execute("""
        CREATE TABLE knowledge_documents (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector(384)
        );
    """)

conn.commit()

print("knowledge_documents table recreated successfully!")

with conn.cursor() as cur:
    for index, item in enumerate(data, start=1):
        source = item["source"]
        text = item["text"]
        embedding = item["embedding"]

        cur.execute(
            """
            INSERT INTO knowledge_documents
            (source, content, embedding)
            VALUES (%s, %s, %s)
            """,
            (source, text, embedding)
        )

        print(f"Inserted {index}/{len(data)}")

conn.commit()

with conn.cursor() as cur:
    cur.execute("""
        SELECT COUNT(*)
        FROM knowledge_documents;
    """)

    count = cur.fetchone()[0]

print("\n===================================")
print("INSERTION COMPLETED SUCCESSFULLY!")
print("===================================")
print(f"Total records in database: {count}")

conn.close()

print("Database connection closed.")