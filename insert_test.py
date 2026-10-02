import psycopg
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

text = """
Temperature inside a greenhouse should be monitored carefully.
High temperatures can stress plants and reduce their growth.
Ventilation can help reduce excessive heat.
"""
embedding = model.encode(text).tolist()

print("Embedding generated!")
print("Dimensions:", len(embedding))

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()

cur.execute(
    """
    INSERT INTO knowledge_documents
    (content, metadata, embedding)
    VALUES (%s, %s, %s)
    """,
    (
        text,
        '{"source": "test", "topic": "greenhouse_temperature"}',
        embedding
    )
)

conn.commit()

print("Document inserted successfully!")


cur.close()
conn.close()