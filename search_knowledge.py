from sentence_transformers import SentenceTransformer
import psycopg

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

question = "How can I manage irrigation in a greenhouse?"

question_embedding = model.encode(question).tolist()

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT
            id,
            source,
            content,
            1 - (embedding <=> %s::vector) AS similarity
        FROM knowledge_documents
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
        """,
        (
            question_embedding,
            question_embedding
        )
    )

    results = cur.fetchall()

conn.close()

print("\n==============================")
print("SEMANTIC SEARCH RESULTS")
print("==============================")

for rank, row in enumerate(results, start=1):
    doc_id, source, content, similarity = row

    print(f"\n--- Result {rank} ---")
    print(f"ID: {doc_id}")
    print(f"Source: {source}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Content:\n{content[:1000]}")