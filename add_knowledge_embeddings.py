import pickle
import psycopg
import numpy as np

# Load CSV knowledge embeddings
with open("greenhouse_knowledge_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

knowledge_df = data["knowledge_df"]
embeddings = data["embeddings"]

print(f"Loaded {len(knowledge_df)} knowledge records.")
print(f"Embeddings shape: {embeddings.shape}")

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

print("PostgreSQL connected successfully!")

with conn.cursor() as cur:

    for i, (_, row) in enumerate(knowledge_df.iterrows()):

        embedding = embeddings[i]

        # Convert numpy array to PostgreSQL vector format
        embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"

        cur.execute(
            """
            INSERT INTO knowledge_documents
            (source, content, embedding)
            VALUES (%s, %s, %s)
            """,
            (
                str(row["source"]),
                str(row["content"]),
                embedding_str
            )
        )

        print(f"Inserted knowledge {i + 1}/{len(knowledge_df)}")

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM knowledge_documents")
    total = cur.fetchone()[0]

print("\n===================================")
print("KNOWLEDGE INSERTION COMPLETED!")
print("===================================")
print(f"Total records in database: {total}")

conn.close()