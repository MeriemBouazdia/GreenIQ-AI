import pickle
import psycopg

with open("greenIQ_knowledge_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

knowledge_df = data["knowledge_df"]
embeddings = data["embeddings"]

print(f"Loaded {len(knowledge_df)} GreenIQ chunks.")
print(f"Embeddings shape: {embeddings.shape}")

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

print("PostgreSQL connected successfully!")

inserted = 0
skipped = 0

with conn.cursor() as cur:

    cur.execute("SELECT COUNT(*) FROM knowledge_documents")
    old_count = cur.fetchone()[0]

    for i, (_, row) in enumerate(knowledge_df.iterrows()):

        source = str(row["source"])
        content = str(row["content"])

        cur.execute(
            """
            SELECT 1
            FROM knowledge_documents
            WHERE source = %s AND content = %s
            LIMIT 1
            """,
            (source, content)
        )

        if cur.fetchone():
            skipped += 1
            continue

        embedding = embeddings[i]

        embedding_str = "[" + ",".join(
            map(str, embedding.tolist())
        ) + "]"

        cur.execute(
            """
            INSERT INTO knowledge_documents
            (source, content, embedding)
            VALUES (%s, %s, %s)
            """,
            (
                source,
                content,
                embedding_str
            )
        )

        inserted += 1

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM knowledge_documents")
    new_count = cur.fetchone()[0]

print()
print("===================================")
print("GREENIQ KNOWLEDGE INSERTION DONE")
print("===================================")
print(f"Old records : {old_count}")
print(f"Inserted    : {inserted}")
print(f"Skipped     : {skipped}")
print(f"New total   : {new_count}")

conn.close()