import numpy as np

from documents import documents
from embedder import get_embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


print("Generating document embeddings...")

document_store = []

for doc in documents:

    embedding = get_embedding(doc)

    document_store.append({
        "text": doc,
        "embedding": embedding
    })


print("Ready!\n")


while True:

    query = input("Question: ")

    query_embedding = get_embedding(query)

    results = []

    for document in document_store:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append(
            (score, document["text"])
        )

    results.sort(reverse=True)

    print("\nTop Matches:\n")

    for score, text in results:

        print(
            f"{score:.4f} -> {text}"
        )

    print()