import requests
import numpy as np
import json

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

    document_store.append({
        "text": doc,
        "embedding": get_embedding(doc)
    })

print("Ready!\n")


while True:

    question = input("Question: ")

    query_embedding = get_embedding(question)

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

    best_score, best_chunk = results[0]

    print(best_score)
    print(best_chunk)

    if best_score < 0.65:
        print("\nI don't have enough information.\n")
        continue

    prompt = f"""
                Answer ONLY using the information provided in the context.

                If the context does not contain the answer,
                say "I don't have enough information."

                Do not make assumptions.
                Do not invent company names.

                Context:
                {best_chunk}

                Question:
                {question}
    """

    print("\nAnswer:\n")

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    answer = ""

    for line in response.iter_lines():

        if line:

            data = json.loads(line)

            token = data["response"]

            print(token, end="", flush=True)

            answer += token

    print("\n")