# documents.py

with open("test.txt", "r") as file:
    policy_text = file.read()

raw_chunks = policy_text.split("---")

documents = []

for chunk in raw_chunks:

    cleaned_chunk = chunk.strip()

    if cleaned_chunk != "":
        documents.append(cleaned_chunk)