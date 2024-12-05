import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

folder_path = "./texts"
model = SentenceTransformer("all-MiniLM-L6-v2")

# fetch all txt files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

file_contents = {}

for file in files:
    with open(os.path.join(folder_path, file), "r") as f:
        file_contents[file] = f.read()

embeddings = {}
for file, content in file_contents.items():
    embeddings[file] = model.encode(content)

user_query = input("Enter your query: ")
query_embedding = model.encode(user_query)

file_scores = {}
for file, embedding in embeddings.items():
    file_scores[file] = cosine_similarity([query_embedding], [embedding])[0][0]

# Build a FAISS index
dimension = len(list(embeddings.values())[0])  # Embedding size
index = faiss.IndexFlatL2(dimension)

# Add embeddings
embeddings_matrix = np.array(list(embeddings.values()))
index.add(embeddings_matrix)

# Query index
_, indices = index.search(np.array([query_embedding]), k=1)
best_match = files[indices[0][0]]

print(f"The most relevant file is: {best_match}")



