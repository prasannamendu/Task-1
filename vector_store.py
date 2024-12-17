import faiss
import numpy as np

def save_to_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, "vector_index.faiss")
    print("Embeddings saved to FAISS.")

def load_faiss_index():
    return faiss.read_index("vector_index.faiss")

def search_faiss(index, query_embedding, k=5):
    D, I = index.search(np.array(query_embedding), k)
    return I
