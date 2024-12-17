from vector_store import load_faiss_index, search_faiss

def handle_query(query, model, chunks, index, k=10):
    """
    Retrieve top-k relevant chunks for a user query.
    """
    query_embedding = model.encode([query])
    indices = search_faiss(index, query_embedding, k=k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    # Combine all retrieved chunks into a single context
    combined_context = "\n".join(relevant_chunks)
    return combined_context
