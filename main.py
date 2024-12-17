from pdf_ingestion import extract_text_from_pdf
from text_processing import chunk_text, generate_embeddings
from vector_store import save_to_faiss, load_faiss_index
from query_pipeline import handle_query
from response_generation import generate_response

def main():
    print("### PDF Chat with Dynamic Data Extraction ###")

    # Step 1: Extract text
    pdf_path = "myfile.pdf"  # Replace with your uploaded PDF file
    text = extract_text_from_pdf(pdf_path)
    print("\nText extraction complete!\n")

    # Step 2: Chunk text and generate embeddings
    chunks = chunk_text(text, chunk_size=300)
    print(f"Total chunks created: {len(chunks)}")

    model, embeddings = generate_embeddings(chunks)
    save_to_faiss(embeddings)

    # Step 3: Query the system
    index = load_faiss_index()
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Retrieve relevant chunks
        combined_context = handle_query(query, model, chunks, index, k=10)

        # Print retrieved context (optional for debugging)
        print("\n### Retrieved Context ###")
        print(combined_context)

        # Generate dynamic response
        response = generate_response(combined_context, query)
        print("\n### Answer ###")
        print(response)

if __name__ == "__main__":
    main()
