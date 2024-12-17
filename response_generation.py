from transformers import pipeline

def generate_response(context, query):
    """
    Use Hugging Face QA model to generate an answer.
    """
    qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    response = qa_pipeline(question=query, context=context)
    return response['answer']
