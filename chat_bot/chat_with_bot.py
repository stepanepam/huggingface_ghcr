import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from openai import OpenAI
from retrieve import get_retriever

from core.embedding import embeddings
from core.vector_store import init_vector_store
from data_pipeline.generate import (
    generate_response_with_gpt4,
    generate_response_with_llama,
    generate_response_with_mistral,
    generate_response_with_roberta,
)
from settings.config import FOLDER_PATH, OPENAI_API_KEY

vector_store = init_vector_store(embeddings, FOLDER_PATH)
retriever = get_retriever(vectorstore=vector_store)

#! IF YOU WANT TO USE OPENAI API
client = OpenAI(api_key=OPENAI_API_KEY)


question = "How many dwarfs was there?"
print("User:", question)
while question != "q":
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print("AI (retrieved):", context)

    query_with_context = f"I'll give you some context and one question. Answear for this question.\
                            \n\nContext:\n{context}\n\nQuestion: {question}"

    ##### Choose one of the models: GPT4, Mistral, Roberta

    ### GPT4 API
    # DONT FORGET UNCOMMENT CLIENT ABOVE
    response = generate_response_with_gpt4(client, context, question)

    ## QnA open source model roberta
    # response = generate_response_with_roberta(question, context)

    ### Mistral 7B
    # response = generate_response_with_mistral(query_with_context)

    # ### Llama 3.1
    # response = generate_response_with_llama(query_with_context)

    print("AI (LLM):", response)
    question = str(input("User: "))
