from generate import (
    generate_response_with_gpt4,
    generate_response_with_llama,
    generate_response_with_mistral,
    generate_response_with_roberta,
)
from indexing import indexing_page
from retrieve import retriever
from split import Splitter
from transformers import pipeline


def main():

    splits = Splitter().split_page()
    vector_store = indexing_page(splits)
    retrieved = retriever(vector_store)

    question = "How many dwarfs was there?"
    print("User:", question)
    while question != "q":
        retrieved_docs = retrieved.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print("AI (retrieved):", context)

        query_with_context = f"I'll give you some context and one question. Answear for this question.\
                                \n\nContext:\n{context}\n\nQuestion: {question}"

        ##### Choose one of the models: GPT4, Mistral, Roberta

        ### GPT4 API
        response = generate_response_with_gpt4(context, question)

        ## QnA open source model roberta
        # response = generate_response_with_roberta(question, context)

        ### Mistral 7B
        # response = generate_response_with_mistral(query_with_context)

        # ### Llama 3.1
        # response = generate_response_with_llama(query_with_context)

        print("AI (LLM):", response)
        question = str(input("User: "))


if __name__ == "__main__":
    main()
