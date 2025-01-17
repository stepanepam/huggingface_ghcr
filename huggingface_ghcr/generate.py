import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from openai import OpenAI
from transformers import pipeline


def generate_response_with_gpt4(context: str, question: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )

    return response.choices[0].message.content


def generate_response_with_roberta(question: str, context: str):
    QA_input = {"question": question, "context": context}
    repo_id = "deepset/roberta-base-squad2"
    pipe = pipeline("question-answering", model=repo_id, tokenizer=repo_id)
    response = pipe(QA_input).get("answer")

    return response


def generate_response_with_mistral(query_with_context: str):
    raise Exception("lack of memory for this model")
    pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
    messages = [
        {"role": "user", "content": query_with_context},
    ]
    response = pipe(messages)
    return response


def generate_response_with_llama(query_with_context: str):
    model_id = "meta-llama/Llama-3.2-1B"
    pipe = pipeline("text-generation", model=model_id)

    response = pipe(query_with_context)[0]["generated_text"]

    return response
