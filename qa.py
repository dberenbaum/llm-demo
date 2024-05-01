"""Ask a question to the notion database."""
import json
import os
import pickle
import pandas as pd
from langchain import hub
from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from ruamel.yaml import YAML


with open("params.yaml") as f:
    params = YAML().load(f)
chat_params = params['ChatLLM']

with open("docs.json", "r") as f:
    docs = json.load(f)
retriever = BM25Retriever.from_texts(docs)

df = pd.read_csv("ground_truths.csv")
sample_questions = df["Q"].to_list()

llm = HuggingFaceEndpoint(**chat_params)
prompt = hub.pull("rlm/rag-prompt").messages[0].prompt

records = []
for question in sample_questions:
    context = retriever.get_relevant_documents(question)
    context = [doc.page_content for doc in context]
    context_str = "\n\n".join(context)
    print(f"Question: {question}")

    input = prompt.invoke({"question": question, "context": context_str})
    result = llm.invoke(input.text) 

    records.append({
        "Q": question,
        "A": result,
        "context": context,
    })

    print(f"Answer: {result}")
    print("\n\n")

with open("results.json", "w") as f:
    json.dump(records, f)
