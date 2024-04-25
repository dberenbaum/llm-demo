"""Ask a question to the notion database."""
import json
import os
import pickle
import pandas as pd
from langchain import hub
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from ruamel.yaml import YAML


with open("params.yaml") as f:
    params = YAML().load(f)
emb_params = params['Embeddings']
chat_params = params['ChatLLM']

# Load the LangChain.
emb = HuggingFaceEmbeddings(**emb_params)

store = FAISS.load_local("docs.index", emb, allow_dangerous_deserialization=True)
retriever = store.as_retriever()

df = pd.read_csv("ground_truths.csv")
sample_questions = df["Q"].to_list()

llm = HuggingFaceEndpoint(**chat_params)
prompt = hub.pull("rlm/rag-prompt").messages[0].prompt

records = []
for question in sample_questions:
    question = question.strip()
    context = retriever.get_relevant_documents(question)
    context = [doc.page_content for doc in context]
    context_str = "\n\n".join(context)
    print(f"Question: {question}")

    input = prompt.invoke({"question": question, "context": context_str})
    result = llm.invoke(input.text) 

    records.append({
        "Q": question,
        "A": result.strip(),
        "context": context,
    })

    print(f"Answer: {result}")
    print("")

with open("results.json", "w") as f:
    json.dump(records, f)
