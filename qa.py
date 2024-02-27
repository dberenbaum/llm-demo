"""Ask a question to the notion database."""
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
import json
import os
import pickle
import pandas as pd
import dvc.api


params = dvc.api.params_show()
emb_params = params['Embeddings']
chat_params = params['ChatLLM']
qa_params = params['Retrieval']
print(chat_params)
print(qa_params)

# Load the LangChain.
emb = HuggingFaceEmbeddings(**emb_params)

store = FAISS.load_local("docs.index", emb)
retriever = store.as_retriever()

df = pd.read_csv("ground_truths.csv")
sample_questions = df["Q"].to_list()

llm = HuggingFaceHub(**chat_params)
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                    retriever=retriever, max_tokens_limit=qa_params['max_tokens_limit'],
                                                    reduce_k_below_max_tokens=qa_params['reduce_k_below_max_tokens'],
                                                    verbose=qa_params['verbose'])

records = []
for question in sample_questions:
    question = question.strip()
    context = retriever.get_relevant_documents(question)
    context = [doc.page_content for doc in context]
    print(f"Question: {question}")

    result = chain({"question": question})
    records.append({
        "Q": question,
        "A": result["answer"].strip(),
        "sources": result['sources'].strip(),
        "context": context,
    })

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print("")

with open("results.json", "w") as f:
    json.dump(records, f)
