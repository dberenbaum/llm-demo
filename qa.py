"""Ask a question to the notion database."""
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle
import pandas as pd
import dvc.api


params = dvc.api.params_show()
emb_params = params['Embeddings']
chat_params = params['Chat']
qa_params = params['Retrieval']
print(chat_params)
print(qa_params)

# Load the LangChain.
emb = OllamaEmbeddings(**emb_params)
store = FAISS.load_local("docs.index", emb)

df = pd.read_csv("canfy.csv")
sample_questions = df["Q"].to_list()

llm = ChatOllama(**chat_params)
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                    retriever=store.as_retriever(), max_tokens_limit=qa_params['max_tokens_limit'],
                                                    reduce_k_below_max_tokens=qa_params['reduce_k_below_max_tokens'],
                                                    verbose=qa_params['verbose'])

records = []
for question in sample_questions:
    question = question.strip()
    print(f"Question: {question}")

    result = chain({"question": question})
    records.append({"Q": question, "A": result["answer"].strip(), "sources": result['sources'].strip()})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print("")

df = pd.DataFrame.from_records(records)
df.to_csv("results.csv", header=True, index=False)
