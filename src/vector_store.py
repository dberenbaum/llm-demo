import faiss
import pickle
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ruamel.yaml import YAML


with open("params.yaml") as f:
    params = YAML().load(f)
emb_params = params['Embeddings']

with open("docs.json", "r") as f:
    docs = json.load(f)

print(f"Processing {len(docs)} documents.")

emb = HuggingFaceEmbeddings(**emb_params)

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, emb)
store.save_local("docs.index")
