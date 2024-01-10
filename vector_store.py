import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
import pickle
import json
import dvc.api


params = dvc.api.params_show()['Embeddings']
print(params)

with open("docs.json", "r") as f:
    docs = json.load(f)

with open("metadatas.json", "r") as f:
    metadatas = json.load(f)

print(f"Processing {len(docs)} documents.")

emb = OllamaEmbeddings(**params)

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, emb, metadatas=metadatas)
store.save_local("docs.index")
