import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from ruamel.yaml import YAML


def split_docs(docs):
    with open("params.yaml") as f:
        params = YAML().load(f)
    split_params = params['TextSplitter']

    text_splitter = RecursiveCharacterTextSplitter(separators=[".\n", ".", "\n"], **split_params)
    return text_splitter.split_text("\n".join(docs))

if __name__ == '__main__':
    path = "dvc_discord_channel.csv"
    df = pd.read_csv(path)

    messages = df["Content"].astype(str).values.tolist()
    docs = split_docs(messages)

    with open("docs.json", "w") as f:
        json.dump(docs, f)
