import json
from ruamel.yaml import YAML

import pandas as pd
from datasets import Dataset
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import metrics
from ragas import evaluate


with open("results.json") as f:
    results = json.load(f)

questions, answers, contexts = [], [], []
for row in results:
    questions.append(row["Q"])
    answers.append(row["A"])
    contexts.append(row["context"])

truth = pd.read_csv("ground_truths.csv")
ground_truth = truth["A"].to_list()

dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth
        })

with open("params.yaml") as f:
    params = YAML().load(f)
emb_params = params['Embeddings']
chat_params = params['ChatLLM']

emb = HuggingFaceEmbeddings(**emb_params)
llm = HuggingFaceHub(**chat_params)

result = evaluate(
    dataset,
    metrics=[metrics.answer_similarity],
    llm=llm,
    embeddings=emb,
)

print(result)
df = result.to_pandas()
df.to_csv("eval_ragas.csv", header=True, index=False)
