import json
import pandas as pd
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from ruamel.yaml import YAML


template = """
You are an expert grading the correctness of answers to questions on a scale of 0 to 4.
A score of 0 means that the answer is not factually correct. A score of 4 means that the
answer is completely factually correct in all details.

You are grading the following question:
{question}

Here is the real answer:
{answer}

You are grading the following predicted answer:
{result}

Respond with a number on the scale of 0 to 4.
"""

with open("params.yaml") as f:
    params = YAML().load(f)
llm = HuggingFaceEndpoint(**params["ChatLLM"])

truth = pd.read_csv("ground_truths.csv")
with open("results.json") as f:
    predictions = json.load(f)

records = []
for row in range(len(predictions)):
    question = truth.loc[row]["Q"]
    answer = truth.loc[row]["A"]
    result = predictions[row]["A"]

    prompt = PromptTemplate.from_template(template)
    text = prompt.format(question=question, answer=answer, result=result)
    print(text)
    print()
    response = llm.invoke(text)
    print(response)
    records.append({"Q": question, "grade": response.strip()})

df = pd.DataFrame.from_records(records)
df.to_csv("eval.csv", header=True, index=False)
