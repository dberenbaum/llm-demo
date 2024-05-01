"""Python file to serve as the frontend"""
import json
import langchain
import streamlit as st
from langchain import hub
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from ruamel.yaml import YAML
from streamlit_chat import message

langchain.debug = True

with open("params.yaml") as f:
    params = YAML().load(f)
emb_params = params['Embeddings']
chat_params = params['ChatLLM']

# Load the LangChain.
emb = HuggingFaceEmbeddings(**emb_params)

store = FAISS.load_local("docs.index", emb)
retriever = store.as_retriever()

llm = HuggingFaceHub(**chat_params)
prompt = hub.pull("rlm/rag-prompt").messages[0].prompt

def chain(question):
    context = retriever.get_relevant_documents(question)
    context = [doc.page_content for doc in context]
    context_str = "\n\n".join(context)
    input = prompt.invoke({"question": question, "context": context_str})
    return llm.invoke(input.text) 

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Git QA Bot", page_icon=":robot:")
st.header("Git QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain(user_input)

    log_data = {'user_input': user_input, 'answer': result}
    log_str = json.dumps(log_data)
    assert len(log_str.splitlines()) == 1
    with open('chat.log', 'a') as f:
        f.write(log_str)
        f.write('\n')

    st.session_state.past.append(user_input)
    st.session_state.generated.append(result)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
