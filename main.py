"""Python file to serve as the frontend"""
import json
import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
import dvc.api

import langchain
langchain.debug = True

params = dvc.api.params_show()
emb_params = params['Embeddings']
chat_params = params['ChatLLM']
qa_params = params['Retrieval']

# Load the LangChain.
emb = HuggingFaceEmbeddings(**emb_params)

store = FAISS.load_local("docs.index", emb)
retriever = store.as_retriever()

llm = HuggingFaceHub(**chat_params)
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                    retriever=retriever,
                                                    **qa_params)

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
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    log_data = {'user_input': user_input, 'answer': result['answer'], 'sources': result['sources']}
    log_str = json.dumps(log_data)
    assert len(log_str.splitlines()) == 1
    with open('chat.log', 'a') as f:
        f.write(log_str)
        f.write('\n')

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
