import streamlit as st

from langchain_helper import create_vector_db, get_qa_chain

st.title("QA for Online Learning Plateform 💡")
knowledge_base_is_created = st.button("Create Knowledge Base")

if knowledge_base_is_created:
    create_vector_db()

question = st.text_input(" 💬 Hi I am the Online Platforme Q&A Assistant. How could I help you ?")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response['result'])