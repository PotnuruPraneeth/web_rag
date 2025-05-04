import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

import streamlit as st

# Set page config
st.set_page_config(page_title="Robot Center", layout="wide")
st.title("WEB APPLICATION RAG")
st.write("Welcome to the Web App RAG")

# Black section with centered emoji
st.markdown(
    """
    <style>
    .black-background {
        background-color: black;
        padding: 50px 0;
        text-align: center;
    }
    .black-background h1 {
        font-size: 100px;
        color: white;
        margin: 0;
    }
    </style>
    <div class="black-background">
        <h1>AI 🤖</h1>
    </div>
    """,
    unsafe_allow_html=True
)
user_prompt = st.chat_input("write the prompt")
if user_prompt:
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    # here vector requeries the embeddings to query the question
    vector_store = Chroma(
    # documents=documents,# document objects no need there beacuse as we already done it in main rag for vector store
    embedding_function=embeddings, # model
    persist_directory="./chroma_Db")
    retriver = vector_store.as_retriever()
    retriver_resultss = retriver.invoke(user_prompt)
    if not  retriver_resultss:  # Works for None or empty list
        st.warning('''I'm sorry, but the provided documents do not
         contain any information relevant to your query.''')
    else:
        llm=init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
        promt_create_template=PromptTemplate.from_template('Answer the question based on the context below {retriver_resultss} and question is {user_prompt}')
        chain=promt_create_template | llm
        output=chain.invoke({'retriver_resultss':retriver_resultss,'user_prompt':user_prompt})
        st.write(f"<span style='color:red; font-size:22px;'> PROMPT:   {user_prompt}</span>", unsafe_allow_html=True)
        st.write(f"<span style='color:blue; font-size:22px;'> Answer:   {output.content}</span>", unsafe_allow_html=True)
    
    