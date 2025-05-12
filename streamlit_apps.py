import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

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
    similarity_threshold = st.slider("Set similarity threshold", 0.0, 1.0, 0.7, 0.01)#Shows a slider labeled "Set similarity threshold" from 0.0 to 1.0
                                 # Default value is 0.7
                                 #Step size is 0.01 (for fine-grained control)
                                 #The user can drag it to set how strict the retrieval filter should be.
    results_with_scores = vector_store.similarity_search_with_score(user_prompt, k=3)

    retriver = vector_store.as_retriever()
    retriver_resultss = retriver.invoke(user_prompt)
    if not  retriver_resultss:  # Works for None or empty list
        st.warning('''I'm sorry, but the provided documents do not
         contain any information relevant to your query.''')
    else:
        top_doc, top_score = results_with_scores[0]
        st.markdown(f"<span style='color:orange;'>Top similarity score: {top_score:.2f}</span>", unsafe_allow_html=True)
        if top_score < similarity_threshold:
            st.warning("I'm sorry, but the similarity score is too low. Skipping LLM.")
            print(f"Skipped prompt due to low score: '{user_prompt}' with score {top_score:.2f}")
        else:
            llm=init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
            promt_create_template=PromptTemplate.from_template('Answer the question based on the context below {retriver_resultss} and question is {user_prompt}')
            chain=promt_create_template | llm
            output=chain.invoke({'retriver_resultss':top_doc.page_content,'user_prompt':user_prompt})
            st.write(f"<span style='color:red; font-size:22px;'> PROMPT:   {user_prompt}</span>", unsafe_allow_html=True)
            st.write(f"<span style='color:blue; font-size:22px;'> Answer:   {output.content}</span>", unsafe_allow_html=True)
        
         # Optionally display all top documents with their scores
        with st.expander("See retrieved documents and scores"):
            for i, (doc, score) in enumerate(results_with_scores):
                st.markdown(f"**Document {i+1} (score: {score:.2f})**")
                st.code(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))
