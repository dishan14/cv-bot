import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1 Load docs
loader1 = PyPDFLoader("rag-cv-bot/data/Dishan_Shukla_new_cv.pdf") 
loader2 = TextLoader("rag-cv-bot/data/job_descriptions.txt")
docs = loader1.load() + loader2.load()

# 2 Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3 Embed + Store
embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embed, persist_directory="db")
vectordb.persist()

# 4 LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.3, groq_api_key=st.secrets["GROQ"])

# 5 Prompt
template = """You are an expert recruiter. Use ONLY the CV and JDs below to answer.
Context: {context}
Question: {question}
Answer in 3 bullets, max 60 words."""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 4}), chain_type_kwargs={"prompt": PROMPT})

# Streamlit UI
st.title("🤖 Why Hire Me?")
jd = st.text_area("Paste Job Description")
if st.button("Ask Bot"):
    ans = qa.run(f"Using the CV, explain why I match this JD: {jd}")
    st.write(ans)