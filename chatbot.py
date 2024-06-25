import os
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Document preprocessing function
def doc_preprocessings():
    loader = DirectoryLoader(
        'content/data',
        glob='**/*.txt',
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

# Embedding and indexing function
@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()
    pc = PineconeClient(
        api_key=PINECONE_API_KEY
    )

    index_name = 'chatbot'
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENV
            )
        )

    docs_split = doc_preprocessings()
    doc_db = Pinecone.from_texts(
        texts=[doc.page_content for doc in docs_split],
        embedding=embeddings,
        index_name=index_name
    )
    return doc_db

# Function to get retrieval answer
def retrieval_answer(query):
    llm = ChatOpenAI()
    doc_db = embedding_db()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever()
    )
    result = qa.run(query)
    return result

# Streamlit app
def main():
    st.title("Chatbot for Personal website")
    text_input = st.text_input("Ask your question...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

if __name__ == "__main__":
    main()
