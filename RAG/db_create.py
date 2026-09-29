#load pdf
#split into chunks
#create the embeddings
#store
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
loader = PyPDFLoader("RAG/document loaders/Mastering Cloud Computing.pdf")
docs = loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
chunks=splitter.split_documents(docs)


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory='chroma_db'
)