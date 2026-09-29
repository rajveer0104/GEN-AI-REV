#load pdf
#split into chunks
#create the embeddings
#store
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
loader = TextLoader("RAG/document loaders/deep.txt")
docs = loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks=splitter.split_documents(docs)


embedding_model = VoyageAIEmbeddings(
    model='voyage-3-lite'
)
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory='text_db'
)
#Similarity search
similarity_retriever=vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k':3}
)

print("+++++++++ Similarity Search results+++++++++++")
similarity_docs=similarity_retriever.invoke("what is gradient descent ?")
for doc in similarity_docs:
    print(doc.page_content)

#max marginal retriever

maxmarginal_retriever=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3}
)

print("+++++++++ MMR results+++++++++++")
mmr_docs=maxmarginal_retriever.invoke("what is gradient descent ?")
for doc in mmr_docs:
    print(doc.page_content)