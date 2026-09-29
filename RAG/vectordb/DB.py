from langchain_community.vectorstores import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
data=TextLoader('RAG/document loaders/deep.txt')
docs=data.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
chunks=splitter.split_documents(docs)
embedding_model = VoyageAIEmbeddings(
    model="voyage-3-lite",
    #voyage_api_key=os.getenv("VOYAGE_API_KEY")
)
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory='chroma-db'
)
result=vectorstore.similarity_search("What is stochastic gradient descent ?",k=2)
for r in result:
    print(r)
print ("Now we are about to go for retriever")
retriever=vectorstore.as_retriever()
x=retriever.invoke("What is loss function")
for i in x:
    print(i)

