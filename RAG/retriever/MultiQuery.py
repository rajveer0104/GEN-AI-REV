from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq



load_dotenv()
data=TextLoader('RAG/document loaders/deep.txt')
docs = data.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10)
chunks=splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='text_db1'
)

retriever = vectorstore.as_retriever()


llm = ChatGroq(model_name="openai/gpt-oss-120b")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)