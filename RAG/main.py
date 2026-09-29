from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vectorstore=Chroma(
    persist_directory='chroma_db',
    embedding_function=embedding_model
)
model=ChatGroq(model_name="openai/gpt-oss-120b")
retriever=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k':4,
        'fetch_k':10,
        "lambda_mult":0.5
    }
)
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","""
You are a helpful AI assistant , answer only from the context given .
If the context is not given state that you could not answer it based on context 


"""),
(
    "human","context:{context}"         "Question:{question}"
)
    ]
)
print("Rag system created ")
print("press 0 to exit")
while True:
    query=input("You:")
    if(query=="0"):
        break
    docs=retriever.invoke(query)
    context="\n\n ".join(doc.page_content for doc in docs)
    final_prompt=prompt.invoke({"context":context,"question":query})
    response=model.invoke(final_prompt)
    print("Model:",response.content)
