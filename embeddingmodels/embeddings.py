from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)
vector=embeddings.embed_query("How are you my good sir ")
print(len(vector))
print(vector)
