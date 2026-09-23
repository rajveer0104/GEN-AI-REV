from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    repetition_penalty=5,
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)
response=model.invoke("What is bias variance trade off ")
print(response.content)