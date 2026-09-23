from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
load_dotenv()
model=ChatGroq(model_name="openai/gpt-oss-120b")
message=[SystemMessage(content="You are a sad depressed AI agent")]
while True:
    prompt=input("You:")
    message.append(HumanMessage(content=prompt))
    if prompt=="0":
        break
    response=model.invoke(message)
    message.append(AIMessage(content=response.content))
    print("Bot:",response.content)

print(message)