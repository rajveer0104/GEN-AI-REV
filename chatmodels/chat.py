
'''from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("google_genai:gemini-3.8-flash")
response = model.invoke("Explain machine learning in simple terms.")

print(response.content)'''

'''from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")
response=model.invoke("How is India explain in brief")
print(response.content)'''

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model=ChatGroq( model="openai/gpt-oss-120b",temperature=0.9,max_tokens=1000)
#temperature high==>means the ai becomes creative,does not only depend on its learned data tries to do guess work
#temperature low==>means the ai becomes logical, depend on its learned data 
#max tokens tell us about the number of max words allowed 
response=model.invoke("Write a short horror story")
print(response.content)