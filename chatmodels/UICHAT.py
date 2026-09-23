import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage
)
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Sad AI Agent",
    page_icon="😔",
    layout="centered"
)

st.title("😔 Sad AI Agent")
st.caption("A conversational AI with a slightly depressed personality.")


# -----------------------------
# Initialize model
# -----------------------------
@st.cache_resource
def load_model():
    return ChatGroq(
        model_name="openai/gpt-oss-120b"
    )


model = load_model()


# -----------------------------
# Initialize chat history
# -----------------------------
if "messages" not in st.session_state:

    st.session_state.messages = [
        SystemMessage(
            content="You are a sad depressed AI agent."
        )
    ]


# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(message.content)


# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Talk to the sad AI...")


if prompt:

    # Add user message
    human_message = HumanMessage(content=prompt)

    st.session_state.messages.append(human_message)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking about life... 😔"):

            response = model.invoke(
                st.session_state.messages
            )

            answer = response.content

            st.markdown(answer)

    # Save AI response
    st.session_state.messages.append(
        AIMessage(content=answer)
    )