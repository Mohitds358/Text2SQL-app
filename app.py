from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from database_utils import init_database, extract_schema
from generate_response import get_response
from chroma_utils import load_schema_to_chroma
import os
import pandas as pd

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

if "technique" not in st.session_state:
    st.session_state.technique = "base"

st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title("Chat with MySQL")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")

    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="mohit", key="Password")
    st.text_input("Database", value="mimiciiiv14", key="Database")

    chroma_dir = os.path.join(os.getcwd(), "chroma_db")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            schema_df = extract_schema(db)
            st.session_state.schema_df = schema_df
            st.session_state.chroma_db = load_schema_to_chroma(schema_df, chroma_dir)
            st.success("Connected to database!")

st.subheader("Choose Prompt Technique")
if st.button("Base Prompt"):
    st.session_state.technique = "base"
if st.button("Chain of Thought (CoT)"):
    st.session_state.technique = "cot"
if st.button("ReAct"):
    st.session_state.technique = "react"
if st.button("Least to Most"):
    st.session_state.technique = "least_to_most"

st.write(f"Current Prompt Technique: {st.session_state.technique}")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(
            user_query, st.session_state.db, st.session_state.chat_history, st.session_state.technique, st.session_state.chroma_db, st.session_state.schema_df
        )
        st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))
