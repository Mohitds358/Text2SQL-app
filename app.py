import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from database_utils import init_database, extract_schema
from generate_response import get_response
from chroma_utils import load_schema_to_chroma

# Set up the page configuration
st.set_page_config(
    page_title="SQLMate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for improved styling
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: #FFFFFF;
            font-family: 'Roboto', sans-serif;
        }
        .main {
            background-color: #303030;
            color: #FFFFFF;
        }
        .stTextInput>div>div>input {
            border: 1px solid #444;
            padding: 10px;
            border-radius: 5px;
            background-color: #3A3A3A;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #444;
            color: #FFFFFF;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #555;
        }
        .sidebar .sidebar-content {
            background-color: #404040;
        }
        .sidebar .sidebar-content .element-container * {
            color: #FFFFFF !important;
        }
        .css-1d391kg {
            color: #FFFFFF;
        }
        .css-15zrgzn {
            background-color: #303030;
            color: #FFFFFF;
        }
        .css-1v0mbdj a {
            color: #FFA500;
        }
        .css-1v0mbdj a:hover {
            color: #FF6347;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        .chat-container {
            background-color: #2C2C2C;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .stChatMessage {
            background-color: #3A3A3A;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .stChatMessage--human {
            background-color: #4A4A4A;
            border: 1px solid #FFA500;
        }
        .stChatMessage--ai {
            background-color: #3A3A3A;
            border: 1px solid #00CED1;
        }
        h1 {
            color: #FFA500 !important;
        }
        h2, h3 {
            color: #FFA500 !important;
        }
        .stRadio > label {
            color: #FFA500 !important;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        .stRadio > div {
            flex-direction: row;
            gap: 10px;
        }
        .stRadio > div > label {
            background-color: #3A3A3A;
            color: #FFFFFF !important;
            padding: 10px 15px;
            border-radius: 25px;
            cursor: pointer;
            border: 1px solid #FFA500;
            transition: all 0.3s ease;
        }
        .stRadio > div > label:hover {
            background-color: #4A4A4A;
        }
        .stRadio > div > label[data-baseweb="radio"] {
            background-color: #FFA500;
            color: #000000 !important;
        }
        .sidebar .stButton>button {
            background-color: #FFA500;
            color: #000000;
            font-weight: bold;
            border-radius: 25px;
            padding: 10px 20px;
            width: 100%;
        }
        .sidebar .stButton>button:hover {
            background-color: #FF8C00;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

if "technique" not in st.session_state:
    st.session_state.technique = "base"

# Sidebar for settings
with st.sidebar:
    st.markdown("# 🤖 SQLMate")
    st.header("Database Connection")

    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="mohit", key="Password")
    st.text_input("Database", value="MIMICIIIv14", key="Database")

    chroma_dir = os.path.join(os.getcwd(), "chroma_db")

    if st.button("Connect", key="connect_button"):
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

# Main chat interface
st.title("SQLMate 🤖")
st.markdown("Your AI-powered SQL Assistant")

# Prompt technique selector
technique = st.radio(
    "Choose Prompt Technique",
    ["Base Prompt", "Chain of Thought (CoT)", "ReAct", "Plan and Execute"],
    horizontal=True
)

# Update the technique in session state based on the selection
technique_map = {
    "Base Prompt": "base",
    "Chain of Thought (CoT)": "cot",
    "ReAct": "react",
    "Plan and Execute": "plan_and_execute"
}
st.session_state.technique = technique_map[technique]

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content, unsafe_allow_html=True)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# User input and response
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(
            user_query, st.session_state.db, st.session_state.chat_history, st.session_state.technique,
            st.session_state.chroma_db, st.session_state.schema_df
        )
        st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))

# Footer
st.markdown("---")
st.markdown(
    "SQLMate v1.0 | [Report an issue](https://github.com/Mohitds358/Text2SQL-app)"
)
