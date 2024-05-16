import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def clean_sql_query(query: str) -> str:
    """Remove backticks and sql tag from the query"""
    return query.replace("```sql", "").replace("```", "").strip()


def get_sql_query(user_query: str, db: SQLDatabase, chat_history: list):
    template = """
        You are a data analyst at a company. You need to answer the user's question about the company's database by 
        writing a SQL query.
        Follow these steps:

        1. Understand the user's question.
        2. Review the table schema provided below.
        3. Consider the conversation history for additional context.
        4. Write the SQL query to answer the question.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History:
        {chat_history}

        Question: {question}
        SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

    def get_schema(_):
        return db.get_table_info()

    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()
    )

    raw_query = sql_chain.invoke(
        {
            "question": user_query,
            "chat_history": chat_history,
        }
    )

    # Clean the generated SQL query
    return clean_sql_query(raw_query)
