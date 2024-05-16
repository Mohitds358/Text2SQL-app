import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign
from langchain_google_genai import ChatGoogleGenerativeAI
from generate_sql_query import get_sql_query
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def get_response(user_query: str, db: SQLDatabase, chat_history: list):

    sql_query = get_sql_query(user_query, db, chat_history)

    sql_response = db.run(sql_query)

    template = """
        You are a data analyst explaining SQL query results to a non-technical user.
        Given the database schema, question, the executed SQL query and its response, write a clear and informative natural language response. Ensure your response accurately addresses the question and interprets the results in a way that is easy for a non-technical user to understand. If the response is empty or does not directly answer the question, explain why.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History:
        {chat_history}

        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}

        Thought:
        1. Summarize the SQL query's purpose.
        2. Explain the key insights from the results.
        3. If relevant, provide additional context or analysis based on the schema.
        4. If the query produced no results, explain why based on the data or query.
        Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

    chain = (
        RunnableAssign(
            {
                "schema": lambda _: db.get_table_info(),
                "query": lambda _: sql_query,
                "response": lambda _: sql_response,
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    llm_response = chain.invoke(
        {
            "question": user_query,
            "chat_history": chat_history,
            "schema": db.get_table_info(),
            "query": sql_query,
            "response": sql_response,
        }
    )

    # Construct the final response
    final_response = (
        f"Question: {user_query}\n\n"
        f"SQL Query: {sql_query}\n\n"
        f"Answer: {llm_response}"
    )

    return final_response
