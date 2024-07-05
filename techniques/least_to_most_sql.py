# techniques/least_to_most_sql.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def get_sql_query(user_query: str, db: SQLDatabase, chat_history: list):
    template_subtask = """
    You are a data analyst working with a MySQL Database. Your task is to answer the user's question about the company's database by writing a SQL query.

    Break down the task into simpler sub-tasks and solve each one.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History:
    {chat_history}

    Sub-task Question: {question}

    SQL Query:
    """

    prompt_subtask = ChatPromptTemplate.from_template(template_subtask)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

    def get_schema(_):
        return db.get_table_info()

    sub_tasks = break_down_query(user_query)
    sql_queries = []

    for sub_task in sub_tasks:
        sql_chain = (
                RunnablePassthrough.assign(schema=get_schema) | prompt_subtask | llm | StrOutputParser()
        )
        raw_query = sql_chain.invoke(
            {
                "question": sub_task,
                "chat_history": chat_history,
            }
        )
        sql_queries.append(clean_sql_query(raw_query))

    final_query = combine_queries(sql_queries)
    return final_query


def break_down_query(query: str) -> list:
    # Logic to break down the main query into simpler sub-tasks
    return ["sub-task 1", "sub-task 2", "sub-task 3"]


def combine_queries(queries: list) -> str:
    # Logic to combine the results of the sub-tasks into a final query
    return " ".join(queries)


def clean_sql_query(query: str) -> str:
    query = query.replace("```sql", "").replace("```", "").strip()
    query = "\n".join([line for line in query.split("\n") if not line.strip().startswith("--")])
    return query.strip()


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    try:
        sql_query = get_sql_query(user_query, db, chat_history)
        sql_response = db.run(sql_query)

        template = """
        You are a data analyst explaining MySQL query results to a non-technical user.
        Given the MySQL database schema, question, the executed SQL query, and its response, write a clear and informative natural language response. Ensure your response accurately addresses the question and interprets the results in a way that is easy for a non-technical user to understand. If the response is empty or does not directly answer the question, explain why.

        Break down the explanation into simpler sub-tasks and solve each one.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History:
        {chat_history}

        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}

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

        final_response = (
            f"User query:\n\n{user_query}\n\n"
            f"SQL Query:\n\n{sql_query}\n\n"
            f"Answer:\n\n{llm_response}"
        )

        return final_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response"
