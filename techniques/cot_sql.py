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
    template = """
        You are a data analyst working with a MySQL Database. Your task is to answer the user's question about the company's database by writing a clean and executable SQL query. Follow these steps:

        1. **Understand the User's Question:**
           - Carefully read the user's question to understand the required data and operations.
           - Identify the key elements such as tables, columns, conditions, and operations (joins, aggregations, etc.).

        2. **Review the Table Schema Provided:**
           - Analyze the schema to understand the relationships between tables.
           - Note down the relevant tables and columns needed to form the query.

        3. **Plan the SQL Query:**
           - Determine the required operations (e.g., joins, aggregations, filters) to answer the question.
           - Construct the query using appropriate SQL operations to ensure accuracy and complexity.

        4. **Write the SQL Query:**
           - Construct the SQL query step-by-step, ensuring each part logically follows from the previous steps.
           - Verify the query against the schema to ensure all columns and tables are correctly referenced.
           - Ensure the SQL syntax is correct, particularly for functions like COUNT() and TIMESTAMPDIFF().
           - Remove any unnecessary comments or instructions within the query.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History:
        {chat_history}

        Question: {question}

        Step-by-step Thought Process:
        1. Understand the user's question and identify key elements.
        2. Review the schema to determine relevant tables and columns.
        3. Plan the query by breaking it down into manageable parts.
        4. Construct the query step-by-step.
        5. Verify the query against the schema.
        6. Ensure the SQL syntax is correct and remove unnecessary comments.

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

    return clean_sql_query(raw_query)


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

        Think step-by-step through the problem before explaining the response.

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
