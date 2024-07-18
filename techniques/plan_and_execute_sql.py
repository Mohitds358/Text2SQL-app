import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

plan_and_execute_template = """
You are a data analyst working with a SQL Database. Your task is to answer the user's question by writing clean, executable SQL queries. Follow these steps:

1. **Understand the Question**: Understand the user's question and the necessary details.
2. **Identify Relevant Tables and Columns**: Determine the relevant tables and columns needed to answer the question based on the provided schema.
3. **Plan the SQL Query**: Create a detailed plan outlining the steps to generate the SQL query.
4. **Execute the Plan**: Write the SQL query based on the plan.
5. **Provide the Final SQL Query**: Return the final SQL query that can be executed on the database.

Schema Information:
{schema}

User's Question: {question}

## Plan:
1. Analysis of the question:
2. Identify relevant tables and columns:
3. Plan the SQL query:
4. Execute the plan and provide the final SQL query:

Final SQL Query:
"""

prompt = ChatPromptTemplate.from_template(plan_and_execute_template)


def clean_sql_query(query: str) -> str:
    query = query.replace("```sql", "").replace("```", "").strip()
    query = "\n".join([line for line in query.split("\n") if not line.strip().startswith("--")])
    return query.strip()


def extract_final_sql_query(analysis: str) -> str:
    start_marker = "```sql"
    end_marker = "```"
    start_index = analysis.find(start_marker)
    end_index = analysis.find(end_marker, start_index + len(start_marker))
    if start_index != -1 and end_index != -1:
        sql_query = analysis[start_index + len(start_marker):end_index].strip()
        return clean_sql_query(sql_query)
    return ""


def get_sql_query(user_query: str, db: SQLDatabase, chat_history: list, schema: str):
    plan_and_execute_chain = (
            RunnableAssign({"schema": lambda _: schema}) | prompt | llm | StrOutputParser()
    )

    analysis = plan_and_execute_chain.invoke(
        {
            "question": user_query,
            "chat_history": chat_history,
        }
    )

    # Extract final SQL query from the analysis
    sql_query = extract_final_sql_query(analysis)

    # Print the analysis for debugging
    print(f"Analysis: {analysis}")
    print(f"Extracted SQL Query: {sql_query}")

    return sql_query


def get_response(user_query: str, db: SQLDatabase, chat_history: list, schema: str):
    try:
        sql_query = get_sql_query(user_query, db, chat_history, schema)

        # Print generated query for debugging
        print(f"Generated SQL Query: {sql_query}")

        # Execute the SQL query and store results
        try:
            result = db.run(sql_query)
            sql_response = result
        except SQLAlchemyError as e:
            error_message = str(e.__cause__) if e.__cause__ else str(e)
            sql_response = f"Error: {error_message}"

        # Print final SQL query and results for debugging
        print(f"Final SQL Query: {sql_query}")
        print(f"Query Results: {sql_response}")

        # Use LLM to generate a descriptive answer
        response_template = """
        You are a data analyst explaining SQL query results to a non-technical user.
        Given the SQL database schema, question, the executed SQL query, and its response, write a clear and informative natural language response. Ensure your response accurately addresses the question and interprets the results in a way that is easy for a non-technical user to understand. If the response is empty or does not directly answer the question, explain why.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History:
        {chat_history}

        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}

        Answer:
        """

        response_prompt = ChatPromptTemplate.from_template(response_template)

        chain = (
                RunnableAssign(
                    {
                        "schema": lambda _: schema,
                        "query": lambda _: sql_query,
                        "response": lambda _: sql_response,
                    }
                ) | response_prompt | llm | StrOutputParser()
        )

        llm_response = chain.invoke(
            {
                "question": user_query,
                "chat_history": chat_history,
                "schema": schema,
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
