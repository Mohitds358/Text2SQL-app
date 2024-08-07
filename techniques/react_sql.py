import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

react_template = """
    You are a data analyst working with a SQL Database. Your task is to answer the user's question about the company's database by writing clean and executable SQL queries. Follow the ReAct (Reasoning and Acting) approach:

    1. **Reasoning Step**:
       - Analyze the user's question.
       - Identify the necessary tables and columns.
       - Plan the SQL operations needed (e.g., joins, aggregations).

    2. **Acting Step**:
       - Generate an initial SQL query based on the reasoning.
       - Execute the query to fetch intermediate results.

    3. **Iteration**:
       - Evaluate the intermediate results.
       - Refine the query if necessary.
       - Repeat the process until the final query is accurate.
       - Ensure that the generated query references only valid tables and columns as per the provided schema.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History:
    {chat_history}

    Question: {question}

    Step-by-step Thought Process:
    1. Reasoning Step: Analyze and plan.
    2. Acting Step: Generate and execute initial query.
    3. Iteration: Refine the query based on results.
    4. Validate the query against the schema.

    Provide only the final SQL query:

    SQL Query:
    """

prompt = ChatPromptTemplate.from_template(react_template)


def get_sql_query(user_query: str, db: SQLDatabase, chat_history: list, schema: str):
    sql_chain = (
            RunnableAssign({"schema": lambda _: schema}) | prompt | llm | StrOutputParser()
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


def get_response(user_query: str, db: SQLDatabase, chat_history: list, schema: str):
    try:
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
        sql_query = get_sql_query(user_query, db, chat_history, schema)
        sql_response = db.run(sql_query)

        response_template = """
        You are a data analyst explaining SQL query results to a non-technical user.
        Given the SQL database schema, question, the executed SQL query, and its response, write a clear and informative natural language response. Ensure your response accurately addresses the question and interprets the results in a way that is easy for a non-technical user to understand. If the response is empty or does not directly answer the question, explain why.

        Think step-by-step through the problem before explaining the response.

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
                )
                | response_prompt
                | llm
                | StrOutputParser()
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
