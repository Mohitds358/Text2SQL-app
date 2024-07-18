from langchain_community.utilities import SQLDatabase
from techniques.base_sql import get_sql_query as get_sql_query_base, get_response as get_response_base
from techniques.cot_sql import get_sql_query as get_sql_query_cot, get_response as get_response_cot
from techniques.react_sql import get_sql_query as get_sql_query_react, get_response as get_response_react
from techniques.plan_and_execute_sql import get_sql_query as get_sql_query_plan_and_execute, get_response as get_response_plan_and_execute
from chroma_utils import query_relevant_schema
import pandas as pd


def get_sql_query(user_query: str, db: SQLDatabase, chat_history: list, technique: str, chroma_db, schema_df: pd.DataFrame):
    schema_str = query_relevant_schema(chroma_db, user_query, schema_df)

    if technique == "base":
        return get_sql_query_base(user_query, db, chat_history, schema_str)
    elif technique == "cot":
        return get_sql_query_cot(user_query, db, chat_history, schema_str)
    elif technique == "react":
        return get_sql_query_react(user_query, db, chat_history, schema_str)
    elif technique == "plan_and_execute":
        return get_sql_query_plan_and_execute(user_query, db, chat_history, schema_str)
    else:
        raise ValueError("Invalid technique specified")


def get_response(user_query: str, db: SQLDatabase, chat_history: list, technique: str, chroma_db, schema_df: pd.DataFrame):
    schema_str = query_relevant_schema(chroma_db, user_query, schema_df)

    if technique == "base":
        return get_response_base(user_query, db, chat_history, schema_str)
    elif technique == "cot":
        return get_response_cot(user_query, db, chat_history, schema_str)
    elif technique == "react":
        return get_response_react(user_query, db, chat_history, schema_str)
    elif technique == "plan_and_execute":
        return get_response_plan_and_execute(user_query, db, chat_history, schema_str)
    else:
        raise ValueError("Invalid technique specified")
