from langchain_community.utilities import SQLDatabase
from techniques.base_sql import get_sql_query as get_sql_query_base, get_response as get_response_base
from techniques.cot_sql import get_sql_query as get_sql_query_cot, get_response as get_response_cot
from techniques.react_sql import get_sql_query as get_sql_query_react, get_response as get_response_react
from techniques.least_to_most_sql import get_sql_query as get_sql_query_least_to_most, \
    get_response as get_response_least_to_most


def get_sql_query(user_query: str, db: SQLDatabase, chat_history: list, technique: str):
    if technique == "base":
        return get_sql_query_base(user_query, db, chat_history)
    elif technique == "cot":
        return get_sql_query_cot(user_query, db, chat_history)
    elif technique == "react":
        return get_sql_query_react(user_query, db, chat_history)
    elif technique == "least_to_most":
        return get_sql_query_least_to_most(user_query, db, chat_history)
    else:
        raise ValueError("Invalid technique specified")


def get_response(user_query: str, db: SQLDatabase, chat_history: list, technique: str):
    if technique == "base":
        return get_response_base(user_query, db, chat_history)
    elif technique == "cot":
        return get_response_cot(user_query, db, chat_history)
    elif technique == "react":
        return get_response_react(user_query, db, chat_history)
    elif technique == "least_to_most":
        return get_response_least_to_most(user_query, db, chat_history)
    else:
        raise ValueError("Invalid technique specified")
