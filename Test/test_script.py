import time
import pandas as pd
from database_utils import init_database
from generate_response import get_response


def main():
    test_queries = pd.read_csv("test_queries2.csv")

    db = init_database(
        user="root",
        password="mohit",
        host="localhost",
        port="3306",
        database="mimiciiiv14",
    )

    for index, row in test_queries.iterrows():
        user_query = row["user_query"]
        chat_history = []

        try:
            # Get the response and the SQL query from the Text2SQL application
            response = get_response(user_query, db, chat_history)

            # Print the results
            print(f"Query: {user_query}\nResponse: {response}\n")
        except Exception as e:
            print(f"Query: {user_query}\nError: {str(e)}\n")

        time.sleep(60)


if __name__ == "__main__":
    main()
    print("Testing completed!!")
