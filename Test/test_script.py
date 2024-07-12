import time
import pandas as pd
from database_utils import init_database
from generate_response import get_response


def extract_sql_and_answer(response):
    try:
        # Split the response to extract SQL query and answer
        sql_query_part = response.split("SQL Query:\n\n")[1].split("\n\nAnswer:\n\n")[0].strip()
        answer_part = response.split("\n\nAnswer:\n\n")[1].strip()
        return sql_query_part, answer_part
    except Exception as e:
        return "", f"Error parsing response: {str(e)}"


def main():
    test_queries = pd.read_csv("test_queries3.csv")

    # Initialize database connection
    db = init_database(
        user="root",
        password="mohit",
        host="localhost",
        port="3306",
        database="mimiciiiv14",
    )

    # Prompt user to select the prompting technique
    techniques = ["base", "cot", "react", "least_to_most"]
    print("Select the prompting technique:")
    for i, technique in enumerate(techniques, 1):
        print(f"{i}. {technique}")
    choice = int(input("Enter the number of your choice: "))
    selected_technique = techniques[choice - 1]

    # Process each query
    for index, row in test_queries.iterrows():
        user_query = row["user_query"]
        chat_history = []

        try:
            # Get the response and the SQL query from the Text2SQL application
            response = get_response(user_query, db, chat_history, selected_technique)
            sql_query, answer = extract_sql_and_answer(response)
        except Exception as e:
            sql_query = ""
            answer = f"Error: {str(e)}"

        test_queries.at[index, 'generated_sql_query'] = sql_query
        test_queries.at[index, 'generated_answer'] = answer

        # Sleep to avoid rate limit issues
        time.sleep(60)

    # Save the updated dataframe to the same CSV file
    test_queries.to_csv("test_queries3.csv", index=False)


if __name__ == "__main__":
    main()
    print("Testing completed!!")
