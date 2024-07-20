import time
import pandas as pd
from database_utils import init_database, extract_schema
from generate_response import get_response
from chroma_utils import load_schema_to_chroma
import os


def extract_sql_and_answer(response):
    try:
        sql_query_start = response.find("SQL Query:\n") + len("SQL Query:\n")
        sql_query_end = response.find("\n\nAnswer:", sql_query_start)
        sql_query = response[sql_query_start:sql_query_end].strip()

        answer_start = response.find("Answer:\n", sql_query_end) + len("Answer:\n")
        answer = response[answer_start:].strip()

        return sql_query, answer
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Full response: {response}")
        return "", f"Error parsing response: {str(e)}"


def main():
    # Load test queries
    test_queries = pd.read_csv("P&E Prompt Test/Medium_TestPE.csv")

    # Initialize database connection
    db = init_database(
        user="root",
        password="mohit",
        host="localhost",
        port="3306",
        database="mimiciiiv14",
    )

    # Extract schema and load into Chroma
    schema_df = extract_schema(db)
    chroma_dir = os.path.join(os.getcwd(), "chroma_db")
    chroma_db = load_schema_to_chroma(schema_df, chroma_dir)

    # Prompt user to select the prompting technique
    techniques = ["base", "cot", "react", "plan_and_execute"]
    print("Select the prompting technique:")
    for i, technique in enumerate(techniques, 1):
        print(f"{i}. {technique}")
    choice = int(input("Enter the number of your choice: "))
    selected_technique = techniques[choice - 1]

    # Process each query
    for index, row in test_queries.iterrows():
        user_query = row["NLQ"]
        chat_history = []

        try:
            # Get the response from the Text2SQL application
            response = get_response(user_query, db, chat_history, selected_technique, chroma_db, schema_df)
            sql_query, answer = extract_sql_and_answer(response)

            # Print the full response to console
            print(f"\nProcessed query {index + 1}/{len(test_queries)}:")
            print(f"NLQ: {user_query}")
            print(f"Response:\n{response}\n")

            # Update the dataframe with results
            test_queries.at[index, 'SQL Query'] = sql_query
            test_queries.at[index, 'Answer'] = answer

            print(f"Stored SQL Query: {sql_query}")
            print(f"Stored Answer: {answer}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            test_queries.at[index, 'SQL Query'] = ""
            test_queries.at[index, 'Answer'] = f"Error: {str(e)}"

        # Save the updated dataframe after each query
        test_queries.to_csv("P&E Prompt Test/Medium_TestPE.csv", index=False)

        # Sleep for 1 minute between queries
        if index < len(test_queries) - 1:
            print("Waiting for 1 minute before next query...")
            time.sleep(60)

    print("Testing completed!")


if __name__ == "__main__":
    main()