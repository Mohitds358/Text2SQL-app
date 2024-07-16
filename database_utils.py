from langchain_community.utilities import SQLDatabase
import pandas as pd
import ast


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def extract_schema(db: SQLDatabase) -> pd.DataFrame:
    query = """
    SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
    """
    result = db.run(query)

    # Print the type and content of result for debugging
    print(f"Type of result: {type(result)}")
    print(f"Content of result: {result}")

    # Ensure result is a list of tuples
    if isinstance(result, str):
        try:
            result = ast.literal_eval(result)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Error parsing result string: {e}")

    if isinstance(result, list) and all(isinstance(item, tuple) for item in result):
        print("Result is a list of tuples, creating DataFrame.")
        try:
            df = pd.DataFrame(result, columns=["TABLE_NAME", "COLUMN_NAME", "COLUMN_TYPE"])
            print("DataFrame created successfully.")
            print(df.head())  # Print the first few rows of the DataFrame for verification
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            raise e
    else:
        raise ValueError(f"Unexpected result format: {result}")
