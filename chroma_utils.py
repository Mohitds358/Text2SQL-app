from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd


def load_schema_to_chroma(schema_df: pd.DataFrame, chroma_dir: str):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [
        Document(
            page_content=f"Table: {row['TABLE_NAME']}, Column: {row['COLUMN_NAME']}, Type: {row['COLUMN_TYPE']}",
            metadata={"table": row['TABLE_NAME'], "column": row['COLUMN_NAME'], "type": row['COLUMN_TYPE']}
        )
        for _, row in schema_df.iterrows()
    ]
    db = Chroma.from_documents(documents, embedding_function, persist_directory=chroma_dir)
    db.persist()
    return db


def get_full_schema(schema_df: pd.DataFrame) -> str:
    return "\n".join(
        [f"Table: {row['TABLE_NAME']}, Columns: {row['COLUMN_NAME']}" for _, row in schema_df.iterrows()]
    )


def query_relevant_schema(db, user_query, schema_df: pd.DataFrame, k=5):
    # Always return the full schema as a single string
    full_schema_str = get_full_schema(schema_df)
    return full_schema_str
