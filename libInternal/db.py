import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

def get_mysql_engine():
    load_dotenv()

    db_host = os.getenv("DB_HOST", "localhost")
    db_user = os.getenv("DB_USER", "root")
    db_pass = quote_plus(os.getenv("DB_PASS", ""))
    db_name = os.getenv("DB_NAME")

    if not db_name:
        raise RuntimeError("DB_NAME is not set in .env")

    engine = create_engine(
        f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}",
        pool_pre_ping=True,
    )
    return engine
