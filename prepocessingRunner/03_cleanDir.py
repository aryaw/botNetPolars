import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from sqlalchemy import text
from libInternal.db import get_mysql_engine

TABLES = [
    # "scenario9",
    # "scenario10",
    "scenario11",
    "scenario12",
]

def ensure_dir_clean(conn, table):
    exists = conn.execute(
        text("""
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :table
              AND COLUMN_NAME = 'dir_clean'
        """),
        {"table": table}
    ).scalar()

    if exists == 1:
        conn.execute(
            text(f"""
                ALTER TABLE `{table}`
                DROP COLUMN `dir_clean`
            """)
        )

    conn.execute(
        text(f"""
            ALTER TABLE `{table}`
            ADD COLUMN `dir_clean` VARCHAR(10)
        """)
    )

def main():
    engine = get_mysql_engine()

    with engine.begin() as conn:
        for table in TABLES:
            print(f"cleaning Dir field: {table}")
            ensure_dir_clean(conn, table)

            conn.execute(
                text(f"""
                    UPDATE `{table}`
                    SET dir_clean = TRIM(Dir)
                """)
            )

            conn.execute(
                text(f"""
                    UPDATE `{table}`
                    SET dir_clean = '<->'
                    WHERE dir_clean IN ('<?>', '< ?>', '<? >')
                """)
            )

            print(f"dir_clean `{table}` updated")

    print("dir_clean completed")


if __name__ == "__main__":
    main()
