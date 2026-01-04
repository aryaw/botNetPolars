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

REQUIRED_COLS = [
    "SrcAddr", "DstAddr", "Dir", "Proto", "Dur",
    "TotBytes", "TotPkts", "Label", "label_as_bot", "dir_clean"
]

COLUMN_NAME = "will_be_drop"


def ensure_column(engine, table):
    with engine.begin() as conn:
        exists = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
            """),
            {"table": table, "column": COLUMN_NAME}
        ).scalar()

        if exists == 0:
            conn.execute(
                text(f"""
                    ALTER TABLE `{table}`
                    ADD COLUMN `{COLUMN_NAME}` TINYINT(1) DEFAULT 0
                """)
            )
            print(f"  + column `{COLUMN_NAME}` added to {table}")
        else:
            conn.execute(
                text(f"""
                    ALTER TABLE `{table}`
                    DROP COLUMN `{COLUMN_NAME}`
                """)
            )

            conn.execute(
                text(f"""
                    ALTER TABLE `{table}`
                    ADD COLUMN `{COLUMN_NAME}` TINYINT(1) DEFAULT 0
                """)
            )
            print(f"  = column `{COLUMN_NAME}` already exists in {table}")


def update_will_be_drop(engine, table):
    null_conditions = []

    for col in REQUIRED_COLS:
        null_conditions.append(f"`{col}` IS NULL")

        if col not in ("Dur", "TotBytes", "TotPkts", "label_as_bot"):
            null_conditions.append(f"`{col}` = ''")

    condition_sql = " OR ".join(null_conditions)

    sql = f"""
        UPDATE `{table}`
        SET `{COLUMN_NAME}` = CASE
            WHEN {condition_sql}
            THEN 1
            ELSE 0
        END
    """

    with engine.begin() as conn:
        result = conn.execute(text(sql))
        print(f"  - will_be_drop updated in {table}")


def main():
    engine = get_mysql_engine()

    for table in TABLES:
        print(f"\nprocessing table: {table}")
        ensure_column(engine, table)
        update_will_be_drop(engine, table)

if __name__ == "__main__":
    main()
