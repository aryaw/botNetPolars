import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import re
from libInternal.db import get_mysql_engine
from sqlalchemy import text, bindparam

bot_pattern = re.compile(
    r"\b(bot|botnet|cnc|c&c|malware|infected|attack|spam|ddos|trojan|worm|zombie|backdoor)\b",
    re.IGNORECASE,
)

TABLES = [
    # "scenario9",
    # "scenario10",
    "scenario11",
    "scenario12",
]

def ensure_column(engine, table):
    with engine.begin() as conn:

        exists = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = 'label_as_bot'
            """),
            {"table": table}
        ).scalar()

        if exists == 1:
            conn.execute(
                text(f"""
                    ALTER TABLE `{table}`
                    DROP COLUMN `label_as_bot`
                """)
            )

        conn.execute(
            text(f"""
                ALTER TABLE `{table}`
                ADD COLUMN `label_as_bot` TINYINT(1) DEFAULT 0
            """)
        )

def update_labels(engine, table):
    with engine.begin() as conn:

        rows = conn.execute(
            text(f"SELECT id, Label FROM `{table}`")
        ).fetchall()

        print(f"-- setTable {table}")

        updates = []
        for r in rows:
            label = r.Label or ""
            is_bot = 1 if bot_pattern.search(label) else 0

            updates.append({
                "id": r.id,
                "label_as_bot": is_bot
            })

            if r.id == 53988:
                print(f"-- checkLabel: {label}")
                print(f"-- isBot: {is_bot}")
                print(f"id: {r.id}, label_as_bot: {is_bot}")
            
            conn.execute(
                text(f"""
                    UPDATE `{table}`
                    SET label_as_bot = :label_as_bot
                    WHERE id = :id
                """),
                {
                    "id": r.id,
                    "label_as_bot": is_bot
                }
            )

        print(f"-- updated {len(rows)} rows")

        # stmt = text(f"""
        #     UPDATE `{table}`
        #     SET label_as_bot = :label_as_bot
        #     WHERE id = :id
        # """).bindparams(
        #     bindparam("id"),
        #     bindparam("label_as_bot")
        # )
        # conn.execute(stmt, updates)
        # print(f"-- updated {len(updates)} rows")

def main():
    engine = get_mysql_engine()

    for table in TABLES:
        print(f"processing: {table}")
        ensure_column(engine, table)
        update_labels(engine, table)

    print("label_as_bot completed")


if __name__ == "__main__":
    main()
