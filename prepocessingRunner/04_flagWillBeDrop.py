from sqlalchemy import text
from libInternal.db import get_mysql_engine

TABLES = [
    "scenario9",
    "scenario10",
    "scenario11",
    "scenario12",
]

REQUIRED_COLS = [
    "SrcAddr", "DstAddr", "Dir", "Proto", "Dur",
    "TotBytes", "TotPkts", "Label", "label_as_bot", "dir_clean"
]

def ensure_column(engine, table):
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE `{table}`
            ADD COLUMN IF NOT EXISTS will_be_drop TINYINT(1) DEFAULT 0;
        """))

def update_will_be_drop(engine, table):
    null_conditions = []

    for col in REQUIRED_COLS:
        null_conditions.append(f"`{col}` IS NULL")

        if col not in ("Dur", "TotBytes", "TotPkts", "label_as_bot"):
            null_conditions.append(f"`{col}` = ''")

    condition_sql = " OR ".join(null_conditions)

    sql = f"""
        UPDATE `{table}`
        SET will_be_drop = CASE
            WHEN {condition_sql}
            THEN 1
            ELSE 0
        END;
    """

    with engine.begin() as conn:
        conn.execute(text(sql))
        print(f"  will_be_drop updated in {table}")

def main():
    engine = get_mysql_engine()

    for table in TABLES:
        print(f"processing table: {table}")
        ensure_column(engine, table)
        update_will_be_drop(engine, table)

    print("task: will_be_drop flagging completed")

if __name__ == "__main__":
    main()
