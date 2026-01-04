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

EDGE_COLUMNS = {
    "edge_weight": "INT DEFAULT 1",
    "src_total_weight": "INT DEFAULT 0",
    "dst_total_weight": "INT DEFAULT 0",
}

def ensure_columns(engine, table):
    with engine.begin() as conn:
        for col, ddl in EDGE_COLUMNS.items():
            exists = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = :table
                      AND COLUMN_NAME = :column
                """),
                {"table": table, "column": col}
            ).scalar()

            if exists == 0:
                conn.execute(
                    text(f"""
                        ALTER TABLE `{table}`
                        ADD COLUMN `{col}` {ddl}
                    """)
                )
                print(f"  + column `{col}` added")
            else:
                print(f"  = column `{col}` exists")

def update_edge_weight(engine, table):
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE `{table}` t
            JOIN (
                SELECT SrcAddr, DstAddr, COUNT(*) AS ew
                FROM `{table}`
                WHERE will_be_drop = 0
                GROUP BY SrcAddr, DstAddr
            ) e
            ON t.SrcAddr = e.SrcAddr
           AND t.DstAddr = e.DstAddr
            SET t.edge_weight = e.ew
        """))

        conn.execute(text(f"""
            UPDATE `{table}` t
            JOIN (
                SELECT SrcAddr, SUM(edge_weight) AS sw
                FROM `{table}`
                WHERE will_be_drop = 0
                GROUP BY SrcAddr
            ) s
            ON t.SrcAddr = s.SrcAddr
            SET t.src_total_weight = s.sw
        """))

        conn.execute(text(f"""
            UPDATE `{table}` t
            JOIN (
                SELECT DstAddr, SUM(edge_weight) AS dw
                FROM `{table}`
                WHERE will_be_drop = 0
                GROUP BY DstAddr
            ) d
            ON t.DstAddr = d.DstAddr
            SET t.dst_total_weight = d.dw
        """))

        print(f"  edge weights updated for {table}")

def main():
    engine = get_mysql_engine()
    for table in TABLES:
        print(f"\nprocessing table: {table}")
        ensure_columns(engine, table)
        update_edge_weight(engine, table)

if __name__ == "__main__":
    main()
