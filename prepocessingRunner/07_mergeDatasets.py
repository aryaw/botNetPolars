import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from sqlalchemy import text
from libInternal.db import get_mysql_engine

SOURCE_TABLES = [
    "scenario9",
    "scenario10",
    "scenario11",
    "scenario12",
    "sensor3",
]

MERGED_TABLE = "merged_ctu13_ncc2"


def main():
    engine = get_mysql_engine()

    with engine.begin() as conn:
        conn.execute(
            text(f"DROP TABLE IF EXISTS `{MERGED_TABLE}`")
        )

        conn.execute(text(f"""
            CREATE TABLE `{MERGED_TABLE}` (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,

                SrcAddr VARCHAR(64),
                DstAddr VARCHAR(64),

                Proto VARCHAR(32),
                Dir VARCHAR(8),
                dir_clean TINYINT,

                State VARCHAR(32),

                Dur DOUBLE,
                TotBytes BIGINT,
                TotPkts BIGINT,
                sTos INT,
                dTos INT,
                SrcBytes BIGINT,

                -- GRAPH FEATURES
                edge_weight INT,
                src_total_weight BIGINT,
                dst_total_weight BIGINT,

                Label TEXT,
                label_as_bot TINYINT(1),

                source_table VARCHAR(32),

                INDEX idx_src (SrcAddr),
                INDEX idx_dst (DstAddr),
                INDEX idx_bot (label_as_bot),
                INDEX idx_dir_clean (dir_clean)
            )
        """))

        total_rows = 0

        for table in SOURCE_TABLES:
            print(f"[MERGE] {table}")

            result = conn.execute(text(f"""
                INSERT INTO `{MERGED_TABLE}` (
                    SrcAddr, DstAddr,
                    Proto, Dir, dir_clean,
                    State,
                    Dur, TotBytes, TotPkts,
                    sTos, dTos, SrcBytes,

                    edge_weight,
                    src_total_weight,
                    dst_total_weight,

                    Label, label_as_bot,
                    source_table
                )
                SELECT
                    SrcAddr, DstAddr,
                    Proto, Dir, dir_clean,
                    State,
                    Dur, TotBytes, TotPkts,
                    sTos, dTos, SrcBytes,

                    edge_weight,
                    src_total_weight,
                    dst_total_weight,

                    Label, label_as_bot,
                    '{table}'
                FROM `{table}`
                WHERE will_be_drop = 0
                  AND dir_clean IS NOT NULL
            """))

            inserted = result.rowcount or 0
            total_rows += inserted
            print(f"  -> {inserted} rows inserted")

        print(f"\n[DONE] Total merged rows: {total_rows}")

if __name__ == "__main__":
    main()