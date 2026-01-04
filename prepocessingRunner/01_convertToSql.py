import polars as pl
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "dataset" / "src"
SCRIPT_NAME = Path(__file__).stem
BASE_DIST_DIR = PROJECT_ROOT / "dataset" / "dist" / SCRIPT_NAME
BASE_DIST_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1_000_000

def sql_value(val):
    if val is None:
        return "NULL"
    if isinstance(val, float) and math.isnan(val):
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    return "'" + str(val).replace("\\", "\\\\").replace("'", "\\'") + "'"

def write_header(f, table_name):
    f.write(f"DROP TABLE IF EXISTS `{table_name}`;\n")
    f.write(f"""
CREATE TABLE `{table_name}` (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    StartTime TEXT,
    Dur DOUBLE,
    Proto TEXT,
    SrcAddr TEXT,
    Sport TEXT,
    Dir TEXT,
    DstAddr TEXT,
    Dport TEXT,
    State TEXT,
    sTos INT,
    dTos INT,
    TotPkts BIGINT,
    TotBytes BIGINT,
    SrcBytes BIGINT,
    Label TEXT
) ENGINE=InnoDB;
\n""")

def convert_binetflow_to_sql(input_file: Path):
    table_name = input_file.stem.lower()
    print(f"processing: {table_name}")

    table_dist_dir = BASE_DIST_DIR / table_name
    table_dist_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(
        input_file,
        separator=",",
        has_header=True,
        ignore_errors=True,
        try_parse_dates=False
    )

    total_rows = df.height
    print(f"  rows   : {total_rows}")

    part = 1
    start = 0

    while start < total_rows:
        end = min(start + CHUNK_SIZE, total_rows)
        chunk = df.slice(start, end - start)
        output_file = table_dist_dir / f"{table_name}_part{part:03d}.sql"
        print(f"  writing {output_file.relative_to(BASE_DIST_DIR)}")

        with open(output_file, "w", encoding="utf-8") as f:
            if part == 1:
                write_header(f, table_name)

            for row in chunk.iter_rows(named=True):
                values = [
                    sql_value(row.get("StartTime")),
                    sql_value(row.get("Dur")),
                    sql_value(row.get("Proto")),
                    sql_value(row.get("SrcAddr")),
                    sql_value(row.get("Sport")),
                    sql_value(row.get("Dir")),
                    sql_value(row.get("DstAddr")),
                    sql_value(row.get("Dport")),
                    sql_value(row.get("State")),
                    sql_value(row.get("sTos")),
                    sql_value(row.get("dTos")),
                    sql_value(row.get("TotPkts")),
                    sql_value(row.get("TotBytes")),
                    sql_value(row.get("SrcBytes")),
                    sql_value(row.get("Label")),
                ]

                f.write(
                    f"INSERT INTO `{table_name}` "
                    f"(StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,"
                    f"sTos,dTos,TotPkts,TotBytes,SrcBytes,Label)\n"
                    f"VALUES ({','.join(values)});\n"
                )

        start = end
        part += 1

    print(f"  done: {part - 1} chunks created\n")

def main():
    files = sorted(SRC_DIR.glob("*.binetflow"))
    if not files:
        print("no .binetflow files found")
        return

    for file in files:
        convert_binetflow_to_sql(file)

    print(f"converted {len(files)} datasets into grouped chunks")

if __name__ == "__main__":
    main()
