from sqlalchemy import text
from libInternal.db import get_mysql_engine

TABLES = ["scenario9", "scenario10", "scenario11", "scenario12"]

FEATURE_COLUMNS = {
    "featureEng_ByteRatio": "DOUBLE",
    "featureEng_DurationRate": "DOUBLE",
    "featureEng_FlowIntensity": "DOUBLE",
    "featureEng_PktByteRatio": "DOUBLE",
    "featureEng_SrcByteRatio": "DOUBLE",
    "featureEng_TrafficBalance": "DOUBLE",
    "featureEng_DurationPerPkt": "DOUBLE",
    "featureEng_Intensity": "DOUBLE",
}

def ensure_columns(engine, table):
    with engine.begin() as conn:
        for col, dtype in FEATURE_COLUMNS.items():
            conn.execute(text(f"""
                ALTER TABLE `{table}`
                ADD COLUMN IF NOT EXISTS `{col}` {dtype};
            """))

def update_features(engine, table):
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE `{table}`
            SET
                featureEng_ByteRatio =
                    TotBytes / (TotPkts + 1),

                featureEng_DurationRate =
                    TotPkts / (Dur + 0.1),

                featureEng_FlowIntensity =
                    SrcBytes / (TotBytes + 1),

                featureEng_PktByteRatio =
                    TotPkts / (TotBytes + 1),

                featureEng_SrcByteRatio =
                    SrcBytes / (TotBytes + 1),

                featureEng_TrafficBalance =
                    ABS(sTos - dTos),

                featureEng_DurationPerPkt =
                    Dur / (TotPkts + 1),

                featureEng_Intensity =
                    TotBytes / (Dur + 1)
            WHERE will_be_drop = 0;
        """))

        print(f"  feature engineering updated for {table}")

def main():
    engine = get_mysql_engine()

    for table in TABLES:
        print(f"processing table: {table}")
        ensure_columns(engine, table)
        update_features(engine, table)

    print("task: feature engineering completed")

if __name__ == "__main__":
    main()
