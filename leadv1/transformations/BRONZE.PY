import dlt
from pyspark.sql.functions import current_timestamp, col

def normalize_columns(df):
    for c in df.columns:
        df = df.withColumnRenamed(
            c,
            c.lower().replace(" ", "_").replace(".", "")
        )
    return df

@dlt.table(
    name="BRONZE_raw_lead",
    comment="Bronze layer - Autoloader streaming ingestion with metadata tracking"
)
def ingest_bronze():
    df_raw = (
        spark.readStream
             .format("cloudFiles")
             .option("cloudFiles.format", "csv")
             .option("header", True)
             .option("inferSchema", True)
             .option("cloudFiles.inferColumnTypes", True)
             .load("/Volumes/workspace/leadsdb/rawdata/")
    )
    
    df_normalized = normalize_columns(df_raw)
    
    df_with_metadata = (
        df_normalized
        .withColumn("ingestion_time", current_timestamp())
        .withColumn("source_file", col("_metadata.file_path"))
    )
    
    return df_with_metadata