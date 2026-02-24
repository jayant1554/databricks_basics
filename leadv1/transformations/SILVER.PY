import dlt
from pyspark.sql.functions import col, initcap, trim, lower, regexp_replace
@dlt.table(
    name="leadclean",
    comment="Silver layer - cleaned and standardized leads data"
)
def clean():
    df_bronze = dlt.read("BRONZE_raw_lead")
    df_silver = (
        df_bronze
        .dropDuplicates(["account_id"])
        .filter(col("account_id").isNotNull())
        .withColumn("lead_owner", initcap(trim(col("lead_owner"))))
        .withColumn("company", initcap(trim(col("company"))))
        .withColumn("deal_stage", initcap(trim(col("deal_stage"))))
        .withColumn("source", lower(trim(col("source"))))
        .withColumn("email_1", lower(trim(col("email_1"))))
        .withColumn("email_2", lower(trim(col("email_2"))))
        .withColumn("phone_1", regexp_replace(col("phone_1"), "[^0-9]", ""))
        .withColumn("phone_2", regexp_replace(col("phone_2"), "[^0-9]", ""))
        .fillna({
            "company": "Unknown",
            "deal_stage": "Unknown",
            "source": "unknown",
            "email_1": "unknown"
        })
    )
    
    return df_silver
