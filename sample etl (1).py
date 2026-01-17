# Databricks notebook source
bronze_df = spark.read.format("delta") \
    .load("/Volumes/workspace/default/raw_people_1000/")


# COMMAND ----------

display(bronze_df)

# COMMAND ----------

from pyspark.sql.functions import col, trim, to_date

silver_df = (
    bronze_df
    .drop("index")
    .withColumnRenamed("user_id", "user_id")
    .withColumn("first_name", trim(col("first_name")))
    .withColumn("last_name", trim(col("last_name")))
    .withColumn("sex", col("sex"))
    .withColumn("email", col("email"))
    .withColumn("phone", col("phone"))
    .withColumn(
        "date_of_birth",
        to_date(col("date_of_birth"), "yyyy-MM-dd")
    )
    .withColumn("job_title", col("job_title"))
)


# COMMAND ----------

display(silver_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS workspace.default.silver_people_1000;
# MAGIC

# COMMAND ----------

silver_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.default.silver_people_1000")


# COMMAND ----------

from pyspark.sql.functions import count

gold_gender_df = (
    spark.table("workspace.default.silver_people_1000")
    .groupBy("sex")
    .agg(count("*").alias("total_people"))
)


# COMMAND ----------

gold_job_df = (
    spark.table("workspace.default.silver_people_1000")
    .groupBy("job_title")
    .agg(count("*").alias("total_people"))
    .orderBy(col("total_people").desc())
)


# COMMAND ----------

gold_gender_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.default.gold_people_by_gender")

gold_job_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.default.gold_people_by_job")


# COMMAND ----------



# COMMAND ----------

display(gold_job_df)

# COMMAND ----------

display(gold_gender_df)

# COMMAND ----------


