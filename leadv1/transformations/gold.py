import dlt
from pyspark.sql.functions import monotonically_increasing_id, lit, count, sum, avg, col

@dlt.table(
    name="dim_lead",
    comment="Gold layer - Lead dimension with surrogate keys",
    table_properties={
        "quality": "gold",
        "layer": "dimension"
    }
)
def build_dim_lead():
    """
    Lead dimension table with unique lead information
    Deduplication ensures one record per unique lead
    """
    df_silver = dlt.read("leadclean")
    
    return (
        df_silver
        .select("account_id", "first_name", "last_name", "email_1", "phone_1")
        .dropDuplicates()
        .withColumn("lead_id", monotonically_increasing_id())
    )

@dlt.table(
    name="dim_owner",
    comment="Gold layer - Lead owner dimension",
    table_properties={
        "quality": "gold",
        "layer": "dimension"
    }
)
def build_dim_owner():
    """Owner/sales rep dimension"""
    df_silver = dlt.read("leadclean")
    
    return (
        df_silver
        .select("lead_owner")
        .dropDuplicates()
        .withColumn("owner_id", monotonically_increasing_id())
    )

@dlt.table(
    name="dim_company",
    comment="Gold layer - Company dimension",
    table_properties={
        "quality": "gold",
        "layer": "dimension"
    }
)
def build_dim_company():
    """Company dimension for lead companies"""
    df_silver = dlt.read("leadclean")
    
    return (
        df_silver
        .select("company")
        .dropDuplicates()
        .withColumn("company_id", monotonically_increasing_id())
    )

@dlt.table(
    name="dim_source",
    comment="Gold layer - Lead source dimension",
    table_properties={
        "quality": "gold",
        "layer": "dimension"
    }
)
def build_dim_source():
    """Source channel dimension (web, referral, etc.)"""
    df_silver = dlt.read("leadclean")
    
    return (
        df_silver
        .select("source")
        .dropDuplicates()
        .withColumn("source_id", monotonically_increasing_id())
    )

@dlt.table(
    name="dim_stage",
    comment="Gold layer - Deal stage dimension",
    table_properties={
        "quality": "gold",
        "layer": "dimension"
    }
)
def build_dim_stage():
    """Sales pipeline stage dimension"""
    df_silver = dlt.read("leadclean")
    
    return (
        df_silver
        .select("deal_stage")
        .dropDuplicates()
        .withColumn("stage_id", monotonically_increasing_id())
    )

# ============================================================================
# FACT TABLE
# ============================================================================

@dlt.table(
    name="fact_leads",
    comment="Gold layer - Central fact table with all dimension foreign keys",
    table_properties={
        "quality": "gold",
        "layer": "fact"
    }
)
def build_fact_leads():
    """
    Star schema fact table
    Links all dimensions and contains measurable metrics
    """
    df_silver = dlt.read("leadclean")
    dim_lead = dlt.read("dim_lead")
    dim_owner = dlt.read("dim_owner")
    dim_company = dlt.read("dim_company")
    dim_source = dlt.read("dim_source")
    dim_stage = dlt.read("dim_stage")
    
    return (
        df_silver
        .join(dim_lead, ["account_id", "first_name", "last_name", "email_1", "phone_1"])
        .join(dim_owner, ["lead_owner"])
        .join(dim_company, ["company"])
        .join(dim_source, ["source"])
        .join(dim_stage, ["deal_stage"])
        .select(
            "lead_id",
            "owner_id",
            "company_id",
            "source_id",
            "stage_id",
            lit(1).alias("lead_count"),
            col("ingestion_time")  # Track when lead was ingested
        )
    )

# ============================================================================
# AGGREGATE/SUMMARY TABLES
# ============================================================================

@dlt.table(
    name="agg_leads_by_source",
    comment="Gold layer - Pre-aggregated metrics by lead source for fast analytics"
)
def aggregate_by_source():
    """
    Summary table: Lead counts and metrics by source
    Optimized for BI dashboards and reporting
    """
    fact = dlt.read("fact_leads")
    dim_source = dlt.read("dim_source")
    
    return (
        fact
        .join(dim_source, "source_id")
        .groupBy("source")
        .agg(
            count("lead_count").alias("total_leads"),
            sum("lead_count").alias("sum_leads")
        )
        .orderBy(col("total_leads").desc())
    )

@dlt.table(
    name="agg_leads_by_stage",
    comment="Gold layer - Lead pipeline metrics by deal stage"
)
def aggregate_by_stage():
    """Pipeline funnel analysis by stage"""
    fact = dlt.read("fact_leads")
    dim_stage = dlt.read("dim_stage")
    
    return (
        fact
        .join(dim_stage, "stage_id")
        .groupBy("deal_stage")
        .agg(
            count("lead_count").alias("total_leads"),
            sum("lead_count").alias("sum_leads")
        )
        .orderBy(col("total_leads").desc())
    )

@dlt.table(
    name="agg_leads_by_owner",
    comment="Gold layer - Sales performance metrics by owner/rep"
)
def aggregate_by_owner():
    """Owner performance leaderboard"""
    fact = dlt.read("fact_leads")
    dim_owner = dlt.read("dim_owner")
    
    return (
        fact
        .join(dim_owner, "owner_id")
        .groupBy("lead_owner")
        .agg(
            count("lead_count").alias("total_leads"),
            sum("lead_count").alias("sum_leads")
        )
        .orderBy(col("total_leads").desc())
    )

@dlt.table(
    name="agg_leads_by_company",
    comment="Gold layer - Lead volume by company"
)
def aggregate_by_company():
    """Company-level lead aggregation"""
    fact = dlt.read("fact_leads")
    dim_company = dlt.read("dim_company")
    
    return (
        fact
        .join(dim_company, "company_id")
        .groupBy("company")
        .agg(
            count("lead_count").alias("total_leads")
        )
        .orderBy(col("total_leads").desc())
    )

# ============================================================================
# CROSS-DIMENSIONAL ANALYTICS
# ============================================================================

@dlt.table(
    name="agg_leads_source_stage_matrix",
    comment="Gold layer - Lead conversion funnel by source and stage"
)
def aggregate_source_stage_matrix():
    fact = dlt.read("fact_leads")
    dim_source = dlt.read("dim_source")
    dim_stage = dlt.read("dim_stage")
    return (
        fact
        .join(dim_source, "source_id")
        .join(dim_stage, "stage_id")
        .groupBy("source", "deal_stage")
        .agg(
            count("lead_count").alias("total_leads")
        )
        .orderBy("source", col("total_leads").desc())
    )

@dlt.table(
    name="agg_leads_owner_source",
    comment="Gold layer - Owner performance by lead source"
)
def aggregate_owner_source():
    fact = dlt.read("fact_leads")
    dim_owner = dlt.read("dim_owner")
    dim_source = dlt.read("dim_source")
    
    return (
        fact
        .join(dim_owner, "owner_id")
        .join(dim_source, "source_id")
        .groupBy("lead_owner", "source")
        .agg(
            count("lead_count").alias("total_leads")
        )
        .orderBy("lead_owner", col("total_leads").desc())
    )

@dlt.table(
    name="executive_summary",
    comment="Gold layer - High-level KPIs for executive dashboard"
)
def build_executive_summary():
    fact = dlt.read("fact_leads")
    dim_source = dlt.read("dim_source")
    dim_stage = dlt.read("dim_stage")
    dim_owner = dlt.read("dim_owner")
    
    return (
        fact
        .agg(
            count("lead_count").alias("total_leads"),
            count(col("source_id")).alias("total_sources"),
            count(col("stage_id")).alias("total_stages"),
            count(col("owner_id")).alias("total_owners")
        )
    )
