# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %pip install databricks-genai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------

from databricks.model_training import foundation_model as fm

from finreganalytics.utils import setup_logging, get_dbutils, get_current_cluster_id, get_user_name

setup_logging()

uc_target_catalog = "dpechi"
uc_target_schema = "test"

if (locals().get("uc_target_catalog") is None
        or locals().get("uc_target_schema") is None):
    uc_target_catalog = get_user_name()
    uc_target_schema = get_user_name()

supported_models = fm.get_models().to_pandas()["name"].to_list()
get_dbutils().widgets.combobox(
    "base_model", "meta-llama/Meta-Llama-3-8B-Instruct", supported_models, "base_model"
)

get_dbutils().widgets.text("training_duration", "1ep", "training_duration")
get_dbutils().widgets.text("learning_rate", "1e-6", "learning_rate")
get_dbutils().widgets.text(
    "custom_weights_path",
    "",
    "custom_weights_path",
)

# COMMAND ----------

base_df = spark.read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset")  # noqa

# COMMAND ----------

base_df = base_df.select("context", "question", "answer")
display(base_df)

# COMMAND ----------

import pyspark.sql.functions as F

# Renaming columns of the second DataFrame
base_df_2 = base_df.select([F.col(c).alias(f"{c}_distract") for c in base_df.columns])

# Performing cross join
result_df = base_df.crossJoin(base_df_2).filter(F.col('context_distract') != F.col('context'))

# Aggregating to include 3 random rows for each question
result_df = result_df.withColumn("rand", F.rand()).groupBy("question", "context", "answer").agg(F.collect_list(F.struct("*")).alias("rows"))
result_df = result_df.withColumn("rows", F.expr("slice(array_sort(rows, (x, y) -> case when x.rand < y.rand then -1 when x.rand > y.rand then 1 else 0 end), 1, 3)"))

# Rotating the 3 rows to be 3 columns for each question and extracting context_distract
result_df = result_df.select("context", 
    "question", "answer",
    F.col("rows")[0].alias("row1"),
    F.col("rows")[1].alias("row2"),
    F.col("rows")[2].alias("row3")
).select("context", "question", "answer",
    F.col("row1.context_distract").alias("context_distract1"),
    F.col("row2.context_distract").alias("context_distract2"),
    F.col("row3.context_distract").alias("context_distract3")
)

# Transforming the context column
result_df = result_df.withColumn("rand", F.rand())
result_df = result_df.withColumn("context_combined", F.when(
    F.col("rand") < 0.000001,
    F.shuffle(F.array("context_distract1", "context_distract2", "context_distract3"))
).otherwise(
    F.shuffle(F.array("context", "context_distract1", "context_distract2"))
))

result_df = result_df.withColumn("context_combined", F.expr("slice(context_combined, 1, 3)"))
result_df = result_df.withColumn("context", F.expr("concat_ws(' ', context_combined)"))

display(result_df)

# COMMAND ----------

n = 3
result_df.write.mode("append").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset{n}")

# COMMAND ----------

from finreganalytics.dataprep.ift_data_prep import (
    prepare_ift_dataset,
)
from finreganalytics.utils import get_spark
from pyspark.sql.functions import rand

#gold answer contained 50% of time, 3 documents each time

qa_train_df, qa_val_df = get_spark().read.table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset{n}").orderBy(
    rand()).randomSplit([0.9, 0.1])
qa_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_train{n}")
qa_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_val{n}")

qa_ift_train_df = prepare_ift_dataset(qa_train_df, limit=-1)
qa_ift_val_df = prepare_ift_dataset(qa_val_df, limit=-1)

qa_ift_train_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_train{n}")
qa_ift_val_df.write.mode("overwrite").saveAsTable(f"{uc_target_catalog}.{uc_target_schema}.qa_instructions_val{n}")

# COMMAND ----------


