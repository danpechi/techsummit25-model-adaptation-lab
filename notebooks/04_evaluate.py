# Databricks notebook source
# MAGIC %md 
# MAGIC # Model Adaptation Demo 
# MAGIC ## Fine-tuning a European Financial Regulation Assistant model 
# MAGIC
# MAGIC In this demo we will generate synthetic question/answer data about Capital Requirements Regulation and after that will use this data to dine tune the Llama 3.0 8B model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating Model
# MAGIC In this notebook we will evaluate the model we have fine-tuned during the previous step

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from finreganalytics.dataprep.evaluation import evaluate_qa_chain
from finreganalytics.dataprep.ift_data_prep import SYSTEM_INSTRUCTION
from finreganalytics.utils import get_spark, get_user_name

# COMMAND ----------

uc_target_catalog = "dpechi"
uc_target_schema = "test"

if (locals().get("uc_target_catalog") is None
        or locals().get("uc_target_schema") is None):
    uc_target_catalog = get_user_name()
    uc_target_schema = get_user_name()


# COMMAND ----------

# MAGIC %md In the next cell we will prepare the functions needed to build the evaluation chain

# COMMAND ----------


def build_retrievalqa_zeroshot_chain(llm: BaseLanguageModel):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTION),
        ("user", """Please answer the user question using the given context:\n {question}"""),
    ])
    chain = prompt | llm | StrOutputParser()

    return chain


def build_retrievalqa_with_context_chain(llm: BaseLanguageModel):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTION),
        ("user", """Context:\n {context}\n\n Please answer the user question using the given context:\n {question}"""),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain


# COMMAND ----------

# MAGIC %md We will need to create a PT endpoint for the model we have fine-tuned during the previous step. I have done this manually using Databricks UI.

# COMMAND ----------

llm = ChatDatabricks(endpoint="llama38b", temperature=0.99)
qa_chain_with_ctx = build_retrievalqa_with_context_chain(llm)

# COMMAND ----------


from pyspark.sql.functions import col

val_qa_eval_sdf = (
    get_spark()
    .read
    .table(f"{uc_target_catalog}.{uc_target_schema}.qa_dataset_val")
    .select(col("context"),
            col("question"),
            col("answer"))
)
display(val_qa_eval_sdf)  # noqa

# COMMAND ----------

limit = 10
val_qa_eval_pdf = val_qa_eval_sdf.toPandas()
if limit > 0:
    val_qa_eval_pdf = val_qa_eval_pdf[:limit]

# COMMAND ----------

eval_results = evaluate_qa_chain(
    val_qa_eval_pdf,
    ["context", "question"],
    qa_chain_with_ctx,
    "FinReg_CRR_Llama38b",
)
print(f"See evaluation metrics below: \n{eval_results.metrics}")
display(eval_results.tables["eval_results_table"])  # noqa

# COMMAND ----------


