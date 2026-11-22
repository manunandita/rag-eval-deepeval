# eval_retriever.py
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from src.retriever import build_retriever
from deepeval.models import OllamaModel
load_dotenv()


GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = OllamaModel(model="llama3.2:3b")   # NOTE: differs from the other evals (gpt-4o-mini)
THRESHOLD = 0.7

retriever = build_retriever()

    # 1. LOAD the golden set --- the fixed, human-authored truth
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)  # list of dicts: id, query, ideal_answer, source

# 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.
test_cases = []
for g in goldens[:3]:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )

# 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True,async_mode=False),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True,async_mode=False),
]

# 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
#    hyperparameters travel with the run so the report is tagged with the config.
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "base_k5",          # vs "reranked" when you swap it in
        "embedding_model": "bge-m3",
        "chunk_size": 500,
        "chunk_overlap": 150,
        "top_k": 3,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)