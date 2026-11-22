from dotenv import load_dotenv
import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate   # your generator: generate(query, context) -> answer

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = "llama3.2:3b"  # NOTE: differs from the other evals (gpt-4o-mini)
THRESHOLD = 0.7



# 1. LOAD the faithfulness golden set (query + ideal_context)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)  # list of dicts: id, query, ideal_context, source

# 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
test_cases = []
for g in goldens:
    context = g["ideal_context"]              # known-good context (list of chunk strings)
    answer = generate(g["query"], context)    # RUN the generator -> actual_output

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,             # the generated answer we're judging
            retrieval_context=context,        # faithfulness checks the answer against THIS
            # no expected_output --- faithfulness never reads it
        )
    )

# 3. THE METRICS --- decompose actual_output into claims, attribute each to context
metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,   # prints WHY each score --- shows which claims were unsupported
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),
]

# 4. EVALUATE --- runs the metrics on every case, prints a report
evaluate(test_cases=test_cases, metrics=metrics)

