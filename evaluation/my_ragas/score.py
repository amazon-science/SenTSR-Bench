import copy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from botocore.exceptions import ClientError
from evaluation.my_ragas.metric import AnswerCorrectness
from evaluation.my_ragas.config import load_llm, load_embeddings, config
from ragas import RunConfig

#–– retry wrappers for the I/O‐heavy loaders ––
@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)  # you can narrow this to your specific I/O exceptions
)
def load_embeddings_with_retry():
    return load_embeddings()

@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def load_llm_with_retry():
    return load_llm()

@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def calculate_ragas_score(question: str, response: str, label: str):
    """
    Compute the RAGAS score for a single QA pair, retrying resource loads on failure.
    Returns:
      - score (float)
      - detail dict (deepcopied)
    """
    # 1) load resources with retry guarantees
    embeddings = load_embeddings_with_retry()
    llm        = load_llm_with_retry()

    # 2) instantiate the scorer
    answer_correctness = AnswerCorrectness(
        embeddings=embeddings,
        llm=llm,
        weights=[1.0, 0.0]
    )
    answer_correctness.answer_detail = {}

    # 3) compute the score
    score = answer_correctness.score(
        row={
            'question':     question,
            'answer':       response,
            'ground_truth': label
        }
    )

    # 4) return both the numeric score and a snapshot of the detail dict
    return float(score), copy.deepcopy(answer_correctness.answer_detail)
