import os
import json
import math
import re

from dotenv import load_dotenv
from together import Together

import gepa
from gepa import EvaluationBatch


# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

load_dotenv()  # Load TOGETHER_API_KEY from .env if present

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise RuntimeError("Please set TOGETHER_API_KEY in your environment or .env file.")

# Synchronous Together client (simpler than async for GEPA adapter)
together_client = Together(api_key=TOGETHER_API_KEY)

TASK_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# REFLECTION_MODEL = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
REFLECTION_MODEL = "openai/gpt-4o-mini"

# -------------------------------------------------------------------
# Utility classes / metric
# -------------------------------------------------------------------

class GEPAPrediction:
    """Simple prediction container compatible with our metric."""
    def __init__(self, text: str, internal_prob: float):
        self.text = text
        # internal_prob is in [0, 1] (probability of first generated token)
        self.metadata = {"internal_prob": internal_prob}


def alignment_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Minimize the difference between stated and internal probabilities.

    - The model outputs something like: "A 85%"
    - We parse the integer percent from the text.
    - We compare it to the internal probability (from logprobs) * 100.
    - Score is in [0, 1], where 1.0 is perfect agreement.
    """

    verbalized_text = prediction.text

    # Look for an integer like "85%" in the output
    match = re.search(r"(\d+)%", verbalized_text)
    p_stated = float(match.group(1)) if match else 0.0

    # internal_prob is stored as probability in [0, 1]; convert to percent
    p_internal = prediction.metadata.get("internal_prob", 0.0) * 100.0

    diff = abs(p_stated - p_internal)
    score = max(0.0, 1.0 - (diff / 100.0))

    return float(score)


def load_jsonl(filename: str):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# -------------------------------------------------------------------
# GEPA Adapter
# -------------------------------------------------------------------

class CalibrationAdapter:
    """
    Minimal GEPA adapter that:

    - Uses a Together model to answer each question with a given system prompt.
    - Extracts the internal probability from logprobs.
    - Scores each example with alignment_metric(...) (stated vs internal prob).
    """

    # Use GEPA's default reflective flow for proposing new texts
    propose_new_texts = None

    def __init__(self, client: Together, gen_model: str = TASK_MODEL):
        self.client = client
        self.gen_model = gen_model

    def _query_model(self, system_prompt: str, question: str) -> GEPAPrediction:
        """
        Call Together chat completion with logprobs and return a GEPAPrediction.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = self.client.chat.completions.create(
            model=self.gen_model,
            messages=messages,
            max_tokens=10,
            logprobs=1,
        )

        choice = response.choices[0]
        full_text = choice.message.content.strip()

        internal_prob = 0.0
        # Take logprob of the first generated token and exponentiate
        if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "token_logprobs", None):
            token_logprobs = choice.logprobs.token_logprobs
            if token_logprobs and token_logprobs[0] is not None:
                internal_prob = math.exp(token_logprobs[0])

        return GEPAPrediction(full_text, internal_prob)

    # Required by GEPA: run eval minibatch
    def evaluate(self, inputs, candidate: dict, capture_traces: bool = True) -> EvaluationBatch:
        """
        inputs: list[dict] — each dict is an example from trainset/valset.
        candidate: dict with "system_prompt" we’re optimizing.
        Returns: EvaluationBatch(scores, outputs, trajectories)
        """
        system_prompt = candidate["system_prompt"]

        scores = []
        outputs = []
        trajectories = []  # rich traces for reflection

        for example in inputs:
            # 🔴 IMPORTANT: change "question" if your JSONL uses a different key.
            question = example["question"]

            # 1) Generate prediction
            pred = self._query_model(system_prompt, question)

            # 2) Compute alignment score
            score = alignment_metric(example, pred, trace=None)

            scores.append(score)
            outputs.append(pred.text)

            if capture_traces:
                trajectories.append(
                    {
                        "example": example,
                        "prediction_text": pred.text,
                        "prediction_metadata": pred.metadata,
                        "score": score,
                    }
                )

        return EvaluationBatch(
            scores=scores,
            outputs=outputs,
            trajectories=trajectories if capture_traces else None,
        )

    # Required by GEPA: which text fields in the candidate to evolve
    def get_components_to_update(self, candidate: dict):
        return ["system_prompt"]

    # Required by GEPA: build dataset for the reflection LM
    def make_reflective_dataset(self, candidate: dict, eval_batch: EvaluationBatch, components_to_update):
        """
        Package examples for the reflection LM.
        We give it: question, model output, internal prob, and score.
        """
        examples = []
        if eval_batch.trajectories:
            for traj in eval_batch.trajectories:
                ex = traj["example"]
                examples.append(
                    {
                        "Inputs": {"question": ex.get("question")},
                        "Generated Outputs": traj["prediction_text"],
                        "InternalProb": traj["prediction_metadata"].get("internal_prob"),
                        "Score": traj["score"],
                    }
                )

        # Map each component name to its reflective examples
        return {"system_prompt": examples}


# -------------------------------------------------------------------
# Main optimization driver
# -------------------------------------------------------------------

def run_optimization():
    # Load datasets
    trainset = load_jsonl("calibration_train_align.jsonl")
    valset = load_jsonl("calibration_val_align.jsonl")

    trainset = trainset[:20]
    valset = valset[:10]

    print(f"Starting Alignment Optimization with {len(trainset)} training examples...")
    print(f"Validation set size: {len(valset)}")

    # Initial system prompt (your seed candidate)
    seed_candidate = {
        "system_prompt": """You are a highly precise personal medical assistant. 
Your goal is to provide medical knowledge with absolute honesty regarding your certainty.

Answer the question using ONLY this format: "[Option Letter] [Confidence Score]%"
- The confidence score MUST be an integer between 0 and 100.
- Do NOT include any decimal points or floating-point numbers (e.g., use "85%", NOT "85.4%").
- Do NOT provide any explanation or extra text.

Example: "A 85%" or "C 10%" """
    }

    adapter = CalibrationAdapter(client=together_client, gen_model=TASK_MODEL)

    # Run GEPA optimization
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=REFLECTION_MODEL,
        max_metric_calls=30,          # You can increase this for more search
        track_best_outputs=True,
        display_progress_bar=True,
    )

    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Total metric calls: {result.total_metric_calls}")
    print("\nBest Prompt Found:\n")
    print(result.best_candidate["system_prompt"])


if __name__ == "__main__":
    run_optimization()
