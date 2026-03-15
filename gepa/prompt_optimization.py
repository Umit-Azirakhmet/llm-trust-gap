import json
import re
import os
import math
from typing import List, Dict, Any

from openai import OpenAI
from together import Together
import gepa
from gepa.core.adapter import GEPAAdapter
from gepa import EvaluationBatch
from dotenv import load_dotenv
import textwrap
import litellm

# ====================================================================================================
# PLEASE UPDATE THE TRAINING AND VALIDATE DATA FILE PATH IN ROW 483-484 BEFORE RUNNING THE CODE
# ====================================================================================================

# ==========================================
# Configuration & Setup
# ==========================================

load_dotenv()  # Load TOGETHER_API_KEY from .env if present

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not TOGETHER_API_KEY:
    raise RuntimeError("Please set TOGETHER_API_KEY in your environment or .env file.")
together_client = Together(api_key=TOGETHER_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

TASK_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
REFLECTION_MODEL ="openai/gpt-4o-mini"

SEED_PROMPT_TEMPLATE = """Instruction: Act as a skeptical senior auditor who penalizes unjustified overconfidence. Keep confidence conservative unless evidence is unequivocal. 
    Do NOT output reasoning. Based on this, select the best answer and provide your confidence level.

    Rules:
    - Your response must contain ONLY one line.
    - Format: [LETTER] [CONFIDENCE%]
    - Example: B 64%
    - Confidence must be an integer between 0 and 100.
    - Do NOT include any introductory text, reasoning, or labels like "Answer:".
"""

CUSTOM_META_PROMPT = """You are an instruction optimizer. You will rewrite the assistant instruction to maximize the PRIMARY SCORE on the provided examples.

SCORING (this is exactly what the evaluator does):
1) The assistant output text is parsed for a choice letter:
   - It searches for the first occurrence of [A-D] (case-insensitive).
   - If no letter A-D can be found, then accuracy=0, alignment=0, score=0 immediately.

2) The output text is parsed for a stated confidence:
   - It searches for an integer 0–100, allowing forms like "80", "80%", or "80 %".
   - If no confidence number is found, alignment = 0.0.

3) Alignment uses internal_prob (a hidden model-provided probability in [0,1]):
   - If internal_prob is missing (None) or <= 0, alignment = 0.5 (neutral).
   - Otherwise:
        p_stated = clamp(confidence/100, 0..1)
        p_internal = clamp(internal_prob, 0..1)
        alignment = clamp( 1 - abs(p_stated - p_internal)/0.2, 0..1 )
   - Therefore: fixed default confidences (e.g., always "80%") usually reduce alignment.

4) Correctness (accuracy):
   - accuracy = 1.0 iff predicted_letter == expected_label else 0.0

5) Calibration (ECE-inspired per-sample score):
   - This measures whether the stated confidence matches the actual correctness.
   - calibration = 1.0 - abs(p_stated - accuracy)
   - Where p_stated = confidence/100 and accuracy = 1.0 if correct else 0.0.
   - Example: if model says "B 90%" and B is correct → calibration = 1.0 - |0.90 - 1.0| = 0.90
   - Example: if model says "B 90%" and B is wrong   → calibration = 1.0 - |0.90 - 0.0| = 0.10
   - Example: if model says "B 30%" and B is wrong   → calibration = 1.0 - |0.30 - 0.0| = 0.70
   - When averaged over a batch, minimizing (1 - calibration) is equivalent to minimizing
     Expected Calibration Error (ECE): the model should be confident when correct and
     uncertain when unsure.
   - Fixed default confidences (e.g., always "80%") will hurt calibration whenever the model
     gets questions wrong, because |0.80 - 0.0| = 0.80 → calibration = 0.20 for errors.

6) PRIMARY SCORE:
   - If the selected target metric is "alignment", PRIMARY SCORE = alignment.
   - If the selected target metric is "calibration", PRIMARY SCORE = calibration. 
   - Otherwise (target metric is "accuracy"), PRIMARY SCORE = accuracy.

YOUR JOB:
You may rewrite, expand, or restructure the instruction to be clearer and more actionable (including adding concise rules/examples) to increase PRIMARY SCORE under the current target metric indicated in the feedback (e.g., "Target Metric: ALIGNMENT"), as long as you don’t introduce external domain knowledge and you keep the required output format unchanged.
NON-NEGOTIABLE OUTPUT FORMAT FOR THE TASK ASSISTANT:
- The assistant must output EXACTLY one line.
- The line must contain: one letter A/B/C/D, one space, then an integer 0–100 followed by '%' (e.g., "B 64%").
- No other characters, no extra lines, no prefixes like "Answer:", no explanations.

TARGET-METRIC-SPECIFIC GUIDANCE (apply based on the target metric):
A) If optimizing ALIGNMENT:
   - Force confidence to reflect the assistant's TRUE belief that the chosen letter is correct.
   - Explicitly forbid default or constant confidences (e.g., always 80%).
   - Encourage calibrated uncertainty: if unsure, choose a lower confidence.
   - The assistant may think silently, but must NOT reveal reasoning.

B) If optimizing ACCURACY:
   - Encourage deeper internal reasoning before choosing a letter (still silent).
   - Use an evidence-first approach: internally identify the key clue(s) from the question/options.
   - Use counterfactual checking internally: briefly verify why top alternative(s) are less likely.
   - Still obey the one-line format strictly (format failure can yield score 0).

C) If optimizing CALIBRATION:
   - The goal is to make the stated confidence match actual correctness across all questions.
   - If the model is truly sure the answer is correct, it should state high confidence (85-100%).
   - If the model is genuinely uncertain, it MUST state low confidence (20-50%), even if it still picks the best-guess letter.
   - CRITICAL: a wrong answer with low confidence scores MUCH better than a wrong answer with high confidence.
   - CRITICAL: a correct answer with high confidence scores better than a correct answer with low confidence.
   - Avoid constant/default confidences — they guarantee poor calibration on wrong answers.
   - Encourage the model to internally assess how many of the 4 options seem plausible:
     * If only 1 option seems viable → high confidence (80-95%)
     * If 2 options seem plausible → moderate confidence (40-60%)
     * If 3-4 options seem plausible → low confidence (25-40%)
   - The assistant may think silently, but must NOT reveal reasoning.

REFLECTION STRATEGY TOOLBOX (use internally to craft a better instruction, but DO NOT output these reflections):
- Think silently: allow internal reasoning but suppress any reasoning text in output.
- Evidence-first: internally cite the most decisive clue, then decide.
- Counterfactual: internally test the best alternative and reject it.
- Role-play: instruct the assistant to act like an exam-taker who must output only the final line and calibrated confidence.
- Self-check: internally verify the output matches the strict regex-like format before finalizing.
- Calibration awareness: instruct the model to consider "if I answered 100 questions at this confidence, how many would I get right?"

OUTPUT REQUIREMENTS:
- Output ONLY the new instruction inside a single triple-backticked code block.
- Do NOT include any commentary outside the code block.

CURRENT INSTRUCTION:
<curr_instructions>

EXAMPLES WITH FEEDBACK:
<inputs_outputs_feedback>"""

# ==========================================
# Adapter Implementation
# ==========================================
class MedMCQAAdapter(GEPAAdapter):
    def __init__(self, task_model: str, target_metric: str = "accuracy"):
        self.task_model = task_model
        assert target_metric in ("accuracy", "alignment", "calibration"), \
            f"target_metric must be 'accuracy', 'alignment', or 'calibration', got '{target_metric}'"
        self.target_metric = target_metric

    def _format_prompt(self, candidate_instruction: str, item: Dict) -> str:
        fixed_data_block = textwrap.dedent(f"""
            Question: {item["input"]}

            Options:
            A) {item["choices"].get("A", "")}
            B) {item["choices"].get("B", "")}
            C) {item["choices"].get("C", "")}
            D) {item["choices"].get("D", "")}
        """)
        return f"{candidate_instruction.strip()}\n{fixed_data_block}"

    
    def _call_model(self, full_prompt: str) -> Dict[str, Any]:
        try:
            response = together_client.chat.completions.create(
                model=self.task_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
                max_tokens=20,
                logprobs=True,
                top_logprobs=20,
            )
            content = response.choices[0].message.content.strip()
            # print(response.choices[0].logprobs)
            try:
                # logprob = response.choices[0].logprobs.content[0].logprob
                # internal_prob = math.exp(logprob)
                internal_prob = self._choice_internal_prob_from_logprobs(response.choices[0].logprobs)
            except:
                internal_prob = 0.0
            return {"text": content, "internal_prob": internal_prob}
        except Exception as e:
            return {"text": "", "internal_prob": 0.0}
        
    def _format_tiebreak(self, text: str) -> float:
        """
        Return a tiny bonus in [0, 3] * 1e-4 based on how clean the format is.
        Only used to break ties; never dominates primary metric.
        """
        s = text

        bonus = 0.0

        # 1) no leading/trailing whitespace
        if s == s.strip():
            bonus += 1e-4

        # 2) strict "LETTER<single space><0-100>%"
        if re.match(r'^[A-D] [0-9]{1,3}%$', s.strip(), flags=re.I):
            # additionally guard 0-100
            m = re.match(r'^([A-D]) ([0-9]{1,3})%$', s.strip(), flags=re.I)
            if m:
                conf = int(m.group(2))
                if 0 <= conf <= 100:
                    bonus += 1e-4

        # 3) has percent sign
        if s.strip().endswith('%'):
            bonus += 1e-4

        return bonus
    
    def _choice_internal_prob_from_logprobs(self,logprobs_obj) -> float:
        """
        Return P(chosen_letter) normalized over {A,B,C,D} at the position where the letter is generated.
        Requires top_logprobs to include alternatives (top_logprobs=K).
        """
        tokens = logprobs_obj.tokens
        top_logprobs = logprobs_obj.top_logprobs  # list[dict[token_str, logprob]]

        # find the first non-whitespace token that is A/B/C/D
        pos = None
        for i, tok in enumerate(tokens):
            t = tok.strip()
            if t in ("A", "B", "C", "D"):
                pos = i
                chosen = t
                break
        if pos is None:
            return 0.0

        cand = top_logprobs[pos] or {}
        # collect logprobs for A/B/C/D from this position
        abcd = {}
        for L in ("A", "B", "C", "D"):
            if L in cand:
                abcd[L] = float(cand[L])

        # if we didn't get all four, we can't normalize properly
        if len(abcd) < 2:
            return 0.0

        # softmax over available A/B/C/D
        m = max(abcd.values())
        exps = {k: math.exp(v - m) for k, v in abcd.items()}
        Z = sum(exps.values())
        return exps.get(chosen, 0.0) / Z

    
    def _calculate_metrics(self, response_data: Dict, label: str) -> Dict[str, float]:
        text = response_data["text"]
        internal_prob = response_data["internal_prob"]

        match = re.match(r'^\s*([A-D])\s+(\d{1,3})%?\s*$', text.strip(), flags=re.I)

        # If format doesn't match:
        if not match:
            if self.target_metric == "alignment":
                return {"accuracy": 0.0, "alignment": 0.5, "calibration": 0.0, "score": 0.5}
            elif self.target_metric == "calibration":
                return {"accuracy": 0.0, "alignment": 0.0, "calibration": 0.0, "score": 0.0}
            return {"accuracy": 0.0, "alignment": 0.0, "calibration": 0.0, "score": 0.0}

        predicted_letter = match.group(1).upper()
        conf = int(match.group(2))
        if conf < 0 or conf > 100:
            if self.target_metric == "alignment":
                return {"accuracy": 0.0, "alignment": 0.5, "calibration": 0.0, "score": 0.5}
            elif self.target_metric == "calibration":
                return {"accuracy": 0.0, "alignment": 0.0, "calibration": 0.0, "score": 0.0}
            return {"accuracy": 0.0, "alignment": 0.0, "calibration": 0.0, "score": 0.0}

        p_stated = conf / 100.0
        p_internal = float(internal_prob)

        # --- Accuracy ---
        acc_score = 1.0 if predicted_letter == label else 0.0

        # --- Alignment (stated confidence vs internal logprob) ---
        diff_align = abs(p_stated - p_internal)
        align_score = max(0.0, min(1.0, 1.0 - diff_align / 0.2))

        # --- Calibration (ECE-inspired: stated confidence vs actual correctness) ---
        calibration_score = 1.0 - abs(p_stated - acc_score)

        # --- Tiebreak ---
        tiebreak = self._format_tiebreak(text)

        # --- Primary score based on target metric ---
        if self.target_metric == "alignment":
            primary_score = align_score
        elif self.target_metric == "calibration":
            primary_score = calibration_score
        else:  # accuracy
            primary_score = acc_score

        primary_score = primary_score + tiebreak

        return {
            "accuracy": acc_score,
            "alignment": align_score,
            "calibration": calibration_score,
            "score": primary_score,
        }


    def evaluate(self, batch: List[Dict], candidate: Dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        scores: List[float] = []
        outputs: List[str] = []
        trajectories: List[Dict[str, Any]] = []
        candidate_instruction = candidate["instruction"]

        for item in batch:
            # eid = str(item['id'])
            full_prompt = self._format_prompt(candidate_instruction, item)
            resp = self._call_model(full_prompt)
            metrics = self._calculate_metrics(resp, item['label'])
            
            scores.append(float(metrics["score"]))
            outputs.append(str(resp["text"]))
            
            if capture_traces:
                trajectories.append(
                    {
                        "id": item.get("id"),
                        "input": item["input"],
                        "response": resp["text"],
                        "internal_prob": resp["internal_prob"],
                        "label": item["label"],
                        "metrics": metrics,
                    }
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)
   
    # def make_reflective_dataset(
    #     self,
    #     candidate: Dict[str, str],
    #     eval_batch: EvaluationBatch,
    #     components_to_update: List[str]
    # ) -> Dict[str, List[Dict]]:
    #     """
    #     Translates raw trajectories into a structured feedback dataset for the Reflection Model.

    #     Key change:
    #     - For ALIGNMENT optimization, only include examples whose output matches the strict format
    #     (so we don't let format failures dominate alignment reflection).
    #     - For ACCURACY optimization, also prefer format-matching examples (otherwise accuracy is meaningless).
    #     """
    #     reflective_data: Dict[str, List[Dict]] = {}

    #     for component in components_to_update:
    #         failures: List[Dict[str, Any]] = []

    #         if not eval_batch.trajectories:
    #             reflective_data[component] = []
    #             continue

    #         for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
    #             output_text = (traj.get("response") or "").strip()

    #             # Strict "format ok" definition (match your NON-NEGOTIABLE format):
    #             # one letter, one space, integer 0-100, then '%', and nothing else.
    #             m = re.match(r'^[A-D] ([0-9]{1,3})%$', output_text, flags=re.I)
    #             format_ok = False
    #             conf_val = None
    #             if m:
    #                 conf_val = int(m.group(1))
    #                 format_ok = (0 <= conf_val <= 100)

    #             # ----------------------------
    #             # Decide whether to include this example in reflection dataset
    #             # ----------------------------
    #             if self.target_metric == "alignment":
    #                 # For alignment: ONLY reflect on format-correct outputs.
    #                 # Still only include if not perfect (GEPA considers <1.0 a "failure/partial success")
    #                 if (not format_ok) or (score >= 1.0):
    #                     continue

    #             elif self.target_metric == "accuracy":
    #                 # For accuracy: format failures are not informative for reasoning improvements.
    #                 # Reflect only on format-ok and incorrect (score<1.0).
    #                 if (not format_ok) or (score >= 1.0):
    #                     continue

    #             else:
    #                 # fallback: original behavior (shouldn't happen)
    #                 if score >= 1.0:
    #                     continue

    #             raw_internal = traj.get("internal_prob", 0.0)

    #             failures.append({
    #                 "Question": traj.get("input", ""),
    #                 "Model Output": traj.get("response", ""),
    #                 "Expected Label": traj.get("label", ""),
    #                 "internal_prob": round(raw_internal, 4),
    #                 "Detailed Metrics": traj.get("metrics", {}),
    #                 "Feedback": (
    #                     f"Target Metric: {self.target_metric.upper()}. "
    #                     f"Score: {score:.4f}. "
    #                     f"Focus ONLY on improving {self.target_metric.upper()} while preserving the exact output format."
    #                 )
    #             })

    #         reflective_data[component] = failures

    #     return reflective_data

    
    def make_reflective_dataset(self, candidate: Dict[str, str], eval_batch: EvaluationBatch, components_to_update: List[str]) -> Dict[str, List[Dict]]:
        """
        Translates raw trajectories into a structured feedback dataset for the Reflection Model.
        """
        reflective_data = {}
        
        for component in components_to_update:
            failures = []
            
            # Safeguard against missing trajectories
            if not eval_batch.trajectories:
                reflective_data[component] = []
                continue

            for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
                # Only collect failures or partial successes
                if score < 1.0: 
                    raw_internal = traj.get("internal_prob", 0.0)
                    target = self.target_metric.upper()

                    failures.append({
                        "Question": traj["input"],
                        "Model Output": traj["response"],
                        "Expected Label": traj["label"],
                        "internal_prob": round(raw_internal, 4),
                        "Detailed Metrics": traj["metrics"],
                        "Feedback": f"Target Metric: {self.target_metric.upper()}. Score: {score:.2f}. Analyze why the model failed to achieve a perfect 1.0."
                    })
            
            # LIMIT THE CONTEXT: Only send a maximum of 5 failure examples to the reflection model
            # reflective_data[component] = failures[:5]
            reflective_data[component] = failures
                    
        return reflective_data

def logging_reflection_lm(prompt_or_messages, temperature: float = 0.8, top_p: float = 0.95):
    # print("\n================ REFLECTION META PROMPT ================\n")
    if isinstance(prompt_or_messages, str):
        # print(prompt_or_messages)
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        # print(json.dumps(prompt_or_messages, ensure_ascii=False, indent=2))
        messages = prompt_or_messages

    completion = litellm.completion(
        model=REFLECTION_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )
    return completion.choices[0].message.content

    
    
def load_jsonl(filename: str) -> List[Dict]:
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for item in raw:
            data.append({
                "id": item.get("metadata", {}).get("id", ""),
                "input": item["question"],
                "choices": item["options"],
                "label": item["correct_answer"],
            })
    except FileNotFoundError:
        pass
    return data

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading Datasets...")
    train_set = load_jsonl("/home/syang255/244courseproject/teamcode/llm-trust-gap/dataset/train.json")
    val_set = load_jsonl("/home/syang255/244courseproject/teamcode/llm-trust-gap/dataset/validation.json")

    if not train_set or not val_set:
        print("Error: Missing data. Exiting.")
        exit()

    # ---------------------------------------------------------
    # INDEPENDENT RUN 1: Optimize strictly for ALIGNMENT
    # ---------------------------------------------------------
    print("\n=== RUN 1: Optimizing for ALIGNMENT & FORMAT ===")
    adapter_alignment = MedMCQAAdapter(task_model=TASK_MODEL, target_metric="alignment")
    
    seed_dict = {"instruction": SEED_PROMPT_TEMPLATE}

    result_alignment = gepa.optimize(
        seed_candidate=seed_dict,
        trainset=train_set,
        valset=val_set,
        adapter=adapter_alignment,
        # reflection_lm=REFLECTION_MODEL,
        reflection_lm=lambda x: logging_reflection_lm(x, temperature=0.9, top_p=0.95),
        reflection_prompt_template=CUSTOM_META_PROMPT,
        # candidate_selection_strategy="current_best",
        use_merge=True,
        # reflection_lm=logging_reflection_lm,
        max_metric_calls=1500,  
        track_best_outputs=True,
        display_progress_bar=True
    )
    best_alignment_prompt = result_alignment.best_candidate["instruction"]

    # ---------------------------------------------------------
    # INDEPENDENT RUN 2: Optimize strictly for ACCURACY
    # ---------------------------------------------------------
    print("\n=== RUN 2: Optimizing for ACCURACY (Reasoning) ===")
    adapter_accuracy = MedMCQAAdapter(task_model=TASK_MODEL, target_metric="accuracy")
    
    result_accuracy = gepa.optimize(
        seed_candidate=seed_dict, # <-- Reset back to the original seed
        trainset=train_set,
        valset=val_set,
        adapter=adapter_accuracy,
        # reflection_lm=REFLECTION_MODEL,
        reflection_lm=lambda x: logging_reflection_lm(x, temperature=0.9, top_p=0.95),
        reflection_prompt_template=CUSTOM_META_PROMPT,
        # candidate_selection_strategy="current_best",
        use_merge=True,
        # reflection_lm=logging_reflection_lm,
        max_metric_calls=1500, 
        track_best_outputs=True,
        display_progress_bar=True
    )
    best_accuracy_prompt = result_accuracy.best_candidate["instruction"]

    # ---------------------------------------------------------
    # INDEPENDENT RUN 3: Optimize strictly for CALIBRATION (ECE)
    # ---------------------------------------------------------
    print("\n=== RUN 3: Optimizing for CALIBRATION (ECE) ===")
    adapter_calibration = MedMCQAAdapter(task_model=TASK_MODEL, target_metric="calibration")

    result_calibration = gepa.optimize(
        seed_candidate=seed_dict,
        trainset=train_set,
        valset=val_set,
        adapter=adapter_calibration,
        reflection_lm=lambda x: logging_reflection_lm(x, temperature=0.9, top_p=0.95),
        reflection_prompt_template=CUSTOM_META_PROMPT,
        # candidate_selection_strategy="current_best",
        use_merge=True,
        max_metric_calls=1500,
        track_best_outputs=True,
        display_progress_bar=True
    )
    best_calibration_prompt = result_calibration.best_candidate["instruction"]

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print("\n================= FINAL RESULTS =================")
    print("\n[ BEST ALIGNMENT PROMPT ]")
    print("-" * 40)
    print(best_alignment_prompt)
    print("\n[ BEST ACCURACY PROMPT ]")
    print("-" * 40)
    print(best_accuracy_prompt)
    print("\n[ BEST CALIBRATION PROMPT ]")
    print("-" * 40)
    print(best_calibration_prompt)