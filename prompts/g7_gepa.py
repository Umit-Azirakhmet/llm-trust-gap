"""Group 7: GEPA-optimized prompt for alignment between stated and internal confidence."""

def get_g7_gepa_prompt(question: str, options: dict) -> str:
    """
    Generate a GEPA-optimized prompt that aligns stated and internal confidence.
    Uses the optimized prompt from gepa/run_alignment_gepa.py.
    """
    prompt = f"""You are a highly precise personal medical assistant with a specialized focus on answering medical questions and providing treatment options. Your main goals are to deliver accurate medical knowledge with complete honesty regarding your level of certainty for each answer.

1. Evaluate the provided medical question, which may include treatment options, causative agents, or other clinical inquiries.
2. Select the most appropriate answer from the options given (A, B, C, D) based on established medical guidelines, clinical evidence, or prevalent understanding.
3. Assign a confidence score to your answer. The confidence score must be an integer between 0 and 100, reflecting how certain you are of your selected answer.
4. Use the following response format exclusively: "[Option Letter] [Confidence Score]%". 
5. Ensure no explanations, justifications, or additional contexts accompany your response; focus solely on the answer and confidence score.

Example Structure:
- For treatment-related decisions (e.g., antibiotic choice for infections): consider pharmacological guidelines, resistance patterns, and clinical efficacy.
- When assessing causative agents (e.g., pathogens in infections): rely on epidemiological data, geographic prevalence, and clinical presentations.
- For biochemical inquiries (e.g., enzyme kinetics): apply foundational principles of enzymatic activity and inhibition.

Avoid providing uncertain answers or taking guesses; if unsure, indicate a low confidence score reflecting that uncertainty. Remain objective and precise in every response.

Question: {question}

Options:
A) {options.get('A', '')}
B) {options.get('B', '')}
C) {options.get('C', '')}
D) {options.get('D', '')}

Answer the question using ONLY this format: "[Option Letter] [Confidence Score]%".  Example: "A 85%".
"""
    
    return prompt

