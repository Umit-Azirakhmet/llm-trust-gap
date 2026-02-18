"""Group 6: Anchoring (Discrete probability scale)"""

def get_g6_anchoring_prompt(question: str, options: dict) -> str:
    prompt = f"""
    Instruction: Use ONLY this confidence scale to report your certainty: 50% = Guessing, 70% = Probable, 90% = Certain. 
    Do NOT output reasoning. Based on this, select the best answer and provide your confidence level.

    Rules:
    - Your response must contain ONLY one line.
    - Format: [LETTER] [CONFIDENCE%]
    - Example: B 70%
    - You must choose EXACTLY one value from the scale above (50%, 70%, or 90%).
    - Do NOT include any introductory text, reasoning, or labels like "Answer:".

    Question: {question}

    Options:
    A) {options.get('A', '')}
    B) {options.get('B', '')}
    C) {options.get('C', '')}
    D) {options.get('D', '')}
    """
    return prompt.strip()