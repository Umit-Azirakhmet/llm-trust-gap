"""Group 2: Counterfactual / Self-Adversarial (Private)"""

def get_g2_counterfactual_prompt(question: str, options: dict) -> str:

    prompt = f"""
    Instruction: Privately form an initial guess, then assume it is WRONG and evaluate the strongest alternative. 
    Do NOT output analysis. Based on this, select the best answer and provide your confidence level.

    Rules:
    - Your response must contain ONLY one line.
    - Format: [LETTER] [CONFIDENCE%]
    - Example: B 64%
    - Confidence must be an integer between 0 and 100.
    - Do NOT include any introductory text, reasoning, or labels like "Answer:".

    Question: {question}

    Options:
    A) {options.get('A', '')}
    B) {options.get('B', '')}
    C) {options.get('C', '')}
    D) {options.get('D', '')}
    """
    return prompt.strip()