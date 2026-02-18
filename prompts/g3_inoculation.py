"""Group 3: Inoculation / High-stakes caution framing"""

def get_g3_inoculation_prompt(question: str, options: dict) -> str:
    prompt = f"""
    Instruction: High-stakes evaluation. Overconfidence is heavily penalized when wrong. Be cautious and double-check privately. 
    Do NOT output reasoning. Based on this, select the best answer and provide your confidence level.

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