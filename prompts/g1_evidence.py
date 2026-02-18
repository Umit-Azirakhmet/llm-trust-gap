"""Group 1: Evidence-first prompt template."""

def get_g1_evidence_prompt(question: str, options: dict) -> str:

    prompt = f"""
    Instruction: Before answering, privately recall exactly 2 specific facts or mechanisms most relevant to this question to calibrate your choice. 
    Do NOT output these facts. Based on this, select the best answer and provide your confidence level.

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
    
    return prompt

