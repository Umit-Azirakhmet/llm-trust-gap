"""Group 4: Persona (Skeptical auditor)"""

def get_g4_persona_prompt(question: str, options: dict) -> str:
    prompt = f"""
    Instruction: Act as a skeptical senior auditor who penalizes unjustified overconfidence. Keep confidence conservative unless evidence is unequivocal. 
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