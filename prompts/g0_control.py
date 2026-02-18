"""Group 0: Baseline vanilla prompt template."""

def get_control_prompt(question: str, options: dict) -> str:

    prompt = f"""
    Instruction: Choose the single best option for the following question. 
    Select the best answer and provide your confidence level.

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

