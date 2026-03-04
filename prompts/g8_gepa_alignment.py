"""Group 8: GEPA-optimized prompt for alignment between stated and internal confidence."""

def get_g8_gepa_prompt(question: str, options: dict) -> str:
    """
    Generate a GEPA-optimized prompt that aligns stated and internal confidence.
    """
    prompt = f"""
    Instruction: Assume a careful approach that encourages accurate self-assessment of confidence in the chosen answer. 
    Confidence must reflect your genuine belief about the correctness of the selected option and should not default to a constant value. 
    If you feel uncertain, choose a lower confidence level. 

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

    Answer the question using ONLY this format: "[Option Letter] [Confidence Score]%".  Example: "A 85%".
    """
    
    return prompt

