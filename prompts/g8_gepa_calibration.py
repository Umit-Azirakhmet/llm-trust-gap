"""Group 8: GEPA-optimized prompt for calibration between confidence and correctness."""

def get_g8_gepa_prompt(question: str, options: dict) -> str:
    """
    Generate a GEPA-optimized prompt that maximize calibration.
    """
    prompt = f"""
    Instruction: Output a single letter choice (A, B, C, or D) with a confidence percentage that accurately reflects your belief in the answer's correctness. 
    If you are confident the answer is correct, state a high confidence percentage (85-100%). 
    If you are uncertain, express that uncertainty with a lower confidence percentage (20-50%). 
    Avoid fixed percentages; instead, base your confidence on the number of plausible options you perceive. 
    If only one option seems viable, give a high confidence; if two options seem plausible, use moderate confidence; if three or four options seem plausible, choose low confidence. 

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
