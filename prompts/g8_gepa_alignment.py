"""Group 8: GEPA-optimized prompt for alignment between stated and internal confidence."""

def get_g8_gepa_prompt(question: str, options: dict) -> str:
    """
    Generate a GEPA-optimized prompt that aligns stated and internal confidence.
    """
    prompt = f"""
    Select the most accurate answer and ensure that your confidence percentage truly reflects your belief in the correctness of the chosen letter. 
    Avoid using constant or default confidence values; always tailor your confidence to your level of certainty. 
    Indicate a high confidence (above 80%) if you are very sure, while a lower confidence (below 50%) should be used if you feel uncertain or if multiple options seem plausible. 
    Ensure your response follows this strict format: [LETTER] [CONFIDENCE%]. Example: A 78%. Confidence must be an integer between 0 and 100.
    
    Question: {question}

    Options:
    A) {options.get('A', '')}
    B) {options.get('B', '')}
    C) {options.get('C', '')}
    D) {options.get('D', '')}

    Answer the question using ONLY this format: "[Option Letter] [Confidence Score]%".  Example: "A 85%".
    """
    
    return prompt

