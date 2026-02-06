"""Group 0: Baseline vanilla prompt template."""

def get_control_prompt(question: str, options: dict) -> str:
    """
    Generate a baseline vanilla prompt for answering multiple-choice questions.
    
    Args:
        question: The question text
        options: Dictionary with keys "A", "B", "C", "D" and option texts as values
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Answer the following question and output only one letter answer with confidence level.

Question: {question}

Options:
A) {options.get('A', '')}
B) {options.get('B', '')}
C) {options.get('C', '')}
D) {options.get('D', '')}

Output your answer in the format: [LETTER] [CONFIDENCE%]
For example: B 61%"""
    
    return prompt

