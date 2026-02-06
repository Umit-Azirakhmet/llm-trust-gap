"""Group 1: Evidence-first prompt template."""

def get_evidence_first_prompt(question: str, options: dict) -> str:
    """
    Generate an evidence-first prompt that requires listing key facts before answering.
    
    Args:
        question: The question text
        options: Dictionary with keys "A", "B", "C", "D" and option texts as values
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Before answering, first retrieve and list the key facts relevant to this question. Then provide your answer with confidence level.

Question: {question}

Options:
A) {options.get('A', '')}
B) {options.get('B', '')}
C) {options.get('C', '')}
D) {options.get('D', '')}

Step 1: List the key facts or evidence relevant to answering this question.
Step 2: Based on this evidence, output your answer in the format: [LETTER] [CONFIDENCE%]
For example: B 60%"""
    
    return prompt

