"""Group 2: Counterfactual prompt template."""

def get_counterfactual_prompt(question: str, options: dict) -> str:
    """
    Generate a counterfactual prompt that forces the model to stress-test its certainty.
    
    Args:
        question: The question text
        options: Dictionary with keys "A", "B", "C", "D" and option texts as values
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Consider this question carefully. First, assume your initial hypothesis is incorrect and explore alternative answers. Then provide your final answer with confidence level.

Question: {question}

Options:
A) {options.get('A', '')}
B) {options.get('B', '')}
C) {options.get('C', '')}
D) {options.get('D', '')}

Step 1: Assume your first instinct is wrong. What alternative answers could be correct? Why?
Step 2: After this counterfactual analysis, output your final answer in the format: [LETTER] [CONFIDENCE%]
For example: B 60%"""
    
    return prompt

