"""Group 3: AI-Optimized prompts using GEPA (Genetic Evolutionary Prompt Algorithms)."""

def get_gepa_prompt(question: str, options: dict) -> str:
    """
    """
    prompt = f"""
Question: {question}

Options:
A) {options.get('A', '')}
B) {options.get('B', '')}
C) {options.get('C', '')}
D) {options.get('D', '')}

"""
    
    return prompt

