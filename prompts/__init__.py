"""Prompt templates for different experimental groups."""

from prompts.control import get_control_prompt
from prompts.evidence_first import get_evidence_first_prompt
from prompts.counterfactual import get_counterfactual_prompt
from prompts.gepa import get_gepa_prompt

PROMPT_GROUPS = {
    "control": get_control_prompt,
    "evidence_first": get_evidence_first_prompt,
    "counterfactual": get_counterfactual_prompt,
    "gepa": get_gepa_prompt
}

def get_prompt(prompt_group: str, question: str, options: dict) -> str:
    """
    Get a prompt for the specified group.
    
    Args:
        prompt_group: One of "control", "evidence_first", "counterfactual", "gepa"
        question: The question text
        options: Dictionary with keys "A", "B", "C", "D" and option texts as values
        
    Returns:
        Formatted prompt string
        
    Raises:
        ValueError: If prompt_group is not recognized
    """
    if prompt_group not in PROMPT_GROUPS:
        raise ValueError(f"Unknown prompt group: {prompt_group}. Must be one of {list(PROMPT_GROUPS.keys())}")
    
    return PROMPT_GROUPS[prompt_group](question, options)

