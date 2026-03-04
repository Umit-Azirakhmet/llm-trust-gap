"""Prompt templates for different experimental groups."""

from prompts.g0_control import get_control_prompt as get_g0_prompt
from prompts.g1_evidence import get_g1_evidence_prompt
from prompts.g2_counterfactual import get_g2_counterfactual_prompt
from prompts.g3_inoculation import get_g3_inoculation_prompt
from prompts.g4_persona import get_g4_persona_prompt
from prompts.g5_scoring_rule import get_g5_scoring_rule_prompt
from prompts.g6_anchoring import get_g6_anchoring_prompt
from prompts.g7_gepa import get_g7_gepa_prompt
from prompts.g8_gepa_alignment import get_g8_gepa_prompt
from prompts.g9_gepa_calibration import get_g9_gepa_prompt

PROMPT_GROUPS = {
    
    # Added by Jessica
    "g0": get_g0_prompt,
    "g1": get_g1_evidence_prompt,
    "g2": get_g2_counterfactual_prompt,
    "g3": get_g3_inoculation_prompt,
    "g4": get_g4_persona_prompt,
    "g5": get_g5_scoring_rule_prompt,
    "g6": get_g6_anchoring_prompt,
    "g7": get_g7_gepa_prompt,
    "g8": get_g8_gepa_prompt,
    "g9": get_g9_gepa_prompt
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

