"""Parser for extracting answer and confidence from model outputs."""

import re
from typing import Optional, Tuple, Dict


def parse_answer_and_confidence(output: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse the model output to extract answer letter and confidence level.
    
    Only accepts simple format:
    - "A 60%" or "A 60" (with or without % sign)
    - Only integer values (no decimals), e.g., "B 64" or "B 64%"
  
    Args:
        output: Raw model output text
        
    Returns:
        Tuple of (answer_letter, confidence_percentage) or (None, None) if parsing fails
    """
    if not output:
        return None, None
    
    #normalize whitespace
    output = " ".join(output.split())
    
    #"A 64" or "A 64%"
    pattern = r'^([ABCD])\s+(\d+)\s*%?\s*$'
    match = re.match(pattern, output.strip())
    
    if match:
        answer = match.group(1).upper()
        confidence = float(match.group(2))
        return answer, confidence
    
    return None, None


def validate_parsed_output(answer: Optional[str], confidence: Optional[float]) -> bool:
    """
    Validate that parsed output is reasonable.
    
    Args:
        answer: Parsed answer letter
        confidence: Parsed confidence percentage
        
    Returns:
        True if valid, False otherwise
    """
    if answer not in ["A", "B", "C", "D"]:
        return False
    if confidence is not None and (confidence < 0 or confidence > 100):
        return False
    return True

