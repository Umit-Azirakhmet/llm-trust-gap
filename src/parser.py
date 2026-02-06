"""Parser for extracting answer and confidence from model outputs."""

import re
from typing import Optional, Tuple, Dict


def parse_answer_and_confidence(output: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse the model output to extract answer letter and confidence level.
    
    Handles various formats like:
    - "B 60%"
    - "Answer: C (75%)"
    - "The answer is A with 80% confidence"
    - "B\n60%"
    
    Args:
        output: Raw model output text
        
    Returns:
        Tuple of (answer_letter, confidence_percentage) or (None, None) if parsing fails
    """
    if not output:
        return None, None
    
    # Normalize whitespace
    output = " ".join(output.split())
    
    # Pattern 1: Simple format "A 60%" or "B 60"
    pattern1 = r'\b([ABCD])\s*(\d+(?:\.\d+)?)\s*%?'
    match1 = re.search(pattern1, output)
    if match1:
        answer = match1.group(1).upper()
        confidence = float(match1.group(2))
        return answer, confidence
    
    # Pattern 2: "Answer: A (60%)" or "Answer is B (75%)"
    pattern2 = r'(?:answer|answer:)\s*([ABCD])\s*[\(]?\s*(\d+(?:\.\d+)?)\s*%?\s*[\)]?'
    match2 = re.search(pattern2, output, re.IGNORECASE)
    if match2:
        answer = match2.group(1).upper()
        confidence = float(match2.group(2))
        return answer, confidence
    
    # Pattern 3: "A with 60% confidence" or "B with confidence 60%"
    pattern3 = r'\b([ABCD])\s+(?:with\s+)?(?:confidence\s+)?(\d+(?:\.\d+)?)\s*%?\s*(?:confidence)?'
    match3 = re.search(pattern3, output, re.IGNORECASE)
    if match3:
        answer = match3.group(1).upper()
        confidence = float(match3.group(2))
        return answer, confidence
    
    # Pattern 4: Just find the answer letter (A, B, C, or D) - confidence might be separate
    pattern4 = r'\b([ABCD])\b'
    match4 = re.search(pattern4, output)
    if match4:
        answer = match4.group(1).upper()
        # Try to find confidence separately
        confidence_pattern = r'(\d+(?:\.\d+)?)\s*%'
        confidence_match = re.search(confidence_pattern, output)
        confidence = float(confidence_match.group(1)) if confidence_match else None
        return answer, confidence
    
    return None, None


def extract_answer_only(output: str) -> Optional[str]:
    """
    Extract only the answer letter from the output.
    
    Args:
        output: Raw model output text
        
    Returns:
        Answer letter (A, B, C, or D) or None
    """
    answer, _ = parse_answer_and_confidence(output)
    return answer


def extract_confidence_only(output: str) -> Optional[float]:
    """
    Extract only the confidence percentage from the output.
    
    Args:
        output: Raw model output text
        
    Returns:
        Confidence percentage (0-100) or None
    """
    _, confidence = parse_answer_and_confidence(output)
    return confidence


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

