"""Utility functions for the project."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_output_filename(model: str, prompt_group: str, dataset: str, timestamp: str = None) -> str:
    """
    Format output filename for experiment results.
    
    Args:
        model: Model identifier
        prompt_group: Prompt group name
        dataset: Dataset name
        timestamp: Optional timestamp (if None, generates one)
        
    Returns:
        Formatted filename
    """
    if timestamp is None:
        timestamp = get_timestamp()
    return f"{model}_{prompt_group}_{dataset}_{timestamp}.json"


def validate_dataset_entry(entry: Dict[str, Any]) -> bool:
    """
    Validate that a dataset entry has the required fields.
    
    Args:
        entry: Dataset entry dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["question", "options", "correct_answer"]
    for field in required_fields:
        if field not in entry:
            return False
    
    # Validate options
    if not isinstance(entry["options"], dict):
        return False
    
    required_options = ["A", "B", "C", "D"]
    for opt in required_options:
        if opt not in entry["options"]:
            return False
    
    # Validate correct_answer
    if entry["correct_answer"] not in ["A", "B", "C", "D"]:
        return False
    
    return True

