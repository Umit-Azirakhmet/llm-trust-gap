"""Script to create train and validation datasets from MedMCQA and MMLU.

Train: 25 MedMCQA rows not in medmcqa_1000.json + 25 MMLU law rows not in mmlu_professional_law_1000.json
Validation: 25 MedMCQA rows not in medmcqa_1000.json + 25 MMLU law rows not in mmlu_professional_law_1000.json
"""

import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Set


def load_existing_medmcqa_ids(existing_file: str) -> Set[str]:
    """Load IDs from existing medmcqa_1000.json file."""
    with open(existing_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['metadata']['id'] for item in data}


def load_existing_mmlu_indices(existing_file: str) -> Set[int]:
    """Load original_index values from existing mmlu_professional_law_1000.json file."""
    with open(existing_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['metadata']['original_index'] for item in data}


def medmcqa_row_to_format(entry: Dict[str, str], original_index: int = None) -> Dict[str, Any]:
    """Convert a MedMCQA CSV row to our JSON format."""
    question = entry.get("question", "").strip()
    
    # Extract options
    opa = entry.get("opa", "").strip()
    opb = entry.get("opb", "").strip()
    opc = entry.get("opc", "").strip()
    opd = entry.get("opd", "").strip()
    
    # Extract correct answer (cop is 0-indexed: 0=A, 1=B, 2=C, 3=D)
    cop_str = entry.get("cop", "").strip()
    cop = int(cop_str)
    correct_answer = ["A", "B", "C", "D"][cop]
    
    # Build options dictionary
    options = {
        "A": opa,
        "B": opb,
        "C": opc,
        "D": opd
    }
    
    # Create result entry
    result_entry = {
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "metadata": {
            "source": "MedMCQA",
            "id": entry.get("id", ""),
            "choice_type": entry.get("choice_type", ""),
            "subject_name": entry.get("subject_name", ""),
            "topic_name": entry.get("topic_name", ""),
            "original_index": original_index if original_index is not None else 0
        }
    }
    
    return result_entry


def mmlu_item_to_format(item: dict, original_index: int) -> dict:
    """Convert one MMLU item to MedMCQA-style format."""
    import uuid
    choices = item["choices"]
    answer_idx = item["answer"]
    letter = chr(ord("A") + answer_idx)
    options = {chr(ord("A") + i): choices[i] for i in range(len(choices))}
    return {
        "question": item["question"],
        "options": options,
        "correct_answer": letter,
        "metadata": {
            "source": "MMLU",
            "id": str(uuid.uuid4()),
            "choice_type": "single",
            "subject_name": item.get("subject", "professional_law"),
            "topic_name": "",
            "original_index": original_index,
        },
    }


def main():
    random.seed(42)  # For reproducibility
    
    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    MEDMCQA_CSV = REPO_ROOT / "scripts" / "medmcqa.csv"
    MEDMCQA_1000 = REPO_ROOT / "dataset" / "medmcqa_1000.json"
    MMLU_JSON = REPO_ROOT / "scripts" / "mmlu_professional_law.json"
    MMLU_1000 = REPO_ROOT / "dataset" / "mmlu_professional_law_1000.json"
    DATASET_DIR = REPO_ROOT / "dataset"
    
    # Load existing datasets to identify what's already used
    existing_medmcqa_ids = load_existing_medmcqa_ids(MEDMCQA_1000)
    existing_mmlu_indices = load_existing_mmlu_indices(MMLU_1000)
    
    # Load full MedMCQA dataset
    all_medmcqa_entries = []
    with open(MEDMCQA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_medmcqa_entries.append(row)
    
    # Filter MedMCQA: only single choice, and not in existing 1000
    available_medmcqa = []
    for entry in all_medmcqa_entries:
        if entry.get("choice_type", "").strip().lower() == "single":
            entry_id = entry.get("id", "").strip()
            if entry_id and entry_id not in existing_medmcqa_ids:
                # Validate that all options are present
                if all([entry.get("opa", "").strip(), 
                       entry.get("opb", "").strip(),
                       entry.get("opc", "").strip(),
                       entry.get("opd", "").strip()]):
                    try:
                        cop = int(entry.get("cop", "").strip())
                        if 0 <= cop <= 3:
                            available_medmcqa.append(entry)
                    except (ValueError, TypeError):
                        continue
    
    if len(available_medmcqa) < 50:
        raise SystemExit(f"Not enough available MedMCQA entries: need 50, have {len(available_medmcqa)}")
    
    # Sample 25 for train and 25 for validation (non-overlapping)
    random.shuffle(available_medmcqa)
    train_medmcqa = available_medmcqa[:25]
    val_medmcqa = available_medmcqa[25:50]
    
    # Convert MedMCQA entries to format
    train_data = [medmcqa_row_to_format(entry, i) for i, entry in enumerate(train_medmcqa)]
    val_data = [medmcqa_row_to_format(entry, i) for i, entry in enumerate(val_medmcqa)]
    
    # Load full MMLU dataset
    with open(MMLU_JSON, 'r', encoding='utf-8') as f:
        all_mmlu_data = json.load(f)
    
    # Filter MMLU: not in existing 1000
    available_mmlu = []
    for idx, item in enumerate(all_mmlu_data):
        if idx not in existing_mmlu_indices:
            available_mmlu.append((idx, item))
    
    if len(available_mmlu) < 50:
        raise SystemExit(f"Not enough available MMLU entries: need 50, have {len(available_mmlu)}")
    
    # Sample 25 for train and 25 for validation (non-overlapping)
    random.shuffle(available_mmlu)
    train_mmlu = available_mmlu[:25]
    val_mmlu = available_mmlu[25:50]
    
    # Convert MMLU entries to format and add to train and validation
    for idx, item in train_mmlu:
        train_data.append(mmlu_item_to_format(item, idx))
    
    for idx, item in val_mmlu:
        val_data.append(mmlu_item_to_format(item, idx))
    
    # Save datasets
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = DATASET_DIR / "train.json"
    val_path = DATASET_DIR / "validation.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created train.json with {len(train_data)} items ({len(train_medmcqa)} MedMCQA, {len(train_mmlu)} MMLU)")
    print(f"✓ Created validation.json with {len(val_data)} items ({len(val_medmcqa)} MedMCQA, {len(val_mmlu)} MMLU)")


if __name__ == "__main__":
    main()

