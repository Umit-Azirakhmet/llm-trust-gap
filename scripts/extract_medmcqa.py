"""Script to extract 1000 random QA pairs from MedMCQA dataset."""
import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Any


def extract_medmcqa_to_json(
    input_file: str,
    output_file: str,
    num_samples: int = 1000,
    seed: int = None
) -> None:
    """
    Extract random QA pairs from MedMCQA dataset and convert to our JSON format.
    
    MedMCQA format:
    - id: unique identifier
    - question: question text
    - opa, opb, opc, opd: four options
    - cop: correct answer index (0=A, 1=B, 2=C, 3=D)
    - choice_type: multi or single
    - exp: explanation
    - subject_name: subject name
    - topic_name: topic name
    
    Args:
        input_file: Path to MedMCQA CSV file
        output_file: Path to save output JSON file
        num_samples: Number of random samples to extract (default: 1000)
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)
    
    results: List[Dict[str, Any]] = []
    
    #Read all entries from CSV
    all_entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_entries.append(row)
    
    print(f"Total entries in dataset: {len(all_entries)}")
    
    #Filter entries to only include choice_type: single
    single_choice_entries = [
        entry for entry in all_entries 
        if entry.get("choice_type", "").strip().lower() == "single"
    ]
    print(f"Entries with choice_type='single': {len(single_choice_entries)}")
    
    #Randomly sample from single choice entries
    if num_samples > len(single_choice_entries):
        print(f"Warning: Requested {num_samples} samples but only {len(single_choice_entries)} available.")
        print(f"Using all {len(single_choice_entries)} entries.")
        sampled_entries = single_choice_entries
    else:
        sampled_entries = random.sample(single_choice_entries, num_samples)
        print(f"Randomly sampled {num_samples} entries from single choice questions.")
    
    #Convert to our JSON format
    for i, entry in enumerate(sampled_entries):
        try:
            question = entry.get("question", "").strip()
            
            #Extract options
            opa = entry.get("opa", "").strip()
            opb = entry.get("opb", "").strip()
            opc = entry.get("opc", "").strip()
            opd = entry.get("opd", "").strip()
            
            #Skip if any option is missing
            if not all([opa, opb, opc, opd]):
                print(f"Warning: Skipping entry {i} due to missing options")
                continue
            
            #Extract correct answer (cop is 0-indexed: 0=A, 1=B, 2=C, 3=D)
            cop_str = entry.get("cop", "").strip()
            try:
                cop = int(cop_str)
                if 0 <= cop <= 3:
                    correct_answer = ["A", "B", "C", "D"][cop]
                else:
                    print(f"Warning: Invalid cop value {cop} for entry {i}, skipping")
                    continue
            except (ValueError, TypeError):
                print(f"Warning: Could not parse cop '{cop_str}' for entry {i}, skipping")
                continue
            
            #Build options dictionary
            options = {
                "A": opa,
                "B": opb,
                "C": opc,
                "D": opd
            }
            
            #Create result entry
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
                    "original_index": i
                }
            }
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"Warning: Skipping entry {i} due to error: {e}")
            continue
    
    #Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted {len(results)} questions from MedMCQA dataset")
    print(f"Saved to: {output_file}")


def main():
    #PATHS
    #input_file = "scripts/medmcqa.csv"
    output_file = "dataset/medmcqa_10.json"
    num_samples = 1000
    seed = None  #if want same random samples(set to a number)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Number of samples: {num_samples}")
    print(f"Filtering for choice_type: single")
    print()
    
    extract_medmcqa_to_json(
        input_file,
        output_file,
        num_samples=num_samples,
        seed=seed
    )

#so it wont run everytime the file is imported
if __name__ == "__main__":
    main()

