"""Main experiment runner for LLM trust gap research."""

import argparse
import sys
import math
from pathlib import Path
from typing import Dict, Any, List
from src.api_client import TogetherAIClient
from src.logit_extractor import LogitExtractor
from src.parser import parse_answer_and_confidence, validate_parsed_output
from src.utils import (
    load_json, save_json, load_yaml, format_output_filename,
    validate_dataset_entry, get_timestamp
)
from prompts import get_prompt


class ExperimentRunner:
    """Run experiments to collect answer, confidence, and logit data."""
    
    def __init__(self, model: str, prompt_group: str, dataset_path: str):
        """
        Initialize the experiment runner.
        
        Args:
            model: Model identifier (e.g., "llama-3.1-8b")
            prompt_group: Prompt group name (e.g., "control")
            dataset_path: Path to JSON dataset file
        """
        self.model = model
        self.prompt_group = prompt_group
        self.dataset_path = dataset_path
        
        # Load model configuration
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        models_config = load_yaml(str(config_path))
        if model not in models_config["models"]:
            raise ValueError(f"Unknown model: {model}. Available: {list(models_config['models'].keys())}")
        
        self.model_id = models_config["models"][model]["id"]
        self.default_params = models_config.get("default_params", {})
        
        # Initialize API client and logit extractor
        self.api_client = TogetherAIClient()
        self.logit_extractor = LogitExtractor(self.api_client)
        
        # Load dataset
        self.dataset = load_json(dataset_path)
        if not isinstance(self.dataset, list):
            raise ValueError("Dataset must be a list of question entries")
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
    
    def process_question(self, entry: Dict[str, Any], question_id: int) -> Dict[str, Any]:
        """
        Process a single question entry.
        
        Args:
            entry: Dataset entry with question, options, and correct_answer
            question_id: Unique identifier for this question
            
        Returns:
            Result dictionary with all extracted information
        """
        # Validate entry
        if not validate_dataset_entry(entry):
            raise ValueError(f"Invalid dataset entry at index {question_id}")
        
        question = entry["question"]
        options = entry["options"]
        gold_answer = entry["correct_answer"]
        
        #get prompt
        prompt = get_prompt(self.prompt_group, question, options)
        
        # Get model response
        try:
            response = self.api_client.generate(
                model=self.model_id,
                prompt=prompt,
                logprobs=self.default_params.get("logprobs", True),
                top_logprobs=self.default_params.get("top_logprobs", 10),
                temperature=self.default_params.get("temperature", 0.7),
                max_tokens=self.default_params.get("max_tokens", 100)
            )
            model_output = response.get("text", "")
        except Exception as e:
            print(f"Error generating response for question {question_id}: {e}")
            model_output = ""
        
        # Parse answer and confidence
        outputted_answer, outputted_confidence = parse_answer_and_confidence(model_output)
        
        # Validate parsed output
        if not validate_parsed_output(outputted_answer, outputted_confidence):
            print(f"Warning: Failed to parse output for question {question_id}: {model_output}")
        
        # Extract internal logit (Together provides token logprobs) for the chosen answer.
        internal_logit = None
        internal_logit_normalized_100 = None
        if outputted_answer:
            try:
                internal_logit = self.logit_extractor.extract_logprob_for_answer_from_generation(
                    generation=response,
                    answer_letter=outputted_answer,
                )
            except Exception as e:
                print(f"Error extracting logit for question {question_id}: {e}")

        # Normalize internal logit (actually token logprob) into a 0-100 score:
        if internal_logit is not None:
            try:
                internal_logit_normalized_100 = max(0.0, min(100.0, math.exp(float(internal_logit)) * 100.0))
            except Exception:
                internal_logit_normalized_100 = None
        
        # Check if answer is correct
        is_correct = (outputted_answer == gold_answer) if outputted_answer else False
        
        # Build result entry
        result = {
            "question_id": question_id,
            "question": question,
            "options": options,
            "gold_answer": gold_answer,
            "outputted_answer": outputted_answer,
            "outputted_confidence": outputted_confidence,
            "is_correct": is_correct,
            "internal_logit": internal_logit,
            "internal_logit_normalized_100": internal_logit_normalized_100,
            "model_output": model_output,
            "metadata": {
                "model": self.model,
                "model_id": self.model_id,
                "prompt_group": self.prompt_group,
                "dataset": Path(self.dataset_path).stem
            }
        }
        
        return result
    
    def run(self, output_dir: str = "outputs/results") -> str:
        """
        Run the experiment on all questions in the dataset.
        
        Args:
            output_dir: Directory to save output JSON file
            
        Returns:
            Path to the output file
        """
        print(f"Running experiment: {self.model} / {self.prompt_group} / {Path(self.dataset_path).stem}")
        print(f"Processing {len(self.dataset)} questions...")
        
        for i, entry in enumerate(self.dataset):
            try:
                result = self.process_question(entry, i)
                self.results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(self.dataset)} questions...")
                    
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                # Add error entry
                self.results.append({
                    "question_id": i,
                    "error": str(e),
                    "metadata": {
                        "model": self.model,
                        "prompt_group": self.prompt_group,
                        "dataset": Path(self.dataset_path).stem
                    }
                })
        
        # Save results
        timestamp = get_timestamp()
        dataset_name = Path(self.dataset_path).stem
        output_filename = format_output_filename(
            self.model, self.prompt_group, dataset_name, timestamp
        )
        output_path = Path(output_dir) / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_json(self.results, str(output_path))
        
        print(f"\nExperiment complete!")
        print(f"Results saved to: {output_path}")
        print(f"Total questions processed: {len(self.results)}")
        
        return str(output_path)


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run LLM trust gap experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., llama-3.1-8b)"
    )
    parser.add_argument(
        "--prompt-group",
        type=str,
        required=True,
        choices=["control", "evidence_first", "counterfactual", "gepa"],
        help="Prompt group to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSON dataset file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Directory to save output JSON file (default: outputs/results)"
    )
    
    args = parser.parse_args()
    
    try:
        runner = ExperimentRunner(
            model=args.model,
            prompt_group=args.prompt_group,
            dataset_path=args.dataset
        )
        output_path = runner.run(output_dir=args.output_dir)
        print(f"\nSuccess! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

