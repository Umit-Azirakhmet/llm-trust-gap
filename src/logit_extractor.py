"""Extract internal logits for chosen answer tokens."""

from typing import Optional, Dict, Any, List
from src.api_client import TogetherAIClient
from src.parser import extract_answer_only


class LogitExtractor:
    """Extract logit values for specific answer tokens."""
    
    def __init__(self, api_client: TogetherAIClient):
        """
        Initialize the logit extractor.
        
        Args:
            api_client: TogetherAI client instance
        """
        self.api_client = api_client
    
    def extract_logit_for_answer(
        self,
        model: str,
        prompt: str,
        answer_letter: str,
        temperature: float = 0.7,
        max_tokens: int = 100
    ) -> Optional[float]:
        """
        Extract the logit value for a specific answer letter token.
        
        This extracts the logit for the chosen answer token only (A, B, C, or D),
        not for confidence tokens or EOS tokens.
        
        Args:
            model: Model identifier
            prompt: Input prompt
            answer_letter: Answer letter to extract logit for (A, B, C, or D)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Logit value for the answer token, or None if not found
        """
        if answer_letter not in ["A", "B", "C", "D"]:
            raise ValueError(f"Answer letter must be A, B, C, or D, got {answer_letter}")
        
        # Get the full response with logprobs
        response = self.api_client.generate(
            model=model,
            prompt=prompt,
            logprobs=1,
            top_logprobs=10,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logprobs = response.get("logprobs", [])
        tokens = response.get("tokens", [])
        
        # Search through the generated tokens to find the answer letter
        # The answer token should appear early in the generation
        for i, logprob_item in enumerate(logprobs):
            token = logprob_item.get("token", "").strip()
            
            # Check if this token matches the answer letter
            # Handle cases where token might be "A", " A", "A ", etc.
            normalized_token = token.strip()
            if normalized_token == answer_letter:
                # Found the answer token; Together provides logprobs (not raw logits).
                # Prefer the realized token logprob at this position.
                token_logprob = logprob_item.get("logprob")
                if token_logprob is not None:
                    return token_logprob
                # Fallback: if top_logprobs exists and includes the answer, use it.
                top_logprobs = logprob_item.get("top_logprobs")
                if isinstance(top_logprobs, dict) and answer_letter in top_logprobs:
                    return top_logprobs.get(answer_letter)
        
        # If we didn't find the exact token, try to get it from the API's logit extraction
        # This handles cases where the token might be tokenized differently
        logit = self.api_client.get_logit_for_token(
            model=model,
            prompt=prompt,
            target_token=answer_letter,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return logit

    def extract_logprob_for_answer_from_generation(
        self,
        generation: Dict[str, Any],
        answer_letter: str,
    ) -> Optional[float]:
        """
        Extract the realized token logprob for `answer_letter` from an existing generation response.

        This avoids making an extra API call (which was causing timeouts).

        Args:
            generation: Return value from TogetherAIClient.generate(...)
            answer_letter: "A"/"B"/"C"/"D"

        Returns:
            Logprob (float) for the first matching answer token, or None.
        """
        if answer_letter not in ["A", "B", "C", "D"]:
            return None

        logprobs: List[Dict[str, Any]] = generation.get("logprobs") or []
        for item in logprobs:
            tok = (item.get("token") or "").strip()
            if tok == answer_letter:
                lp = item.get("logprob")
                if lp is not None:
                    return lp
                top = item.get("top_logprobs")
                if isinstance(top, dict) and answer_letter in top:
                    return top.get(answer_letter)
        return None

