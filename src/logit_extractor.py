"""Extract internal logits for chosen answer tokens."""

from typing import Optional, Dict, Any, List


class LogitExtractor:
    """Extract logit values for specific answer tokens."""
    
    def extract_logprob(
        self,
        generation: Dict[str, Any],
        answer_letter: str,
    ) -> Optional[float]:
        """
        Extract the logprob for an answer letter from an existing generation response.
        
        This avoids making an extra API call by reusing the generation response.

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
        return None

