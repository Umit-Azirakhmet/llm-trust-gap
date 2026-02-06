"""TogetherAI API client for making inference calls with logit extraction."""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import requests

load_dotenv()


class TogetherAIClient:
    """Client for interacting with TogetherAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the TogetherAI client.
        
        Args:
            api_key: TogetherAI API key. If None, loads from TOGETHER_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TogetherAI API key not found. Set TOGETHER_API_KEY in .env file.")
        
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(
        self,
        model: str,
        prompt: str,
        logprobs: bool = True,
        top_logprobs: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Generate a completion with logit information.
        
        Args:
            model: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            prompt: Input prompt
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top logprobs to return
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing:
                - 'text': Generated text
                - 'logprobs': List of logprobs for each token
                - 'tokens': List of token strings
                - 'raw_response': Full API response
        """

        want_logprobs = bool(logprobs)
        logprobs_k = None
        if want_logprobs:
            try:
                logprobs_k = int(top_logprobs)
            except (TypeError, ValueError):
                logprobs_k = 1
            logprobs_k = max(1, min(5, logprobs_k))
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs_k if want_logprobs else None,
        }
        
        #Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=180
            )
            # If Together returns a 4xx/5xx, include response text for debugging.
            if not response.ok:
                raise Exception(
                    "API request failed "
                    f"(status={response.status_code}): {response.text}"
                )
            data = response.json()
            
            #Extract generated text and logprobs
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message = choice.get("message", {})
                text = message.get("content", "")
                
                # Extract logprobs if available.
                #
                # Together returns a structure like:
                #   choice["logprobs"] = {
                #     "token_ids": [...],
                #     "tokens": [...],
                #     "token_logprobs": [...],
                #     "top_logprobs": [ {tokenStr: logprob, ...}, ... ]
                #   }
                #
                # We normalize to a list of per-token dicts:
                #   [{"token": str, "logprob": float, "top_logprobs": dict}, ...]
                logprobs_data = choice.get("logprobs") or {}
                token_logprobs = []
                tokens = []
                if isinstance(logprobs_data, dict) and "tokens" in logprobs_data and "token_logprobs" in logprobs_data:
                    tokens = list(logprobs_data.get("tokens") or [])
                    token_logprob_vals = list(logprobs_data.get("token_logprobs") or [])
                    top_logprobs_vals = list(logprobs_data.get("top_logprobs") or [])

                    n = min(len(tokens), len(token_logprob_vals))
                    for i in range(n):
                        token_logprobs.append(
                            {
                                "token": tokens[i],
                                "logprob": token_logprob_vals[i],
                                "top_logprobs": top_logprobs_vals[i] if i < len(top_logprobs_vals) else None,
                            }
                        )
                
                return {
                    "text": text,
                    "logprobs": token_logprobs,
                    "tokens": tokens,
                    "raw_response": data
                }
            else:
                raise ValueError("No choices in API response")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def get_logit_for_token(
        self,
        model: str,
        prompt: str,
        target_token: str,
        temperature: float = 0.7,
        max_tokens: int = 100
    ) -> Optional[float]:
        """
        Get the logit value for a specific token in the generated output.
        
        Args:
            model: Model identifier
            prompt: Input prompt
            target_token: Token to extract logit for (e.g., "A", "B", "C", "D")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Logit value for the target token, or None if not found
        """
        response = self.generate(
            model=model,
            prompt=prompt,
            logprobs=True,
            top_logprobs=10,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logprobs = response.get("logprobs", [])
        
        #Search for the target token in the logprobs
        for logprob_item in logprobs:
            token = logprob_item.get("token", "").strip()
            #Check if this is the target token (handle whitespace variations)
            if token == target_token or token.strip() == target_token:
                top_logprobs = logprob_item.get("top_logprobs")
                if isinstance(top_logprobs, dict) and target_token in top_logprobs:
                    return top_logprobs.get(target_token)
                # Fallback to the realized token logprob
                return logprob_item.get("logprob")
        
        return None

