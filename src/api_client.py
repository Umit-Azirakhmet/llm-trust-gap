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
        logprobs: bool,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Generate a completion with logit information.
        
        Args:
            model: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            prompt: Input prompt
            logprobs: Whether to return log probabilities (only for generated tokens)
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
        logprobs_k = 1 if want_logprobs else None
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
                # Together currently uses two different shapes for chat logprobs:
                #
                # 1) Llama-style:
                #    choice["logprobs"] = {
                #      "token_ids": [...],
                #      "tokens": [...],
                #      "token_logprobs": [...],
                #    }
                #
                # 2) Mistral / Gemma / Qwen-style:
                #    choice["logprobs"] = {
                #      "content": [
                #        {"token": str, "logprob": float, ...},
                #        ...
                #      ]
                #    }
                #
                # We normalize both into a flat list of per-token dicts:
                #   [{"token": str, "logprob": float}, ...]
                logprobs_data = choice.get("logprobs") or {}
                token_logprobs = []
                tokens = []

                if isinstance(logprobs_data, dict):
                    # Case 1: Llama-style tokens + token_logprobs arrays
                    if "tokens" in logprobs_data and "token_logprobs" in logprobs_data:
                        tokens = list(logprobs_data.get("tokens") or [])
                        token_logprob_vals = list(logprobs_data.get("token_logprobs") or [])

                        n = min(len(tokens), len(token_logprob_vals))
                        for i in range(n):
                            token_logprobs.append(
                                {
                                    "token": tokens[i],
                                    "logprob": token_logprob_vals[i],
                                }
                            )
                    # Case 2: content list of token/logprob dicts
                    elif "content" in logprobs_data and isinstance(logprobs_data.get("content"), list):
                        for item in logprobs_data.get("content", []):
                            tok = item.get("token")
                            lp = item.get("logprob")
                            if tok is None or lp is None:
                                continue
                            tokens.append(tok)
                            token_logprobs.append(
                                {
                                    "token": tok,
                                    "logprob": lp,
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

