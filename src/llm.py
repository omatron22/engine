# src/llm.py
import requests
import time
import json
import os
from typing import Optional, Dict, Any, List
from src.config import OLLAMA_BASE_URL, DEFAULT_MODEL, REASONING_MODEL

class LLMManager:
    def __init__(self, base_url="http://localhost:11434"):
        """
        Initialize the LLM manager for interacting with Ollama API.
        
        Args:
            base_url: Base URL for the Ollama server
        """
        self.base_url = base_url
        self.available_models = []
        self.session = requests.Session()  # Use a session for connection pooling
        self.model_info = {}
        
        # Attempt to connect to Ollama on initialization
        if self.check_ollama_running():
            print(f"Connected to Ollama server at {base_url}")
            self._refresh_available_models()
        else:
            print(f"Warning: Could not connect to Ollama server at {base_url}")
    
    def check_ollama_running(self) -> bool:
        """
        Check if Ollama server is running.
        
        Returns:
            True if Ollama is running, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error checking Ollama server: {e}")
            return False
            
    def _refresh_available_models(self) -> None:
        """Update the list of available models from Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                if 'models' in models_data:
                    self.available_models = [model['name'] for model in models_data['models']]
                    
                    # Store model details
                    for model in models_data['models']:
                        self.model_info[model['name']] = {
                            'size': model.get('size', 'Unknown'),
                            'modified_at': model.get('modified_at', 'Unknown'),
                            'digest': model.get('digest', 'Unknown')
                        }
                        
                    print(f"Available models: {', '.join(self.available_models)}")
                else:
                    print("Warning: Unexpected response format from Ollama API")
                    self.available_models = []
            else:
                print(f"Error refreshing models: {response.status_code}")
                self.available_models = []
        except Exception as e:
            print(f"Error refreshing available models: {e}")
            self.available_models = []
            
    def ensure_model_available(self, model_name="deepseek-coder:reasoning") -> bool:
        """
        Ensure the specified model is available, pulling it if necessary.
        
        Args:
            model_name: Name of the model to ensure is available
            
        Returns:
            True if model is available (or was pulled successfully), False otherwise
        """
        # Refresh available models
        self._refresh_available_models()
        
        # Check if model is already available
        if model_name in self.available_models:
            print(f"Model {model_name} is available")
            return True
            
        # Try to pull the model
        print(f"Model {model_name} not found. Attempting to pull...")
        try:
            pull_response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # Longer timeout for model pulling
            )
            
            if pull_response.status_code == 200:
                print(f"Successfully pulled model: {model_name}")
                # Update available models list
                self._refresh_available_models()
                return model_name in self.available_models
            else:
                print(f"Error pulling model {model_name}: {pull_response.status_code} - {pull_response.text}")
                return False
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models with their details.
        
        Returns:
            List of dictionaries with model information
        """
        self._refresh_available_models()
        
        model_list = []
        for model_name in self.available_models:
            model_list.append({
                'name': model_name,
                'details': self.model_info.get(model_name, {})
            })
            
        return model_list
            
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                         model: Optional[str] = None, temperature: float = 0.7, 
                         max_tokens: int = 2048, retry_count: int = 2) -> Optional[str]:
        """
        Generate a response using the Ollama API.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to control model behavior
            model: Model to use, defaults to DEFAULT_MODEL from config
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            retry_count: Number of retries on failure
            
        Returns:
            Generated response or None if an error occurred
        """
        if model is None:
            model = DEFAULT_MODEL
        
        # Check if Ollama is running
        if not self.check_ollama_running():
            print("Error: Ollama server is not running")
            return None
            
        # Ensure model is available
        if not self.ensure_model_available(model):
            print(f"Warning: Model {model} not available. Trying fallback model.")
            # Try using the DEFAULT_MODEL as fallback
            if model != DEFAULT_MODEL and self.ensure_model_available(DEFAULT_MODEL):
                model = DEFAULT_MODEL
                print(f"Using fallback model: {model}")
            else:
                print("Error: No suitable model available.")
                return None
        
        # Prepare the request
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": max_tokens
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        # Try multiple times in case of temporary issues
        for attempt in range(retry_count + 1):
            try:
                # Send the request
                print(f"Generating response with model: {model} (attempt {attempt+1}/{retry_count+1})")
                response = self.session.post(
                    f"{self.base_url}/api/generate", 
                    json=data,
                    timeout=120  # 2 minutes timeout for generation
                )
                
                if response.status_code == 200:
                    return response.json().get('response', '')
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    
                    # Only retry for server errors
                    if response.status_code >= 500 and attempt < retry_count:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        return None
            except Exception as e:
                print(f"Error generating response (attempt {attempt+1}/{retry_count+1}): {e}")
                if attempt < retry_count:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return None
        
        return None
    
    def generate_reasoning_response(self, prompt: str, system_prompt: Optional[str] = None,
                                   max_tokens: int = 4096) -> Optional[str]:
        """
        Generate a response using the reasoning-optimized model.
        
        This is specifically designed for complex strategic reasoning tasks.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to control model behavior
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response or None if an error occurred
        """
        return self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            model=REASONING_MODEL,
            temperature=0.4,  # Lower temperature for more deterministic reasoning
            max_tokens=max_tokens
        )
        
    def analyze_business_strategy(self, context: str, query: str, 
                                risk_tolerance: Optional[str] = None) -> Optional[str]:
        """
        Specialized method for business strategy analysis.
        
        Args:
            context: Business context and data
            query: Strategic question or analysis request
            risk_tolerance: Optional risk tolerance level (High/Medium/Low)
            
        Returns:
            Strategic analysis response
        """
        system_prompt = """
        You are the QmiracTM Strategic Advisor, an expert business strategist with decades of experience.
        Your task is to analyze business data and provide insightful, actionable strategic recommendations.
        Focus on being specific, practical, and data-driven. Avoid generic advice.
        
        Structure your analysis clearly with:
        1. Assessment of the current situation
        2. Strategic opportunities and challenges
        3. Concrete, implementable recommendations
        4. Expected outcomes and success metrics
        """
        
        # Add risk tolerance context if provided
        if risk_tolerance:
            risk_context = f"\nBusiness Risk Tolerance: {risk_tolerance}"
            if risk_tolerance.lower() == "high":
                risk_context += "\nConsider more aggressive strategies with higher risk/reward profiles."
            elif risk_tolerance.lower() == "low":
                risk_context += "\nFocus on conservative approaches with steady, reliable outcomes."
            else:  # Medium
                risk_context += "\nBalance opportunity and risk with moderate approaches."
                
            system_prompt += risk_context
        
        prompt = f"""
        Based on the following business context and data, please analyze and respond to this strategic query.
        
        ## BUSINESS CONTEXT
        {context}
        
        ## STRATEGIC QUERY
        {query}
        
        Provide a comprehensive strategic analysis with practical, actionable recommendations.
        """
        
        return self.generate_reasoning_response(prompt, system_prompt)