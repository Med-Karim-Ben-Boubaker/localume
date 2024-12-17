from dataclasses import dataclass
from typing import Optional
import google.generativeai as genai
import time
import requests

@dataclass
class GeminiConfig:
    """Configuration settings for Gemini model"""
    model_name: str = "gemini-1.5-flash-8b"
    temperature: float = 0.2
    top_p: float = 0.4
    top_k: int = 40
    max_output_tokens: int = 1024
    response_mime_type: str = "text/plain"
    max_retries: int = 3
    retry_delay: int = 10

class GeminiService:
    """
    Service for handling interactions with Google's Gemini API.
    Provides query optimization and content generation capabilities.
    """
    
    def __init__(self, api_key: str, config: Optional[GeminiConfig] = None) -> None:
        """
        Initialize the Gemini service.
        
        Args:
            api_key (str): API key for Gemini
            config (Optional[GeminiConfig]): Configuration settings
        """
        self.config = config or GeminiConfig()
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens,
                "response_mime_type": self.config.response_mime_type,
            }
        )

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using Gemini model.
        
        Args:
            prompt (str): The prompt for content generation
            
        Returns:
            str: Generated content
            
        Raises:
            RuntimeError: If content generation fails
        """
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            raise ValueError("Invalid response format or empty text")
        except Exception as e:
            raise RuntimeError(f"Content generation failed: {str(e)}")

    def optimize_search_query(self, user_query: str) -> str:
        """
        Optimize a search query by extracting main topics and removing unnecessary words.
        
        Args:
            user_query (str): The original search query
            
        Returns:
            str: The optimized query
        """
        prompt = f"""
        Extract the main topic from the query by removing all unnecessary words.
        Return only the extracted topic without any additional text or explanation.

        Examples:
        Input: give me the document that talks about Review and Evaluation of Clinical Data
        Output: Review and Evaluation of Clinical Data

        Input: there is a document that talks about office of state fire marshal give it to me
        Output: office of state fire marshal

        Input: I need to find information about Hanford RCRA permits in the documents
        Output: Hanford RCRA permits

        Input: {user_query}
        Output:"""

        try:
            response = self.model.generate_content(prompt)

            if hasattr(response, "text") and response.text:
                optimized_query = response.text.strip()
                optimized_query = optimized_query.replace("Output:", "").strip()
                return optimized_query

            raise ValueError("Invalid response format or empty text")

        except ValueError as e:
            print(f"Value Error: {e}")
            return user_query

        except Exception as e:
            print(f"Error in query optimization: {str(e)}")
            return user_query

        return user_query