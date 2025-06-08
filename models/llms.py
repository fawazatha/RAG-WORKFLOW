from langchain_google_genai import ChatGoogleGenerativeAI
from setup import GOOGLE_API_KEY
from typing import Optional

class LLMModels:
    """
    Define LLM models with OpenAI
    """
    def __init__(
            self, 
            temperature: Optional[float] = 0.5, 
            max_tokens: Optional[int] = 8000
            ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_cv = self.create_llm_cv(self.temperature, self.max_tokens)
    # ***** Define a function for intialize OpenAI LLM
    def create_llm_cv(
            self, 
            temperature: Optional[float], 
            max_tokens: Optional[int]
            ):
        llm_model = ChatGoogleGenerativeAI(
            temperature=temperature,
            max_tokens=max_tokens,
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY, 
        )

        return llm_model