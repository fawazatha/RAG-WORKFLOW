from langchain_google_genai import GoogleGenerativeAIEmbeddings
from setup import GOOGLE_API_KEY
from dotenv import load_dotenv 
import os 

load_dotenv(override=True)

os.getenv('GOOGLE_API_KEY')

class EmbeddingsModels():
    def __init__(self):
        # Initialize the Gemini embedding model
        self.embedding_large_gemini = self.create_embedding_large_gemini()

    def create_embedding_large_gemini(self):
        """
        Creates and returns a GoogleGenerativeAIEmbeddings instance configured for Gemini.
        """
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        )
        return embedding