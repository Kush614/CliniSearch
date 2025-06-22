# utils/api_clients.py

import os
import google.generativeai as genai
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GeminiClient:
    """
    A client for interacting with the Google Gemini API.
    This client uses the 'gemini-1.5-flash-latest' model, which is multimodal
    and can handle both text-only and text-with-image inputs.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API Key not provided. Please set it in your .env file.")
        
        try:
            genai.configure(api_key=api_key)
            # Use the latest fast and powerful multimodal model for all tasks.
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("Gemini Client initialized successfully with model 'gemini-1.5-flash-latest'.")
        except Exception as e:
            print(f"Failed to configure Gemini Client: {e}")
            raise

    def generate_text(self, prompt: str, temperature: float = 0.3) -> str:
        """Generates text using the Gemini model."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            return response.text
        except Exception as e:
            error_message = f"Error during Gemini text generation: {e}"
            print(error_message)
            return f"Error: Could not get a response from Gemini. Details: {e}"

    def analyze_image(self, prompt: str, image_bytes: bytes) -> str:
        """Analyzes an image and text prompt using the Gemini multimodal model."""
        try:
            # The model automatically detects image mime type, but specifying is good practice.
            # Assuming JPEG for this example, but could be PNG, WEBP, etc.
            image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
            prompt_parts = [prompt, image_parts[0]]
            
            response = self.model.generate_content(prompt_parts)
            return response.text
        except Exception as e:
            error_message = f"Error during Gemini image analysis: {e}"
            print(error_message)
            return f"Error: Could not analyze the image with Gemini. Details: {e}"

class ClaudeClient:
    """
    An optional client for interacting with the Anthropic Claude API.
    It will be disabled if the API key is not found.
    """
    def __init__(self, api_key: str):
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not found. Claude client will be unavailable.")
            self.client = None
        else:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                print("Claude Client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Claude Client: {e}")
                self.client = None

    def generate_text(self, prompt: str, model: str = "claude-3-sonnet-20240229") -> str:
        """
        Generates text using a Claude model.
        Available models:
        - "claude-3-opus-20240229" (Most powerful)
        - "claude-3-sonnet-20240229" (Balanced)
        - "claude-3-haiku-20240307" (Fastest, most compact)
        """
        if not self.client:
            return "Error: Claude client is not configured. Please provide an ANTHROPIC_API_KEY."
        
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=2048, # Max output tokens
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # The response is a list of content blocks; we extract the text from the first one.
            return message.content[0].text
        except Exception as e:
            error_message = f"Error during Claude text generation: {e}"
            print(error_message)
            return f"Error: Could not get a response from Claude. Details: {e}"

# --- Global Client Initialization ---
# These clients are initialized once when the module is imported,
# making them available to the rest of the application.

try:
    gemini_client = GeminiClient(api_key=os.getenv("GOOGLE_API_KEY"))
except ValueError as e:
    # This will still raise the error and stop the app if the Gemini key is missing,
    # as Gemini is a core component of this app.
    print(f"CRITICAL ERROR: {e}")
    gemini_client = None 

claude_client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY"))