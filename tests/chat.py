import unittest
import openai
import os
# from sglang.utils import print_highlight # Keep if needed, or replace with standard print

class TestSGLangChatCompletion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the OpenAI client to connect to the local SGLang server."""
        # It might be better to get the base_url from environment variables or config
        cls.base_url = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1")
        cls.api_key = "None" # Or get from env/config if needed
        try:
            cls.client = openai.Client(base_url=cls.base_url, api_key=cls.api_key)
            # Optional: Add a simple ping or check to ensure the server is running before tests
            # cls.client.models.list() # Example check
        except Exception as e:
            # Fail fast if client can't be created or server isn't reachable
            raise ConnectionError(f"Failed to connect to SGLang server at {cls.base_url}: {e}") from e

    def test_simple_chat_completion(self):
        """Test a basic chat completion request."""
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Or make configurable
        messages = [
            {"role": "user", "content": "List 3 countries and their capitals."},
        ]

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=64,
            )

            print(f"Raw Response: {response}") # Use standard print or logging

            self.assertIsNotNone(response, "Response should not be None.")
            self.assertIsInstance(response, openai.types.chat.chat_completion.ChatCompletion, "Response type should be ChatCompletion.")
            self.assertGreater(len(response.choices), 0, "Should receive at least one choice.")

            choice = response.choices[0]
            self.assertIsNotNone(choice.message, "Choice message should not be None.")
            self.assertIsInstance(choice.message.content, str, "Message content should be a string.")
            self.assertGreater(len(choice.message.content), 0, "Message content should not be empty.")

            print(f"Generated Text: {choice.message.content}")

            # Optional: Add more specific assertions about the content
            # e.g., self.assertIn("France", choice.message.content)

        except openai.APIConnectionError as e:
            self.fail(f"Failed to connect to SGLang server at {self.base_url}: {e}")
        except Exception as e:
            self.fail(f"Chat completion request failed: {e}")

if __name__ == '__main__':
    # Ensure the SGLang server is running before executing tests
    print(f"Attempting to connect to SGLang server at: {os.environ.get('SGLANG_BASE_URL', 'http://127.0.0.1:30000/v1')}")
    print("Please ensure the SGLang OpenAI-compatible server is running.")
    unittest.main()