import unittest
from vllm import RequestOutput

from src.utils.enviroment_utils import huggingface_use_domestic_endpoint
from src.models.vllm_loader import load_model

class TestVLLMModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the model once for all tests in this class."""
        huggingface_use_domestic_endpoint()
        cls.llm = load_model("Qwen/Qwen2.5-7B-Instruct")

    def test_model_loading(self):
        """Test if the model loads without errors."""
        self.assertIsNotNone(self.llm, "Model should be loaded.")

    def test_simple_chat(self):
        """Test a simple chat interaction."""
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": "Hello"
            },
        ]
        outputs = self.llm.chat(conversation)
        self.assertIsInstance(outputs, list, "Output should be a list.")
        self.assertGreater(len(outputs), 0, "Should receive at least one output.")
        # Check the structure of the first output
        output = outputs[0]
        self.assertIsInstance(output, RequestOutput, "Output item should be a RequestOutput object.")
        self.assertIsInstance(output.prompt, str, "Prompt should be a string.")
        self.assertGreater(len(output.outputs), 0, "Should have at least one generation.")
        self.assertIsInstance(output.outputs[0].text, str, "Generated text should be a string.")
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

    def test_multi_turn_chat(self):
        """Test a multi-turn conversation."""
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": "Hello"
            },
            {
                "role": "assistant",
                "content": "Hello! How can I assist you today?"
            },
            {
                "role": "user",
                "content": "Write an essay about the importance of higher education.",
            },
        ]
        outputs = self.llm.chat(conversation)
        self.assertIsInstance(outputs, list, "Output should be a list.")
        self.assertGreater(len(outputs), 0, "Should receive at least one output.")
        # Check the structure of the first output
        output = outputs[0]
        self.assertIsInstance(output, RequestOutput, "Output item should be a RequestOutput object.")
        self.assertIsInstance(output.prompt, str, "Prompt should be a string.")
        self.assertGreater(len(output.outputs), 0, "Should have at least one generation.")
        self.assertIsInstance(output.outputs[0].text, str, "Generated text should be a string.")
        # Optional: Add assertions about the content if possible/needed
        self.assertTrue(len(output.outputs[0].text) > 10, "Generated text should be reasonably long.")
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

if __name__ == '__main__':
    unittest.main()