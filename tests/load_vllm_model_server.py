import unittest
import time
import threading

from src.utils.enviroment_utils import huggingface_use_domestic_endpoint, set_python_path
from src.utils.gpu_utils import find_available_gpu
from src.models.vllm_loader import VLLMServer

class TestVLLMServerConcurrency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up environment and load the server once."""
        huggingface_use_domestic_endpoint()
        set_python_path()
        # Consider using a smaller/faster model for testing if possible
        cls.llm = VLLMServer("meta-llama/Llama-3.1-8B-Instruct", cuda=find_available_gpu())
        # Wait a bit for the server to potentially initialize if it runs in background
        time.sleep(5) # Adjust sleep time if necessary

    def test_concurrent_requests(self):
        """Test handling multiple concurrent chat requests."""
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": "What is the capital of France?"
            },
        ]

        results = {}
        num_threads = 10 # Reduce number of threads for faster testing
        threads = []

        def worker(idx):
            try:
                outputs = self.llm.chat(messages=conversation)
                results[idx] = outputs # Store results
                self.assertIsInstance(outputs, list, f"Thread {idx}: Output should be a list.")
                self.assertGreater(len(outputs), 0, f"Thread {idx}: Should receive at least one output.")
                # Basic check on output structure (assuming similar structure to non-server chat)
                output = outputs[0]
                self.assertIsInstance(output, dict, f"Thread {idx}: Output item should be a dict.")
                self.assertIn("generated_text", output, f"Thread {idx}: Output dict should contain 'generated_text'.")
                self.assertIsInstance(output["generated_text"], str, f"Thread {idx}: Generated text should be a string.")
                print(f"Thread {idx} successful. Output length: {len(output['generated_text'])}")
            except Exception as e:
                results[idx] = e # Store exception
                print(f"Thread {idx} failed: {e}")

        start_time = time.time()
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        elapsed_time = time.time() - start_time
        print(f"All {num_threads} threads done. Elapsed: {elapsed_time:.2f}s")

        # Check if all threads completed without raising an exception
        failed_threads = {idx: res for idx, res in results.items() if isinstance(res, Exception)}
        successful_threads = num_threads - len(failed_threads)

        print(f"Successful threads: {successful_threads}/{num_threads}")
        if failed_threads:
            print("Failed threads:")
            for idx, error in failed_threads.items():
                print(f"  Thread {idx}: {error}")

        self.assertEqual(len(failed_threads), 0, f"{len(failed_threads)} threads failed execution.")
        self.assertEqual(successful_threads, num_threads, "All threads should complete successfully.")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources if necessary (e.g., explicitly stop the server if possible)."""
        # If VLLMServer has a shutdown method, call it here.
        # e.g., cls.llm.shutdown()
        print("Finished testing VLLM Server.")
        pass # Add cleanup if needed

if __name__ == '__main__':
    unittest.main()