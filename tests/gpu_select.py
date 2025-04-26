import unittest
import os
from src.utils.gpu_utils import find_available_gpu

class TestGPUSelection(unittest.TestCase):

    def test_find_available_gpu(self):
        """Test the find_available_gpu function."""
        gpus = find_available_gpu()
        print(f"Available GPUs found: {gpus}")

        # Basic assertion: The result should be a string
        self.assertIsInstance(gpus, str, "Result should be a string.")

        # More specific assertion: Check if it's a comma-separated list of integers or an empty string
        if gpus: # If not empty
            gpu_ids = gpus.split(',')
            for gpu_id in gpu_ids:
                self.assertTrue(gpu_id.strip().isdigit(), f"GPU ID '{gpu_id}' should be an integer.")
        else:
            # If it's empty, it's also a valid result (no GPUs available or selected)
            pass

    def test_find_available_gpu_with_env_var(self):
        """Test if CUDA_VISIBLE_DEVICES environment variable is respected."""
        # Set a specific GPU ID
        test_gpu_id = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = test_gpu_id

        gpus = find_available_gpu()
        print(f"Available GPUs with CUDA_VISIBLE_DEVICES={test_gpu_id}: {gpus}")

        # The function should return the value set in the environment variable
        self.assertEqual(gpus, test_gpu_id, f"Should return the GPU ID from CUDA_VISIBLE_DEVICES ({test_gpu_id}).")

        # Clean up the environment variable
        del os.environ["CUDA_VISIBLE_DEVICES"]

        # Test with multiple GPUs
        test_gpu_ids = "0,3"
        os.environ["CUDA_VISIBLE_DEVICES"] = test_gpu_ids
        gpus = find_available_gpu()
        print(f"Available GPUs with CUDA_VISIBLE_DEVICES={test_gpu_ids}: {gpus}")
        self.assertEqual(gpus, test_gpu_ids, f"Should return the GPU IDs from CUDA_VISIBLE_DEVICES ({test_gpu_ids}).")
        del os.environ["CUDA_VISIBLE_DEVICES"]

if __name__ == '__main__':
    unittest.main()