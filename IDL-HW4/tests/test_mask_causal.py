from typing import Callable
import torch

def test_mask_causal(mask_gen_fn: Callable[[torch.Tensor], torch.Tensor]):
    '''
    Test the causal mask generation function.
    Args:   
        mask_gen_fn (Callable[[torch.Tensor], torch.Tensor]): The function to generate the causal mask.
    '''

    print("Testing Causal Mask ...") 

    # Test Cases    
    batch_sizes = [1, 2, 4]
    seq_lengths = [2, 4, 8]
    feat_lens   = [0, 5, 10]

    # Expected Masks    
    expected_mask1 = torch.tensor([
        [False, True],
        [False, False] 
    ])

    expected_mask2 = torch.tensor([
        [False, True,  True,  True ],
        [False, False, True,  True ],
        [False, False, False, True ],
        [False, False, False, False]
    ])  

    expected_mask3 = torch.tensor([
        [False, True,  True,  True,  True,  True,  True,  True ],
        [False, False, True,  True,  True,  True,  True,  True ],
        [False, False, False, True,  True,  True,  True,  True ],
        [False, False, False, False, True,  True,  True,  True ],
        [False, False, False, False, False, True,  True,  True ],
        [False, False, False, False, False, False, True,  True ],
        [False, False, False, False, False, False, False, True ],
        [False, False, False, False, False, False, False, False]
    ])

    # Test Cases        
    masks = [expected_mask1, expected_mask2, expected_mask3]

    for (batch_size, seq_length, feat_len, expected_mask) in zip(batch_sizes, seq_lengths, feat_lens, masks):
        if feat_len == 0:
            input_tensor = torch.randn(batch_size, seq_length)
        else:   
            input_tensor = torch.randn(batch_size, seq_length, feat_len)
        mask = mask_gen_fn(input_tensor)
        assert torch.equal(mask, expected_mask), f"Causal mask generation failed for batch size {batch_size}, sequence length {seq_length}, and feature length {feat_len}."
    
    print("Test Passed: Causal mask generation is correct.")

def main():
    """
    Main function to run the causal mask tests using the testing framework.
    """
    import sys
    import os

    sys.path.append("C:/Users/Nhu Ngo/Documents/cmu/IDL/HW4/HW4P1/IDL-HW4/IDL-HW4/hw4lib")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from tests directory
    
    # Add the project root to the Python path
    sys.path.append(project_root)
    
    from hw4lib.model.masks import CausalMask
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'CausalMask': [
                {
                    'func': lambda: test_mask_causal(CausalMask),
                    'description': 'Test the causal mask generation'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()