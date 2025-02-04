"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch

from torch import nn

class YourModel(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        #TODO: your code here
        pass
    
    def forward(self, x: torch.Tensor):
        #TODO: your code here
        pass
