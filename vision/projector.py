from torch import nn

def load_vision_projector(input, output):
    return nn.Sequential(
        nn.Linear(input, input+output),
        nn.ReLU(),
        nn.Linear(input+output, output),
        # nn.Dropout(0.1)
    )
