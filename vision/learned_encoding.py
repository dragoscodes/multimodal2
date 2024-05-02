from torch import nn

def load_learned_encoding(vision_dim, llm_dim, option):
    if(option == 'linear'):
        return nn.Sequential(
            nn.Linear(vision_dim, llm_dim)
        )
    else:
        raise ValueError(f'Unexpected learned encoding type: {option}')