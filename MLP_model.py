import torch
from torch import nn

class MLP_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256), # Equivalent to dense layer in TF/Keras
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # Indicates to PyTorch how manipulate the data and secuence
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        
        return predictions
    
    def get_model(self):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        print(f"Using {device} device")
        
        return MLP_Net.to(device)
    

if __name__ == '__main__':
    mlp_net = MLP_Net()