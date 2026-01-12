#Importing all the main libraries.
#Our simulation requires both PyTorch + Flower.
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from typing import Dict, Optional, Tuple

#Laying down the blueprint for the Neural Network.
#The network should match the architecture exactly used by the local clients for training.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Defining the layers and their functions.
        #Simple convultional layers to introduce multi-channel images.
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #Dropout for regularization.
        self.dropout = nn.Dropout(0.25)
        #Standard linear layer for converting 1D array input into 1D array outputs.
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    #We define a single forward pass for the input x.
    def forward(self, x):
        #Apply ReLu function to the convolutional layer. Repeat twice.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #Pooling using 2x2 window and max-value selection.
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        #We flatten x into a 1D array before passing it into the linear layer.
        x = torch.flatten(x, 1)
        #Applying ReLu for the last time for non-negative activation values.
        x = F.relu(self.fc1(x))
        return self.fc2(x)

#We define the main evaluative function that prepares, organizes and feeds the testset into the model.
#Also gets the results such as evaluation loss and accuracy.
def get_evaluate_fn():
    #Pre-preparing the data. We convert the PIL image into PyTorch tensor format and normalize the matrix values.
    #The transform function lays down the blueprint for the pre-processing of the image.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #We execute the transform function after downloading the dataset.
    trainset = torchvision.datasets.MNIST("./data", train = True, download = True, transform = transform)
    
    #We use 40% of the leftover dataset for our simulation, since the rest will be used in client training.
    indices = list(range(36000, 60000))
    testset = Subset(trainset, indices)
    #We define the Dataloader that feeds the batches into our neural network for evaluation purpose.
    testloader = DataLoader(testset, batch_size = 32)

    #Define the actual evaluation process. We get flower parameters as Numpy arrays and an integer mentioning server round.
    #The parameter arrays have been aggregated already via the use of flower.
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        #Initialize the model blueprint.
        model = Net()
        #Map the name of the layers to their respective parameter arrays (for now) for preparing the network.
        params_dict = zip(model.state_dict().keys(), parameters)
        #Converts the Numpy arrays into PyTorch tensors, and store them in an ordered dictionary.
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        #Running the evaluation. Tells the model to not use Dropout function, since it is only needed during training time. 
        model.eval()
        loss, correct = 0.0, 0
        #Tells model not to record gradients, saving memory.
        with torch.no_grad():
            #Use the testLoader to load out the images in batches.
            for images, labels in testloader:
                outputs = model(images)
                loss += F.cross_entropy(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        #Print the accuracy and loss.
        accuracy = correct / len(testset)
        print(f"\n[ROUND {server_round}] Server-side Evaluation on 40% Leftover Data")
        print(f"Accuracy: {accuracy:.4f}, Loss: {loss/len(testloader):.4f}\n")
        return loss, {"accuracy": accuracy}

    return evaluate

#The main execution block.
if __name__ == "__main__":
    #Flower requires us to initialize the "strategy" for the implementation.
    #This is basically specifying all the parameter like num_clients, num_rounds, etc. for the model run.
    strategy = fl.server.strategy.FedAvg(
        #Tells python NOT to start the training round unless until 2 clients have atleast joined the port.
        min_fit_clients = 2,
        min_available_clients = 2,
        min_evaluate_clients = 2,
        #Running the evaluation function.
        evaluate_fn = get_evaluate_fn(),
    )
    print("Global Server starting at 0.0.0.0:8080...")
    #Starting the actual server.
    fl.server.start_server(
        #Telling it to listen to all containers. Especially important since we are using Docker.
        #Connecting to port 8080.
        server_address = "0.0.0.0:8080",
        #Configuring no. of FedAvg rounds.
        config = fl.server.ServerConfig(num_rounds = 5),
        #Implementing the strategy.
        strategy = strategy,
    )