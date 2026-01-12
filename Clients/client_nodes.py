#Importing the main libraries.
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import flwr as fl
from collections import OrderedDict

#The neural network blueprint.
#Same as one referred in the global_node.py file.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

#Depending on the client_id we partition and load the data for training.
#Our base goal is to create non-IID datasets for our case.
def load_data(client_id):
    #Basic pipeline for dataset processing.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #Loading both the train and the testset.
    trainset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    #Let 1st client have 35% of the first images of the dataset.
    #Do so by index splitting.
    if client_id == "1":
        indices = list(range(0, 21000))
        print(f"Node 1: Training on images 0 to 21000")
    else:
    #Let 2nd client have 25% of the images selected from the remaining images.
        indices = list(range(21000, 36000))
        print(f"Node 2: Training on images 21000 to 36000")
        
    #We define the dataloader for both our training and test batches, and shuffle images to increase entropy.
    trainloader = DataLoader(Subset(trainset, indices), batch_size = 32, shuffle = True)
    testloader = DataLoader(testset, batch_size = 32)
    return trainloader, testloader

#Defining the MNIST client for our flower framework.
class MNISTClient(fl.client.NumPyClient):
    #Load up the trainloader and testloader.
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    #Extracting the PyTorch weights and turning them into Numpy arrays for calculation.
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    #Assigning those PyTorch arrays to the respective layers + Coverting them into tensors.
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    #The actual training loop.
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        #Setting up the Adam SGD optimizer for our training loop.
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        #Turns the training mode on for the model architecture.
        self.model.train()
        for epoch in range(1):
            #Trainloader loads the batches into the SGD loop. Local training begins.
            for images, labels in self.trainloader:
                #Sets the gradient to 0 accordingly adjusting parameter tensor.
                optimizer.zero_grad()
                #Initiating the backprop.
                F.cross_entropy(self.model(images), labels).backward()
                #Accordingly get the optimizer step.
                optimizer.step()
        return self.get_parameters(config = {}), len(self.trainloader.dataset), {}

    #Evaluation from the client side for recording progress.
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in self.testloader:
                outputs = self.model(images)
                loss += F.cross_entropy(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(self.testloader.dataset)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

#The main execution part.
if __name__ == "__main__":
    #We figure out the client according the environment in which we run the setup.
    #This is docker + flower logic.
    cid = os.getenv("CLIENT_ID", "1")
    server_addr = os.getenv("SERVER_ADDRESS", "server:8080")
    
    #Setting up the model + the dataloaders.
    model = Net()
    trainloader, testloader = load_data(cid)
    
    #Delay to let the server start.
    time.sleep(15) 
    #Flower training loop starts here.
    fl.client.start_numpy_client(server_address=server_addr, client = MNISTClient(model, trainloader, testloader))