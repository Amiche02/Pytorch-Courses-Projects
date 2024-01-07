# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Create fully connected neural network
class NN(nn.Module):
    def __init__(self, input, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784
number_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 10

#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True,
                               transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False,
                              transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Initialize model
model = NN(input=input_size, output_size=number_classes).to(device)

#Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optim = optim.Adam(params=model.parameters(), lr=learning_rate)

#Train network
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(images.shape[0], -1).to(device=device)
        labels = labels.to(device=device)

        #Forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        #Backward
        optim.zero_grad()
        loss.backward()

        #gradient descent step
        optim.step()


#Evaluate network
def check_accuracy(loader, model_):
    num_correct = 0
    num_samples = 0
    model_.eval()

    with torch.no_grad():
        if loader.dataset.train:
            print("Checking accuracy on training set...")
        else:
            print("Checking accuracy on test set...")
        for x, y in loader:
            x = x.view(x.shape[0], -1).to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f' Got {num_correct} / {num_samples} with Accuracy: {float(num_correct) / float(num_samples)*100:.3f} ')
        #acc = float(num_correct) / float(num_samples)
    model.train()
    #return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
