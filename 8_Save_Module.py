# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784
in_channels = 1
number_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 15
load_model = False

#TODO : Create a simple CNN
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['optimizer_state_dict'])


#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True,
                               transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False,
                              transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Initialize model
model = CNN(input_channels=in_channels, num_classes=number_classes).to(device)

#Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optim = optim.Adam(params=model.parameters(), lr=learning_rate)

#Train network
for epoch in range(epochs):
    losses = []

    if (epoch+1) % 3:
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optim}
        save_checkpoint(checkpoint, 'checkpoint.pth')


    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device=device)
        labels = labels.to(device=device)

        #Forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        #Backward
        optim.zero_grad()
        loss.backward()
        #gradient descent step
        optim.step()

    mean_loss = sum(losses) / len(losses)
    print(f'Epoch {epoch+1}/{epochs} Loss: {mean_loss:.3f}')

if load_model:
    load_checkpoint(torch.load('checkpoint.pth'))

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
            x = x.to(device=device)
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
