import numpy as np
import cv2
import torch
import torchvision
import os
import torch
# import torch.nn as nn
# import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import *
from matplotlib import pyplot as plt

X_train= np.load('train_X64_bal.npy')
targets_train= np.load('train_Y64_bal.npy')
X_test= np.load('test_X64_bal.npy')
targets_test= np.load('test_Y64_bal.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train= X_train.reshape(7166,3,7,64,64)
X_test= X_test.reshape(2779,3,7,64,64)

test_x = torch.from_numpy(np.array(X_test)).float()
test_y = torch.from_numpy(np.array(targets_test)).long()
train_x = torch.from_numpy(np.array(X_train)).float()
train_y = torch.from_numpy(np.array(targets_train)).long()
batch_size = 16 #We pick beforehand a batch_size that we will use for the training


# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
test = torch.utils.data.TensorDataset(test_x,test_y)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)
num_classes = 5

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1,1,1))

        self.conv3a = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1,1,1))

        self.conv4a = nn.Conv3d(64,32 , kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2))

        self.conv5a = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.25)

        self.relu = nn.ReLU()


        
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
#         x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
#         x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
#         x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
#         x = self.relu(self.fc7(x))
#         x = self.dropout(x)

        logits = self.fc8(x)
#         print(logits.shape)
        return logits


#Definition of hyperparameters
num_epoch=50

# Create CNN
model = CNNModel()
#model.cuda()
print(model)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 1e-4 #0.001
optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning
# CNN model training
train_losses=[]
test_losses=[]
epochs=[]
accuracy_list=[]
for epoch in range(num_epoch): #I decided to train the model for 50 epochs
    print("Epoch -------->",epoch)
    loss_ep = 0
    loss_test=0
    epochs.append(epoch)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data
        targets = targets
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(data)
        loss = error(scores,targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
    print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_loader)}")
    train_losses.append(loss_ep/len(train_loader))
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data,targets) in enumerate(test_loader):
            data = data 
            targets = targets
            ## Forward Pass
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            loss = error(scores,targets)
            loss_test+=loss.item()
        accuracy=float(num_correct) / float(num_samples) * 100
        print(
            f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}"
        )
        accuracy_list.append(accuracy)
        test_losses.append(loss_test/len(train_loader))
# visualization loss 
# iteration_list=[i for i in range(30)]
plt.plot(epochs,train_losses,label="train loss")
plt.plot(epochs,test_losses,label="test loss")
plt.legend()
plt.xlabel("Number of epoch")
plt.ylabel("Loss")
plt.title("3DCNN: Loss vs Number of epoch")
plt.savefig("deep3d_trainloss50.png")
plt.figure()
# visualization accuracy 
plt.plot(epochs,accuracy_list,color = "red")
plt.xlabel("Number of epoch")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of epoch")
plt.savefig("deep3d_train_accuracy50.png")
torch.save(model.state_dict(), "deep3d_50.pth")
print("Test loss :",loss_test/num_samples)
print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")