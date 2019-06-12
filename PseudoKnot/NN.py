import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from sklearn.preprocessing import normalize


train_dataset = np.genfromtxt("./feature103_Train.txt", delimiter="\t", dtype=None)
# inp = pd.DataFrame(train_dataset)
pdb.set_trace()

temp = []
labels = []
load_size = 100
batch_size = load_size
# temp = np.asarray(inp.iloc[0][1:])
# for i in range(1,inp.shape[0]):
#     # zz = np.append([x.iloc[1][1:]],zz,0)
#     temp.append(inp.iloc[i][2:])       # MUCH faster (5 seconds vs 8-10 mins than the other commented methods)
#     labels.append(inp.iloc[i][1])
#     # temp = np.vstack((temp, inp.iloc[i][1:]))

# # data = np.asarray(temp)
# data = torch.FloatTensor(temp)
# labels = torch.FloatTensor(labels)

data_location = "./feature103_Train.txt"
test_location = "./features103_test.txt"

train_size = train_dataset.shape[0]

class DatasetPseudoKnot (torch.utils.data.Dataset):

    def __init__(self, file_path, transform = None):
        # train_dataset = np.genfromtxt(file_path, delimiter="\t", dtype=None)
        trainColIdx = np.arange(1,105)
        allData = np.loadtxt("feature103_Train.txt", dtype='float', delimiter = None, skiprows = 1, usecols = trainColIdx)
        # trainData = allData[:,1:]
        testData = allData[:,0]
        x_norm = (allData[:,1:] - allData[:,1:].min(0)) / allData[:,1:].ptp(0)
        x_norm = np.column_stack((testData, x_norm))
        # pdb.set_trace()
        # self.data = torch.from_numpy(np.random.rand(10000,104))
        self.data = torch.from_numpy(x_norm)
        # self.data = pd.DataFrame(train_dataset)
        self.transform = None

    def __getitem__(self, index):
        temp = []
        temp_labels = []
        # for i in range(1,self.data.shape[0]):
        #     # zz = np.append([x.iloc[1][1:]],zz,0)x
        #     temp.append(self.data.iloc[i][2:])       # MUCH faster (5 seconds vs 8-10 mins than the other commented methods)
        #     temp_labels.append(self.data.iloc[i][1])
        # rows = None
        # label = None
        # pdb.set_trace()
        return self.data[index,1:].type(torch.FloatTensor), self.data[index][0].type(torch.LongTensor)

        # if (self.data.iloc[index].size != 104):
        #     # df_norm = (df - df.mean()) / (df.max() - df.min())
        #     rows = self.data.iloc[index][2:]
        #     label = torch.LongTensor(self.data.iloc[index][1:])
        #     norm_rows = torch.FloatTensor((rows - rows.mean())/rows.max() - rows.min())
        #     if self.transform is not None:
        #         norm_rows = self.transform(norm_rows)
        #     return norm_rows, label[0]
        # else:
        #     rows = self.data.iloc[index][1:]
        #     norm_rows = torch.FloatTensor((rows - rows.mean())/rows.max() - rows.min())
        #     return norm_rows
            # label = torch.LongTensor(self.data.iloc[index][1:])
        # print(label)
        # print(self.data.iloc[index][0])
        

    def __len__(self):
        return(len(self.data))

train_dataset = DatasetPseudoKnot(file_path=data_location)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=load_size, shuffle=True)

test_dataset = DatasetPseudoKnot(file_path=test_location)

test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)
# img, lab = train_dataset.__getitem__(0)

num_epochs = 20
train_errs = torch.zeros(num_epochs)
val_errs = torch.zeros(num_epochs)

# pdb.set_trace()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(103, 200)
        self.fc1_drop = nn.Dropout(float(sys.argv[1]))
        self.fc2 = nn.Linear(200, 2)
        # self.fc2_drop = nn.Dropout(float(sys.argv[1]))
        # self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1,103)
        x = F.leaky_relu(self.fc1(x), 0.2, True)
        x = self.fc1_drop(x)
        # x = F.leaky_relu(self.fc2(x), 0.2, True)
        # x = self.fc2_drop(x)
        return F.log_softmax(self.fc2(x),dim=1)

#identify device for pytorch, use gpu if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')        

model = Net().to(device)

def train(epoch, log_interval,optimizer, criterion,):
    global device
    global model
    global val_errs
    global train_errs
    #set model to training mode
    model.train()
    val_correct = 0
    train_correct = 0
    # Loop over each batch from the training set
    inside = 0
    has_knot = 0
    no_knot = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx+1)*load_size <= train_size:
            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad() 
            
            # Pass data through the network
            output = model(data)    
            # pdb.set_trace()
            # print(batch_idx)
            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            # pdb.set_trace()
            # target = target.squeeze()
            loss = criterion(output, target)

            # Backpropagate
            loss.backward()
            
            # Update weights
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * load_size, len(train_loader.dataset),
                load_size * batch_idx / len(train_loader), loss.data.item()))
        else:
            output = model(data)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            has_knot += torch.nonzero(pred == 1).shape[0]
            no_knot += torch.nonzero(pred == 0).shape[0]
            print("Has_knot is", has_knot, no_knot)            
            val_correct += pred.eq(target.data).sum()
    print(val_correct)
    print(inside)
    # pdb.set_trace()
    # val_errs[epoch] = ((27348 - train_size) - val_correct.item())/ (27348 - train_size)
    # val_errs[epoch] = (7358 - val_correct.item())/ 7358
    has_knot = 0
    no_knot = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx+1)*load_size <= train_size:
            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1)[1]
            train_correct += pred.eq(target.data).sum()
            has_knot += torch.nonzero(pred == 1).shape[0]
            no_knot += torch.nonzero(pred == 0).shape[0]
            print("Has_knot is", has_knot, no_knot)            
            # pdb.set_trace()
    print("train_correct is ", train_correct)
    train_errs[epoch] = (train_size-train_correct.item())/(train_size)
    print("training error for epoch "+str(epoch+1)+": "+str(train_errs[epoch].item()))
    print("validation error for epoch " +str(epoch+1)+": "+str(val_errs[epoch].item()))
    # pdb.set_trace()

def test():
    global model
    test_correct = 0
    has_knot = 0
    no_knot = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        pred = output.data.max(1)[1]
        test_correct += pred.eq(target.data).sum()
        # pdb.set_trace()
        has_knot += torch.nonzero(pred == 1).shape[0]
        no_knot += torch.nonzero(pred == 0).shape[0]
    print("Entire testing set accuracy is ", test_correct.type(torch.FloatTensor)/27348)
    # pdb.set_trace()

def actual_test():
    global model
    test_correct = 0
    has_knot = 0
    no_knot = 0
    pdb.set_trace()
    for batch_idx, (data) in enumerate(test_loader):
        output = model(data)
        pred = output.data.max(1)[1]
        # test_correct += pred.eq(target.data).sum()
        # pdb.set_trace()
        has_knot += torch.nonzero(pred == 1).shape[0]
        no_knot += torch.nonzero(pred == 0).shape[0]
    pdb.set_trace()
    print("test accuracy")
    # print(test_correct)
    # print(str(test_correct.item()/len(test_dataset)))

def main():
    global device
    #check for valid number of arguments
    if(len(sys.argv) != 4):
        print("please enter dropout, momentum, and weight decay in command line")
        return
    #check for valid float as learning rate
    #read in learning rate from command line
    d = float(sys.argv[1])
    m = float(sys.argv[2])
    wd = float(sys.argv[3])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000007, momentum=m, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    print(model)
    for i in range(num_epochs):
        train(i,200,optimizer,criterion)
    print("training errors")
    #plot results
    test()
    xvals = torch.arange(1,num_epochs+1)
    trainingLine = plt.plot(xvals.numpy(),train_errs.numpy(), label='Training Error')
    valLine = plt.plot(xvals.numpy(),val_errs.numpy(), label='Validation Error')
    plt.title("Error vs Epochs for momentum="+str(m)+" using ReLu")
    plt.ylabel("Error Percentage")
    plt.xlabel("Epochs")
    plt.legend(loc='best')
    plt.show()



main()