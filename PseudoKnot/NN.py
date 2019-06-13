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

class DatasetPseudoKnot (torch.utils.data.Dataset):

    def __init__(self, file_path, transform = None):
        trainColIdx = np.arange(1,len(train_dataset_raw[0]))
        allData = np.loadtxt(sys.argv[4], dtype='float', delimiter = None, skiprows = 1, usecols = trainColIdx)
        if (sys.argv[4].lower().find("train") != -1):
            testData = allData[:,0]
            x_norm = ((allData[:,1:] - allData[:,1:].min(0))) / (allData[:,1:].ptp(0)+1)
            x_norm = np.column_stack((testData, x_norm))
            self.data = torch.from_numpy(x_norm)
            self.transform = None
        else:
            x_norm = (allData - allData.min(0)) / allData.ptp(0)
            self.data = torch.from_numpy(x_norm)
            self.transform = None

    def __getitem__(self, index):
        temp = []
        temp_labels = []
        if (self.data.shape[1] == num_features+1):  # 104 = with label
            return self.data[index,1:].type(torch.FloatTensor), self.data[index][0].type(torch.LongTensor)
        elif (self.data.shape[1] == num_features):  # 103 = without label
            return self.data[index,0:].type(torch.FloatTensor)

    def __len__(self):
        return(len(self.data))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 200)
        self.fc1_drop = nn.Dropout(float(sys.argv[1]))
        self.fc2 = nn.Linear(200, 400)
        self.fc2_drop = nn.Dropout(float(sys.argv[1]))
        self.fc3 = nn.Linear(400, 100)
        self.fc3_drop = nn.Dropout(float(sys.argv[1]))
        self.fc4 = nn.Linear(100, 10)
        self.fc4_drop = nn.Dropout(float(sys.argv[1]))
        self.fc5 = nn.Linear(10, 2)

    def forward(self, x):
        x = x.view(-1, num_features)
        x = F.leaky_relu(self.fc1(x), 0.2, True)
        x = self.fc1_drop(x)
        x = F.leaky_relu(self.fc2(x), 0.2, True)
        x = self.fc2_drop(x)
        x = F.leaky_relu(self.fc3(x), 0.2, True)
        x = self.fc3_drop(x)
        x = F.leaky_relu(self.fc4(x), 0.2, True)
        x = self.fc4_drop(x)
        return F.log_softmax(self.fc5(x),dim=1)



data_set_raw = sys.argv[4]
train_dataset_raw = np.genfromtxt(sys.argv[4], delimiter="\t", dtype=None)
# inp = pd.DataFrame(train_dataset)
test_dataset_raw = None
num_features = None

if (sys.argv[4].lower().find("train") != -1):
    num_features = len(train_dataset_raw[0])-2
else:
    num_features = len(train_dataset_raw[0])-1

temp = []
labels = []
load_size = 100
batch_size = load_size

data_location = "./feature103_Train.txt"
test_location = "./features103_test.txt"
# 1053 features for allfeatures.txt

# train_size = 6000
train_size = train_dataset_raw.shape[0]

train_dataset = DatasetPseudoKnot(file_path=sys.argv[4])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=load_size, shuffle=False)

test_dataset = DatasetPseudoKnot(file_path=sys.argv[4])
test_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False)

if (len(sys.argv) == 6):
    test_dataset = DatasetPseudoKnot(file_path=sys.argv[5])
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)
    test_dataset_raw = np.genfromtxt(sys.argv[5], delimiter="\t", dtype=None)

# img, lab = train_dataset.__getitem__(0)

num_epochs = 50
train_errs = torch.zeros(num_epochs)
val_errs = torch.zeros(num_epochs)


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
    global num_iterations_without_change    
    global epochs_label_results_last_four
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
    # print(val_correct)
    # print(inside)
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
    # print("validation error for epoch " +str(epoch+1)+": "+str(val_errs[epoch].item()))
    epochs_label_results_last_four.append(has_knot)
    if (len(epochs_label_results_last_four) == 3):
        average = (epochs_label_results_last_four[0] + epochs_label_results_last_four[1] + epochs_label_results_last_four[2])/3
        if (average - has_knot > 600):
            model = Net().to(device)
            print("model not generalizing well, restarting")        
        epochs_label_results_last_four = []
    if (epoch == num_iterations_without_change and no_knot == 0):
        num_iterations_without_change += 2
        model = Net().to(device)
        print("model not generalizing well, restarting")        
    if (has_knot < 2500):
        if (epoch == 1 or has_knot == 0):
            model = Net().to(device)
            return False
        return True
    if (epoch >= num_iterations_without_change and no_knot < 200):
        num_iterations_without_change = epoch + 2
        model = Net().to(device)

epochs_label_results_last_four = []
num_iterations_without_change = 3
    # pdb.set_trace()

def test():
    global model
    test_correct = 0
    has_knot = 0
    no_knot = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        # pdb.set_trace()
        output = model(data)
        pred = output.data.max(1)[1]
        test_correct += pred.eq(target.data).sum()
        # pdb.set_trace()
        has_knot += torch.nonzero(pred == 1).shape[0]
        no_knot += torch.nonzero(pred == 0).shape[0]
    print("Entire testing set accuracy is ", test_correct.type(torch.FloatTensor)/test_dataset.__len__())
    # pdb.set_trace()

def actual_test():
    global model
    f = open("results.txt", "w")
    test_correct = 0
    has_knot = 0
    no_knot = 0
    pdb.set_trace()
    for batch_idx, (data) in enumerate(test_loader):
        output = model(data)
        pred = output.data.max(1)[1]
        # test_correct += pred.eq(target.data).sum()
        # pdb.set_trace()
        has_knot = torch.nonzero(pred == 1).shape[0]
        no_knot = torch.nonzero(pred == 0).shape[0]
        if (has_knot == 1):
            f.write(str(test_dataset_raw[batch_idx][0]).replace("\'", "") + ",1")
        else:
            f.write(str(test_dataset_raw[batch_idx][0]).replace("\'", "") + ",0")

    pdb.set_trace()
    print("test accuracy")
    # print(test_correct)
    # print(str(test_correct.item()/len(test_dataset)))

def main():
    global device
    #check for valid number of arguments
    if(len(sys.argv) <= 4):
        print("please enter dropout, momentum, and weight decay in command line")
        return
    #check for valid float as learning rate
    #read in learning rate from command line
    d = float(sys.argv[1])      # dropout
    m = float(sys.argv[2])      # momentum
    wd = float(sys.argv[3])     # wieght-decay
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0000032558, momentum=m, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    print(model)
    stopped = 0
    for i in range(num_epochs):
        stop = train(i,200,optimizer,criterion)
        if (stop == True): 
            break
            stopped = i
        stopped = i

    print("training errors")
    #plot results
    if (len(sys.argv) == 6):
        actual_test()
    else:
        test()
    xvals = torch.arange(1,stopped+2)
    pdb.set_trace()
    trainingLine = plt.plot(xvals.numpy(),train_errs.numpy()[:stopped+1], label='Training Error')
    # valLine = plt.plot(xvals.numpy(),val_errs.numpy(), label='Validation Error')
    # pdb.set_trace()
    plt.title("Error vs Epochs for momentum="+str(m)+" using leaky_ReLu for TRAINING")
    plt.ylabel("Error Percentage")
    plt.xlabel("Epochs")
    plt.legend(loc='best')
    plt.show()



main()