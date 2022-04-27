#import some libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import torch 
import torch.nn as nn
from torch.autograd import Variable

#import the data into a dataframe

df = pd.read_csv('return.csv', index_col = 'date', parse_dates=True)

#define our inputs and output

X = df.iloc[:, 0:8] #Inputs
y = df.iloc[:, 8:9] #Output => daily return

#Preprocess the inputs. Output can stay under their original "form".

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
y = y.values

#define our data

starting_data_number = 0
cut_point = 1850

#split the data into two parts (training data & test data)

X_train = X[starting_data_number:cut_point, :]
X_test = X[cut_point:, :]

y_train = y[starting_data_number:cut_point, :]
y_test = y[cut_point:, :] 

#convert the Numpy Arrays to Tensors and to Variables

X_train_tensors = Variable(torch.Tensor(X_train)) #size : [1870, 8]
X_test_tensors = Variable(torch.Tensor(X_test)) #size : [19, 8]

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

#reshaping to rows, timestamps, features

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

#define the LSTM

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
#define the parameters of the LSTM

num_epochs = 5000 #10000 epochs
learning_rate = 0.01 #0.01 lr

input_size = 8 #number of inputs of the model.
hidden_size = 5 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 

#instantiate the class

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])

#loss function & optimizer

criterion = torch.nn.MSELoss() # We use the mean-squared error as loss function
#criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) #Between RMSProp, AdaDelta and Adam, Adam was found to slightly outperform, so we've chosen Adam. 

#train the model

for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train_tensors_final) #forward pass
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)
 
    loss.backward() #calculates the loss of the loss function
 
    optimizer.step() #improve from loss, i.e backprop
    if epoch % 500 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
        
#bring the original dataset into the model suitable format

df_X_ss = ss.transform(df.iloc[:, 0:8]) #old transformers
df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors

y = Variable(torch.Tensor(y))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

#make the predictions

train_predict = lstm1(df_X_ss) #forward pass
data_predict = train_predict.data.numpy() #numpy conversion

#compute R squared

print(sklearn.metrics.r2_score(y,data_predict)) #whole data set
print(sklearn.metrics.r2_score(y[cut_point:,:],data_predict[cut_point:,:])) #only predicted data

#Predcted and actual value

print(y[cut_point:,:],data_predict[cut_point:,:])