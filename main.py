#import some libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import torch 
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

#import the data into a dataframe

df = pd.read_csv('final_data.csv', index_col = 'date', parse_dates=True)

#define our inputs and output

inputs = df.iloc[:, 0:7] #Inputs
output = df.iloc[:, 7:8].values #Output => closing price

#Preprocess the inputs. Output can stay under their original "form".


ss = StandardScaler()
inputs_ss = ss.fit_transform(inputs)

#define our data

starting_data_number = 0
cut_point = 1867

#split the data into two parts (training data & test data)

inputs_train = inputs_ss[starting_data_number:cut_point, :]
inputs_test = inputs_ss[cut_point:, :]

output_train = output[starting_data_number:cut_point, :]
output_test = output[cut_point:, :]  

#convert the Numpy Arrays to Tensors and to Variables

inputs_train_tensors = Variable(torch.Tensor(inputs_train)) #size : [1870, 8]
inputs_test_tensors = Variable(torch.Tensor(inputs_test)) #size : [19, 8]

output_train_tensors = Variable(torch.Tensor(output_train))
output_test_tensors = Variable(torch.Tensor(output_test))

#reshaping to rows, timestamps, features

inputs_train_tensors_final = torch.reshape(inputs_train_tensors,   (inputs_train_tensors.shape[0], 1, inputs_train_tensors.shape[1])) #size : [1870, 1, 8]
inputs_test_tensors_final = torch.reshape(inputs_test_tensors,  (inputs_test_tensors.shape[0], 1, inputs_test_tensors.shape[1])) #size : [19, 1, 8]

#define the LSTM

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #lstm
        self.lstm_bis = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer
        self.sigmoid = nn.Sigmoid() #sigmoid activation function
        self.silu = nn.SiLU() #swish activation function
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #1st lstm
        output_bis, (hn1, cn1) = self.lstm_bis(hn, (h_1, c_1)) #2nd lstm
        hn1 = hn1.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.sigmoid(hn1) #sigmoid activation function
        out = self.fc(out) #Final Output
        return out
    
#define the parameters of the LSTM

num_epochs = 1000 # Define the number of epochs of the model
learning_rate = 0.01 #Define the learning rate of the optimizer

input_size = 7 #Define the number of inputs
hidden_size = 4 #Define the hidden size
num_layers = 1 #Define the number of layers of the LSTM

num_classes = 1 #Define the targeted size of the input

n = int(input("How many times do you want to run the model?")) #Define the number of times you want to run the model. 
r2_total = []
r2_predictions = []
predictions = pd.DataFrame()

for i in range(n):
    #instantiate the class

    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, inputs_train_tensors_final.shape[1])
    
    #loss function & optimizer

    criterion = torch.nn.MSELoss() # We use the mean-squared error as loss function

    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) #Between RMSProp, AdaDelta and Adam, Adam was found to slightly outperform, so we've chosen Adam.
    
    #train the model

    for epoch in range(num_epochs):
        outputs_train = lstm1.forward(inputs_train_tensors_final) #forward pass 
        loss_train = criterion(outputs_train, output_train_tensors) # calculate the loss (in our case based on MSE)
        #outputs_test = lstm1.forward(inputs_test_tensors_final) #forward pass 
        #loss_test = criterion(outputs_test, output_test_tensors) # calculate the loss (in our case based on MSE)

        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        loss_train.backward() # backwards pass
        optimizer.step() # update model parameters
        
    #bring the original dataset into the model suitable format

    df_X_ss = ss.transform(df.iloc[:, 0:7]) #old transformers
    df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors

    y = Variable(torch.Tensor(output))
    #reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
    
    #make the predictions

    train_predict = lstm1(df_X_ss) #forward pass
    data_predict = train_predict.data.numpy() #numpy conversion
    test_predic = pd.DataFrame(data_predict[cut_point:,:])
    
    #compute R squared
    r2_total.append(sklearn.metrics.r2_score(y,data_predict))
    r2_predictions.append(sklearn.metrics.r2_score(y[cut_point:,:],data_predict[cut_point:,:]))
    
    #compute predicted data
    predictions = pd.concat([predictions,test_predic], axis=1)
    
plt.hist(r2_predictions)
plt.savefig('r2_distribution.png')

r2 = pd.DataFrame(r2_predictions)
r2.describe()

plt.plot(predictions)
plt.plot(y[cut_point:,:], lw = 5)
plt.savefig('predictions.png')
