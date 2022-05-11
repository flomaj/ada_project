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

data = pd.read_csv('final_data.csv', index_col = 'date', parse_dates=True)

#Define our inputs and output

inputs = data.iloc[:, 0:7] #Inputs, in our case : MVRV, transactions count, daily active addresses, mean hash rate, gas price, 30D volatility & Google Trends Score.
output = data.iloc[:, 7:8].values #Output, in our case : Closing price.

#Preprocessing the inputs. Output can stay under its original "form".
#We use a Standard Scaler transformation for our inputs.

ss = StandardScaler() 
inputs_ss = ss.fit_transform(inputs)

#Split the data. In our case, we want to predict the return of the next ten days. Thus, we set our cut point at n-10. 
#Our train dataset is made of all the data expected the last ten. Our test dataset contains the inputs for our prediction.

starting_data_number = 0
cut_point = 1867

inputs_train = inputs_ss[starting_data_number:cut_point, :]
inputs_test = inputs_ss[cut_point:, :]

output_train = output[starting_data_number:cut_point, :] 

#Because we are using PyTorch, we have to convert our dataset to tensors, and then to variables. 

inputs_train_tensors = Variable(torch.Tensor(inputs_train)) #size : [1867, 8]
inputs_test_tensors = Variable(torch.Tensor(inputs_test)) #size : [10, 8]

output_train_tensors = Variable(torch.Tensor(output_train))

#These data have 2 dimensions. To perform the LSTM, we have to reshape them in 3D to include the timestamps. 

inputs_train_tensors_final = torch.reshape(inputs_train_tensors,   (inputs_train_tensors.shape[0], 1, inputs_train_tensors.shape[1])) #size : [1867, 1, 8]
inputs_test_tensors_final = torch.reshape(inputs_test_tensors,  (inputs_test_tensors.shape[0], 1, inputs_test_tensors.shape[1])) #size : [10, 1, 8]

#define the LSTM

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #First LSTM (8 -> hidden size)
        self.lstm_bis = nn.LSTM(hidden_size, hidden_size, num_layers) #Second LSTM (hidden size -> hidden size)
        self.fc = nn.Linear(hidden_size, num_classes) #Last fully connect layer (hidden size -> 1)
        self.sigmoid = nn.Sigmoid() #sigmoid activation function
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #First hidden state related to first LSTM.
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #First internal state related to first LSTM.
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #Second hidden state related to second LSTM.
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #Second internal state related to second LSTM.
        
        #Propagate input through LSTM
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #First LSTM.
        output_bis, (hn1, cn1) = self.lstm_bis(hn, (h_1, c_1)) #Second LSTM.
        hn1 = hn1.view(-1, self.hidden_size) #Reshape data for the last layer.
        out = self.sigmoid(hn1) #Go through the activation function.
        out = self.fc(out) #Last layer, give the outputs.
        return out
    
#define the parameters of the LSTM

num_epochs = 1000 # Define the number of epochs of the model
learning_rate = 0.01 #Define the learning rate of the optimizer

input_size = 7 #Define the number of inputs
hidden_size = 4 #Define the hidden size
num_layers = 1 #Define the number of layers of the LSTM

num_classes = 1 #Define the targeted size of the input

#Now, we will run our model.

n = int(input("How many times do you want to run the model?")) #Define the number of times you want to run the model. 
r2_total = []
r2_predictions = []
predictions = pd.DataFrame()

for i in range(n):
    
    #Instantiate the LSTM model with our parameters and our inputs.

    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, inputs_train_tensors_final.shape[1])
    
    #Determine the loss function & the optimizer

    criterion = torch.nn.MSELoss() #We use the mean-squared error as loss function

    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) #For this project, we use the "Adam" optimizer.
    
    #Train the model the find the optimal weights.

    for epoch in range(num_epochs):
        
        outputs_forward = lstm1.forward(inputs_train_tensors_final) #The forward pass goes on.
        loss_train = criterion(outputs_forward, output_train_tensors) #It computes the loss for each epoch.

        optimizer.zero_grad() #It computes the gradient that we previously set to 0.
        loss_train.backward() #The backward pass goes on.
        optimizer.step() #The weights are updated with regards to our optimizer function.
        
    #bring the original dataset into the model suitable format

    inputs_trans_ss = ss.transform(inputs) #We transform the whole inputs dataset with Standard Scaler.
    inputs_trans_ss_tensor = Variable(torch.Tensor(inputs_trans_ss)) #We convert the transformed inputs to tensors and then variable.

    output_ready_tensor = Variable(torch.Tensor(output)) #We convert the wole output dataset to tensors and then variable.
    
    #We reshape the inputs dataset to make it suitable for the LSTM.
    total_inputs_reshap = torch.reshape(inputs_trans_ss_tensor, (inputs_trans_ss_tensor.shape[0], 1, inputs_trans_ss_tensor.shape[1]))
    
    #We perform our prediction on the whole dataset regarding to our model.
    train_predict = lstm1(total_inputs_reshap) #The forward pass goes on and we get our predictions.
    data_predict = train_predict.data.numpy() #We transform  our predictions (tensors to numpy arrays)
    test_predic = pd.DataFrame(data_predict[cut_point:,:]) #We create a second dataframe with only the predictions for the next 10 days.
    
    #compute R squared for eachtime we run.
    r2_total.append(sklearn.metrics.r2_score(output,data_predict))
    r2_predictions.append(sklearn.metrics.r2_score(output[cut_point:,:],data_predict[cut_point:,:]))
    
    #keep the predictions for eatc time we run the model in a single dataframe.
    predictions = pd.concat([predictions,test_predic], axis=1)

#Plot the R squared.
plt.hist(r2_predictions)
plt.savefig('r2_distribution.png')

#Plot the predictions and the "true" result.
plt.plot(predictions)
plt.plot(output[cut_point:,:], lw = 5)
plt.savefig('predictions.png')
