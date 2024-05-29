# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:32:20 2024

@author: umroot
"""
import time
import seaborn as sns
import random
from torchsummary import summary
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from copy import deepcopy as dc
from train_validation import *
from dataloader import *
from explainability_graphs import *
from models import *
#from functions import train_one_epoch_withoutDA,transformer_encoder,prepare_dataframe_for_lstm,TimeSeriesDataset,train_one_epoch,validate_one_epoch,evaluation,CNN_feature_extractor,Discriminator,LSTM_decoder


#1-choose the target building from the following list
# Moose_education_Ricardo , Robin_education_Billi , Robin_office_Maryann, Robin_office_Antonina 
tgt_building='Moose_education_Ricardo'


#2-choose the number of weeks from the following list: 5, 10, 20 
#the number of weeks is the total number of weeks in the target data
weeks=10


#3-choose the target batch size
#tgt_batch_size depends on the scenario: =8 if weeks=5, =16 if weeks=10, =32 if weeks=20
tgt_batch_size=16


#4-fix the seed
#for reproducibility of the results of Table 7 set seeds=[700]
#for reproducibility of the results of Tables 8 to 12 set seeds=random.sample(range(1, 100), 10)

seeds=[700]



#source building is fixed
src_building='Robin_education_Julius'
src_data=pd.read_csv('clean_genome_meters.csv',sep=',')[['timestamp',src_building]]
src_data['timestamp'] = pd.to_datetime(src_data['timestamp'])

#upload target data
tgt_data=pd.read_csv('clean_genome_meters.csv',sep=',')[['timestamp',tgt_building]][:24*7*weeks]
tgt_data['timestamp'] = pd.to_datetime(tgt_data['timestamp'])


#if you want to choose to work with Malaysian data as your target data then uncomment the next 3 lines and 
# tgt_data = pd.read_csv('malaysia_all_data_for_paper.csv',sep=';')
# tgt_data = tgt_data[['time', 'load']][:24*7*weeks]
# tgt_data['time'] = pd.to_datetime(tgt_data['time'])



device = 'cuda' if torch.cuda.is_available() else 'cpu'



# set the historical length T and the future horizon H

src_lookback = 24*7
src_horizon=24*1

tgt_lookback = 24*7
tgt_horizon=24*1


# prepare the features(the electricity load lags) and the target (the future load vector) 

src_shifted_df = prepare_dataframe_for_lstm(src_data, src_lookback+src_horizon-1,'timestamp',src_building)
tgt_shifted_df = prepare_dataframe_for_lstm(tgt_data, tgt_lookback+src_horizon-1,'timestamp',tgt_building)#timestamp,tgt_building

src_shifted_df_as_np = src_shifted_df.to_numpy()
tgt_shifted_df_as_np = tgt_shifted_df.to_numpy()


#data normalization


scaler = MinMaxScaler(feature_range=(-1, 1))
src_shifted_df_as_np = scaler.fit_transform(src_shifted_df_as_np)
tgt_shifted_df_as_np = scaler.fit_transform(tgt_shifted_df_as_np)

#set the feature matrix in each domain
X_src = src_shifted_df_as_np[:, src_horizon:]
X_tgt = tgt_shifted_df_as_np[:, tgt_horizon:]

X_src=dc(np.flip(X_src, axis=1))
X_tgt=dc(np.flip(X_tgt, axis=1))

#set the target
y_src = src_shifted_df_as_np[:, :src_horizon]
y_src=dc(np.flip(y_src, axis=1))
y_tgt = tgt_shifted_df_as_np[:, :tgt_horizon]
y_tgt=dc(np.flip(y_tgt, axis=1))


# we use the source data only in training, we don't need it in validation and test
X_train_src = X_src
y_train_src = y_src


#train-validation-test split
X_train_tgt = X_tgt[:int(len(X_tgt)*0.6)]
y_train_tgt = y_tgt[:int(len(X_tgt)*0.6)]
X_valid_tgt = X_tgt[int(len(X_tgt)*0.6):int(len(X_tgt)*0.8)]
y_valid_tgt = y_tgt[int(len(X_tgt)*0.6):int(len(X_tgt)*0.8)]
X_test_tgt = X_tgt[int(len(X_tgt)*0.8):]
y_test_tgt = y_tgt[int(len(X_tgt)*0.8):]


#prepare our torch tensors

X_train_src = torch.tensor(X_train_src,dtype=torch.float32)
X_train_tgt = torch.tensor(X_train_tgt,dtype=torch.float32)
y_train_src = torch.tensor(y_train_src,dtype=torch.float32)
y_train_tgt = torch.tensor(y_train_tgt,dtype=torch.float32)


X_valid_tgt = torch.tensor(X_valid_tgt,dtype=torch.float32)
y_valid_tgt = torch.tensor(y_valid_tgt,dtype=torch.float32)

X_test_tgt = torch.tensor(X_test_tgt,dtype=torch.float32)
y_test_tgt = torch.tensor(y_test_tgt,dtype=torch.float32)

#add a dimension to each tensor
X_train_src=torch.unsqueeze(X_train_src,2)
X_train_tgt=torch.unsqueeze(X_train_tgt,2)
X_valid_tgt=torch.unsqueeze(X_valid_tgt,2)
X_test_tgt=torch.unsqueeze(X_test_tgt,2)


# Create Custom Datasets
src_train_dataset = TimeSeriesDataset(X_train_src, y_train_src)
tgt_train_dataset = TimeSeriesDataset(X_train_tgt, y_train_tgt)
tgt_valid_dataset = TimeSeriesDataset(X_valid_tgt, y_valid_tgt)
tgt_test_dataset = TimeSeriesDataset(X_test_tgt, y_test_tgt)


src_batch_size = 128



#prepare our data for training with dataloader

src_train_loader = DataLoader(src_train_dataset, batch_size=src_batch_size, shuffle=True)
tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=tgt_batch_size, shuffle=True)
tgt_valid_loader = DataLoader(tgt_valid_dataset, batch_size=tgt_batch_size, shuffle=False)
tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=tgt_batch_size, shuffle=False)

#print the shape of a batch of data

print("source data\n")
i=0
for _, batch in enumerate(src_train_loader):
    print(i)
    i+=1
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break
print("target data\n")
for _, batch in enumerate(tgt_train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break


learning_rate = 0.001
num_epochs = 20
best_feature_extractor_path = 'best_feature_extractor.pth'
best_tgt_generator_path = 'best_tgt_generator.pth'

#compute the trainnig time
start_time = time.time()


rmses = []
mapes=[]
r2scores=[]
i=0
for random_seed in seeds:
    i+=1
    print("seed number "+str(i))
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    #choose the encoder between:
    #transformer_encoder(64,2,0.3).to(device) & CNN_feature_extractor(1).to(device)
    feature_extractor=transformer_encoder(64,2,0.3).to(device)
    #define the discriminator, the source decoder and the target decoder
    discriminator=Discriminator().to(device)
    src_generator=LSTM_decoder(200,2,src_horizon).to(device)
    tgt_generator=LSTM_decoder(200,2,tgt_horizon).to(device)
    
    
    #define the loss functions
    gen_loss_function = nn.MSELoss()
    disc_loss_function = nn.CrossEntropyLoss()
    
    #define the optimizers
    disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
    gen_optimizer = torch.optim.Adam(list(tgt_generator.parameters())+list(src_generator.parameters())
                                     + list(feature_extractor.parameters()), lr=learning_rate)
    
    #schedule the learning rate
    scheduler = lr_scheduler.LinearLR(gen_optimizer, start_factor=1.0, end_factor=0.5, total_iters=15)
    
    
    #initialize the training loss, validation loss and lowest validation loss
    training_loss,validation_loss=[],[]
    best_val_loss = float('inf') 
    
    
    
    
    for epoch in range(num_epochs):
        training_loss.append(train_one_epoch(feature_extractor,src_generator,tgt_generator,discriminator,src_train_loader,
                            tgt_train_loader,gen_loss_function,disc_loss_function,gen_optimizer,
                            disc_optimizer,scheduler,epoch,num_epochs))
        #if you want to train the TF-LSTM without DA then uncomment the following 3 lines, comment the previous 3 lines and remove the source decoder parameters in the optimizer
        # training_loss.append(train_one_epoch_withoutDA(feature_extractor,tgt_generator,
        #                     tgt_train_loader,gen_loss_function,gen_optimizer,
        #                     scheduler,epoch,num_epochs))
        val_loss=validate_one_epoch(feature_extractor,tgt_generator,discriminator,epoch,
                                                  tgt_valid_loader,gen_loss_function,disc_loss_function)
        validation_loss.append(val_loss)
        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss  
            #save the best model
            torch.save(feature_extractor.state_dict(), best_feature_extractor_path)
            torch.save(tgt_generator.state_dict(), best_tgt_generator_path)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(training_time)
    
    #load the best model for the evaluation step
    with torch.no_grad():
        feature_extractor_state_dict = torch.load(best_feature_extractor_path)
        tgt_generator_state_dict = torch.load(best_tgt_generator_path)
        
        feature_extractor.load_state_dict(feature_extractor_state_dict)
        tgt_generator.load_state_dict(tgt_generator_state_dict)
        
        #uncomment the [0] if your encoder is a transformer
        #comment the [0] if your encoder is a CNN
        predicted = feature_extractor(X_train_tgt.to(device))[0]
        predicted = (tgt_generator(predicted)).cpu().numpy()
    train_predictions = predicted
    
    dummies = np.zeros((X_train_tgt.shape[0], tgt_lookback+tgt_horizon))
    dummies[:, :24] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    
    #train predictions
    train_predictions = dc(dummies[:, :24])
    
    
    dummies = np.zeros((X_train_tgt.shape[0], tgt_lookback+tgt_horizon))
    dummies[:, :24] = y_train_tgt[:,:24] 
    dummies = scaler.inverse_transform(dummies)
    
    # actual train data
    new_y_train = dc(dummies[:, :24])
    
    #plot y_train vs train_prediction
    plt.figure(5)
    plt.plot(new_y_train[::24,:].flatten(), label='Actual load')
    plt.plot(train_predictions[::24,:].flatten(), label='Predicted load')
    plt.xlabel('time')
    plt.ylabel('load')
    plt.title('training')
    plt.legend(loc='lower left')
    plt.show()
    
    
    with torch.no_grad():
        #uncomment the [0] if your encoder is a transformer
        #comment the [0] if your encoder is a CNN
        predicted = feature_extractor(X_test_tgt.to(device))[0]
        test_predictions = (tgt_generator(predicted)).cpu().numpy()
    
    #test predictions
    dummies = np.zeros((X_test_tgt.shape[0], tgt_lookback+tgt_horizon))
    dummies[:, :24] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    
    test_predictions = dc(dummies[:, :24])
    
    
    #actual test data
    dummies = np.zeros((X_test_tgt.shape[0], tgt_lookback+tgt_horizon))
    dummies[:, :24] = y_test_tgt[:,:24]
    dummies = scaler.inverse_transform(dummies)
    
    new_y_test = dc(dummies[:, :24])
    
    plt.figure(6)
    plt.plot(new_y_test[::24,:].flatten()[:24*7], label='Actual')#
    plt.plot(test_predictions[::24,:].flatten()[:24*7], label='TF')#
    #plt.plot(cnn_test_predictions[::24,:].flatten()[:24*7], label='CNN')
    plt.xlabel('hour')
    plt.ylabel('load')
    #plt.title('testing')
    plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
    #plt.savefig('paper/plots/'+dataset+str(weeks), bbox_inches='tight')
    plt.show()
    #######################################################################
    epochs=[i for i in range(1,21)]
    plt.figure(7)
    plt.plot(epochs,training_loss,label='training loss')
    plt.plot(epochs,validation_loss, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.xticks(epochs, epochs)
    plt.show()
    
    rmses.append(evaluation(test_predictions, new_y_test)[0])
    mapes.append(evaluation(test_predictions, new_y_test)[1])
    r2scores.append(evaluation(test_predictions, new_y_test)[2])
    print("training metrics:\n")
    print("rmse: ", evaluation(train_predictions,new_y_train)[0])
    print("mape: ", evaluation(train_predictions,new_y_train)[1])
    print("r2-score: ", evaluation(train_predictions,new_y_train)[2])
    
    print()
    print("test predictions: \n")
    print("rmse: ", evaluation(test_predictions, new_y_test)[0])
    print("mape: ", evaluation(test_predictions, new_y_test)[1])
    print("r2-score: ", evaluation(test_predictions, new_y_test)[2])
    
    
    
    
    print(f"Total training time: {training_time:.2f} seconds")
    
#uncomment the following lines if you run the script on multiple seeds

# plt.figure(8)
# plt.boxplot(rmses)
# plt.figure(9)
# plt.boxplot(mapes)
# plt.figure(10)
# plt.boxplot(r2scores)
# print()
# print(np.mean(r2scores))
# print(np.std(r2scores))
# print()
# print(np.mean(rmses))
# print(np.std(rmses))
# print()
# print(np.mean(mapes))
# print(np.std(mapes))


# if encoder = transformer
# uncomment the next two lines to visualize the attention heatmap
sample_index=0 #choose a random sample index
heatmap(feature_extractor,x_batch,sample_index)


# # if encoder = CNN
# uncomment the following code to visualize the filters and their activation mappings
# visualize_filters(feature_extractor)
# sample_index=0
# activation_mapping(feature_extractor,x_batch[sample_index])