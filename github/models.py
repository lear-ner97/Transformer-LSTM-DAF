# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:05:02 2024

@author: umroot
"""

import torch
import torch.nn as nn
from typing import Tuple



device = 'cuda' if torch.cuda.is_available() else 'cpu'



    


class CNN_feature_extractor(nn.Module):
    def __init__(self, input_size):
        super(CNN_feature_extractor, self).__init__()
        #self.horizon=horizon
        #stride is the # of jumps
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.activ = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=1,padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        out = x.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.activ(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)
        out = self.activ(out)
        out = self.maxpool(out)
        
        # prepare the output for the lstm decoder
        #lstm takes input of shape (batch_size, seq_len, input_size)
        
        out = out.permute(0, 2, 1)

        return out



    
class transformer_encoder(nn.Module):

    def __init__(self,embed_size,num_heads,drop_prob):
        super(transformer_encoder, self).__init__()
        
        self.fc1 = nn.Sequential(nn.Linear(1, 8*embed_size),
                                 nn.LeakyReLU(),
                                 nn.Linear(8 * embed_size, embed_size))
        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.fc2 = nn.Sequential(nn.Linear(embed_size, 8*embed_size),
                                 nn.LeakyReLU(),
                                 nn.Linear(8 * embed_size, embed_size))
        self.dropout = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x):
        x=self.fc1(x)
        x=self.ln1(x)
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True,average_attn_weights=True)#False,True
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        fc_out = self.fc2(x)
        x = x + self.dropout(fc_out)
        x = self.ln2(x)

        return x,attn_weights

    





class LSTM_decoder(nn.Module):
    def __init__(self, hidden_size,num_layers,horizon): #num_layers, to be put back
        super(LSTM_decoder, self).__init__()
        self.horizon=horizon
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_size, horizon)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-6)
        
    def forward(self, x):
        out, _ = self.lstm(x) #128,168,100
        out=self.ln(out)
        # #print(out.shape)
        out = self.fc(out[:, -self.horizon, :]) #out[:, -self.horizon, :], 128,24

        return out
    #the output of lstm(x) shape: (batch_size,LookBack,#lstm units)=(128,24*7,200)
    # the output of fc(out) shape: (batch_size,LookBack,Horizon)=(128,24*7,24*1)
    # if we want to consider only the last element of the output (to base our prediction on the newest
    # value) then the output shape of fc(out[:,-1,:]) is (batch_size,Horizon)=(128,24*1)
    
    
    
    
    
    
class Discriminator(nn.Module):
    """Discriminator model from adatime models/model.py/Discriminator class"""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
           # nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
    
    


