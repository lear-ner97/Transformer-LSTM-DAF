# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:05:02 2024

@author: umroot
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple



device = 'cuda' if torch.cuda.is_available() else 'cpu'



    


class CNN_encoder(nn.Module):
    def __init__(self, input_size):
        super(CNN_encoder, self).__init__()
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):#5000
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        #self.norm_layer=nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class traditional_transformer_encoder(nn.Module):
    def __init__(self, d_model=64, nhead=2, dropout=0.2):
        super(traditional_transformer_encoder, self).__init__()

        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)#dim_feedforward=2048,dropout=0.1 by default
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 2)#2=num_layers 
        #self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.pos_encoder(x1)#x
        #transformer_encoder = attention+feedforward in the transformer encoder progonal paper (2017)
        x = self.transformer_encoder(x2)#x1+x2
        #x = self.decoder(x[:, -24, :])
        return x


 



   
class modified_transformer_encoder(nn.Module):#_layer
    #the modified tf-encoder is composed of a signle encoder layer, whereas the traditional tf-encoder 
    #is composed of 2.
    def __init__(self,embed_size,num_heads,drop_prob):
        super(modified_transformer_encoder, self).__init__()
        
        self.fc1 = nn.Sequential(nn.Linear(1, 8*embed_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(8 * embed_size, embed_size))
        # self.fc3 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=64*8, kernel_size=3),#, stride=1, padding=1),
        #     nn.LeakyReLU(),
        #     nn.Conv1d(in_channels=64*8, out_channels=64, kernel_size=3),#, stride=1, padding=1),
        #     nn.LeakyReLU())#,
            #nn.MaxPool1d(kernel_size=5, stride=1,padding=2))
        # self.fc3 = nn.Sequential(nn.Linear(64, 4*embed_size),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(4 * embed_size, embed_size))
        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        #self.attention=RelativeMultiheadAttention(embed_size, num_heads,173, drop_prob)
        self.fc2 = nn.Sequential(nn.Linear(embed_size, 8*embed_size),
                                  nn.LeakyReLU(),
                                  #nn.Dropout(drop_prob),
                                    # nn.Linear(8*embed_size, 4*embed_size),
                                    #   nn.LeakyReLU(),#nn.GELU(),
                                    # nn.Dropout(drop_prob),
                                    nn.Linear(8 * embed_size, embed_size))
        self.dropout = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(1, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x):
        #input embedding
        #x=self.ln1(x)
        x=self.fc1(x)#.permute(0,2,1)).permute(0,2,1)
        #modif1: layer norm
        x=self.ln2(x)
        attn_out, attn_weights = self.attention(x, x, x)#, need_weights=True,average_attn_weights=True)#False,True
        x = x + self.dropout(attn_out)
        x = self.ln2(x)
        #modif2: modified ffn with gelu & more layers
        fc_out = self.fc2(x)#.permute(0,2,1)).permute(0,2,1)
        x = x + self.dropout(fc_out)
        #x= self.fc3(x)#.permute(0,2,1)).permute(0,2,1)
        x = self.ln2(x)
    
        return x#,attn_weights


    
class linear_decoder(nn.Module):
    def __init__(self, d_model,horizon,n_features):
        super(linear_decoder, self).__init__()
        self.d_model=d_model
        self.n_features=n_features
        #self.horizon=horizon
        # self.decoder1= nn.Linear(d_model, 24)
        # self.decoder2= nn.Linear(24, 24)
        self.linear = nn.Linear(d_model*n_features, horizon)
    def forward(self, x):
        # x = self.decoder1(x[:,-24,:])
        # x = self.decoder2(torch.squeeze(x))
        pooled_output = x.reshape(x.shape[0],self.n_features*self.d_model)
        output = self.linear(pooled_output)
        return output




class LSTM_decoder(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,horizon,bidirectional,hidden_coef): #num_layers, to be put back
        super(LSTM_decoder, self).__init__()
        self.horizon=horizon
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout=0.2,bidirectional=bidirectional)
        # self.gru=nn.GRU(input_size=64, hidden_size=hidden_size,
        #                     num_layers=num_layers, batch_first=True,dropout=0.2,bidirectional=False)
        self.fc = nn.Linear(hidden_size*hidden_coef, horizon)#hidden_size*2 if bilstm
        self.ln = nn.LayerNorm(hidden_size*hidden_coef, eps=1e-6)#*2
        
    def forward(self, x):
        out, _ = self.lstm(x) #128,168,100
        #out, _ = self.gru(x)
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

    def __init__(self,input_size):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
           # nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
    
    


