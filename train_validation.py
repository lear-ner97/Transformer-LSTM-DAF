# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:06:30 2024

@author: umroot
"""
from torch.autograd import Function
import itertools
from typing import Tuple
from copy import deepcopy as dc
from sklearn.metrics import mean_absolute_percentage_error, r2_score ,mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset



device = 'cuda' if torch.cuda.is_available() else 'cpu'
    



def train_one_epoch(feature_extractor,src_generator,tgt_generator,discriminator,src_train_loader,
                    tgt_train_loader,gen_loss_function,dom_loss_function,optimizer_gen,
                    optimizer_disc,scheduler,epoch,num_epochs):
    torch.cuda.empty_cache()
    feature_extractor.train(True)
    src_generator.train(True)
    tgt_generator.train(True)
    discriminator.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    num_batches=len(tgt_train_loader)
    for (src_batch_index, src_batch),(tgt_batch_index, tgt_batch) in zip(
            enumerate(src_train_loader),enumerate(tgt_train_loader)):
        src_x_batch, src_y_batch = src_batch[0].to(device), src_batch[1].to(device)
        tgt_x_batch, tgt_y_batch = tgt_batch[0].to(device), tgt_batch[1].to(device)
        
        p = float(tgt_batch_index + epoch * num_batches) / num_epochs + 1 / num_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        optimizer_disc.zero_grad()
        optimizer_gen.zero_grad()
        
        
        #feature exrtraction
        #uncomment the [0] if your encoder is a transformer
        #comment the [0] if your encoder is a CNN
        src_features = feature_extractor(src_x_batch)[0] 
        tgt_features = feature_extractor(tgt_x_batch)[0]
        
        #decoding
        src_generator_pred=src_generator(src_features)
        tgt_generator_pred=tgt_generator(tgt_features)
        
        #domain classification
        src_features = ReverseLayerF.apply(src_features, alpha)
        src_domain_pred=discriminator(src_features) 
        tgt_features = ReverseLayerF.apply(tgt_features, alpha)
        tgt_domain_pred=discriminator(tgt_features)
        
        #true labels
        domain_label_src, domain_label_tgt = make_true_dom(src_domain_pred, tgt_domain_pred)
        
        
        src_gen_loss = gen_loss_function(src_generator_pred, src_y_batch)
        tgt_gen_loss = gen_loss_function(tgt_generator_pred, tgt_y_batch)
        src_dom_loss = dom_loss_function(src_domain_pred,domain_label_src)
        tgt_dom_loss = dom_loss_function(tgt_domain_pred,domain_label_tgt)

        
        loss=src_gen_loss+tgt_gen_loss+src_dom_loss+tgt_dom_loss
        loss.backward()
        
        #running_loss += src_gen_loss.item()+tgt_gen_loss.item()+src_dom_loss.item()+tgt_dom_loss.item()
        running_loss += src_gen_loss.item()
        
        optimizer_disc.step()
        optimizer_gen.step()
        
        #len(src_train_loader) is supposed to be > than len(tgt_train_loader)
        if (src_batch_index==len(tgt_train_loader)-1):  
            avg_loss_across_batches = running_loss / len(tgt_train_loader) 
            print('Batch {0}, Loss: {1:.3f}'.format(src_batch_index+1,
                                                    avg_loss_across_batches))

    scheduler.step()
    print()
    return avg_loss_across_batches


def validate_one_epoch(feature_extractor,generator,discriminator,epoch,test_loader,
                       gen_loss_function,dom_loss_function):
    feature_extractor.train(False)
    generator.train(False)
    discriminator.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            #uncomment the [0] if your encoder is a transformer
            #comment the [0] if your encoder is a CNN
            features = feature_extractor(x_batch)[0]
            generator_pred = generator(features)
            loss = gen_loss_function(generator_pred, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()
    return avg_loss_across_batches
    
def evaluation(y_pred,y_true):
    rmse=np.sqrt(mean_squared_error(y_pred,y_true))
    mape=mean_absolute_percentage_error(y_pred,y_true)
    r2score=r2_score(y_pred,y_true)
    return rmse,mape,r2score

    
    

    

def train_one_epoch_withoutDA(feature_extractor,tgt_generator,
                    tgt_train_loader,gen_loss_function,optimizer_gen,
                    scheduler,epoch,num_epochs):
    torch.cuda.empty_cache()
    feature_extractor.train(True)
    tgt_generator.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    
    for (tgt_batch_index, tgt_batch) in enumerate(tgt_train_loader):
        tgt_x_batch, tgt_y_batch = tgt_batch[0].to(device), tgt_batch[1].to(device)

        optimizer_gen.zero_grad()
        
        
        #feature exrtraction
        tgt_features = feature_extractor(tgt_x_batch)#[0]
        
        #decoding
        tgt_generator_pred=tgt_generator(tgt_features)
        
        #compute loss

        tgt_gen_loss = gen_loss_function(tgt_generator_pred, tgt_y_batch)
        
        #backpropagation

        tgt_gen_loss.backward()
        
        running_loss += tgt_gen_loss.item()
        
        optimizer_gen.step()
        
        if (tgt_batch_index==len(tgt_train_loader)-1):  
            avg_loss_across_batches = running_loss / len(tgt_train_loader)
            print('Batch {0}, Loss: {1:.3f}'.format(tgt_batch_index+1,
                                                    avg_loss_across_batches))

    scheduler.step()
    print()
    return avg_loss_across_batches


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def make_true_dom(
    src_dom_: torch.Tensor, tgt_dom_: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make true labels for domain classification"""
    src_dom, tgt_dom = (
        torch.zeros_like(src_dom_, device=src_dom_.device),
        torch.zeros_like(tgt_dom_, device=tgt_dom_.device),
    )
    src_dom[:, 0, :], tgt_dom[:, 1, :] = 1, 1
    return src_dom, tgt_dom






