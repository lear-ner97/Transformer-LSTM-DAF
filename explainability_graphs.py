# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:55:55 2024

@author: umroot
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch



def heatmap(feature_extractor,batch,sample_index):
    sns.heatmap(feature_extractor(batch)[1][sample_index,:,:].cpu().detach().numpy())
    


def visualize_filters(feature_extractor):
    # # Get the weights of the convolutional layer
    weights = feature_extractor.conv1.weight.data #conv1 (1*64 filters)or conv2(for conv2 we have 64 *64 fil)

    # Normalize the weights
    normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Plot the filters
    plt.figure(figsize=(10, 5))
    num_filters = normalized_weights.shape[0]
    for i in range(int(num_filters/4)):
        plt.subplot(4, 4, i+1)  # Change the subplot arrangement according to your number of filters
        plt.plot(normalized_weights[i].squeeze().cpu().numpy())  # Assuming 1D time series data
        plt.title('Filter {}'.format(i + 1))
        plt.axis('off')
    plt.show()
    
    
def hook_fn(module, input, output):
    global layer_output
    layer_output = output

def activation_mapping(feature_extractor,x):
    global layer_output
    layer=feature_extractor.conv1
    # Register the hook
    handle = layer.register_forward_hook(hook_fn)
    # Forward pass the image through the model
    feature_extractor.eval()
    
    with torch.inference_mode():
        preds = feature_extractor(x.unsqueeze(0))
    

    layer_output = layer_output.squeeze()
    
    rows, cols = 4, 4
    
    fig = plt.figure(figsize=(20, 6))
    
    for i in range(1, (rows * cols)+1):
        feature_map = layer_output[i-1, :].detach().cpu().numpy()
        fig.add_subplot(rows, cols, i)
        plt.imshow(np.expand_dims(feature_map,0), cmap='viridis')
        plt.title('filter '+str(i))
        plt.colorbar()
    

        plt.tight_layout()
        plt.axis(False)