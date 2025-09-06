# 🔬 Official Code for 
[**"One-day-ahead electricity load forecasting of non-residential buildings using a modified Transformer-BiLSTM adversarial domain adaptation forecaster"**](https://doi.org/10.1007/s40435-025-01701-x)  
Published in *International Journal of Dynamics and Control, 2025*


## step-by-step guideline to reproduce the results

## install dependencies
put all the github repo in a single folder and install the packages listed in the main file, lines 7-25

## choose the experiment to run
6 hyperparameters to fix in main.py, lines 32-70 
For reproducibility of the results, choose the translation term as mentioned below in section "reproducibility" below

## description of each file
boxplot.py : used to plot the boxplots (figures 10, 11, 12)
correlation_analysis.py: used to choose the independent variables of the model
data_cleaning.py: clean the data using IQR method
data_visualization.py: visualization of load data
dataloader.py: used to prepare the data for training
feature_engineering.py: contains the detailed feature engineering process
models.py: contains the architecture of our DAF model and the benchmarks
train_validation.py: describes the training process (with DA and without DA)
pvalue.py: computes the pvalues (table 9)

## reproducibility
choose random_seed=700

if target_data=="Billi":
    *weeks=20: translation=2
    *weeks=10: translation=2
    *weeks=5: translation=2

if target_data=="Maryann":
    *weeks=20: translation=4
    *weeks=10: translation=4
    *weeks=5: translation=8

if target_data=="Antonina":
    *weeks=20: translation=1
    *weeks=10: translation=5
    *weeks=5: translation=2
    
## Training
run main.py
The expected output: 
1- training & validation metrics per epoch
2- training & testing metrics at the end of the training
3- plot of the actual data vs predictions

## 📖 Citation

If you use this code or find our work helpful, please consider citing:

```bibtex
@article{article,
author = {Brahim, Sami and Amayri, Manar and Bouguila, Nizar},
year = {2025},
month = {04},
pages = {},
title = {One-day-ahead electricity load forecasting of non-residential buildings using a modified Transformer-BiLSTM adversarial domain adaptation forecaster},
volume = {13},
journal = {International Journal of Dynamics and Control},
doi = {10.1007/s40435-025-01701-x}
}

## 📬 Contact
If you have questions or encounter issues, please [open an issue](https://github.com/lear-ner97/Transformer-LSTM-DAF/issues) or contact us at **sami.benbrahim@mail.concordia.ca**.

