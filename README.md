# 🔬 Official Code for 
[**"One-day-ahead electricity load forecasting of non-residential buildings using a modified Transformer-BiLSTM adversarial domain adaptation forecaster"**](https://doi.org/10.1007/s40435-025-01701-x)  
Published in *International Journal of Dynamics and Control, 2025*


## step-by-step guidelines to reproduce the results

## install dependencies
put all the github repo in a single folder and install the packages listed in the main file, lines 7-25

## choose the experiment to run
6 hyperparameters to fix in main.py, lines 32-70 <br>
For reproducibility of the results, choose the translation term as mentioned below in section "reproducibility" below

## description of each file
boxplot.py : used to plot the boxplots (figures 10, 11, 12)<br>
correlation_analysis.py: used to choose the independent variables of the model<br>
data_cleaning.py: clean the data using IQR method<br>
data_visualization.py: visualization of load data<br>
dataloader.py: used to prepare the data for training<br>
feature_engineering.py: contains the detailed feature engineering process<br>
models.py: contains the architecture of our DAF model and the benchmarks<br>
train_validation.py: describes the training process (with DA and without DA)<br>
pvalue.py: computes the pvalues (table 9)<br>

## reproducibility
choose random_seed=700<br>

if target_data=="Billi":<br>
    *weeks=20: translation=2<br>
    *weeks=10: translation=2<br>
    *weeks=5: translation=2<br>

if target_data=="Maryann":<br>
    *weeks=20: translation=4<br>
    *weeks=10: translation=4<br>
    *weeks=5: translation=8<br>

if target_data=="Antonina":<br>
    *weeks=20: translation=1<br>
    *weeks=10: translation=5<br>
    *weeks=5: translation=2<br>
    
## Training
run main.py.  <br>    The expected output:   <br> 1- training & validation metrics per epoch  <br>  2- training & testing metrics at the end of the training   <br>   3- plot of the actual data vs predictions

## 📬 Contact
If you have questions or encounter issues, please [open an issue](https://github.com/lear-ner97/Transformer-LSTM-DAF/issues).


