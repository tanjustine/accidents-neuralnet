﻿# Accidents Neural Network Readme

This is an autoencoder and classification neural network program built using Python for our Pattern Recognition class. The Python libraries numpy and matplotlib were used for this program. 

The dataset constitutes of accident reports in Metro Manila. Label 1 signifies accidents that are damage to property. Label 2 signifies accidents with fatal injuries, while label 3 signifies accidents with non-fatal injuries. The dataset has already been filtered and normalized beforehand. I do not own the rights to this dataset.


## Experiments

Two different experiments were also implemented in the program in order to compare their accuracy.
- _Experiment A_: only a classification neural network was used for this experiment.
- _Experiment B_: autoencoders and a classification neural network were used for this experiment.


## Data Files

The program also needs the following data file to run the model (files should be in .csv file format, with comma-delimited data format):

- **trainset.csv​** : Holds all 70 datapoints for training
- **trainset_actual.csv** : Holds the actual value of all 70 datapoints. The labels are translated to (1,0,0) for label 1, (0,1,0) for label 2, and (0,0,1) for label 3.
- **testset.csv​** : Holds all 30 datapoints for testing
- **testset_actual.csv**​ : Holds the actual label of the 30 test datapoints
- **atrain.csv​** : Holds all the train datapoints under label 1 
- **btrain.csv**​ : Holds all the train datapoints under label 2
- **ctrain.csv**​ : Holds all the train datapoints under label 3
- **test1.csv​** : Holds all test datapoints under label 1
- **test2.csv**​ : Holds all test datapoints under label 2
- **test3.csv​** : Holds all test datapoints under label 3

_Note_: All files, such as .csv and .py, should be in the same directory.


## Configurations and Preferences

- **Number of epochs/rounds**: set the number of epochs for the model by inputting a valid value on the “rounds” function.
- **Topology**: ​The topology of each experiment can be configured in the program by changing the integers inside the respective arrays.
- **Activation**: ​Activation can be configured by inputting 1 for Relu or 0 for Sigmoid
- **Learning Rate and Momentum**: ​This can be set by inputting the desired values in their respective functions. The program accepts decimal inputs. By default, both are set to 1.


## Acknowledgement

I would like to thank Mr. Raphael Alampay for his guidance in this project and Dr. Kardi Teknomo for the dataset.