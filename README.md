# DSN-Paper

Here offers the basic realizations of algorithms under security case. It contains the main components described in our paper.

# Usage
1. Requirement: Ubuntu 20.04, Python v3.5+, Pytorch and CUDA environment
2. "./Main.py" is about configurations and the basic Federated Learning framework
3. "./Sims.py" describes the simulators for clients and central server
4. "./Utils.py" contains all necessary functions and discusses how to get training and testing data
5. Folder "./Models" includes codes for AlexNet, FC, VGG-11, ResNet and LSTM
6. Folder "./CompFIM" is the package used to compute Fisher Information Matrix (FIM)

# Implementation
 1. Should use "./Main.py" to run results, the command is '''python3 ./Main.py'''
 2. Parameters can be configured in "./Main.py"
