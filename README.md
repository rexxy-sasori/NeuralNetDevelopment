## Running the Script
This repository contains template src code developed in python3 for training neural networks. 
To run the code, run the command:

python main_train.py [YAML File Path]

This command will train a neural network model either from scratch or from a 
previously trained model. In the yaml folder, 1 YAML file for Alexnet on CIFAR10 has been prepared.
In the yaml file, the result_dir needs to be set properly along with any possible path if necessary.
The training script will save the save the best model as
specified in the yaml file. It also offers an option to split the training data to create
a validation set. If use_validation is set to False, then the script will save the model with
either the best test error or test loss. 

When the training is completed, a model with the extension .pt.tar can be found in directory 
specified in result_dir. To evalute the model, run:

python main_eval.py [Model File Path]

The script will print the classification accuracy and draw the confusion matrix. Note: currently
the API only supports training on GPU and evaluating on GPU. If the feature of training on GPU and 
evaluating on CPU is desired, please submit the issue in wiki.

## How to Develop New Neural Network Model
To define new neural network model, the definition should be put in nn/models.py. If  the model
needs some submodule, it is advised to use modules.py. After new model is defined, it must be declared in
the lookup dictionary found in nn/model.py along with its corresponding input for profiling. 

If the new model needs a new training mechanism, it is advised to inherit the Trainer class 
in nn/trainer.py. Two functions need to be overwritten to define the computation done for single batch
during both training and evaluation. 

If a new dataset is needed, the dataset class definition should be in nn/dataset.py. 

If a new nn transform is needed, it is advised to define them in nn/transforms.py

Preprocessing for 2d (Img) data is put in ./vision and that for 1d data is put in ./audio

You can also leverage the API defined in nn/quantization.py to try out fixed-point techniques


