# Control, Generate, Augment

The repository contains the following foldes::

single_attribute_control: Contains the scripts to train the Variational Autoencoder and to generate sentences
    ``run Analisys.py`` to train the model
    run Generation.py to generate sentences 

Code_multiple_attribute_hotOneEncoder: Works exactly as the single_Attribute the only difference is that there you can actually generate               sentences where the control is on more than one attribute

Evaluation: Contains the script to perform data augmentation and check the performance a Bi-LSTM on a downstream task(sentiment classification)

