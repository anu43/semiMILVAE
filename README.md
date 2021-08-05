# Semi-supervised Multiple Instance Learning using Variational Auto-Encoders

## Overview
Pytorch implementation of my Master Thesis project "Semi-supervised Multiple Instance Learning using Variational Auto-Encoders". The experimental findings and the method descriptions can be accessed in `./ssMILVAE.pdf`.

## Installation
Installing PyTorch 1.7.1, using pip or conda, should resolve all dependencies. Tested with Python 3.8. Tested on GPU only. It should work on the CPU as well but considering the size of the model, it is not recommended. PyTorch installation can be followed from https://pytorch.org/get-started/locally/

## Content
In this thesis, a hybrid model comprised of an Attention-based MIL classifier (AD-MIL) and a Variational Autoencoder (VAE) is defined in the direction of developing a deep generative framework for multiple-instance learning (MIL). For such purpose, this dissertation follows the subsequent research questions; (1) Integrating a VAE and an Attention-based Deep MIL classifier, (2) Investigating whether the hybrid learning approach can provide state-of-the-art performance in the semi-supervised setting, (3) Evaluating the proposed approach on the semi-supervised scenario and comparing it with baselines. The experiments are evaluated on the MNIST-BAGS and one real-life histopathology (the COLON CANCER) datasets. One appealing aspect of the results is that the ssMILVAE can integrate the density of the features _p(x)_ and the predictive distribution _p(y|x)_. The availability of the joint density _p(x,y)_ brings an opportunity to gain additional capabilities such as detecting uncertain decision-making of the model and enabling semi-supervised learning. In addition to this, the ssMILVAE achieves better accuracy than the purely predictive baseline model and samples in a similar quality to a VAE when the model encounters an insufficient number of labeled training data. Nevertheless, the attention mechanism provides more accurate attention weights over the instances of a bag which can be used to create heatmaps for representing the region of interests (ROIs).

## Usage
`dataloaders/*`: Constructs the bags from the chosen dataset (either the MNIST or the COLON CANCER).

**NOTE**: To run the experiments on the histopathology dataset, please download [the COLON CANCER](http://www.warwick.ac.uk/BIAlab/data/CRChistoLabeledNucleiHE) dataset.

`models/AttentionMILClassifier.py`: Constructs the Attention-based Deep MIL classifier (Ilse et al., 2018) as described in `ssMILVAE.pdf`.

`models/SSMILVAE.py`: Integrates the VAE and Attention-based Deep MIL.

`models/config.py`: The configuration script to keep the parameters and model constructions consistent.

`main.py`: Trains either baseline models or the ssMILVAE model due to the given parameters in `config.py`.

`train.py`: Contains the training, the validation, and the testing functions including reconstruction and sampling methods.

## Questions and Issues
In case of a bug-finding or a question about the project, you can reach me out by [mailing](mailto:anuzunalioglu@hotmail.com) or opening an issue with a relating subject. Thank you.
