# Import main modules
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Tuple

import numpy as np
from torch import load
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloaders.load_data import load_data

# Import own modules
from models import SSMILVAE, AttentionMILClassifier
from models.config import update_args
from train import (
    reconstruct_image,
    sample_image,
    train,
    train_base,
    train_full,
    validate,
    validate_base,
    validate_full,
)


def create_model_optimizer(args: Namespace) -> Tuple[SSMILVAE.SSMILVAE, Adam]:
    # If the mode is equal to base attention
    if args.model == "base_att":
        # Create the Attention MIL model based on the given dataset
        model = (
            AttentionMILClassifier.AttentionBase_MNIST().to(device=args.device)
            if args.data == "MNIST"
            else AttentionMILClassifier.AttentionBase_Colon().to(device=args.device)
        )
        # Declare the optimizer parameters
        # Declare the learning rate
        args.lr = 0.0005 if args.data == "MNIST" else 0.0001
        # Declare the weight decay
        args.wd = 0.0001 if args.data == "MNIST" else 0.0005
        # Declare the optimizer
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Otherwise
    else:
        # If it is the separate optimizers version
        if args.model == "sepopt":
            # Create the model
            model = SSMILVAE.SSMILVAE(args).to(device=args.device)
            # Create an additional optimizer for the classifier
            optimizer_classifier = Adam(
                model.att.parameters(),
                lr=0.0001,
                betas=(0.9, 0.999),
                weight_decay=0.0005,
            )
            # Declare the optimizer
            optimizer_vae = Adam(
                list(model.enc.parameters()) + list(model.dec.parameters()),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=args.wd,
            )
            # Declare the multiple optimizers object
            optimizer = SSMILVAE.MultipleOptimizer(optimizer_vae, optimizer_classifier)
        # Otherwise
        else:
            # Create the model
            model = SSMILVAE.SSMILVAE(args).to(device=args.device)
            # Declare the optimizer
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Return the model and the optimizer
    return model, optimizer


def main(
    model: SSMILVAE.SSMILVAE,
    args: Namespace,
    optimizer: Adam,
    trainloader: DataLoader,
    labeledloader: DataLoader,
    unlabeledloader: DataLoader,
    validationloader: DataLoader,
    testloader: DataLoader,
):
    """
    The steps are as follows.
        1. Training
            1.1. Validation of the training set.
            1.2. Validation of the validation set.
        2. Validation of the test set.
        3. Reconstruction of the images by the best model.
        4. Sampling of the images by the best model.

    Returns
    -------
    losses_train: list
        ELBO + MIL losses of the training set.
    losses_VAE_train: list
        ELBO losses of the training set.
    accs_train: list
        The accuracy rates of the training set.
    losses_val: list
        ELBO + MIL losses of the validation set.
    losses_VAE_val: list
        ELBO losses of the validation set.
    accs_val: list
        The accuracy rates of the validation set.
    """
    # Declare training trackers
    losses_train = list()
    losses_VAE_train = list()
    losses_val = list()
    losses_VAE_val = list()
    accs_train = list()
    accs_val = list()
    # Declare accuracy and loss trackers
    args.max_acc = -1.0
    args.least_loss = 1e9
    # Declare real image counter to save
    args.real_image_counter = 0
    # Trace args
    print(args)
    # Train
    for epoch in range(1, args.epochs + 1):
        # Trace
        print(f"Epoch {epoch}")
        # Set the epoch value to args
        args.epoch = epoch
        # If run with all data as labeled
        if args.full_data_train:
            # Run training
            train_full(model, args, optimizer, trainloader)
            # Run validation [for training set]
            loss_train, loss_VAE_train, acc_train = validate_full(
                model, args, trainloader, loadername="-Train-", train=True
            )
            # Run validation [for validation set]
            loss_val, loss_VAE_val, acc_val = validate_full(
                model, args, validationloader, loadername="-Validation-", train=False
            )
        # Otherwise
        else:
            # Run training
            train(model, args, optimizer, labeledloader, unlabeledloader)
            # Run validation [for training set]
            loss_train, loss_VAE_train, acc_train = validate(
                model, args, trainloader, loadername="-Train-", train=True
            )
            # Run validation [for validation set]
            loss_val, loss_VAE_val, acc_val = validate(
                model, args, validationloader, loadername="-Validation-", train=False
            )
        # Record validation results [for training set]
        losses_train.append(loss_train)
        losses_VAE_train.append(loss_VAE_train)
        accs_train.append(acc_train)
        # Record validation results [for validation set]
        losses_val.append(loss_val)
        losses_VAE_val.append(loss_VAE_val)
        accs_val.append(acc_val)

    # Declare a test accuracy and ELBO + MIL tracker
    args.max_test_acc = -1.0
    args.min_test_loss = 1e8
    # Test each one of the saved model to find the best
    for modelname in os.listdir(f"{args.MODELPATH}"):
        # # Update the arguments
        # args = update_args(args=args)
        # Create the model again
        model, _ = create_model_optimizer(args=args)
        # Load the state dict
        model.load_state_dict(load(f"{args.MODELPATH}/{modelname}"))
        # If run with all data as labeled
        if args.full_data_train:
            # Run test
            loss_test, loss_VAE_test, acc_test = validate_full(
                model, args, testloader, loadername=f"-Test[{modelname}]-", train=True
            )
        # Otherwise
        else:
            # Run test
            loss_test, loss_VAE_test, acc_test = validate(
                model, args, testloader, loadername=f"-Test[{modelname}]-", train=True
            )
        # Check if the model accuracy was higher than before
        if acc_test > args.max_test_acc:
            # Keep the model name
            args.best_model = modelname
            # Update the maximum accuracy
            args.max_test_acc = acc_test
            # Update the minimum loss
            args.min_test_loss = loss_test
        # Check if the model accuracy was the same but less loss rate
        elif acc_test == args.max_test_acc and loss_test < args.min_test_loss:
            # Keep the model name
            args.best_model = modelname
            # Update the maximum accuracy
            args.max_test_acc = acc_test
            # Update the minimum loss
            args.min_test_loss = loss_test

    # Trace the best model
    print(f"Picked {args.best_model}.")
    # # Update the arguments
    # args = update_args(args=args)
    # Create the model again
    model, _ = create_model_optimizer(args=args)
    # Load the state dict for the last time for the best model
    model.load_state_dict(load(f"{args.MODELPATH}/{args.best_model}"))
    # Reconstruct images
    reconstruct_image(args, model, trainloader)
    # Sample images
    for i in range(1, args.numsamp + 1):
        sample_image(args=args, model=model, idx=i, epoch=None)

    # Return the results
    return (
        losses_train,
        losses_VAE_train,
        accs_train,
        losses_val,
        losses_VAE_val,
        accs_val,
        loss_test,
        loss_VAE_test,
        acc_test,
    )


def main_base(
    model: SSMILVAE.SSMILVAE,
    args: Namespace,
    optimizer: Adam,
    trainloader: DataLoader,
    validationloader: DataLoader,
    testloader: DataLoader,
):
    """
    The steps are as follows.
        1. Training
            1.1. Validation of the training set.
            1.2. Validation of the validation set.
        2. Validation of the test set.
        3. Reconstruction of the images by the best model.
        4. Sampling of the images by the best model.

    Returns
    -------
    losses_train: list
        Either the ELBO (from base VAE) or the accuracy rate (from the base MIL) of the training set.
    losses_val: list
        Either the ELBO (from base VAE) or the accuracy rate (from the base MIL) of the validation set.
    """
    # Declare training trackers
    losses_train = list()
    losses_val = list()
    args.prev_val = -1.0 if args.model == "base_att" else 1e9
    # Declare real image counter to save
    args.real_image_counter = 0
    # Trace args
    print(args)
    # Train
    for epoch in range(1, args.epochs + 1):
        # Trace
        print(f"Epoch {epoch}")
        # Set the epoch value to args
        args.epoch = epoch
        # Run training
        train_base(model, args, optimizer, trainloader)
        # Run validation [for training set]
        loss_train = validate_base(model, args, trainloader, loadername="-Train-")
        # Run validation
        # Record validation results
        losses_train.append(loss_train)
        # Run validation [for validation set]
        loss_val = validate_base(
            model, args, validationloader, loadername="-Validation-", train=False
        )
        # Record validation results
        losses_val.append(loss_val)

    # Declare a test accuracy or ELBO loss tracker
    args.max_test_acc = -1.0 if args.model == "base_att" else 1e9
    # Test each one of the saved model to find the best
    for modelname in os.listdir(f"{args.MODELPATH}"):
        # Load the best model before testing
        # Update the arguments
        args = update_args(args=args)
        # Create the model again
        model, _ = create_model_optimizer(args=args)
        # Load the state dict
        model.load_state_dict(load(f"{args.MODELPATH}/{modelname}"))
        # Run test
        loss_test = validate_base(
            model, args, testloader, loadername=f"-Test[{modelname}]-", train=True
        )

        # If the model is 'base_att'
        if args.model == "base_att":
            # Check if the model accuracy was higher than before
            if loss_test >= args.max_test_acc:
                # Keep the model name
                args.best_model = modelname
                # Update the maximum accuracy
                args.max_test_acc = loss_test
        # If the model is 'base'
        elif args.model == "base":
            # Check if the model accuracy was higher than before
            if loss_test <= args.max_test_acc:
                # Keep the model name
                args.best_model = modelname
                # Update the maximum accuracy
                args.max_test_acc = loss_test
    # Trace the best model
    print(f"Picked {args.best_model}.")

    # If the mode is not 'base_att'
    if args.model != "base_att":
        # Load the best model before testing
        # Update the arguments
        args = update_args(args=args)
        # Create the model again
        model, _ = create_model_optimizer(args=args)
        # Load the state dict for the last time for the best model
        model.load_state_dict(load(f"{args.MODELPATH}/{args.best_model}"))
        # Reconstruct images
        reconstruct_image(args, model, trainloader)
        # Sample images
        for i in range(1, args.numsamp + 1):
            sample_image(args=args, model=model, idx=i, epoch=None)

    # Return the results
    return losses_train, losses_val


if __name__ == "__main__":
    # Declare the arguments
    parser = ArgumentParser(description="Semi-supervised MIL using VAE main script")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Enable the test mode to see if everything works properly.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Colon",
        help="Choose b/w MNIST or Colon Cancer dataset",
    )
    # Parse the arguments
    args = parser.parse_args()
    # Update the arguments
    args = update_args(args=args)

    # Trace
    print(f"Running ssMILVAE main script on {args.device.upper()}.")

    # Set the time before starting the training
    args.TIME = datetime.now().strftime("%y.%m.%d-%H.%M")
    # Declare paths
    args.TRAININGPATH = (
        f"./records/train-test/{args.data}/{args.mode}/{args.TIME}/{args.model}"
        if not args.test_mode
        else f"./records/train-test/{args.data}/{args.mode}/{args.TIME}-test/{args.model}"
    )
    args.RECONPATH = (
        f"./records/reconstructions/{args.data}/{args.mode}/{args.TIME}/{args.model}"
        if not args.test_mode
        else f"./records/reconstructions/{args.data}/{args.mode}/{args.TIME}-test/{args.model}"
    )
    args.MODELPATH = (
        f"./records/models/{args.data}/{args.mode}/{args.TIME}/{args.model}"
        if not args.test_mode
        else f"./records/models/{args.data}/{args.mode}/{args.TIME}-test/{args.model}"
    )
    args.LOADERPATH = "./dataloaders/savedLoaders"
    # Create necessary data folder(s) if it does not exist
    for path in [
        getattr(args, p)
        for p in ["TRAININGPATH", "RECONPATH", "MODELPATH", "LOADERPATH"]
    ]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Load the DataLoaders
    (
        trainloader,
        labeledloader,
        unlabeledloader,
        validationloader,
        testloader,
    ) = load_data(args=args)
    # Trace
    print("Dataloaders are loaded.")

    # Create the model, and the optimizer
    model, optimizer = create_model_optimizer(args=args)
    # Print the model summary
    # print(model.modelsummary())

    # If the mode is base
    if args.model == "base":
        # Delete labeled and unlabeled DataLoaders
        del labeledloader
        del unlabeledloader
        # Run the training & the testing on VAE
        losses_train, losses_val = main_base(
            model, args, optimizer, trainloader, validationloader, testloader
        )
        # Save the training & the validation results
        np.save(
            file=f"{args.TRAININGPATH}/loss-acc_results.npy",
            arr=np.array(
                [
                    losses_train,
                    losses_val,
                ]
            ),
            allow_pickle=False,
        )
    # If the mode is base_att
    elif args.model == "base_att":
        # Delete the train and unlabeled DataLoaders
        del trainloader
        del unlabeledloader
        # Run the training & the testing on VAE
        losses_train, losses_val = main_base(
            model, args, optimizer, labeledloader, validationloader, testloader
        )
        # Save the training & the validation results
        np.save(
            file=f"{args.TRAININGPATH}/loss-acc_results.npy",
            arr=np.array(
                [
                    losses_train,
                    losses_val,
                ]
            ),
            allow_pickle=False,
        )
    # Otherwise
    else:
        # Run the training & the testing on SSMILVAE
        (
            losses_train,
            losses_VAE_train,
            accs_train,
            losses_val,
            losses_VAE_val,
            accs_val,
            _,
            _,
            _,
        ) = main(
            model,
            args,
            optimizer,
            trainloader,
            labeledloader,
            unlabeledloader,
            validationloader,
            testloader,
        )
        # Save the training & the validation results
        np.save(
            file=f"{args.TRAININGPATH}/loss-acc_results.npy",
            arr=np.array(
                [
                    losses_train,
                    losses_VAE_train,
                    accs_train,
                    losses_val,
                    losses_VAE_val,
                    accs_val,
                ]
            ),
            allow_pickle=False,
        )
