# Import modules
from torchvision.utils import save_image
from numpy import random, arange
import torch.nn.functional as F
from itertools import cycle
from tqdm import tqdm
import torch


def train(model,
          args,
          optimizer,
          labeledloader,
          unlabeledloader):
    """
    The training function.

    Returns
    -------
    Nothing.
    """
    # Model: train
    model.train()

    # Loop through the data
    if len(labeledloader) > len(unlabeledloader):
        for (l, y), (u, _) in tqdm(zip(cycle(unlabeledloader), labeledloader), desc='  Training'):
            # Convert the data to cuda if available
            l = l.to(device=args.device).squeeze(0)
            y = y[0].to(device=args.device)
            u = u.to(device=args.device).squeeze(0)
            if args.real_image_counter < 1:
                # Save the image
                # from the labeled
                save_image(l,
                           f'{args.RECONPATH}/realImageExamples_Labeled_E{args.epoch}.png',
                           nrow=5)
                # from the unlabeled
                save_image(u,
                           f'{args.RECONPATH}/realImageExamples_Unlabeled_E{args.epoch}.png',
                           nrow=5)
                args.real_image_counter += 1
            # Refresh the gradients
            optimizer.zero_grad(set_to_none=True)
            # Calculate ELBO for labeled data [backward mean]
            elbo_l_mean, acc = model(l, y, train=True)
            # Calculate ELBO for unlabeled data [backward mean]
            elbo_u_mean = model(u, train=True)
            # Track elbo results together [mean]
            elbo = (elbo_l_mean + elbo_u_mean) / (l.shape[0] + u.shape[0])
            # Backward [mean]
            elbo.backward()
            # Optimizer step
            optimizer.step()

            if args.test_mode:
                break
    else:
        for (l, y), (u, _) in tqdm(zip(cycle(labeledloader), unlabeledloader), desc='  Training'):
            # Convert the data to cuda if available
            l = l.to(device=args.device).squeeze(0)
            y = y[0].to(device=args.device)
            u = u.to(device=args.device).squeeze(0)
            if args.real_image_counter < 1:
                # Save the image
                # from the labeled
                save_image(l,
                           f'{args.RECONPATH}/realImageExamples_Labeled_E{args.epoch}.png',
                           nrow=5)
                # from the unlabeled
                save_image(u,
                           f'{args.RECONPATH}/realImageExamples_Unlabeled_E{args.epoch}.png',
                           nrow=5)
                args.real_image_counter += 1
            # Refresh the gradients
            optimizer.zero_grad(set_to_none=True)
            # Calculate ELBO for labeled data [backward mean]
            elbo_l_mean, acc = model(l, y, train=True)
            # Calculate ELBO for unlabeled data [backward mean]
            elbo_u_mean = model(u, train=True)
            # Track elbo results together [mean]
            elbo = (elbo_l_mean + elbo_u_mean) / (l.shape[0] + u.shape[0])
            # Backward [mean]
            elbo.backward()
            # Optimizer step
            optimizer.step()

            if args.test_mode:
                break


def validate(model,
             args,
             loader,
             loadername,
             train=True):
    """
    The validation function. Validates the ELBO + MIL, ELBO, and the accuracy
    of the given [training, validation or test] loader.

    Returns
    -------
    loss: list
        ELBO + MIL losses of the training set.
    loss_VAE: list
        ELBO losses of the training set.
    acc_val: list
        The accuracy rates of the training set.
    """
    # Model: validate
    model.eval()

    # Declare loss, accuracy trackers
    loss_val, loss_VAE, acc_val = 0., 0., 0.
    # Initialize the number of points
    N = 0
    # Loop through the data
    for (x, y) in tqdm(loader, desc=f'  Validation[{loadername}]'):
        # Convert the data to cuda if available
        x = x.to(device=args.device).squeeze(0)
        y = y[0].to(device=args.device)
        # Update the N
        N += 2 * x.shape[0]

        # Calculate ELBO for labeled data
        elbo, elbo_l_sum, acc = model(x, y)
        # Calculate ELBO for unlabeled data
        elbo_u_sum = model(x)
        # Track elbo results together [sum]
        # VAE + MIL loss track
        loss_val += (elbo_l_sum + elbo_u_sum).item()
        # Plain VAE loss track
        loss_VAE += (elbo + elbo_u_sum).item()
        # Track accuracy
        acc_val += acc

        if args.test_mode:
            break

        # Try to free some memory on GPU if it is the Colon Cancer dataset
        if args.data == 'Colon':
            # Delete loss
            del elbo_l_sum
            del elbo_u_sum
            del elbo
            del acc
            # Free deleted memory
            torch.cuda.empty_cache()

    # Divide the loss by the number of points
    loss = loss_val / N
    # Divide the plain ELBO by the number of points
    loss_VAE = loss_VAE / N
    # Divide the accuracy by the length of the loader
    acc_val = acc_val / len(loader)

    print(
        f'  [Valid {loadername}]\t elbo + MIL loss: {loss: .2f}, ',
        f'elbo: {loss_VAE: .2f} ',
        f'accuracy: {acc_val:.2f}'
    )

    # If the loader is not the training loader
    if not train:
        # If the validation accuracy is higher than the previous one
        if acc_val > args.max_acc:
            # Save the model
            torch.save(model.state_dict(),
                       f'{args.MODELPATH}/{args.mode}_E{args.epoch}.pt')
            # Update the accuracy value
            args.max_acc = acc_val
            # Update the loss value
            args.least_loss = loss
        # If the accuracy is the same but the loss is less
        elif acc_val == args.max_acc and loss <= args.least_loss:
            # Save the model
            torch.save(model.state_dict(),
                       f'{args.MODELPATH}/{args.mode}_E{args.epoch}.pt')
            # Update the accuracy value
            args.max_acc = acc_val
            # Update the loss value
            args.least_loss = loss

    # Return validation records
    return loss, loss_VAE, acc_val


def train_full(model,
               args,
               optimizer,
               trainloader):
    """
    The training function.

    Returns
    -------
    Nothing.
    """
    # Model: train
    model.train()

    # Loop through the data
    for (l, y) in tqdm(trainloader, desc='  Training'):
        # Convert the data to cuda if available
        l = l.to(device=args.device).squeeze(0)
        y = y[0].to(device=args.device)
        if args.real_image_counter < 1:
            # Save the image
            # from the labeled
            save_image(l,
                       f'{args.RECONPATH}/realImageExamples_Labeled_E{args.epoch}.png',
                       nrow=5)
            args.real_image_counter += 1
        # Refresh the gradients
        optimizer.zero_grad(set_to_none=True)
        # Calculate ELBO for labeled data [backward mean]
        elbo_l_mean, acc = model(l, y, train=True)
        # Track elbo results together [mean]
        elbo = elbo_l_mean / l.shape[0]
        # Backward [mean]
        elbo.backward()
        # Optimizer step
        optimizer.step()

        if args.test_mode:
            break


def validate_full(model,
                  args,
                  loader,
                  loadername,
                  train=True):
    """
    The validation function. Validates the ELBO + MIL, ELBO, and the accuracy
    of the given [training, validation or test] loader.

    Returns
    -------
    loss: list
        ELBO + MIL losses of the training set.
    loss_VAE: list
        ELBO losses of the training set.
    acc_val: list
        The accuracy rates of the training set.
    """
    # Model: validate
    model.eval()

    # Declare loss, accuracy trackers
    loss_val, loss_VAE, acc_val = 0., 0., 0.
    # Initialize the number of points
    N = 0
    # Loop through the data
    for (x, y) in tqdm(loader, desc=f'  Validation[{loadername}]'):
        # Convert the data to cuda if available
        x = x.to(device=args.device).squeeze(0)
        y = y[0].to(device=args.device)
        # Update the N
        N += x.shape[0]

        # Calculate ELBO for labeled data
        elbo, elbo_l_sum, acc = model(x, y)
        # Track elbo results together [sum]
        # VAE + MIL loss track
        loss_val += elbo_l_sum.item()
        # Plain VAE loss track
        loss_VAE += elbo.item()
        # Track accuracy
        acc_val += acc

        # Try to free some memory on GPU if it is the Colon Cancer dataset
        if args.data == 'Colon':
            # Delete loss
            del elbo_l_sum
            del elbo
            del acc
            # Free deleted memory
            torch.cuda.empty_cache()

        if args.test_mode:
            break

    # Divide the loss by the number of points
    loss = loss_val / N
    # Divide the plain ELBO by the number of points
    loss_VAE = loss_VAE / N
    # Divide the accuracy by the length of the loader
    acc_val = acc_val / len(loader)

    print(
        f'  [Valid {loadername}]\t elbo + MIL loss: {loss: .2f}, ',
        f'elbo: {loss_VAE: .2f} ',
        f'accuracy: {acc_val:.2f}'
    )

    # If the loader is not the training loader
    if not train:
        # If the validation accuracy is higher than the previous one
        if acc_val > args.max_acc:
            # Save the model
            torch.save(model.state_dict(),
                       f'{args.MODELPATH}/{args.mode}_E{args.epoch}.pt')
            # Update the accuracy value
            args.max_acc = acc_val
            # Update the loss value
            args.least_loss = loss
        # If the accuracy is the same but the loss is less
        elif acc_val == args.max_acc and loss <= args.least_loss:
            # Save the model
            torch.save(model.state_dict(),
                       f'{args.MODELPATH}/{args.mode}_E{args.epoch}.pt')
            # Update the accuracy value
            args.max_acc = acc_val
            # Update the loss value
            args.least_loss = loss

    # Return validation records
    return loss, loss_VAE, acc_val


def train_base(model,
               args,
               optimizer,
               trainloader):
    """
    The training function. Trains the base models [either base VAE or the base MIL].

    Returns
    -------
    Nothing.
    """
    # Model: train
    model.train()

    # Trace
    print('  ---Training!---')
    # Loop through the data
    for data, label in tqdm(trainloader, desc='  Training'):
        # Convert the data to cuda if available
        data = data.to(device=args.device).squeeze(0)
        if args.real_image_counter < 1:
            # Save the image
            save_image(data,
                       f'{args.RECONPATH}/realImageExamples_Labeled_E{args.epoch}.png',
                       nrow=5)
            args.real_image_counter += 1
        # Refresh the gradients
        optimizer.zero_grad(set_to_none=True)
        # If args.mode is 'base_att'
        if args.model == 'base_att':
            # Convert the label to cuda if available
            label = label[0].to(device=args.device)
            # Calculate the objective for the Attention MIL
            # (name kept the same not to duplicate the code blocks)
            elbo_u_mean = model.calculate_objective(data, label)[0]
        # Otherwise
        else:
            # Calculate ELBO for unlabeled data [backward mean]
            elbo_u_mean = model(data, train=True)

        # Backward [mean]
        elbo_u_mean.backward()
        # Optimizer step
        optimizer.step()

        if args.test_mode:
            break


def validate_base(model,
                  args,
                  loader,
                  loadername,
                  train=True):
    """
    The validation function. Validates the ELBO + MIL, ELBO, and the accuracy
    of the given [training, validation or test] loader.

    Returns
    -------
    loss: list
        Either the ELBO (from base VAE) or the accuracy rate (from the base MIL).
    """
    # Model: validate
    model.eval()

    # Declare loss tracker
    loss_val = 0.
    # Initialize the number of points
    N = 0
    # Loop through the data
    for data, label in tqdm(loader, desc=f'  Validation[{loadername}]'):
        # Convert the data to cuda if available
        data = data.to(device=args.device).squeeze(0)
        # Update the N
        N += data.shape[0]
        # If args.mode is 'base_att'
        if args.model == 'base_att':
            # Convert the label to cuda if available
            label = label[0].to(device=args.device)
            # Calculate the objective for the Attention MIL
            # (name kept the same not to duplicate the code blocks)
            elbo_u_sum = model.calculate_classification_error(data, label)[0]
        # Otherwise
        else:
            # Calculate ELBO for unlabeled data
            elbo_u_sum = model(data)

        # Track elbo results together [sum]
        loss_val += elbo_u_sum.item()

        if args.test_mode:
            break

    # If the mode is base_att
    if args.model == 'base_att':
        # Divide the accuracy by the length of the loader
        loss_val = loss_val / len(loader)
        # Trace
        print(f'  [Valid {loadername}]\t accuracy: {loss_val: .2f}')
        # If the loader is not the training loader
        if not train:
            # If the validation accuracy is higher than the previous one
            if loss_val >= args.prev_val:
                # Save the model
                torch.save(model.state_dict(),
                           f'{args.MODELPATH}/{args.mode}_E{args.epoch}.pt')
                # Update the accuracy value
                args.prev_val = loss_val
    # If the mode is base
    elif args.model == 'base':
        # Divide the loss by the number of points
        loss_val = loss_val / N
        # Trace
        print(f'  [Valid {loadername}]\t elbo: {loss_val: .2f}')
        # If the loader is not the training loader
        if not train:
            # If the validation loss is lower than the previous one
            if loss_val <= args.prev_val:
                # Save the model
                torch.save(model.state_dict(),
                           f'{args.MODELPATH}/{args.mode}_E{args.epoch}.pt')
                # Update the accuracy value
                args.prev_val = loss_val

    # Return validation records
    return loss_val


def sample_from_dmol(x_mean, nc=3, nmix=2, random_sample=False):
    """
    Sample from Discretized mix of logistic distribution.

    Credit: https://github.com/ioangatop/srVAE/blob/master/src/modules/distributions.py#L119
    """
    scale_min, scale_max = [0., 1.]

    # unpack values
    # pi
    logit_probs = x_mean[:, :nmix]
    batch_size, nmix, H, W = logit_probs.size()
    means = x_mean[:, nmix:(nc + 1) * nmix].view(batch_size, nmix,
                                                 nc, H, W)  # mean
    logscales = x_mean[:, (nc + 1) * nmix:(nc * 2 + 1) *
                       nmix].view(batch_size, nmix, nc, H, W)  # log_var
    coeffs = x_mean[:, (nc * 2 + 1) * nmix:(nc * 2 + 4) *
                    nmix].view(batch_size, nmix, nc, H, W)  # chan_coeff

    # activation functions
    logscales = logscales.clamp(min=-7.)
    logit_probs = F.log_softmax(logit_probs, dim=1)
    coeffs = coeffs.tanh()

    # sample mixture
    index = logit_probs.argmax(dim=1, keepdim=True) + \
        logit_probs.new_zeros(means.size(0), *means.size()[2:]).long()
    one_hot = means.new_zeros(means.size()).scatter_(1, index.unsqueeze(1), 1)
    means = (means * one_hot).sum(dim=1)
    logscales = (logscales * one_hot).sum(dim=1)
    coeffs = (coeffs * one_hot).sum(dim=1)
    x = means

    if random_sample:
        # sample y from CDF
        u = means.new_zeros(means.size()).uniform_(1e-5, 1 - 1e-5)
        # from y map it to the corresponing x
        x = x + logscales.exp() * (torch.log(u) - torch.log(1.0 - u))

    # concat image channels
    x0 = (x[:, 0]).clamp(min=scale_min, max=scale_max)
    x1 = (x[:, 1] + coeffs[:, 0] * x0).clamp(min=scale_min, max=scale_max)
    x2 = (x[:, 2] + coeffs[:, 1] * x0 + coeffs[:, 2] * x1).clamp(min=scale_min, max=scale_max)
    x = torch.stack([x0, x1, x2], dim=1)
    return x


def sample_image(model, args, idx, epoch):
    """
    The sampling function. Saves the image as png and pdf format.

    Returns
    -------
    Nothing.
    """
    # No gradient
    model.eval()

    # If it is the Colon Cancer dataset
    if args.data == 'Colon':
        # Declare randomly a bag size
        bag_size = random.choice(arange(10, 60, 10))
        # Generate random z
        z = torch.randn(bag_size, args.dec_inp, *args.latent_img_size, device=args.device)
    # If it the MNIST dataset
    elif args.data == 'MNIST':
        # Generate random z
        z = torch.randn(args.mean_bag_length, args.enc_out,
                        *args.latent_img_size, device=args.device)
    # If it is the disentaglement model
    if args.model == 'disent':
        # Chunk z into 3
        z1, z2, z3 = torch.chunk(z, 3, 1)
        # Create z for decoder
        z_dec = torch.cat((z1, z2), dim=1)
        # Create z for the attention MIL
        z_mil = torch.cat((z2, z3), dim=1)
        # Reconstruct the image
        img = model.dec(z_dec)
        # If it is the Colon Cancer dataset
        if args.data == 'Colon':
            # Sample from dMoL loss
            img = sample_from_dmol(img, nc=args.input_channels, nmix=args.num_components)
        # Predict the target value
        label = int(model.att(z_mil)[1][0].item())
    # Otherwise
    else:
        # Predict the target value
        label = int(model.att(z)[1][0].item()) \
            if (args.model != 'auxil') and (args.model != 'base') \
            else None
        # Reconstruct the image
        img = model.dec(z)
        # If it is the Colon Cancer dataset
        if args.data == 'Colon':
            # Sample from dMoL loss
            img = sample_from_dmol(img, nc=args.input_channels, nmix=args.num_components)
    # If epoch is not None:
    if epoch is not None:
        # Save the image as png
        save_image(img,
                   f'{args.RECONPATH}/sampleImage{idx}_E{args.epoch}_Y{label}.png',
                   nrow=5)
        # and as pdf
        save_image(img,
                   f'{args.RECONPATH}/sampleImage{idx}_E{args.epoch}_Y{label}.pdf',
                   nrow=5)
    # If epoch is None
    else:
        # Save the image as png
        save_image(img,
                   f'{args.RECONPATH}/sampleTestImage{idx}_Y{label}.png',
                   nrow=5)
        # and as pdf
        save_image(img,
                   f'{args.RECONPATH}/sampleTestImage{idx}_Y{label}.pdf',
                   nrow=5)


def reconstruct_image(args, model, loader):
    """
    The reconstruction function. Saves the image as png and pdf format.

    Returns
    -------
    Nothing.
    """
    # No gradient
    model.eval()

    # Set a counter to stop
    counter = 1
    # Loop through the data
    for data, label in tqdm(loader, desc='Reconstruction'):
        # Convert the data to cuda if available
        data = data.to(device=args.device).squeeze(0)
        label = label[0].to(device=args.device)
        # If data batch is divisible by 5 and lower than 60
        # and the counter is lower than 11
        if data.shape[0] % 5 == 0 and data.shape[0] < 60 and counter < 11:
            # Forward pass of the model
            # Zs
            z_mean, z_logvar = model.enc(data)
            # Sample z
            z = model.enc._sample(z_mean, z_logvar)
            # If the model type is equal to disent
            if args.model == 'disent':
                # Chunk into 3
                z1, z2, z3 = torch.chunk(z, 3, 1)
                # Create z for decoder
                z = torch.cat((z1, z2), dim=1)
            # Decode z
            img = model.dec(z)
            # If it is the Colon Cancer dataset
            if args.data == 'Colon':
                # Sample from dMoL loss
                img = sample_from_dmol(img, nc=args.input_channels, nmix=args.num_components)
            # Save the reconstructed image along with the original ones
            # as png
            save_image(torch.cat([data, img], dim=0),
                       f'{args.RECONPATH}/reconImage{counter}_Y{int(label[0])}.png',
                       nrow=5)
            # and as pdf
            save_image(torch.cat([data, img], dim=0),
                       f'{args.RECONPATH}/reconImage{counter}_Y{int(label[0])}.pdf',
                       nrow=5)
            # Increase the counter
            counter += 1

            # If it is test mode
            if args.test_mode:
                break
        # If the counter is equal to 11
        elif counter == 11:
            # Break
            break
