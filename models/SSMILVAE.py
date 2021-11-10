# Import class modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Import own modules
from models.AttentionMILClassifier import (
    Attention_Colon,
    AttentionRes_Colon,
    Attention_MNIST,
)

# Declare PI
PI = torch.from_numpy(np.asarray(np.pi))


def log_bernoulli(x, p, eps=1.0e-5):
    """
    Auxiliary function to calculate the reconstruction loss.
    Used in the MNIST Dataset.
    """
    pp = torch.clamp(p, eps, 1.0 - eps)
    log_p = x * torch.log(pp) + (1.0 - x) * torch.log(1.0 - pp)
    log_p = log_p.flatten(1)
    return torch.sum(log_p, dim=-1)


def log_normal_diag(x, mu, log_var):
    """
    Auxiliary function to calculate p(x|z).
    Used in the MNIST Dataset.
    """
    return (
        -0.5 * x.flatten(1).shape[1] * torch.log(2.0 * PI)
        - 0.5 * log_var
        - 0.5 * torch.exp(-log_var) * (x - mu) ** 2.0
    )


def log_standard_normal(x):
    """
    Auxiliary function to calculate p(z).
    Used in the MNIST Dataset.
    """
    return -0.5 * x.flatten(1).shape[1] * torch.log(2.0 * PI) - 0.5 * x ** 2.0


def logsumexp(x, dim=None):
    """
    Numerically stable logsum op. Used in the Colon Cancer Dataset.

    Credit: https://github.com/ioangatop/srVAE/blob/master/src/modules/distributions.py#L25
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))


def dmol_loss(x, output, nc=3, nmix=5, nbits=8):
    """
    Discretized mix of logistic distributions loss. Used in the Colon Cancer Dataset.

    Credit: https://github.com/ioangatop/srVAE/blob/master/src/modules/distributions.py#L53
    """
    bits = 2.0 ** nbits
    scale_min, scale_max = [0.0, 1.0]

    bin_size = (scale_max - scale_min) / (bits - 1.0)
    eps = 1e-12

    # unpack values
    batch_size, nmix, H, W = output[:, :nmix].size()
    logit_probs = output[:, :nmix]
    means = output[:, nmix : (nc + 1) * nmix].view(batch_size, nmix, nc, H, W)
    logscales = output[:, (nc + 1) * nmix : (nc * 2 + 1) * nmix].view(
        batch_size, nmix, nc, H, W
    )
    coeffs = output[:, (nc * 2 + 1) * nmix : (nc * 2 + 4) * nmix].view(
        batch_size, nmix, nc, H, W
    )

    # activation functions and resize
    logit_probs = F.log_softmax(logit_probs, dim=1)
    logscales = logscales.clamp(min=-7.0)
    coeffs = coeffs.tanh()

    x = x.unsqueeze(1)
    means = means.view(batch_size, *means.size()[1:])
    logscales = logscales.view(batch_size, *logscales.size()[1:])
    coeffs = coeffs.view(batch_size, *coeffs.size()[1:])
    logit_probs = logit_probs.view(batch_size, *logit_probs.size()[1:])

    # channel-wise conditional modelling sub-pixels
    mean0 = means[:, :, 0]
    mean1 = means[:, :, 1] + coeffs[:, :, 0] * x[:, :, 0]
    mean2 = means[:, :, 2] + coeffs[:, :, 1] * x[:, :, 0] + coeffs[:, :, 2] * x[:, :, 1]
    means = torch.stack([mean0, mean1, mean2], dim=2)

    # compute log CDF for the normal cases (lower < x < upper)
    x_plus = torch.exp(-logscales) * (x - means + 0.5 * bin_size)
    x_minus = torch.exp(-logscales) * (x - means - 0.5 * bin_size)
    cdf_delta = torch.sigmoid(x_plus) - torch.sigmoid(x_minus)
    log_cdf_mid = torch.log(cdf_delta.clamp(min=eps))

    # Extreme Case #1: x > upper (before scaling)
    upper = scale_max - 0.5 * bin_size
    mask_upper = x.le(upper).float()
    log_cdf_up = -F.softplus(x_minus)

    # Extreme Case #2: x < lower (before scaling)
    lower = scale_min + 0.5 * bin_size
    mask_lower = x.ge(lower).float()
    log_cdf_low = x_plus - F.softplus(x_plus)

    # Extreme Case #3: probability on a sub-pixel is below 1e-5
    #   --> If the probability on a sub-pixel is below 1e-5, we use an approximation
    #       based on the assumption that the log-density is constant in the bin of
    #       the observed sub-pixel value
    x_in = torch.exp(-logscales) * (x - means)
    mask_delta = cdf_delta.gt(1e-5).float()
    log_cdf_approx = x_in - logscales - 2.0 * F.softplus(x_in) + np.log(bin_size)

    # Compute log CDF w/ extrime cases
    log_cdf = log_cdf_mid * mask_delta + log_cdf_approx * (1.0 - mask_delta)
    log_cdf = log_cdf_low * (1.0 - mask_lower) + log_cdf * mask_lower
    log_cdf = log_cdf_up * (1.0 - mask_upper) + log_cdf * mask_upper
    # Compute log loss
    loss = logsumexp(log_cdf.sum(dim=2) + logit_probs, dim=1)
    # Return the loss
    return loss.view(loss.shape[0], -1).sum(1)


class MultipleOptimizer(object):
    """Constructs a multiple optimizer object for the sepopt model."""

    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self, set_to_none=True):
        for op in self.optimizers:
            op.zero_grad(set_to_none=set_to_none)

    def step(self):
        for op in self.optimizers:
            op.step()


class ConvEncoder_MNIST(nn.Module):
    """
    The Encoder of the VAE that will be used in the MNIST Dataset.
    """

    # Declare init
    def __init__(self, args):
        super(ConvEncoder_MNIST, self).__init__()

        # The number of input channels
        # MNIST: 1
        self.input_channels = args.input_channels
        # Declare input dims for the model summary method
        self.recon_dims = args.recon_dims
        # The latent dimension size
        self.z_dim = args.enc_out
        # The hidden layer dimensions
        self.hidden_dims = args.hidden_dims

        # Declare modules for encoder
        modules = []

        # Build Encoder modules
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.input_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
            self.input_channels = h_dim
        # Build Encoder
        self.encoder = nn.Sequential(*modules)
        # Build the latent mean layer
        self.z_mean = nn.Conv2d(
            in_channels=self.hidden_dims[-1], out_channels=self.z_dim, kernel_size=1
        )
        # Build the latent logvar layer
        self.z_logvar = nn.Conv2d(
            in_channels=self.hidden_dims[-1], out_channels=self.z_dim, kernel_size=1
        )

    def _encode(self, x):
        # [B, C:#instancesInBag, H:28, W:28] -> [B, C:h_dim[-1], H:1, W:1]
        x = self.encoder(x)
        # [B, C:h_dim[-1], H:1, W:1] -> [B, C:z_dim, H:1, W:1]
        z_mean = self.z_mean(x)
        # [B, C:h_dim[-1], H:1, W:1] -> [B, C:z_dim, H:1, W:1]
        z_logvar = self.z_logvar(x)
        # Return mean and logvar
        return z_mean, z_logvar

    @staticmethod
    def _sample(z_mean, z_logvar):
        # Declare the standard deviation
        std = torch.exp(0.5 * z_logvar)
        # Declare the epsilon
        eps = torch.randn_like(std)
        # Return the sample z
        return z_mean + std * eps

    def modelsummary(self):
        return summary(self, self.recon_dims)

    def forward(self, x):
        # Encode the given batch and receive mean & variance
        z_mean, z_logvar = self._encode(x)
        # Return latent mean, and log variance
        return z_mean, z_logvar


class ConvEncoder_Colon(nn.Module):
    """
    The Encoder of the VAE that will be used in the Colon Cancer Dataset.
    """

    def __init__(self, args):
        super(ConvEncoder_Colon, self).__init__()
        # The number of input channels
        # Colon: 3
        self.input_channels = args.input_channels
        # Declare input dims for the model summary method
        self.recon_dims = args.recon_dims
        # The latent dimension size
        self.z_dim = args.enc_out
        # The hidden layer dimensions
        self.hidden_dims = args.hidden_dims
        # The kernel sizes
        self.kernel_sizes = args.kernel_sizes

        # Build the encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, self.hidden_dims[0], self.kernel_sizes[0], 1, 0),
        #     nn.LeakyReLU(0.2),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(
        #         self.hidden_dims[0], self.hidden_dims[1], self.kernel_sizes[1], 1, 0
        #     ),
        #     nn.LeakyReLU(0.2),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(self.hidden_dims[1], self.z_dim, self.kernel_sizes[2], 1, 0),
        # )
        self.encoder = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        self.encoder.fc = nn.Linear(in_features=512, out_features=self.z_dim)

    def _encode(self, x):
        # Encode, then chunk into two [z_mean, z_logvar]
        x = self.encoder(x)
        x = x.reshape(*x.shape, 1, 1)
        return torch.chunk(x, 2, 1)

    @staticmethod
    def _sample(z_mean, z_logvar):
        # Declare the standard deviation
        std = torch.exp(0.5 * z_logvar)
        # Declare the epsilon
        eps = torch.randn_like(std)
        # Return the sample z
        return z_mean + std * eps

    def modelsummary(self):
        return summary(self, self.recon_dims)

    def forward(self, x):
        # Encode the given batch and receive mean & variance
        z_mean, z_logvar = self._encode(x)
        # Return latent mean, and log variance
        return z_mean, z_logvar


class ConvDecoder_MNIST(nn.Module):
    """
    The Decoder of the VAE that will be used in the MNIST Dataset.
    """

    # Declare init

    def __init__(self, args):
        super(ConvDecoder_MNIST, self).__init__()

        # The latent dimension size
        self.z_dim = args.dec_inp
        # The latent dimension size [input size of the decoder]
        self.input_channels = args.dec_inp
        # The number of output channels [reconstruction dimension]
        self.output_channels = (
            args.input_channels if args.data == "MNIST" else args.dec_out
        )
        # The hidden layer dimensions
        self.hidden_dims = args.hidden_dims[::-1]
        # # Reverse the hidden layers
        # self.hidden_dims.reverse()

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.z_dim,
                out_channels=self.hidden_dims[0],
                kernel_size=1,
                stride=1,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[1], 4, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[2], 4, 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[3], 4, 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[4], 4, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(self.hidden_dims[4], self.output_channels, 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Decode the latent space
        return self.decoder(x)


class ConvDecoder_Colon(nn.Module):
    """
    The Decoder of the VAE that will be used in the Colon Cancer Dataset.
    """

    def __init__(self, args):
        super(ConvDecoder_Colon, self).__init__()
        # The latent dimension size
        self.z_dim = args.dec_inp
        # The latent dimension size [input size of the decoder]
        self.input_channels = args.dec_inp
        # The number of output channels [reconstruction dimension]
        self.output_channels = args.dec_out
        # The hidden layer dimensions
        self.hidden_dims = args.hidden_dims[::-1]
        # The kernel sizes
        self.kernel_sizes = args.kernel_sizes[::-1]
        # # Reverse the hidden layers
        # self.hidden_dims.reverse()
        # # Reverse the kernel sizes
        # self.kernel_sizes.reverse()

        # Build the decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         self.input_channels, self.hidden_dims[0], self.kernel_sizes[0], 1, 0
        #     ),
        #     nn.LeakyReLU(0.2),
        #     nn.Upsample(scale_factor=2),
        #     nn.ConvTranspose2d(
        #         self.hidden_dims[0], self.hidden_dims[1], self.kernel_sizes[1], 1, 0
        #     ),
        #     nn.LeakyReLU(0.2),
        #     nn.Upsample(scale_factor=2),
        #     nn.ConvTranspose2d(self.hidden_dims[1], self.output_channels, 7, 1, 0),
        # )
        self.decoder = ResNet18Dec(z_dim=self.z_dim, out_channels=self.output_channels)

    def modelsummary(self):
        return summary(self, self.recon_dims)

    def forward(self, x):
        return self.decoder(x)


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3, out_channels=100):
        super().__init__()
        self.in_planes = 512
        self.out_channels = out_channels

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, self.out_channels, kernel_size=7, scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        h = self.linear(z.view(*z.shape[:2]))
        h = h.view(z.size(0), 512, 1, 1)
        h = F.interpolate(h, scale_factor=4)
        h = self.layer4(h)
        h = self.layer3(h)
        h = self.layer2(h)
        h = self.layer1(h)
        h = torch.sigmoid(self.conv1(h))
        h = h.view(h.size(0), self.out_channels, 28, 28)
        return h


class SSMILVAE(nn.Module):
    """
    Semi-supervised MIL VAE architecture. Contains the plain, the disentanglement,
    the separate optimizers, and the auxiliary network setups.
    """

    def __init__(self, args):
        super(SSMILVAE, self).__init__()
        # Declare the arguments
        self.args = args

        # Declare Encoder
        self.enc = (
            ConvEncoder_MNIST(args) if args.data == "MNIST" else ConvEncoder_Colon(args)
        )
        # Declare Decoder
        self.dec = (
            ConvDecoder_MNIST(args) if args.data == "MNIST" else ConvDecoder_Colon(args)
        )
        # If it is not the base VAE model
        if args.model != "base":
            # Declare Attention MIL Classifier
            self.att = (
                Attention_MNIST(args) if args.data == "MNIST" else AttentionRes_Colon()
            )
        # If the model type is equal to auxil
        if self.args.model == "auxil":
            # Declare the auxiliary network
            self.auxil = AuxiliaryConvNet(args)
        # Trace
        print(
            f"Constructed the {args.model.upper()} model based on the {args.data} dataset."
        )

    def modelsummary(self):
        return summary(self, self.args.recon_dims)

    @staticmethod
    def _compute_re_for_MNIST(x, recon_x):
        return log_bernoulli(x, recon_x)

    @staticmethod
    def _compute_re_for_Colon(x, x_logits, nmix=5):
        return dmol_loss(x, x_logits, nmix=nmix)

    @staticmethod
    def _compute_kl(z, z_mean, z_logvar):
        """use sum output."""
        kl = log_standard_normal(z) - log_normal_diag(z, z_mean, z_logvar)
        return kl.flatten(1).sum(-1)

    def forward(self, x, y=None, train=False):
        # MNIST: [Bag size, 1, 28, 28] -> [Bag size, z_dim, 1, 1]
        z_mean, z_logvar = self.enc(x)
        # Sample z
        z = self.enc._sample(z_mean, z_logvar)
        # If the model type is equal to disent
        if self.args.model == "disent":
            # Chunk into 3
            z1, z2, z3 = torch.chunk(z, 3, 1)
            # Create z for decoder
            z_dec = torch.cat((z1, z2), dim=1)
            # Create z for the attention MIL
            z_mil = torch.cat((z2, z3), dim=1)
            # MNIST: [Bag size, 1, 28, 28]
            recon_x = self.dec(z_dec)
        # If the model type is equal to auxil
        elif self.args.model == "auxil":
            # Put x to the auxiliary convolutional network
            v = self.auxil(x)
            # MNIST: [Bag size, 1, 28, 28]
            recon_x = self.dec(z)
        # Otherwise
        else:
            # Decode z
            recon_x = self.dec(z)
        # Calculate RE
        RE_sum = (
            self._compute_re_for_MNIST(x, recon_x)
            if self.args.data == "MNIST"
            else self._compute_re_for_Colon(x, recon_x, nmix=self.args.num_components)
        )
        # Calculate KL
        KL_sum = self.args.beta * self._compute_kl(z, z_mean, z_logvar)
        # Calculate ELBO
        elbo = -(RE_sum + KL_sum)
        # If it is the labeled data
        if y is not None:
            # If the model type is equal to disent
            if self.args.model == "disent":
                # Classify
                y_prob, y_hat, _ = self.att(z_mil)
            # If the model type is equal to auxil
            elif self.args.model == "auxil":
                # Classify
                y_prob, y_hat, _ = self.att(torch.cat((z, v), dim=1))
            # Otherwise
            else:
                # Classify
                y_prob, y_hat, _ = self.att(z)
            # Calculate the classifier objective
            cls_loss = (self.args.alpha * self.att._compute_att_loss(y, y_prob)).repeat(
                elbo.shape[0]
            )
            # ELBO + the classification loss
            # First, with ELBO mean [used in the training]
            if train:
                elbo_mil_mean = (elbo + cls_loss).mean(dim=-1)
                # Return the (ELBO + MIL).mean, and the accuracy
                return elbo_mil_mean, self.att._compute_cls_error(y, y_hat)
            # Second, with ELBO sum [used in the validation]
            elbo_mil_sum = (elbo + cls_loss).sum(dim=-1)
            # Then, return plain ELBO, (ELBO + MIL).sum, and the accuracy
            return elbo.sum(dim=-1), elbo_mil_sum, self.att._compute_cls_error(y, y_hat)
        # If it is the unlabeled data
        else:
            # If it is the training session, return the ELBO.mean
            if train:
                return elbo.mean(dim=-1)
            # If it is not the training session, then return the ELBO.sum
            return elbo.sum(dim=-1)


class AuxiliaryConvNet(nn.Module):
    """
    Auxiliary Network to encode the dataset along with the encoder and,
    then produce the concat version with the latent variables to the MIL classifier.
    """

    def __init__(self, args):
        super(AuxiliaryConvNet, self).__init__()

        # The number of input channels
        # MNIST: 1
        self.input_channels = args.input_channels
        # Declare input dims for the model summary method
        self.recon_dims = args.recon_dims
        # The latent dimension size
        self.z_dim = args.enc_out
        # The hidden layer dimensions
        self.hidden_dims = args.hidden_dims

        # Build the auxiliary network
        self.auxil = nn.Sequential(
            nn.Conv2d(args.input_channels, self.hidden_dims[2], 7, 1, 0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[0], 5, 1, 0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.hidden_dims[0], self.z_dim, 3, 1, 0),
        )

    def forward(self, x):
        return self.auxil(x)
