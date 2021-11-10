# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def calc_output_size_from_Conv2d(
    height: int, width: int, kernels: tuple, strides: tuple, paddings: tuple
) -> tuple:
    # Upsample
    height, width = height * 10, width * 10
    # Loop over the kernel size
    for kernel, stride, padding in zip(kernels, strides, paddings):
        # Calculate the height
        height = int((height + 2 * padding - (kernel - 1) - 1) / stride + 1)
        # Calculate the width
        width = int((width + 2 * padding - (kernel - 1) - 1) / stride + 1)
    # Return the image dimension in the latent space
    return height, width


class Attention_MNIST(nn.Module):
    """
    A neural network-based permutation-invariant aggregation
    operator that corresponds to the attention mechanism.

    @article{ITW:2018,
      title={Attention-based Deep Multiple Instance Learning},
      author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
      journal={arXiv preprint arXiv:1802.04712},
      year={2018}
    }

    Adapted to the SSMILVAE architecture for the MNIST Dataset.
    """

    def __init__(self, args):
        super(Attention_MNIST, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.z_dim = args.dec_inp * 2 if args.model == "auxil" else args.dec_inp
        # The output size from the feature_extractor_part1
        self.fep2_height, self.fep2_width = calc_output_size_from_Conv2d(
            *args.latent_img_size,
            kernels=(3, 2, 3, 2),
            strides=(1, 2, 1, 2),
            paddings=(2, 1, 2, 1)
        )

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(self.z_dim, 20, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(20, 50, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Flatten(),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(450, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        # [#InstanceInBag, C, W, H] -> [#InstanceInBag, 50, 4, 4]
        H = self.feature_extractor_part1(x)
        # [#InstanceInBag, 800] -> [#InstanceInBag, 500]
        H = self.feature_extractor_part2(H)  # NxL
        # [#InstanceInBag, 500] -> [#InstanceInBag, 1]
        A = self.attention(H)  # NxK
        # [#InstanceInBag, 1] -> [1, #InstanceInBag]
        A = torch.transpose(A, 1, 0)  # KxN
        # [1, #InstanceInBag] -> [1, #InstanceInBag]
        A = F.softmax(A, dim=1)  # softmax over N
        # [1, #InstanceInBag] -> [1, 500]
        M = torch.mm(A, H)  # KxL
        # [1, 500] -> [B, y_dim[one_hot]] // Former: [1, 1]
        Y_prob = self.classifier(M)
        # [1, 1] -> [1, 1]
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    @staticmethod
    def _compute_att_loss(y, y_prob):
        y = y.float()
        y_prob = torch.clamp(y_prob, min=1e-5, max=1.0 - 1e-5)
        return (
            -1.0 * (y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob))
        ).squeeze(0)

    @staticmethod
    def _compute_cls_error(y, y_hat):
        y = y.float()
        error = y_hat.eq(y).cpu().float().mean().item()

        return error


class Attention_Colon(nn.Module):
    """
    A neural network-based permutation-invariant aggregation
    operator that corresponds to the attention mechanism.

    @article{ITW:2018,
      title={Attention-based Deep Multiple Instance Learning},
      author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
      journal={arXiv preprint arXiv:1802.04712},
      year={2018}
    }

    Adapted to the SSMILVAE architecture for the Colon Cancer Dataset.
    """

    def __init__(self, args):
        super(Attention_Colon, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.recon_dims = (args.dec_inp, *args.latent_img_size)
        self.z_dim = args.dec_inp * 2 if args.model == "auxil" else args.dec_inp
        # The output size from the feature_extractor_part1
        self.fep2_height, self.fep2_width = calc_output_size_from_Conv2d(
            *args.latent_img_size,
            kernels=(4, 2, 3, 2),
            strides=(1, 2, 1, 2),
            paddings=(0, 0, 0, 0)
        )

        self.feature_extractor_part1 = nn.Sequential(
            nn.Upsample(scale_factor=10),
            nn.Conv2d(self.z_dim, 36, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(36, 48, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(48 * self.fep2_height * self.fep2_width, self.L),
            nn.ReLU(),
        )

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        # [#InstanceInBag, args.dec_inp, *args.latent_img_size] -> [#InstanceInBag, ?]
        H = self.feature_extractor_part1(x)
        # [#InstanceInBag, 800] -> [#InstanceInBag, 500]
        H = self.feature_extractor_part2(H)  # NxL
        # 512 -> 512
        H = self.feature_extractor_part3(H)
        # [#InstanceInBag, 500] -> [#InstanceInBag, 1]
        A = self.attention(H)  # NxK
        # [#InstanceInBag, 1] -> [1, #InstanceInBag]
        A = torch.transpose(A, 1, 0)  # KxN
        # [1, #InstanceInBag] -> [1, #InstanceInBag]
        A = F.softmax(A, dim=1)  # softmax over N
        # [1, #InstanceInBag] -> [1, 500]
        M = torch.mm(A, H)  # KxL
        # [1, 500] -> [B, y_dim[one_hot]] // Former: [1, 1]
        Y_prob = self.classifier(M)
        # [1, 1] -> [1, 1]
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def modelsummary(self):
        return summary(self, self.recon_dims)

    @staticmethod
    def _compute_att_loss(y, y_prob):
        y = y.float()
        y_prob = torch.clamp(y_prob, min=1e-5, max=1.0 - 1e-5)
        return (
            -1.0 * (y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob))
        ).squeeze(0)

    @staticmethod
    def _compute_cls_error(y, y_hat):
        y = y.float()
        error = y_hat.eq(y).cpu().float().mean().item()

        return error


class AttentionRes_Colon(nn.Module):
    def __init__(self, args):
        super(AttentionRes_Colon, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.inp_channels = args.dec_inp

        self.feature_extractor_part1 = nn.Sequential(
            nn.ConvTranspose2d(int(self.inp_channels / 16), 16, kernel_size=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 1, kernel_size=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(1 * 28 * 28, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, x):
        # x = x.squeeze(0)
        x = x.view(-1, 2, 4, 4)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 1 * 28 * 28)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    @staticmethod
    def _compute_att_loss(y, y_prob):
        y = y.float()
        y_prob = torch.clamp(y_prob, min=1e-5, max=1.0 - 1e-5)
        return (
            -1.0 * (y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob))
        ).squeeze(0)

    @staticmethod
    def _compute_cls_error(y, y_hat):
        y = y.float()
        error = y_hat.eq(y).cpu().float().mean().item()

        return error


class AttentionBase_MNIST(nn.Module):
    """
    A neural network-based permutation-invariant aggregation
    operator that corresponds to the attention mechanism.

    @article{ITW:2018,
      title={Attention-based Deep Multiple Instance Learning},
      author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
      journal={arXiv preprint arXiv:1802.04712},
      year={2018}
    }

    The original architecture for the MNIST Dataset from the paper.
    """

    def __init__(self):
        super(AttentionBase_MNIST, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.recon_dims = (1, 28, 28)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

        # Trace
        print("Constructed the Attention Base model based on the MNIST dataset.")

    def forward(self, x):
        # [B, #InstanceInBag, C, W, H] -> [#InstanceInBag, C, W, H]
        x = x.squeeze(0)
        # [#InstanceInBag, C, W, H] -> [#InstanceInBag, 50, 4, 4]
        H = self.feature_extractor_part1(x)
        # [#InstanceInBag, 50, 4, 4] -> [#InstanceInBag, 800]
        H = H.view(-1, 50 * 4 * 4)  #: -> out size: [#InstanceInBag, 800]
        # [#InstanceInBag, 800] -> [#InstanceInBag, 500]
        H = self.feature_extractor_part2(H)  # NxL
        # [#InstanceInBag, 500] -> [#InstanceInBag, 1]
        A = self.attention(H)  # NxK
        # [#InstanceInBag, 1] -> [1, #InstanceInBag]
        A = torch.transpose(A, 1, 0)  # KxN
        # [1, #InstanceInBag] -> [1, #InstanceInBag]
        A = F.softmax(A, dim=1)  # softmax over N
        # [1, #InstanceInBag] -> [1, 500]
        M = torch.mm(A, H)  # KxL
        # [1, 500] -> [1, 1]
        Y_prob = self.classifier(M)
        # [1, 1] -> [1, 1]
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def modelsummary(self):
        return summary(self, self.recon_dims)

    # AUXILIARY METHODS
    def calculate_classification_error(self, x, y):
        y = y.float()
        _, y_hat, _ = self.forward(x)
        error = y_hat.eq(y).cpu().float()
        return error, y_hat

    def calculate_objective(self, x, y):
        y = y.float()
        y_prob, _, A = self.forward(x)
        y_prob = torch.clamp(y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood, A


class AttentionBase_Colon(nn.Module):
    """
    A neural network-based permutation-invariant aggregation
    operator that corresponds to the attention mechanism.

    @article{ITW:2018,
      title={Attention-based Deep Multiple Instance Learning},
      author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
      journal={arXiv preprint arXiv:1802.04712},
      year={2018}
    }

    The original architecture for the Colon Cancer Dataset from the paper.
    """

    def __init__(self):
        super(AttentionBase_Colon, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.recon_dims = (3, 27, 27)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(36, 48, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(1200, self.L),
            nn.ReLU(),
        )

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

        self.dropout = nn.Dropout(p=0.25)

        # Trace
        print("Constructed the Attention Base model based on the Colon dataset.")

    def forward(self, x):
        # [#InstanceInBag, C, W, H] -> [#InstanceInBag, 50, 4, 4]
        H = self.feature_extractor_part1(x)
        # 1200 -> 512
        H = self.feature_extractor_part2(H)
        # Dropout
        H = self.dropout(H)
        # 512 -> 512
        H = self.feature_extractor_part3(H)
        # Dropout
        H = self.dropout(H)
        # [#InstanceInBag, 500] -> [#InstanceInBag, 1]
        A = self.attention(H)  # NxK
        # [#InstanceInBag, 1] -> [1, #InstanceInBag]
        A = torch.transpose(A, 1, 0)  # KxN
        # [1, #InstanceInBag] -> [1, #InstanceInBag]
        A = F.softmax(A, dim=1)  # softmax over N
        # [1, #InstanceInBag] -> [1, 500]
        M = torch.mm(A, H)  # KxL
        # [1, 500] -> [B, y_dim[one_hot]] // Former: [1, 1]
        Y_prob = self.classifier(M)
        # [1, 1] -> [1, 1]
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def modelsummary(self):
        return summary(self, self.recon_dims)

    # AUXILIARY METHODS
    def calculate_classification_error(self, x, y):
        y = y.float()
        _, y_hat, _ = self.forward(x)
        error = y_hat.eq(y).cpu().float()
        return error, y_hat

    def calculate_objective(self, x, y):
        y = y.float()
        y_prob, _, A = self.forward(x)
        y_prob = torch.clamp(y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob)
        )  # negative log bernoulli

        return neg_log_likelihood, A
