import torch
import torch.nn as nn
import numpy as np

class QuantileLoss(nn.Module):
    def __init__(self, quantiles, device):
        super(QuantileLoss, self).__init__()
        """
        Loss function (Class) for quantile regression. The loss is the sum of the quantile losses for each quantile.

        Input arguments:
        quantiles: list of quantiles for which the loss is calculated
        
        device: Str device on which the loss is calculated (cpu or cuda)
        """

        self.quantiles = quantiles
        self.device = device

    def forward(self, y_pred, y_true):
        """
        Forward pass of the loss function. The loss is the sum of the quantile losses for each quantile.

        Input arguments:
        y_pred: torch.Tensor (batch_size, sequence_length, len(quantiles))

        y_true: torch.Tensor (batch_size, sequence_length)

        Output:
        loss: torch.Tensor
        """

        losses = torch.empty_like(y_pred, device=self.device)
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[:, :, i]
            losses[:, :, i] = torch.max(q * error, (q - 1) * error)
        loss = torch.mean(losses)
        return loss
    
class GMMLoss(nn.Module):
    def __init__(self, variance_regularization=0, weights_regularization=0, return_mixture=False):
        super(GMMLoss, self).__init__()
        """
        Loss function (Class) for Gaussian Mixture Models. The loss is the negative log likelihood of the GMM.

        Input arguments:
        variance_regularization: float: Regularization term for the variance of the GMM components
        weights_regularization: float: Regularization term for the weights of the GMM components

        return_mixture: bool: If True, the function returns the negative log likelihood, the variance loss, the weights loss and the GMM object
        """
        self.eps = 1e-15
        self.variance_regularization = variance_regularization
        self.weights_regularization = weights_regularization
        self.return_mixture=return_mixture

    def forward(self, outputs, y_true):
        """
        Forward pass of the loss function. The loss is the negative log likelihood of the GMM.

        Input arguments:
        outputs: tuple: Tuple containing the weights, means and sigmas of the GMM components
        y_true: torch.Tensor (batch_size, sequence_length)

        Returns:
        negative_log_likelihood: torch.Tensor: Negative log likelihood of the GMM

        If return_mixture is True in the class:

        variance_loss: torch.Tensor: Regularization term for the variance of the GMM components

        weights_loss: torch.Tensor: Regularization term for the weights of the GMM components

        mixed: torch.distributions.MixtureSameFamily: GMM object

        """

        weights, means, sigmas = outputs
        mixture = torch.distributions.Categorical(weights)
        components = torch.distributions.Normal(means, sigmas)
        mixed = torch.distributions.MixtureSameFamily(mixture, components)
        negative_log_likelihood = -torch.mean(mixed.log_prob(y_true))
        variance_loss = self.variance_regularization * torch.mean(sigmas**(-2))
        weights_loss = self.weights_regularization * torch.sqrt(torch.sum(weights**(2)))

        if self.return_mixture:
            return negative_log_likelihood, variance_loss, weights_loss, mixed
        return negative_log_likelihood + variance_loss + weights_loss

