import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAELoss(nn.Module):
    def __init__(self, beta=1.0, recon_loss_type='mse'):
        """
        Initialize the BetaVAE Loss module.
        
        Args:
            beta (float): Weight of the KL divergence term. Default is 1.0.
            recon_loss_type (str): Type of reconstruction loss ('mse' or 'bce'). Default is 'mse'.
        """
        super(BetaVAELoss, self).__init__()
        self.beta = beta
        self.recon_loss_type = recon_loss_type
    
    def forward(self, outputs, x):
        """
        Compute the BetaVAE loss.
        
        Args:
            x: Original input.
            x_recon: Reconstructed input from the decoder.
            mu: Mean of the latent variable distribution.
            logvar: Log variance of the latent variable distribution.
        
        Returns:
            loss: Total loss (reconstruction + KL divergence).
            recon_loss: Reconstruction loss.
            kl_loss: KL divergence loss.
        """
        
        x_recon, mu, logvar = outputs
        # Reconstruction loss
        if self.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        elif self.recon_loss_type == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.recon_loss_type}")

        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting on KL divergence
        loss = recon_loss + self.beta * kl_loss
        
        return loss