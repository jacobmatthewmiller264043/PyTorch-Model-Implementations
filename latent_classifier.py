# import torch
# from torch import nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# An encoder/classifier pair which first encodes features into latent space
# and then classifies based upon the expected value of the encoding. 
# We use a 10 sample Monte Carlo estimate for the expected value of the latent representation
# Encoder parameters here are optimized simultaneous to the classifier

class Latent_classifier(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Latent_classifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4 , latent_dim * 8),
            nn.ReLU(),
            nn.Linear(latent_dim * 8 , latent_dim * 16),
            nn.ReLU(),
            nn.Linear(latent_dim * 16 , latent_dim * 8),
            nn.ReLU(),
            nn.Linear(latent_dim * 8 , latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4 , latent_dim * 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim *4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim *8),
            nn.ReLU(),
            nn.Linear(latent_dim*8, latent_dim *4),
            nn.ReLU(),
            nn.Linear(latent_dim *4 , latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim //2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, 1)
        )

        self.softplus = nn.Softplus()


    def encode(self, x, eps: float = 1e-8):

        # Get parameter vectors for latent Gaussian, ensure positive definite entries for covariance, embed entries in matrix
        # return distribution object, mean, log variance  

        z_params = self.encoder(x)
        mu, logvar = torch.chunk(z_params, 2, dim = -1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
    
    
        return [torch.distributions.MultivariateNormal(mu, scale_tril = scale_tril), mu, logvar]

    
    def reparam(self, dist):

        # Use built-in Pytorch reparameterization trick functionality to sample from distribution object  


        return dist.rsample()

    
    def forward(self, x):
        sample = self.reparam(self.encode(x)[0])
        for i in range(0,9):
            sample += self.reparam(self.encode(x)[0])
        latent = sample/10
    
        return self.classifier(latent)