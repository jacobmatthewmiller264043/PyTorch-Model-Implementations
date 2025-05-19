# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset


# Create VAE class, subclassing nn.Module from Pytorch. 

class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        
        # Input tensor size B x input_dim
        # Output tensor size B x latent_dim * 2
        # Output is mean and diagonal covariance matrix entries for Gaussian in latent space ( R^{latent_dim} )


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


        # Input tensor size B x latent_dim
        # Output tensor size B x input_dim * 2
        # Output is mean and diagonal precision matrix (we use precision matrix for
        # numerical stability) entries for Gaussian in input space ( R^{input_dim} )
        # This model allows for non-constant entries in the diagonal precision matrix for input space

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4 , input_dim * 8),
            nn.ReLU(),
            nn.Linear(input_dim * 8 , input_dim * 16),
            nn.ReLU(),
            nn.Linear(input_dim * 16 , input_dim * 8),
            nn.ReLU(),
            nn.Linear(input_dim * 8 , input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4 , input_dim * 2),
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

        # Use built-in Pytorch rsample() reparameterization trick functionality to sample from distribution object  
        # See "Reparameterization Trick" repository for theoretical discussion of this well-known but rarely
        # technically explained method

        return dist.rsample()


    

    def decode(self, z):

        # Get mean and log precision for input space from latent representation

        x_params = self.decoder(z)
        mu, logprec = torch.chunk(x_params, 2, dim = -1)
        

        return [mu, logprec]

    
    

    def forward(self, x):

        # Get latent space parameter vectors, sample from latent space using these vectors, get reconstructed input parameter
        # vectors from latent space sample, return all vectors

        z_dist, z_mu, z_logvar = self.encode(x)
        z = self.reparam(z_dist)
        x_mu, x_logprec = self.decode(z)


        return [x_mu, x_logprec, z_mu, z_logvar] 



    def loss_function(self, x, params, prior_scale,
                                        recon_penalty,
                                        kld_penalty): 
                                        
                                        # prior_scale is float to constrain or generalize the Gaussian prior over latent space
                                        # recon_penalty constrains the recon_loss
                                        # kld_penalty constrains the kld loss

        # Minimize the negative elbo (reconstruction score + KL Divergence between Gaussion prior and posterior
        # over latent sapce)

        x_mu, x_logprec, z_mu, z_logvar = params

        l = x - x_mu
        l2 = l**2

        recon_loss = torch.mean(0.5 * torch.sum(x_logprec.exp() * l2 - x_logprec  , dim = 1), dim = 0)
        scale = z_mu ** 2 + z_logvar.exp()
        scale = prior_scale * scale
        kld = torch.sum(1 + z_logvar - scale, dim = 1)
        kld =  - np.log(prior_scale) + kld
        kld = torch.mean(- 0.5 * kld, dim = 0)


        return recon_penalty * recon_loss +  kld_penalty * kld   

     
    def test_reconstruction_loss(self, x):

        # Get reconstruction scores for input tensor. Input is tensor of size N x input_dim where
        # N is the size of either train or test set. Output is tensor of size N. Average over 10
        # reconstruction samples to obtain stable estimate

       x_mu, x_logprec, z_mu, z_logvar = self.forward(x)

       l = x - x_mu
       l2 = l**2

       recon_loss = 0.5 * torch.sum(x_logprec.exp() * l2 - x_logprec  , dim = 1)

       for i in range(0,9):
              x_mu, x_logprec, z_mu, z_logvar = self.forward(x)

              l = x - x_mu
              l2 = l**2

              recon_loss_2 = 0.5 * torch.sum(x_logprec.exp() * l2 - x_logprec  , dim = 1)
              recon_loss = recon_loss + recon_loss_2

       
       return recon_loss/10
    

    def reconstruct(x, model):

        # Reconstruct a given input tensor according to the generative model and additionally return its associated
        # reconstruction score tensor
        
        z_dist, z_mu, z_logvar = model.encode(x)
        z = z_dist.sample()
        x_mu, x_logprec = model.decode(z)

        x_prec = torch.diag_embed(x_logprec.exp())
        x = torch.distributions.MultivariateNormal(x_mu, precision_matrix= x_prec).sample()

        l = x - x_mu
        l2 = l**2

        recon_loss = 0.5 * torch.sum(x_logprec.exp() * l2 - x_logprec, dim = 1)


        return [x, recon_loss]