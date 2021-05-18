# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:02:12 2020

@author: Miriam
"""
from typing import *
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
# %matplotlib nbagg
# %matplotli# inline
import seaborn as sns
import pandas as pd
import random
sns.set_style("whitegrid")

import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution
from scipy.sparse import coo_matrix
from plotting import make_vae_plots

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma*self.sample_epsilon() # <- your code
        #: let z ∼ p(z|x) = N (µ, σ2). In this case, a valid reparameterization is z = µ + σ eps, where eps is an auxiliary noise variable eps - N (0, 1).
   
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
       # d=Normal(loc=self.mu, scale=self.sigma)
        return  -((z - self.mu)** 2) / (2 * self.sigma**2) - self.sigma.log() -math.log(math.sqrt(2 * math.pi))  # <- your code
    
    
from torch.distributions import Bernoulli #<- your code

p = Bernoulli(logits=torch.zeros((1000,)))


#########################################################################################################################

# from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
   
from importDatasets import train_loader, test_loader , test_loader_politic  , test_loader_UK, embeding_matching,y,wordEmbedingModel


###  power of 2 = 256, 512 ...

#plot a few tfidf examples
#f, axarr = plt.subplots(4, 16, figsize=(16, 4))

# Load a batch of images into memory
batch = next(iter(train_loader))

# for i, ax in enumerate(axarr.flat):
#     ax.spy(tfidf_m[i])
#     ax.axis('off')
    
# plt.show()
            
#########################################################################################################################


class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int, embedding_size, rnn_type, hidden_size, num_layers=1, bidirectional=False) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # self.word_dropout_rate = word_dropout
        # self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.conv=nn.Sequential(
            
             nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size= 9, stride= 3, padding  = 4),
             nn.ReLU(),
            )
        
        self.encoder_rnn =  rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.hidden2rep = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.latent2hidden = nn.Linear(latent_features, hidden_size * self.hidden_factor)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), embedding_size)

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        _, h_x = self.encoder_rnn(x)
        
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            h_x = h_x.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            h_x = h_x.squeeze()
            
        h_x = self.hidden2rep(h_x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        
        h_x = self.latent2hidden(z)
        
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            h_x = h_x.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            h_x = h_x.unsqueeze(0)
            
        outputs,_ = self.decoder_rnn(x, h_z)
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output
        return Bernoulli(logits=px_logits)
        
        # b,s,_ = outputs.size()
        # # project outputs to vocab
        # px_logits = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=-1)
        # return px_logits.view(b, s, embedding_size)
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        # sample the prior 
        z = pz.rsample()
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        return {'px': px, 'pz': pz, 'z': z}

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        #x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x) 
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # REPARAMETERIZATION
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    



###########################################################################################################################


def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta
        
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        
        # compute the ELBO with and without the beta parameter: 
        # L = E_q [ log p(x|z) -  D_KL(q(z|x) | p(z))`
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl  # <- your code here
        beta_elbo = log_px - (self.beta*kl) # <- your code here
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs
        
from collections import defaultdict
# define the models, evaluator and optimizer

##########################################################################################################################


# VAE
vae = VariationalAutoencoder(input_shape=batch[0].shape, latent_features=64, embedding_size=300, rnn_type='rnn', hidden_size=2, num_layers=1, bidirectional=False)
# print(vae)

# Evaluator: Variational Inference
beta = 10
vi = VariationalInference(beta=beta)

# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# define dictionary to store the training curves
training_data = defaultdict(list)
validation_data = defaultdict(list)
validation_data_politic = defaultdict(list)
validation_data_UK = defaultdict(list)

epoch = 0

#########################################################################################################



num_epochs =15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# move the model to the device
vae = vae.to(device)
matches=matches.to(device)

# training..
while epoch < num_epochs:
    epoch+= 1
    print("epoch "+str(epoch))
    training_epoch_data = defaultdict(list)
    vae.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    i=0
    for x in train_loader:
        x = x.to(device)
        
        if(i*batch_size>mUSA_train.shape[0]):
            break
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # gather data for the current bach
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
        i+=1

    # gather data for the full epoch
    if(epoch>-1):
        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]

    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        vae.eval()
        
        # Just load a single batch from the test loader
        x = next(iter(test_loader))
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        
        # x_politic = next(iter(test_loader_politic))
        # x_politic= x_politic.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        #loss_politic, diagnostics_politic, outputs_politic = vi(vae, x_politic)
        
        # x_UK = next(iter(test_loader_UK))
        # x_UK = x_UK.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        #loss_UK, diagnostics_UK, outputs_UK = vi(vae, x_UK)
        # gather data for the validation step
        
        loss_matches, diagnostics_matches, outputs_matches = vi(vae, matches)
        
        # gather data for the validation step
        if(epoch>-1):
            for k, v in diagnostics.items():
                validation_data[k] += [v.mean().item()]
            # for k, v in diagnostics_politic.items():
            #     validation_data_politic[k] += [v.mean().item()]
            # for k, v in diagnostics_UK.items():
            #     validation_data_UK[k] += [v.mean().item()]
            
    ##Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
    if(epoch>-1):
        make_vae_plots(vae, x, y, outputs, outputs_matches, training_data, validation_data,validation_data_politic,validation_data_UK)
          
        #make_vae_plots(vae, x, y, outputs, outputs_matches, training_data, validation_data,validation_data_politic,validation_data_UK)
    # print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
    # for key, tensor in diagnostics.items():
    #     print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")






make_vae_plots(vae, x, y, outputs, outputs_matches, training_data, validation_data,validation_data_politic,validation_data_UK)
loss, diagnostics, outputs = vi(vae, x)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")
    
    
    
loss_matches, diagnostics_matches, outputs_matches = vi(vae, matches)
px=outputs_matches['px']
output=px.sample()
matchescpu=matches.cpu()
outputcpu=output.cpu()


#print(pz)
#print(qz)
for i in range(55) :
  print(wordEmbedingModel.similar_by_vector(matchescpu[i],1))
  print(wordEmbedingModel.similar_by_vector(outputcpu[i],1))
  print(i)