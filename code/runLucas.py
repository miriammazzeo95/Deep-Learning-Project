# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:02:12 2020

@author: Miriam
"""
from typing import *
from torch.utils.data import DataLoader
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
# plt.figure(figsize=(12, 3))
# sns.distplot(p.sample())
# plt.title(r"$\mathcal{B}(\mathbf{y} \mid \mathbf{\theta})$")
# plt.show() 


#########################################################################################################################

# from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
from importDatasets import mUSA_train, mUSA_test,tfidf_politic,tfidf_UK, tfidf_matching, y

# # # Flatten the images into a vector
# flatten = lambda x: ToTensor()(x).view(28**2)

# # Define the train and test sets
# dset_train = MNIST("./", train=True,  transform=flatten, download=True)
# dset_test  = MNIST("./", train=False, transform=flatten)

# # The digit classes to use
# classes = [ 4, 9]
# #[3, 7]

# def stratified_sampler(labels):
#     """Sampler that only picks datapoints corresponding to the specified classes"""
#     (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
#     indices = torch.from_numpy(indices)
#     return SubsetRandomSampler(indices)

# batch_size = 64
# eval_batch_size = 100
# # The loaders perform the actual work
# train_loader = DataLoader(dset_train, batch_size=batch_size,
#                           sampler=stratified_sampler(dset_train.train_labels))
# test_loader  = DataLoader(dset_test, batch_size=eval_batch_size, 
#                           sampler=str

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

#DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           # batch_sampler=None, num_workers=0, collate_fn=None,
           # pin_memory=False, drop_last=False, timeout=0,
           # worker_init_fn=None, *, prefetch_factor=2,
           # persistent_workers=False)

 # create your datset from list of tensors
my_dataloader = DataLoader( TensorDataset(*convert_sparse_matrix_to_sparse_tensor(X_data), batch_size=batch_size) )

def batch_generator(X_data, batch_size, infinite=None ):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = (samples_per_epoch/batch_size)-1
    counter=0
    index = np.arange(np.shape(X_data)[0])
    if infinite == True:
        while 1:
            index_batch = index[batch_size*counter:batch_size*(counter+1)]
            #idx = np.random.permutation(len())
            X_batch = convert_sparse_matrix_to_sparse_tensor(X_data[index_batch,:])
            counter += 1
            yield X_batch
            if counter > number_of_batches:
                index = np.arange(np.shape(X_data)[0])
                np.random.shuffle(index)
                X_data[index, :]     
                counter= 0
    while counter < number_of_batches:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = convert_sparse_matrix_to_sparse_tensor(X_data[index_batch,:])
        counter += 1
        yield X_batch
        

batch_size = 256
eval_batch_size = 256
# The loaders perform the actual work
train_loader = batch_generator(mUSA_train, batch_size, infinite = True)

test_loader  = batch_generator(mUSA_test, eval_batch_size, infinite = True)
test_loader_politic  = batch_generator(tfidf_politic, eval_batch_size, infinite = True)
test_loader_UK  = batch_generator(tfidf_UK, eval_batch_size, infinite = True)

matches = convert_sparse_matrix_to_sparse_tensor(tfidf_matching)

###  power of 2 = 256, 512 ...

#plot a few tfidf examples
#f, axarr = plt.subplots(4, 16, figsize=(16, 4))

# Load a batch of images into memory
tfidf_m = next(iter(train_loader))

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
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.conv=nn.Sequential(
            
             nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size= 9, stride= 3, padding  = 4),
             nn.ReLU(),
            )
        self.encoder = nn.Sequential(
            
            nn.Linear(in_features=int((self.observation_features/3)+1), out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=64, out_features=2*latent_features) # <- note the 2*latent_features
        )
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
           
            nn.Linear(in_features=latent_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.observation_features)
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        
        x=x.reshape((x.size(0),1,x.size(1)))
        x=self.conv(x)
        x=x.reshape((x.size(0),x.size(2)))
        h_x = self.encoder(x)
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
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output
        return Bernoulli(logits=px_logits)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        #x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}


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
latent_features = 128
vae = VariationalAutoencoder(tfidf_m[0].shape, latent_features)
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
        make_vae_plots(vae, x, y, outputs, outputs_matches, training_data, validation_data)
        
        #make_vae_plots(vae, x, y, outputs, outputs_matches, training_data, validation_data,validation_data_politic,validation_data_UK)
    # print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
    # for key, tensor in diagnostics.items():
    #     print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")





make_vae_plots(vae, x, y, outputs, outputs_matches, training_data, validation_data)
loss, diagnostics, outputs = vi(vae, x)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")