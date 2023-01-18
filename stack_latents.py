import argparse
import numpy as np

from os.path import join, exists
from os import mkdir
from tqdm import tqdm

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE


BSIZE = 16
SEQ_LEN = 32
stack_num = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


stacked_obs = np.zeros(stack_num)  # 8 observations to be stacked

transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)

train_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, shuffle=False)


#vae_file = join(args.logdir, 'vae', 'best.tar')
#assert exists(vae_file), "No trained VAE in the logdir..."
#state = torch.load(vae_file)
#print("Loading VAE at epoch {} "
#      "with test error {}".format(
#          state['epoch'], state['precision']))


vae = VAE(3, LSIZE).to(device)
#vae.load_state_dict(state['state_dict'])

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        obs, next_obs = [
            f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                       mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs


# VAE was already trained independently.
# When we update controller(NN), concatenation of k(=8 for example) observations(latents) can be regarded as a state.

# What kind of controller(NN) would we use?
