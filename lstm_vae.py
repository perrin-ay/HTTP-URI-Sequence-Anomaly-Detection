# -*- coding: utf-8 -*-
"""LSTM VAE"""

import datasets
from datasets import load_dataset,Value, Sequence, Features
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoModelForMaskedLM
from transformers import default_data_collator
from transformers import DataCollatorForLanguageModeling
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoConfig, AutoModelForMaskedLM, BertForMaskedLM
from transformers import pipeline
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import time,sys,os
import math
import collections
import itertools
import random

from ANNhelper import set_all_seeds,log_setup,configuration, ANN,VAE, Net, custReshape, Trim, CNN1D, RNN_classification
from ANNhelper import BatchSamplerSimilarLength, hiddenBidirectional, hiddenUnidirectional, squeezeOnes
from ANNhelper import BidirectionextractHiddenfinal, UnidirectionextractHiddenfinal, permuteTensor, globalMaxpool, MultiNet
from ANNhelper import LSTMhc,UnidirectionextractHiddenCell,UnidirectionalextractOutput,Linearhc,RNNhc,unsqueezeOnes
from ANNhelper import concatTwotensors, concatThreetensors, UnidirectionextractHidden, decoder_cho, hcHiddenonlyBidirectional
from ANNhelper import hcBidirectional, BidirectionextractHCfinal, Bidirectionfullprocess, activationhc, Linearhchiddencell
from ANNhelper import Attention, decoderGRU_attn_bahdanau, decoderGRU_cho, Seq2SeqAttnGRU, decoder_attn_bahdanau
from ANNhelper import GRULinearhchidden, activationh, standin,UnpackpackedOutputHidden, GRUBidirectionfullprocess
from ANNhelper import SamplerSimilarLengthHFDataset, Seq2SeqAttnGRUPacked, Seq2SeqAttnLSTMPacked, Seq2SeqLSTMPacked
from AttentionHelper import Seq2SeqSelfAttn, DecoderLayer, DecoderSelfAttn, PositionwiseFeedforwardLayer
from AttentionHelper import MultiHeadAttentionLayer, EncoderLayer, EncoderSelfAttn, TransformerFTseqclassifier

from ANNdebug import CNNparamaterStats,FCNparameterStats, hook_prnt_activations, hook_prnt_activation_norms
from ANNdebug import hook_prnt_inputs, hook_prnt_weights_grad_stats, callback_prnt_allweights_stats
from ANNdebug import callback_prnt_allgrads_stats, callback_prnt_weights_stats
from ANNdebug import hook_prnt_inputs_stats, hook_prnt_activations_stats, hook_prnt_inputs_norms, hook_return_activations, hook_return_inputs
from ANNdebug import activation as hookactivations
from ANNdebug import inputs as hookinputs

"""**Load training dataset**

- These contain embeddings of each HTTP request URI by passing the URI through previously trained model and then mean pooling the hidden states.
"""

ds_concat = load_from_disk("ds_concat_classif_meanpooled")
ds_good = ds_concat.filter(lambda example: example["label"] == 0)

train_test = ds_good.train_test_split(test_size=0.2, seed =2018)


train = train_test['train']
test = train_test['test']

ds = datasets.DatasetDict({
    'train': train,
    'test': test})

ds.set_format("torch",columns=["hidden_state"], output_all_columns= True )
ds = ds.remove_columns("label")

"""**Setup batch sampler, collate function and data loader**"""

batch_size = 32
shuffle = True
drop_last =False

def collateseq2seqHFDatasetPack(batch):

    encoder_list = []
    x_lens = []
    label_list = []
    encoder_list = [example["hidden_state"] for example in batch]
    batchedsrc = pad_sequence(encoder_list, batch_first=True, padding_value=0)
    collated = {"hidden_state" : batchedsrc}
    return collated

train_loader = DataLoader(ds,
                          batch_sampler=SamplerSimilarLengthHFDataset(dataset = ds,
                                                                  batch_size=batch_size, shuffle=shuffle, drop_last= drop_last,
                                                                      keyname= 'hidden_state'), collate_fn=collateseq2seqHFDatasetPack)

"""**LSTM encoder and decoder for the VAE**"""

class VAEEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_layers=2):
        super(VAEEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x) # hidden ([numlayers, batchs, hidden_size])
        return (hidden, cell)


class VAEDecoder(nn.Module):

    def __init__(self, input_size=768, hidden_size=512, output_size=768, num_layers=2):
        super(VAEDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x ([64, 14, 16]) (batch, seqlen, latentz)
        output, (hidden, cell) = self.lstm(x, hidden)# hidden torch.Size([2, 64, 1024])
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class LSTMVAE(nn.Module):

    def __init__(self, enc, dec, hidden_size, latent_size, device=None):

        super(LSTMVAE, self).__init__()

        self.device = device
        self.encoder = enc
        self.decoder = dec
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.z_mean = nn.Linear(self.hidden_size, latent_size)
        self.z_log_var = nn.Linear(self.hidden_size, latent_size)

        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)

    def reparameterization(self, mean, exponentvar):
        epsilon = torch.randn_like(exponentvar).to(self.device)
        z = mean + exponentvar*epsilon
        return z

    def forward(self, x):

        batch_size, seq_len, feature_dim = x.shape

        hc = self.encoder(x)


        x =hc[0][-1, :, :].to(self.device) # [batchsize, hidden_size] [64, 512]

        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterization(z_mean, torch.exp(0.5 * z_log_var))
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)

        reconstruct_output, hidden = self.decoder(z, hc)

        x_hat = reconstruct_output # ([64, 14, 768])
        return z, z_mean, z_log_var,x_hat

"""**LSTM VAE**

"""

hidden_size =  1024
latent_size =  16
encoder = VAEEncoder(input_size=768, hidden_size=hidden_size, num_layers=2)
decoder = VAEDecoder(input_size=latent_size, hidden_size=hidden_size, output_size=768, num_layers=2)
vae  = Net()
vae.setupCuda()
device = vae.device
vae.net = LSTMVAE(encoder, decoder, hidden_size, latent_size, device=device)
vae.net.to(device)
vae.train_loader =  train_loader

"""**Training the VAE network**"""

optimizer = AdamW(vae.net.parameters(), lr=0.0005)
epochs = 150

vae.configureTraining(epochs=epochs, lossfun=nn.MSELoss(reduction='mean'), prntsummary = False, gpu= True)
vae.optimizer = optimizer

result = vae.trainVAEHFds(batchlogs=False,testset=False, lossreduction = True, Xkey ="hidden_state")

savedmodel ={}
savedmodel['net'] = vae.net.state_dict()
savedmodel['opt'] = optimizer.state_dict()
torch.save(savedmodel,"Anomaly_vae_save3")



