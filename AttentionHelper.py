import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



import spacy
import tqdm
import evaluate
import datasets


from torch.utils.data import Sampler, Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, Dataset



from ANNdebug import CNNparamaterStats,FCNparameterStats, hook_prnt_activations, hook_prnt_activation_norms
from ANNdebug import hook_prnt_inputs, hook_prnt_weights_grad_stats, callback_prnt_allweights_stats, callback_prnt_allgrads_stats
from ANNdebug import callback_prnt_weights_stats, hook_prnt_inputs_stats, hook_prnt_activations_stats, hook_prnt_inputs_norms, hook_return_activations, hook_return_inputs

import seaborn as sns
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
import plotly.express as px
import plotly.graph_objects as go


import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import PredictionError, ResidualsPlot

import copy
import random
import time
import sys
import os
import datetime
import logging
logging.raiseExceptions = False
import logging.handlers
from packaging import version

import collections
import unicodedata
import unidecode
import string
import re


def make_src_mask_transformerencoder(src, embed_dims=False):
    """
    Here assuming padding is always a 0
    src = [batch size, src len] or with embed dims
    src = [batch size, src len, embed_dims]
    """

    src_mask = (src == 0) # since it needs only padded indexes as True. this is opposite of implemented from scratch
    #src_mask = [batch size,src len]
    #src_mask = [batch size, src len,embed_dims]
    
    if embed_dims:
        src_mask = src_mask[:,:,0]
        
    return src_mask

def make_src_mask_scratch(src, embed_dims=False):
    """
    Here assuming padding is always a 0
    src = [batch size, src len] or with embed dims
    src = [batch size, src len, embed_dims]
    """
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    #src_mask = [batch size, 1, 1, src len]
    #src_mask = [batch size, 1, 1, src len,embed_dims]
    
    if embed_dims:
        src_mask = src_mask[:,:,:,:,0]
        
    return src_mask

class TransformerEncoderclassifier(nn.Module):

    def __init__(self , transformermodel, device, model_dims = 768, num_labels= 1):
        
        """
        num_labels gets the last out_features of linear layer, for binary choose 1, for multiclass choose 3 or more
        
        """
        super().__init__()
        self.device = device
        self.net = transformermodel
        self.num_labels = num_labels
        self.model_dims = model_dims
        self.linear1 = nn.Linear(self.model_dims,512)
        self.linear2 = nn.Linear(512,self.num_labels)
    
    def src_mask(self, src, embed_dims=True):
    
        """
        Here assuming padding is always a 0
        src = [batch size, src len] or with embed dims
        src = [batch size, src len, embed_dims]
        """

        src_mask = (src == 0).to(self.device) # since it needs only padded indexes as True. this is opposite of implemented from scratch or mask in pooling
        #src_mask = [batch size,src len]
        #src_mask = [batch size, src len,embed_dims]

        if embed_dims:
            src_mask = src_mask[:,:,0]

        return src_mask
    
    def src_mean_mask(self, src, embed_dims=True):
        
        """
        Here assuming padding is always a 0
        """
        
    #src = [batch size, src len]
     #src = [batch size, src len, embed_dims]
        
        src_mask = (src != 0).to(self.device)

        #src_mask = [batch size, 1, 1, src len]
        #src_mask = [batch size, 1, 1, src len,embed_dims]
        if embed_dims:
            src_mask = src_mask[:,:,0]

        return src_mask
    
    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        
    def forward(self,src):
        
        src_mask = self.src_mask(src,embed_dims=True)
        mean_mask = self.src_mean_mask(src,embed_dims=True)
        
        x = self.net(src, src_key_padding_mask = src_mask) # (batchs, seqln, hidden_dims) ( 2, 11, 768)
        x = self.mean_pooling(x,mean_mask) # (batchs,  hidden_dims) ( 2, 768)
        
        x = self.linear1(x)
        x = nn.Dropout(p=0.1)(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x= x.squeeze() # for binary class as x is [batchsize, 1] and labels is [batchsize]
        return x


class TransformerFTseqclassifier(nn.Module):
    """
     Due to selfattn , cls ( the 0th element) contains info about other tokens in the sequence. So we just use cls token vector 
     as a represenation of the entire sequence and feed to classifier head.  
     
    """
    def __init__(self , transformermodel, device, num_labels= 1):
        
        """
        num_labels gets the last out_features of linear layer, for binary choose 1, for multiclass choose 3 or more
        
        """
        super().__init__()
        self.device = device
        self.net = transformermodel
        self.num_labels = num_labels
        self.linear1 = nn.Linear(768,512)
        self.linear2 = nn.Linear(512,self.num_labels)
        
        
    def forward(self,x,attn_mask):
        
        last_hidden_state = self.net(x,attention_mask = attn_mask).last_hidden_state
        x =  last_hidden_state[:,0]
        
        x = self.linear1(x)
        x = nn.Dropout(p=0.2)(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x= x.squeeze() # for binary class as x is [batchsize, 1] and labels is [batchsize]
        return x
        
        
class TransformerEncoder(nn.Module):
    """
    in hierarchical encoders, the dims of the first encoder are the input and hid dims of the second encoder
    so if first encoder was a distilbert, the input_dim=hid_dim of second encoder = 768
    """
    
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()

        self.device = device
        
        ### NO TOKEN EMBEDDINGS CAN BE USED WITH INPUT AS ENCODED SRC ###
########### BUT alternatively could add linear layer if wanted###############
    
        self.position_embedding = nn.Embedding(max_length, hid_dim) # positional vocabulary size is statically 100 here
        # this means max size of input sentence is 100 tokens
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,n_heads,pf_dim,dropout,device) for _ in range(n_layers)])
        # see if you wanna make this deepcopy instead.
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask is only used when computing self attention
        #src_mask = [batch size, 1, 1, src len] 
        # This is simply 1 where the token is not a <pad> and 0 where it a pad token. 
        #It is then unsqueezed so it can be correctly broadcast when applying the mask to the energy, which of shape 
        #[batch size, n heads, seq len, seq len].
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) # this is supposed to tell us the position of
        # the sequence tokens
        
        # if for example seq len were 5 and batchsize was 2, this would produce
        # pos = tensor([[0, 1, 2, 3, 4],
        #               [0, 1, 2, 3, 4]])
        
        #pos = [batch size, src len]

#        src = self.dropout((self.token_embedding(src) * self.scale) + self.position_embedding(pos))       
        src = self.dropout((src * self.scale) + self.position_embedding(pos)) # element wise summation
        
        #src = [batch size, src len, hid dim]
        
       #### this is where the pytorch module for transformerencoderlayer actually starts#####
    
        for layer in self.layers: # paper had 6 layers
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src
    

class EncoderSelfAttn(nn.Module):
    
    def __init__(self,input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()

        self.device = device
        self.token_embedding = nn.Embedding(input_dim, hid_dim)
        self.position_embedding = nn.Embedding(max_length, hid_dim) # positional vocabulary size is statically 100 here
        # this means max size of input sentence is 100 tokens
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,n_heads,pf_dim,dropout,device) for _ in range(n_layers)])
        # see if you wanna make this deepcopy instead.
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask is only used when computing self attention
        #src_mask = [batch size, 1, 1, src len] # This is simply 1 where the token is not a <pad> and 0 where it is. It is then unsqueezed so it can be correctly broadcast when applying the mask to the energy, which of shape [batch size, n heads, seq len, seq len].
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) # this is supposed to tell us the position of
        # the sequence tokens
        
        # if for example seq len were 5 and batchsize was 2, this would produce
        # pos = tensor([[0, 1, 2, 3, 4],
        #               [0, 1, 2, 3, 4]])
        
        #pos = [batch size, src len]
                
        src = self.dropout((self.token_embedding(src) * self.scale) + self.position_embedding(pos)) # element wise summation
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src
    
    
class EncoderLayer(nn.Module):
    
    def __init__(self, hid_dim, n_heads, pf_dim,  dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        
        #src_mask = [batch size, 1, 1, src len] 
        ## the encoder src mask is only used to identify where in src there is padding ,ask is false wher padding and true where not
        ## the src mask is only used in final attention weights 
        ## its dims are [batch size, 1, 1, src len], because we use it as fill mask for unnomalized attention weights which are of dims
        # [batch size, n heads, query len, key len] # energy is unnomalized attention
        ## the 1, 1 in src mask are singleton dimensions and its shape will be broadcasted to the shape of energy. the key len for energy 
        #is the padded seqlen 
        # fillmask Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)  # self.attention is an object of multhead attention class
        
##### below we are only implmenting norm later rule, torch has option for norm first as well which is false by default########        
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src)) # the src + dropou(_src) is the residual connection
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, n_heads, dropout, device):
        
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None): # here the query , key , value are the same (src, src, src)
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        
        # hidden dims is split into N heads, which creates N heads with reduced dimensionality compared to original Q, K ,V 
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # K.permute(0, 1, 3, 2) [batch size, n heads, head dim, key len]
        
        #energy = [batch size, n heads, query len, key len] # energy is unnomalized attention
        
        if mask is not None: # this is finally where the mask is used - on energy, 
            #which is made very small such that softmax would make it 0
            # Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.
            energy = energy.masked_fill(mask == 0, -1e10) ## NOTE here we are ignoring 0/false in attention. In pytorch doing opposite
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V) # final output is simply weighted V, weighted with learned attention weights
        
        #x = [batch size, n heads, query len, head dim]
        
        # below reorganize dims and recombine the n heads to get hidden dims back
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim] so back to same shape as of embeddings
        
        x = self.fc_o(x) # pass through a final FCN layer
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) # addition dropout to mimic pytorch transformerencoder
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        ############### Additional dropout from pytorch implementation #######
        x = self.dropout2(x)
        
        return x
    
class DecoderSelfAttn(nn.Module):
    
    def __init__(self, output_dim,hid_dim,n_layers,n_heads,pf_dim,dropout,device,max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.token_embedding = nn.Embedding(output_dim, hid_dim) # hid_dims seems to be the embedding dims
        self.position_embedding = nn.Embedding(max_length, hid_dim) # max_length is the vocab size of the traget seq len (100)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,n_heads,pf_dim,dropout, device)  for _ in range(n_layers)]) # again repeated layers of the decoder, usually same number as in the encoder
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device) # scaling embedding for reasons discussed in encoder
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.token_embedding(trg) * self.scale) + self.position_embedding(pos))   
         #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention       
        
class DecoderLayer(nn.Module):
    
    def __init__(self, hid_dim,n_heads,pf_dim,dropout,device):
     
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class Seq2SeqSelfAttn(nn.Module):
    
    def __init__(self,encoder,decoder,src_pad_idx,trg_pad_idx,device ):                
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention