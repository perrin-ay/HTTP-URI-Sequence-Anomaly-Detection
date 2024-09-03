import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#import torchtext
#import torchdata


import spacy
import tqdm
import evaluate
import datasets

#from torchtext.vocab import build_vocab_from_iterator, GloVe, vocab, Vectors
from torch.utils.data import Sampler, Dataset
#from torchtext.data.utils import get_tokenizer
#from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, Dataset
#from torchtext.data import get_tokenizer


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



def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_setup(loglvl,logtodisk,write_file):
    logger = logging.getLogger('myLogger')
    formatter=logging.Formatter('%(asctime)s %(name)s %(process)d %(message)s')
    if logtodisk:
        filehandler=logging.FileHandler(write_file)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    if loglvl=='DEBUG':
        logger.setLevel(logging.DEBUG)
        return logger
    else:
        logger.setLevel(logging.INFO)
        return logger
    #streamhandler doesnt seem to work on jupyter-notebook
    
class custReshape(nn.Module):
    def __init__(self, *args):
        
        """
               self.decoder = nn.Sequential(
                torch.nn.Linear(2, 3136),
                custReshape(-1, 64, 7, 7) # can be then called within model instead of using some if statements in forward to reshape.
        so args here are = (-1, 64, 7, 7)
        """
        super().__init__()
        self.shape = args

    def forward(self, x): #when subclass is instantiated with an object only init is called. when object is called with a (), then
        # internally __call__ is called which calls hooks if present and the forward function. here the forward then the returns appropriate reshape
        return x.view(self.shape)
    
class permuteTensor(nn.Module):
    def __init__(self, *args):
        super().__init__()
        """
        call like permuteTensor(0,2,1) to rearrange the the dimensions
        """
        self.dims = args
     
    def forward(self, x):
        return x.permute(self.dims)
    
class globalMaxpool(nn.Module):
    def __init__(self, dim =2):
        super().__init__()
        """
        returns one token which has the max vector in a sequence of tokens 
        dim  = the dimension which corresponds to sequence length
        
        """
        self.dim = dim
     
    def forward(self, x):
        x,_= torch.max(x, self.dim)
        return x
    
class concatTwotensors(nn.Module):
    
    def __init__(self, dim =None):
        super().__init__()
        """
        concats tensors over specified dims
        """
        self.dim = dim
     
    def forward(self,tens1,tens2):
        return torch.cat((tens1,tens2), dim = self.dim)
    
class concatThreetensors(nn.Module):
    def __init__(self, dim =None):
        super().__init__()
        """
        concats tensors over specified dims
        """
        self.dim = dim
     
    def forward(self,tens1,tens2,tens3):
        return torch.cat((tens1,tens2,tens3), dim = self.dim)

    
class squeezeOnes(nn.Module):
    def __init__(self, *args):
        super().__init__()    
    
    def forward(self, x):
        return x.squeeze()
    
class standin(nn.Module):
    def __init__(self):
        super().__init__()    
    
    def forward(self, x):
        return x   
    
class unsqueezeOnes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)
    
    
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.w = args[0]
        self.h = args[1]

    def forward(self, x):
        return x[:, :, :self.w, :self.h] # trim to first 28 pixels witdh and helight


class Linearhc(nn.Module): 
    
    """
    This is a linear layer which takes output/hidden/cell of lstm with (NOTE) seq len 1 (decoder in seq2seq)
    and returns output passed throgh linear layer and hidden/cell. NOTE the squeeze is over seq len as it is one
    """
   
    
    def __init__(self,infeatures,outfeatures):
        super().__init__()
        self.linearhc = nn.Linear(infeatures,outfeatures)
        
    def forward(self,x):
        output, (hidden, cell) = x
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [batch size, 1, hid dim]
        output = self.linearhc(output.squeeze(1)) # only squeeze seq len 1. without dims it will squeeze batchsize for inference ( which is 1 as well)
        
        return output, (hidden, cell)

class Linearhchiddencell(nn.Module): 
    
    """
    This is a linear layer which takes output/hidden/cell of lstm with
    and returns hidden and cell passed throgh linear layer and output untouched. 
    """
   
    
    def __init__(self,infeatures,outfeatures):
        super().__init__()
        self.linearhc = nn.Linear(infeatures,outfeatures)
        
    def forward(self,x):
        output, (hidden, cell) = x
        #expects hidden,cell in form of [batchsize,infeatures]
        hidden = self.linearhc(hidden)
        cell = self.linearhc(cell)

        return output, (hidden, cell)
    
class GRULinearhchidden(nn.Module): 

    
    def __init__(self,infeatures,outfeatures):
        super().__init__()
        self.linearhc = nn.Linear(infeatures,outfeatures)
        
    def forward(self,x):
        output, hidden = x
        #expects hidden in form of [batchsize,infeatures]
        hidden = self.linearhc(hidden)

        return output, hidden
        
class UnpackpackedOutput(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        packed_outputs, hidden = x
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs,batch_first=True)
        return outputs
        
class UnpackpackedOutputHidden(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        packed_outputs, hidden = x
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden    

class activationhc(nn.Module): 
    
    """
    This takes in the output of lstm and only processes the hc through act func and returns out, (h,c)
    """
    def __init__(self,actfunc):
        super().__init__()
        self.actfunc = actfunc
        
    def forward(self,x):
        output, (hidden, cell) = x
        hidden = self.actfunc(hidden)
        cell = self.actfunc(cell)
        
        return output, (hidden, cell)
    
class activationh(nn.Module): 

    def __init__(self,actfunc):
        super().__init__()
        self.actfunc = actfunc
        
    def forward(self,x):
        output,hidden = x
        hidden = self.actfunc(hidden)

        return output, hidden
        
class Bidirectionfullprocess(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        
        output, (hidden, cell) = x
        # output of BiLSTM: [ batch size,src len, hid dim * num directions] torch.Size([128, 20, 1024]) 
        # hidden [n layers * num directions, batch size, hid dim] torch.Size([2, 128, 512]) 
        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            #[num_layers,num_directions, batchsize, hidden_dims] = torch.Size([1, 2, 128, 512])
            hidden = hidden[-1] # last layer of RNN
            # len of hidden is 2 and hidden [0] = [batchs, hidden_dims]= torch.Size([128, 512])
            cell = cell.view(int(cell.shape[0]/2),2,cell.shape[1], cell.shape[2])
            cell = cell[-1] # last layer of RNN
        except Exception as e:
            print ("debug: ", output.shape, hidden.shape, cell.shape)
            sys.exit(0)
            
        h_fwd, h_bwd = hidden[0], hidden[1]
        # h_fwd = [batchsize, hidden_dims] torch.Size([128, 512])
        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        #h_n = [batchs, hidden_dims*2] due to concatanations torch.Size([128, 1024])
        c_fwd, c_bwd = cell[0], cell[1]
        c_n = torch.cat((c_fwd, c_bwd), dim=1)  
        
        return output, (h_n, c_n) 
        # output of BiLSTM: [ batch size,src len, hid dim * num directions] torch.Size([128, 20, 1024]) 
        #h_n [batch size, hid dim*2] torch.Size([128, 1024])

class GRUBidirectionfullprocess(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        
        output, hidden = x
        # output of BiGRU: [ batch size,src len, hid dim * num directions] torch.Size([128, 20, 1024]) 
        # hidden [n layers * num directions, batch size, hid dim] torch.Size([2, 128, 512]) 
        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            #[num_layers,num_directions, batchsize, hidden_dims] = torch.Size([1, 2, 128, 512])
            hidden = hidden[-1] # last layer of RNN
            # len of hidden is 2 and hidden [0] = [batchs, hidden_dims]= torch.Size([128, 512])
        except Exception as e:
            print ("debug: ", output.shape, hidden.shape, cell.shape)
            sys.exit(0)
            
        h_fwd, h_bwd = hidden[0], hidden[1]
        # h_fwd = [batchsize, hidden_dims] torch.Size([128, 512])
        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        #h_n = [batchs, hidden_dims*2] due to concatanations torch.Size([128, 1024])

        
        return output, h_n
        # output of BiGRU: [ batch size,src len, hid dim * num directions] torch.Size([128, 20, 1024]) 
        #h_n [batch size, hid dim*2] torch.Size([128, 1024])        
        
class BidirectionextractHiddenfinal(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            hidden = hidden[-1] # last layer of RNN
        except Exception as e:
            print ("debug: ", output.shape, hidden.shape)
            sys.exit(0)
        return hidden
    
class hiddenBidirectional(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        h_fwd, h_bwd = x[0], x[1]
        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        return h_n

    
class BidirectionextractHCfinal(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            hidden = hidden[-1] # last layer of RNN
            cell = cell.view(int(cell.shape[0]/2),2,cell.shape[1], cell.shape[2])
            cell = cell[-1] # last layer of RNN
        except Exception as e:
            print ("debug: ", output.shape, hidden.shape, cell.shape)
            sys.exit(0)
        return (hidden, cell)    

class hcBidirectional(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, h,c):
        h_fwd, h_bwd = h[0], h[1]
        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        c_fwd, c_bwd = c[0], c[1]
        c_n = torch.cat((c_fwd, c_bwd), dim=1)        
        return (h_n, c_n)
    
class hcHiddenonlyBidirectional(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, h,c):
        h_fwd, h_bwd = h[0], h[1]
        h_n = torch.cat((h_fwd, h_bwd), dim=1)      
        return h_n

class LSTMhc(nn.Module):
    
    def __init__(self,input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False):
        super().__init__()
        self.lstmhc = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
    def forward(self,x,hc):

        output, (hidden, cell) = self.lstmhc(x,hc)
        return output, (hidden, cell)
    

class UnidirectionalextractOutput(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        return output.squeeze(0)
           

        
class UnidirectionextractHiddenCell(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        return (hidden,cell)

class UnidirectionextractHidden(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        return hidden
 
    
class UnidirectionextractHiddenfinal(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        try:
            hidden = hidden.view(hidden.shape[0],1,hidden.shape[1], hidden.shape[2])
            hidden = hidden[-1] # while this is hidden state for final timestep it is still stacked for each layer. -1 for last layer of RNN
        except Exception as e:
            print ("debug: ", output.shape, hidden.shape)
            sys.exit(0)
        return hidden
    


        
class hiddenUnidirectional(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.squeeze(0)


class configuration(object): 
    
    def __init__(self):
        pass
    
    def configureRNN(self, rnnlayers = {}, params = {}):
        
        """
        rnnlayers = {1:embed,2:lstm, 3:fcn}
        params = {"batch_size":128,"pretrained_embeddings" : False}
        """
        self.rnnnet = rnnlayers, params

    def configureFCN(self,layers={1:[4,12],2:[12,12],3:[12,3]}):
        
        """
           FCNlayer = {1:linear1, 2:relu, 3:linear2, 4:relu, 5:linear3, 6:relu, 7:linear4,
                8:sigmoid} # for AE we add sigmoid
        """
        self.fcnnet = layers
        
    def configureCNN1d(self,convlayers = {}, inputsize = 0):
        
        assert inputsize > 0, "Check input size" 
        
        outsize = {} # each key will contain list of [width and height]
        finaloutsize =[]
        insize = {}
        #assuming the index 1 and that the min value in the dictionary is a conv layer
        insize[1] = inputsize # [width, height]
        outsize[1] = inputsize # initi so that if conv is not present in a layer and its just pool
        finaloutchannels = 0
        conv_key = list(convlayers.keys())

        for i in convlayers:

#            for l in convlayers[i]:

            if 'Conv1d' in str(type(convlayers[i])):
# note below calc assumes dilation as 1
# convlayers[i].padding[0] return tuple (0,) so have to take first element
                outsize[i] = np.floor((inputsize + 2*convlayers[i].padding[0] - convlayers[i].kernel_size[0])/convlayers[i].stride[0]) +1
                # stride (1, 1) kernel (5, 5) padding (1, 1)
                finaloutchannels = convlayers[i].out_channels  
                finaloutsize = outsize[i]
                inputsize = outsize[i]
                insize[i+1] = outsize[i]
#                print ("layer :", convlayers[i], finaloutchannels, finaloutimgsize, inputsize)        
        
            if 'pooling' in str(type(convlayers[i])):

                # stride 2 kernel 2 padding 0 - padding assumed to be zero in below calcs
                outsize[i] = np.floor((inputsize -convlayers[i].kernel_size)/convlayers[i].stride) +1
                finaloutsize = outsize[i]
                inputsize = outsize[i]
                insize[i+1] = outsize[i] 
                
#        self.fcin = int(finaloutchannels*finaloutsize)
        self.convoutsize = finaloutsize
        self.finaloutconvchannels = finaloutchannels
        self.convsizeperlayer = outsize
        print (" Final outsize: ", self.convoutsize)
        
        self.cnn1dnet = convlayers 

    def configureCNN(self, convlayers = {},deconvlayers = {} ,inputsize = [], deconv_inputsize =[]):
        """
        NOTE: one conv or/and deconv layer per configurecnn, where each have to be consecutive in order- not sure about this anymore
        inputsize [width, height]
        conv blocks convolution -> pool -> batchnorm -> relu
        convlayer = {1:conv1,2:relu, 3:pool1, 
             4:conv2,5:relu, 6:pool2}
        deconvlayer = {7:deconv1,8:relu,
               9:deconv2}
        
        Returns vars : self.cnnnet, self.fcnnet, self.fcin, self.convoutimgsize(list for each conv block), self.finaloutconvchannels,self.deconvfcin, self.deconvoutimgsize(list for each conv block), self.finaloutdeconvchannels
        
        """

        # everything below is just to figure out downsampled sizes
        if convlayers:
            outimgsize = {} # each key will contain list of [width and height]
            finaloutimgsize =[]
            inimgsize = {}
            #assuming the index 1 and that the min value in the dictionary is a conv layer
            inimgsize[1] = inputsize # [width, height]
            outimgsize[1] = inputsize # initi so that if conv is not present in a layer and its just pool
            finaloutchannels = 0
            conv_key = list(convlayers.keys())


            for i in convlayers:

    #            for l in convlayers[i]:

                if 'conv' in str(type(convlayers[i])):
    # note below calc assumes dilation as 1
                    outimgsize[i] = [np.floor((inputsize[0] + 2*convlayers[i].padding[0] -
                                               convlayers[i].kernel_size[0])/convlayers[i].stride[0]) +1,
                                      np.floor((inputsize[1] + 2*convlayers[i].padding[0] -
                                               convlayers[i].kernel_size[0])/convlayers[i].stride[0]) +1
                                               ]
                    # stride (1, 1) kernel (5, 5) padding (1, 1)
                    finaloutchannels = convlayers[i].out_channels  
                    finaloutimgsize = outimgsize[i]
                    inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]
    #                print ("layer :", convlayers[i], finaloutchannels, finaloutimgsize, inputsize)

                if 'pool' in str(type(convlayers[i])):

                    # stride 2 kernel 2 padding 0  padding assumed to be zero in below calcs
                    outimgsize[i] = [np.floor((inputsize[0] -convlayers[i].kernel_size)/convlayers[i].stride) +1,
                                          np.floor((inputsize[1] -convlayers[i].kernel_size)/convlayers[i].stride) +1
                                           ]
                    finaloutimgsize = outimgsize[i]
                    inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]

            self.fcin = int(finaloutchannels*finaloutimgsize[0]*finaloutimgsize[1])
            self.convoutimgsize = finaloutimgsize
            self.finaloutconvchannels = finaloutchannels
            self.convimgsizeperlayer = outimgsize

        
        
        #for deconv now
        if deconvlayers:
            deconv_key = list(deconvlayers.keys())
            finaloutimgsize = []
            outimgsize = {} # each key will contain list of [width and height]
            inimgsize = {}
            finaloutchannels = 0

            if not deconv_inputsize: # when we are not specifiying deconv input size taje it as the final size out of conv
                deconv_inputsize = self.convoutimgsize

        #assuming the first is a deconv layer and the index corresponding to min of deconvkey is a deconv layer
            inimgsize[min(deconv_key)] = deconv_inputsize # lowest key/index of deconv layer is given input of [width, height]
            outimgsize[min(deconv_key)] = deconv_inputsize # initi so that if conv is not present in a layer and its just pool

            for i in deconvlayers:

#                for l in deconvlayers[i]: 

                if 'conv' in str(type(deconvlayers[i])): 

                    outimgsize[i] = [((deconv_inputsize[0] -1)*deconvlayers[i].stride[0] - 2*deconvlayers[i].padding[0] + (deconvlayers[i].kernel_size[0] -1) + deconvlayers[i].output_padding[0]+1),
                                     ((deconv_inputsize[1] -1)*deconvlayers[i].stride[1] - 2*deconvlayers[i].padding[1] + (deconvlayers[i].kernel_size[1] -1) + deconvlayers[i].output_padding[1]+1)]


                    # stride (1, 1) kernel (5, 5) padding (1, 1)
                    finaloutchannels = deconvlayers[i].out_channels
                    finaloutimgsize = outimgsize[i]
                    deconv_inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]

                if 'pool' in str(type(deconvlayers[i])):


                    outimgsize[i] = [np.floor((deconv_inputsize[0] -deconvlayers[i].kernel_size)/deconvlayers[i].stride) +1,
                                          np.floor((deconv_inputsize[1] -deconvlayers[i].kernel_size)/deconvlayers[i].stride) +1
                                           ]
                    finaloutimgsize = outimgsize[i]
                    deconv_inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]
                # stride 2 kernel 2 padding 0

            self.deconvfcin = int(finaloutchannels*finaloutimgsize[0]*finaloutimgsize[1])
            self.deconvoutimgsize = finaloutimgsize
            self.finaloutdeconvchannels = finaloutchannels
            self.deconvimgsizeperlayer = outimgsize
            
            
        self.cnnnet = convlayers, deconvlayers
        
    def conv(self,inchannels = 0, outchannels = 0, kernel_size = 5, stride = 1, padding = 0):
        
        return nn.Conv2d(inchannels, outchannels, kernel_size, stride, padding)
    
    def maxpool(self,kernel_size =2, stride = None):
        
        return nn.MaxPool2d(kernel_size, stride = stride)
    
    def avgpool(self,kernel_size =2, stride = None):
        
        return nn.AvgPool2d(kernel_size, stride = stride)
    
    def conv1d(self,inchannels = 0, outchannels = 0, kernel_size = 5, stride = 1, padding = 0):
        return nn.Conv1d(inchannels, outchannels, kernel_size, stride, padding)
    
    def maxpool1d(self,kernel_size =2, stride = None):
        return nn.MaxPool1d(kernel_size, stride = stride)
        
    
    def batchnorm2d(self,channels =0):
        
        return nn.BatchNorm2d(channels)
    
    def convtranspose(self,inchannels = 0, outchannels = 0, kernel_size = 5, stride = 1, padding = 0,output_padding =0):
        return nn.ConvTranspose2d(inchannels, outchannels, kernel_size, stride, padding,output_padding)
    
    def batchnorm1d(self,num_features=0):
        return nn.BatchNorm1d(num_features =  num_features) #m = nn.BatchNorm1d(100), m(x)
    
    def relu(self, inplace = False):
        return torch.nn.ReLU(inplace = inplace)
    
    def tanh(self):
        return torch.nn.Tanh()
    
    
    def leaky_relu(self,negative_slope=0.01, inplace = False):
        return nn.LeakyReLU(negative_slope=negative_slope, inplace = inplace)   # m = nn.LeakyReLU(0.1), m(x)
    
    def sigmoid(self):
        return nn.Sigmoid() # m = nn.Sigmoid(), m(x)
    
    def dropout(self,p=0.5):
        return nn.Dropout(p=p) #m = nn.Dropout(p=0.2), m(x)
    
    def dropout1d(self,p=0.25):
        return nn.Dropout1d(p=p)
    
    def dropout2d(self,p=0.25):
        return nn.Dropout2d(p=p)
    
    def linear(self, infeatures = 0, outfeatures = 0):
        return nn.Linear(infeatures,outfeatures)
    
    def flatten(self, start_dim=1, end_dim=-1):
        return nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    
    def unflatten(self, dim, unflattened_size):
        return nn.Unflatten(dim, unflattened_size) # typically something like dim =1, unflattened_size = (1,18,18)
    
    def embeddings(self, num_embeddings, embedding_dim):
        return nn.Embedding(num_embeddings, embedding_dim)
    
    def pretrained_embeddings(self, weights, freeze =  True):
        return nn.Embedding.from_pretrained(weights, freeze=freeze)
    
    def lstm(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False):
        return nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def gru(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False):
        return nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def rnn(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=False, dropout=0.0, bidirectional=False):
        return nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    
    def modulelist(self,modls):
        """
        pass modules in a python list
        """
        tmpmodls = nn.ModuleList()
        for m in modls:
            tmpmodls.append(m)
        return tmpmodls 

#conv1 = conv(1, 10, 5, 1,1)
#pool1= maxpool(2)
#conv2 = conv(10, 20, 5, 1,1)
#pool2= maxpool(2)
#linear1 = linear(5000,50), linear2 = linear(50,10)


class VAE(nn.Module):
    def __init__(self, encoder = None, decoder = None, fcin = 0, zdims =2):
        """
        provide full decoder and encoder net as ANN objects
        zmean and zlogvar can nets themselves or use single layer linear with fcin and zdims values
        """
        super().__init__()
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        for i in encoder:
            self.encoder[str(i)] = encoder[i]
        
        self.z_mean = nn.Linear(fcin, zdims)
        self.z_log_var = nn.Linear(fcin, zdims)
        
        for i in decoder:
            self.decoder[str(i)] = decoder[i]
     
    def embedding(self,x):
        for i in self.encoder:
            x = self.encoder[i](x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x) # get mean and logvar activations out of the two linear layers
        encoding = self.reparameterization(z_mean, torch.exp(0.5 * z_log_var)) # to get samples of z
        return encoding
    
    def generate_from_latent_space(self,x):
        for i in self.decoder:
            x = self.decoder[i](x)
        return x
 
        
    def reparameterization(self, mean, exponentvar): # this is the function that handles creating samples
        epsilon = torch.randn_like(exponentvar) # torch.randn creates samples from N(0,1) and randn_like can pass var whose shape it will copy to create samples of same shape. this is needed since training will be hapening in batches              
        z = mean + exponentvar*epsilon                          
        return z 

    
    def forward(self, x):
        for i in self.encoder:
#            print ("layer", self.encoder[i])
#            print("input size", x.shape)
            x = self.encoder[i](x)
           
            
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x) # get mean and logvar activations out of the two linear layers
#        print (z_mean.shape, z_log_var.shape)
        z = self.reparameterization(z_mean, torch.exp(0.5 * z_log_var)) # to get samples of z
#        print(z.shape)
        
        for c,i in enumerate(self.decoder):
            if c ==0:
#                print("layer", self.decoder[i])
#                print("input size", z.shape)
                xhat= self.decoder[i](z)
            else:
#                print("layer", self.decoder[i])
#                print("input size", xhat.shape)
                xhat = self.decoder[i](xhat)
#        print ("xhat", xhat)
        
        return z, z_mean, z_log_var, xhat    
    

class ANN(nn.Module):   
    
    def __init__(self, confobj ={}):
        """
        confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.layers = nn.ModuleDict()
        self.layersdict = {}
        if confobj:
            for network in confobj:
                if network == 'CNN':
                    self.CNN =  True
                    for cfg in confobj[network]:
                        self.cnn(cfg.cnnnet[0],cfg.cnnnet[1])
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)

            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

## TODO give an option to convert to sequential which is the fastest for fprward pass since utilizes no python for loops



    def cnn(self, convlayers={}, deconvlayers ={}):

        assert len(convlayers) >0 , "Check Layers"

        for i in convlayers:
            self.layersdict[i] = convlayers[i] 

        for i in deconvlayers:                   
            self.layersdict[i] = deconvlayers[i] # Note index i's might not be continous. for example conv 1-2 and deconv 4-5, middle some fcn's
                 
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" # also faclayers input should be populated as it is directly used later

#FClayers = {4:linear1,5:relu,6:batchnorm1d_1])}

        for i in fclayers:
            self.layersdict[i] = fclayers[i]

    def forward(self,x):

        for i in self.layers: 
#            print ('index',i)
#           print("layer: ", self.layers[i])
#            print ("input size", x.shape)
            x = self.layers[i](x)
        return x


class CNN1D(nn.Module):   
    
    def __init__(self, confobj ={}):
        """
        confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.layers = nn.ModuleDict()
        self.layersdict = {}
        if confobj:
            for network in confobj:
                if network == 'CNN':
                    self.CNN =  True
                    for cfg in confobj[network]:
                        self.cnn(cfg.cnn1dnet)
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)

            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

## TODO give an option to convert to sequential which is the fastest for fprward pass since utilizes no python for loops

    def cnn(self, convlayers={}):

        assert len(convlayers) >0 , "Check Layers"

        for i in convlayers:
            self.layersdict[i] = convlayers[i] 
                 
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" # also faclayers input should be populated as it is directly used later
#FClayers = {4:linear1,5:relu,6:batchnorm1d_1])}
        for i in fclayers:
            self.layersdict[i] = fclayers[i]

    def forward(self,x):

        for i in self.layers: 
#            print ('index',i)
#           print("layer: ", self.layers[i])
#            print ("input size", x.shape)
            x = self.layers[i](x)
        return x
    
    
class RNN_classification(nn.Module):   
    
    def __init__(self,confobj ={}):
        
        """confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.layers = nn.ModuleDict()
        self.layersdict = {}
#        self.input_dims = None
        self.bidirectional = None
        self.directions = None
        self.batch_size = None
#        self.embedded_dims =None
        self.hidden_dims = None
        self.num_layers = None
        self.rnnlayeridx = None
        if confobj:
            for network in confobj:
                if network == 'RNN' or network == 'LSTM':
                    self.RNN =  True
                    for cfg in confobj[network]:
#                        print (confobj[network], cfg)
#                        print (cfg.rnnnet[0])
                        self.rnn(cfg.rnnnet[0])               
              
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)
                       
            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

            if self.RNN:
                params = cfg.rnnnet[1]
            for p in params:
                if "batch_size" in p:
                    self.batch_size = params[p]
                    
            # this below for loop is not necessary anymore i dont think 
            for l in self.layers:
                if 'rnn.RNN' in str(type(self.layers[l])) or 'rnn.LSTM' in str(type(self.layers[l])):
                    self.rnnlayeridx = l
                    self.hidden_dims = self.layers[l].hidden_size
                    self.num_layers = self.layers[l].num_layers
                    if self.layers[l].bidirectional:
                        self.bidirectional = True
                        self.directions =  2
                    else:
                        self.bidirectional = False
                        self.directions =  1
                        
            
    def rnn(self, rnnlayers = {}):
        assert len(rnnlayers) >0 , "Check Layers" 
        
        
        for i in rnnlayers:
            self.layersdict[i] = rnnlayers[i]  
                        
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 
        
#FClayers = {4:linear1,5:relu,6:batchnorm1d_1])}

        for i in fclayers:
            self.layersdict[i] = fclayers[i]        

    def forward(self,x):

        for i in self.layers:
            
            ## can all of below simply be added to init and conf ?? this will more floexibility and take the below block of if statements
            # away
            x = self.layers[i](x)
            
            """
            if i == self.rnnlayeridx:
                
                output, (hidden, cell) = self.layers[i](x)
                hidden = hidden.view(self.num_layers,self.directions,self.batch_size,self.hidden_dims)
                hidden = hidden[-1] # last layer of RNN
                if self.directions ==2:
                    h_fwd, h_bwd = hidden[0], hidden[1]
                    h_n = torch.cat((h_fwd, h_bwd), dim=1)
                    x = h_n.view(self.batch_size,self.hidden_dims*2)
                elif self.directions == 1:
                    x = hidden.view(self.batch_size,self.hidden_dims)
            
            else:
                
                x = self.layers[i](x)
            """
                
        return x 
    
    
class RNNhc(nn.Module):   
    
    ##### NOTE DONT USE THIS , NOT SURE HOW THIS IS WORKING , I THINK MIGHT BE TAKING H, C AS NONE instead of what is passed from encoder
    #####################################################################################################################################
    def __init__(self,confobj ={}):
        
        """confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.layers = nn.ModuleDict()
        self.layersdict = {}
#        self.input_dims = None
        self.bidirectional = None
        self.directions = None
        self.batch_size = None
#        self.embedded_dims =None
        self.hidden_dims = None
        self.num_layers = None
        self.rnnlayeridx = None
        if confobj:
            for network in confobj:
                if network == 'RNN' or network == 'LSTM':
                    self.RNN =  True
                    for cfg in confobj[network]:
#                        print (confobj[network], cfg)
#                        print (cfg.rnnnet[0])
                        self.rnn(cfg.rnnnet[0])               
              
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)
                       
            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

            if self.RNN:
                params = cfg.rnnnet[1]
            for p in params:
                if "batch_size" in p:
                    self.batch_size = params[p]
                    
            # this below for loop is not necessary anymore i dont think 
            for l in self.layers:
                if 'rnn.RNN' in str(type(self.layers[l])) or 'rnn.LSTM' in str(type(self.layers[l])):
                    self.rnnlayeridx = l
                    self.hidden_dims = self.layers[l].hidden_size
                    self.num_layers = self.layers[l].num_layers
                    if self.layers[l].bidirectional:
                        self.bidirectional = True
                        self.directions =  2
                    else:
                        self.bidirectional = False
                        self.directions =  1
                        
            
    def rnn(self, rnnlayers = {}):
        assert len(rnnlayers) >0 , "Check Layers" 
        
        
        for i in rnnlayers:
            self.layersdict[i] = rnnlayers[i]  
                        
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 
        
#FClayers = {4:linear1,5:relu,6:batchnorm1d_1])}

        for i in fclayers:
            self.layersdict[i] = fclayers[i]        

    def forward(self,x, h= None, c = None, contextvec = None):
        
        #[torch.cuda.FloatTensor [128, 1, 512]]
        
        # x will be [ batch size, sentence length]  # due to batch first. ex ([128, 193]) . For decoder since its 1 word, seqlen =1
        # after passing through embeddings it will be embedded dim: [ batch size, sentence length,embedding dim]
        # after passing through lstm it will be:
        # output dims [batch size, sentence length,  hidden dim *D] [128, 193, 256]), where D =2 for bidirectional
        # hidden dim: [D*num_layers, batch size, hidden dim] ([1, 128, 256])
        # the last layer hidden [1, 128, 256] is usually sent to fcn as [batch_size, infeatures], where infeatures are usually the hidden_dims
        for i in self.layers:

            if 'LSTM' in str(type(self.layers[i])) or 'RNN' in str(type(self.layers[i])):
                x= self.layers[i](x,(h,c))   
            else:
                x = self.layers[i](x)

        return x 

    
class RNNpacked(nn.Module):   

    def __init__(self,confobj ={}):
        
        """confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        print ("RNNpacked class USED")
        self.layers = nn.ModuleDict()
        self.layersdict = {}
#        self.input_dims = None
        self.bidirectional = None
        self.directions = None
        self.batch_size = None
#        self.embedded_dims =None
        self.hidden_dims = None
        self.num_layers = None
        self.rnnlayeridx = None
        if confobj:
            for network in confobj:
                if network == 'RNN' or network == 'LSTM':
                    self.RNN =  True
                    for cfg in confobj[network]:
#                        print (confobj[network], cfg)
#                        print (cfg.rnnnet[0])
                        self.rnn(cfg.rnnnet[0])               
              
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)
                       
            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

            if self.RNN:
                params = cfg.rnnnet[1]
            for p in params:
                if "batch_size" in p:
                    self.batch_size = params[p]
                if "pack" in p:
                    self.packidx = params[p]# str(key#)

                    
            # this below for loop is not necessary anymore i dont think 
            for l in self.layers:
                if 'rnn.RNN' in str(type(self.layers[l])) or 'rnn.LSTM' in str(type(self.layers[l])):
                    self.rnnlayeridx = l
                    self.hidden_dims = self.layers[l].hidden_size
                    self.num_layers = self.layers[l].num_layers
                    if self.layers[l].bidirectional:
                        self.bidirectional = True
                        self.directions =  2
                    else:
                        self.bidirectional = False
                        self.directions =  1
                        
            
    def rnn(self, rnnlayers = {}):
        assert len(rnnlayers) >0 , "Check Layers" 
        
        for i in rnnlayers:
            self.layersdict[i] = rnnlayers[i]  
                        
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 
        
#FClayers = {4:linear1,5:relu,6:batchnorm1d_1])}

        for i in fclayers:
            self.layersdict[i] = fclayers[i]        

    def forward(self,x,x_len=None):
        
        #[torch.cuda.FloatTensor [128, 1, 512]]
        
        # x will be [ batch size, sentence length]  # due to batch first. ex ([128, 193]) . For decoder since its 1 word, seqlen =1
        # after passing through embeddings it will be embedded dim: [ batch size, sentence length,embedding dim]
        # after passing through lstm it will be:
        # output dims [batch size, sentence length,  hidden dim *D] [128, 193, 256]), where D =2 for bidirectional
        # hidden dim: [D*num_layers, batch size, hidden dim] ([1, 128, 256])
        # the last layer hidden [1, 128, 256] is usually sent to fcn as [batch_size, infeatures], where infeatures are usually the hidden_dims
        assert isinstance(x_len, torch.Tensor), "This is a RNNpacked class, src len cannot be None"
        
        for i in self.layers:
            if i == self.packidx:
                x = nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(),batch_first=True,enforce_sorted=False)
            else:
                x = self.layers[i](x)

        return x 
    
class decoderGRU_cho(nn.Module): 
    """
    Decoder with GRU implemented from https://arxiv.org/abs/1406.1078
    """
    
    def __init__(self,output_dim, emb_dim, hid_dim,num_layers, dropout = 0.0):
        
        """
        output_dims = output vocab size. This is used as the outfeatures in final linear layer
        
        """
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        
        self.hid_dim = hid_dim
        self.output_dim = output_dim # again the trg vocab size
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=num_layers, batch_first= True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)  

    def forward(self,input, hidden, contextvec = None):
        
        
        # x will be [ batch size, sentence length]  # due to batch first. ex ([128, 193]) . For decoder since its 1 word, seqlen =1
        # after passing through embeddings it will be embedded dim: [ batch size, sentence length,embedding dim]
        # after passing through lstm it will be:
        # output dims [batch size, sentence length,  hidden dim *D] [128, 193, 256]), where D =2 for bidirectional
        # hidden dim: [D*num_layers, batch size, hidden dim] ([1, 128, 256])
        # the last layer hidden [1, 128, 256] is usually sent to fcn as [batch_size, infeatures], where infeatures are usually the hidden_dims
        input = input.unsqueeze(1) # in seq2seq class we only send inout in shape of batchsize to decoder so this is needed.
        embedded = self.dropout(self.embedding(input)) # [ batch size, sentence length =1 ,embedding dim]
        
        # taking contexvec to be just the hidden and not the cells state from encoder
        
        ####### add context vec to each timestep as input
        """
        [[batchsize , seqlen, cat (embed dims, context vec)]] as a number samples, each sample having some seq len ( number of words)
and each element in the sequence ( each token) having a number of features given by embed dims + context vec.
Intutiively this makes sence as now each input seq to the decoder whose next word is to be predicted by model carries info
about its own embeddings and also each word carries context vector info from encoder.
        """
        #since contextvec from encoder is of size: [D*num_layers, batch size, hidden dim] ([1, 128, 256]) and concat docs say all tensors
        # must be of same shape except concat dim , we may need to permuste the contexvec to [1,0,2]
        contextvec = contextvec.permute(1,0,2) # [ batch size, D*num_layers, hidden dim] ([128, 1, 256])
        emb_con = torch.cat((embedded, contextvec), dim = 2) # [[batchsize , seqlen, cat (embed dims, context vec)= embed dims + context vec]]  
        output, hidden = self.rnn(emb_con, hidden)
        # hidden dim: [D*num_layers, batch size, hidden dim] ([1, 128, 256])
       
    
    ###### adding input current word embeddings to input of linear layer
        """
embed input , output of lstm = hidden since there is 1 token predic, encoder contextvec
embed input =  [batch, 1, embed dims], hidden = [1, batch size, hid dim], 
contextvec = [D*num_layers = 1, batch size, hidden dim] converted to [batch size,1, hidden dim]

This needs to be fed as input to linear layer so needs to be squeezed out of 1 and into general form of [batchs, features]
embedded.squeeze(1) = [batch, 1, embed dims],  hidden.squeeze(0) = [batch size, hid dim]
contextvec.squeeze(1) = [batch size, hid dim] 
output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), context.squeeze(1)), dim = 1)
concat over dim1  ouput =  [batchsize, emb dim + hid dim * 2]
        """
        output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), contextvec.squeeze(1)), dim = 1) #[batchsize, emb dim + hid dim * 2]
        prediction = self.fc_out(output) # #prediction = [batch size, output dim] = [batch size, vocab en len]


        return prediction, hidden   

class decoder_cho(nn.Module): 
    """
    Decoder with LSTM implemented from https://arxiv.org/abs/1406.1078
    """
    
    def __init__(self,output_dim, emb_dim, hid_dim,num_layers, dropout = 0.0):
        
        """
        output_dims = output vocab size. This is used as the outfeatures in final linear layer
        
        """
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        
        self.hid_dim = hid_dim
        self.output_dim = output_dim # again the trg vocab size
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, num_layers=num_layers, batch_first= True)
        #nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)  

    def forward(self,input, hc, contextvec = None):
        
        
        # x will be [ batch size, sentence length]  # due to batch first. ex ([128, 193]) . For decoder since its 1 word, seqlen =1
        # after passing through embeddings it will be embedded dim: [ batch size, sentence length,embedding dim]
        # after passing through lstm it will be:
        # output dims [batch size, sentence length,  hidden dim *D] [128, 193, 256]), where D =2 for bidirectional
        # hidden dim: [D*num_layers, batch size, hidden dim] ([1, 128, 256])
        # the last layer hidden [1, 128, 256] is usually sent to fcn as [batch_size, infeatures], where infeatures are usually the hidden_dims
        input = input.unsqueeze(1) # in seq2seq class we only send inout in shape of batchsize to decoder so this is needed.
        embedded = self.dropout(self.embedding(input)) # [ batch size, sentence length =1 ,embedding dim]
        
        # taking contexvec to be just the hidden and not the cells state from encoder
        
        ####### add context vec to each timestep as input
        """
        [[batchsize , seqlen, cat (embed dims, context vec)]] as a number samples, each sample having some seq len ( number of words)
and each element in the sequence ( each token) having a number of features given by embed dims + context vec.
Intutiively this makes sence as now each input seq to the decoder whose next word is to be predicted by model carries info
about its own embeddings and also each word carries context vector info from encoder.
        """
        #since contextvec from encoder is of size: [D*num_layers, batch size, hidden dim] ([1, 128, 256]) and concat docs say all tensors
        # must be of same shape except concat dim , we may need to permuste the contexvec to [1,0,2]
        contextvec = contextvec.permute(1,0,2) # [ batch size, D*num_layers, hidden dim] ([128, 1, 256])
        emb_con = torch.cat((embedded, contextvec), dim = 2) # [[batchsize , seqlen, cat (embed dims, context vec)= embed dims + context vec]]  
        output, (hidden, cell) = self.rnn(emb_con, (hc[0],hc[1]))
        # hidden dim: [D*num_layers, batch size, hidden dim] ([1, 128, 256])
       
    
    ###### adding input current word embeddings to input of linear layer
        """
embed input , output of lstm = hidden since there is 1 token predic, encoder contextvec
embed input =  [batch, 1, embed dims], hidden = [1, batch size, hid dim], 
contextvec = [D*num_layers = 1, batch size, hidden dim] converted to [batch size,1, hidden dim]

This needs to be fed as input to linear layer so needs to be squeezed out of 1 and into general form of [batchs, features]
embedded.squeeze(1) = [batch, 1, embed dims],  hidden.squeeze(0) = [batch size, hid dim]
contextvec.squeeze(1) = [batch size, hid dim] 
output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), context.squeeze(1)), dim = 1)
concat over dim1  ouput =  [batchsize, emb dim + hid dim * 2]
        """
        output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), contextvec.squeeze(1)), dim = 1) #[batchsize, emb dim + hid dim * 2]
        prediction = self.fc_out(output) # #prediction = [batch size, output dim] = [batch size, vocab en len]


        return prediction, (hidden, cell)         

class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask = None):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]


        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.squeeze(0) # not needed if hidden is already [batch size, dec hid dim], otherwise will turn [1, batch size, dec hid dim] it into this shape
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch size, src len, dec hid dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
  
        #attention= [batch size, src len]
        if mask is not None: # mask  = [batch size, source sentence length]  1 when the source sentence token is not a padding token, and 0 when it is a padding token
            attention = attention.masked_fill(mask == 0, -1e10) # where 0 is true in tensor, fill it with value -1e10, which is so low that on softmax it is zeroed. IMPORTNAT : assumes that padding vocab index == 0!!!!
        
        return F.softmax(attention, dim=1)

class decoderGRU_attn_bahdanau(nn.Module):
    
    """
    Decoder GRU for attention 
    """
    
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim # typically trg vocab len
        self.attention = attention # attention class - return atten scores of size seq len
        
        self.embedding = nn.Embedding(output_dim, emb_dim)

            
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim,num_layers=num_layers,batch_first=True)

        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        

        
    def forward(self, input, hc, encoder_outputs, mask =None):
             
        #input = [batch size] yt token of seq len 1
        #hc = (h,c) = hc[0] [batch size, dec hid dim]
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        #mask = [batchs, src len]


        input = input.unsqueeze(1) # adds 1 as dim [batch size,1]
        
        embedded = self.dropout(self.embedding(input)) # [batch size,1, emb dim]

        hidden = hc

        a = self.attention(hidden, encoder_outputs, mask) # take previous decoder state St-1 and encoder outputs H and returns attention score
        # whose size is same as src ln [batch size, src len]

        a = a.unsqueeze(1) # [batch size, 1, src len]

        weighted = torch.bmm(a, encoder_outputs) # batch matrix matrix product Wt = At x H to create attentive context vec
        # mat1 and mat2 must be 3-D tensors each containing the same number of matrices. [b,n,m] [b,m,p] results in [b,n,p]
        # [batch size, 1, src len] x [batch size, src len, enc hid dim * 2] = [batchs,1, enc hid dim * 2]

        
        rnn_input = torch.cat((embedded, weighted), dim = 2) # embedded is processed input yt through dropou and embed [batch size,1 emb dim]. Cat on dim =2 makes this [batch size, 1 (enc hid dim * 2) + emb dim]
        
        # doing this since somtimes we get hidden as [batchs, hiddn_dim] and other times as [1, batchs, hiddn_dim]
        #since we are hardcoding unsqueeze when passing into rnn as that has to be of shape [1, batchs, hiddn_dim], we are squeezing 
        # hc here in case it is of size [1, batchs, hiddn_dim]. if not and is [batchs, hiddn_dim] no harm done it remains same
        
        
        hidden = hidden.squeeze(0)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        # chnage hc from [batch size, dec hid dim] to [n layers * n directions, batch size, dec hid dim] as a stock LSTM WILL ALWAYS TAKE IN and spit OUT HIDDEN IN FORM OF 
        #[n layers * n directions, batch size, dec hid dim]
                                         
        #output = [batch size, seq len, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [batch size,1, dec hid dim]
        #hidden = [1, batch size, dec hid dim]

        #this also means that output == hidden
        assert (output.permute(1,0,2) == hidden).all()
        
        embedded = embedded.squeeze(1)  # [batch size,emb dim]
        output = output.squeeze(1) # [batch size, dec hid dim]
        weighted = weighted.squeeze(1) # [batchs, enc hid dim * 2]

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim] ,hidden = [n layers * n directions, batch size, dec hid dim]
        
        return prediction, hidden , a.squeeze(1) # additionally sending back attention matrix to view values of atten during inference
         # the a.squee chnages # [batch size, 1, src len] to  [batch size, src len]
        


class decoder_attn_bahdanau(nn.Module):
    
    """
    Decoder LSTM for attention 
    """
    
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        
        super().__init__()

        self.output_dim = output_dim # typically trg vocab len
        self.attention = attention # attention class - return atten scores of size seq len
        
        self.embedding = nn.Embedding(output_dim, emb_dim)



        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim,num_layers=num_layers,batch_first=True)


        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        

        
    def forward(self, input, hc, encoder_outputs, mask = None):
             
        #input = [batch size] yt token of seq len 1
        #hc = (h,c) = hc[0] [batch size, dec hid dim]
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        #mask = [batchs, src len]


        input = input.unsqueeze(1) # adds 1 as dim [batch size,1]
        
        embedded = self.dropout(self.embedding(input)) # [batch size,1, emb dim]

        hidden = hc[0]
        cell = hc[1]


        a = self.attention(hidden, encoder_outputs, mask) # take previous decoder state St-1 and encoder outputs H and returns attention score
        # whose size is same as src ln [batch size, src len]

        a = a.unsqueeze(1) # [batch size, 1, src len]

        weighted = torch.bmm(a, encoder_outputs) # batch matrix matrix product Wt = At x H to create attentive context vec
        # mat1 and mat2 must be 3-D tensors each containing the same number of matrices. [b,n,m] [b,m,p] results in [b,n,p]
        # [batch size, 1, src len] x [batch size, src len, enc hid dim * 2] = [batchs,1, enc hid dim * 2]

        
        rnn_input = torch.cat((embedded, weighted), dim = 2) # embedded is processed input yt through dropou and embed [batch size,1 emb dim]. Cat on dim =2 makes this [batch size, 1 (enc hid dim * 2) + emb dim]
        
        # doing this since somtimes we get hidden as [batchs, hiddn_dim] and other times as [1, batchs, hiddn_dim]
        #since we are hardcoding unsqueeze when passing into rnn as that has to be of shape [1, batchs, hiddn_dim], we are squeezing 
        # hc here in case it is of size [1, batchs, hiddn_dim]. if not and is [batchs, hiddn_dim] no harm done it remains same
        
       
        
        hidden = hidden.squeeze(0) 
        cell = cell.squeeze(0)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0))) # chnage hc from [batch size, dec hid dim] to [n layers * n directions, batch size, dec hid dim] as a stock LSTM WILL ALWAYS TAKE IN and spit OUT HIDDEN IN FORM OF 
        #[n layers * n directions, batch size, dec hid dim]
                                         
        #output = [batch size, seq len, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [batch size,1, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        

        #this also means that output == hidden
        assert (output.permute(1,0,2) == hidden).all()
        
        embedded = embedded.squeeze(1)  # [batch size,emb dim]
        output = output.squeeze(1) # [batch size, dec hid dim]
        weighted = weighted.squeeze(1) # [batchs, enc hid dim * 2]

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim] ,hidden = [n layers * n directions, batch size, dec hid dim]

        
        return prediction, (hidden, cell), a.squeeze(1) # additionally sending back attention matrix to view values of atten during inference
         # the a.squee chnages # [batch size, 1, src len] to  [batch size, src len]
        
     
class Seq2SeqLSTMPacked(nn.Module):
    
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}):
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.encoder = encoder 
        self.decoder = decoder
        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for p in params:
#            if 'batch_size' in p:
#                self.bs = params[p]
            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        
#        assert self.bs> 0, "No batch size"
        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"

    
    def forward(self, src, src_len =None, trg = None, teacher_forcing= 0.5): #new
        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        assert isinstance(src_len, torch.Tensor), "This is a seq2seq packed class, src len cannot be None"
        
        if trg == None: # this bit has been added again to be able to use the model for inference seamlessly
            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            # 25 seq len randonly chosen. This gives trg shape of [batchsize, 25]. it is filled with value = index of vocab for <sos> = 0
            # in our vocab ( the tutorial was 2).Why SOS ??!!! as the network decoder to get started still needs H,C from encoder ( which is fine), but still need trg sentence first token of SOS to be fed into it
            assert teacher_forcing == 0, "Must be zero during inference"
            
        # src and trg are in batch first mode - [batchsize, seq len] torch.Size([128, 23]) torch.Size([128, 31])
        trg_len = trg.shape[1] # the second dim has the seqlen = 31
        bs = trg.shape[0] # dynamically get batch size - usefult chnage for using same mode for inference.
        
#        outputs = torch.zeros(trg_len, self.bs, self.trg_vocab_len).to(self.device) # create a outputs tensor which will store for every batch torch.Size([31, 128, 10837])
                 
        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)
        
        hidden = self.encoder(src, src_len)

 #Note hidden it is not [D*layers, batchs, hidden_dim] as at various place we are taking only final layer which squeezes out the 1 in 0th dim

        context = hidden # additional code im adding since we are using lstm and paper was using gru and im deciding to take
        # just hidden as the context

        inp = trg[:,0] # [batch_size] as we are taking the single tokens and here the first element of every batch which is sos

        for t in range(1,trg_len): # for the entire sentence or stepping through each time step in sequence skipping sos

                # output is the prediction from decoder output.shape = [batchsize, trg_vocab_len] [128, 10837], 
                #hidden shape [1,batch size, dec hid dim]

            output, (hidden, cell) = self.decoder(inp, (hidden, cell), context)
                


            #place predictions in a tensor holding predictions for each token. and prediction are for each token in vocab, this is where 
            # the cross entropy loss is calculated for each batch with this outputs and another matrix of same dims with actual outputs
            outputs[t] = output 

            
             # in the first iteration inp = sos of trg english and hidden cell are context vectors of entire encoded 23 sequence length german statement. GIven this we find the first english token output which is basically the trg[:,1] and insert it in outputs[1]
                # Again to restate the decoder loop which populates outputs starts at 1
                # so this means the 0th element in outputs remains 0. Bascially things will look like this
                #trg =[sos, y1, y2..,<eos>]
                #outputs = [0, yhat1, yhat2..,<eos>]
                #to calculate loss , we cut off the first element from both these matrices.

            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing
            
            #get the vocabulary index ( from 0-10836) whihch holds the hiughest value which corresponds to the predicted token - one for each same in batch - which is why the shape returned is batchsize
            

            top1 = output.argmax(1) # [batchsize] [128]
      
            
            #if teacher forcing, use actual next token as next input - so t[:,1] in the first iteration
            #if not, use predicted token
            inp = trg[:,t] if teacher_force else top1 # either case needs to be [batchsize] =128 as one for each batch sample
            # since trg is [batchsize, seq len], we have to use trg[:,1], trg[:,2], trg[:,3]

        
        return outputs
        
    
class Seq2SeqAttnLSTMPacked(nn.Module):      
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}, src_pad_idx=0):
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.encoder = encoder 
        self.decoder = decoder
        sel.src_pad_idx= src_pad_idx
        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for p in params:
#            if 'batch_size' in p:
#                self.bs = params[p]
            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        
#        assert self.bs> 0, "No batch size"
        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"
        
    def create_mask(self, src): # so any value pad idx as long as its specified here.
        mask = (src != self.src_pad_idx) # [batch size, src len]
        return mask
    
    
    def forward(self, src, src_len =None, trg = None, teacher_forcing= 0.5): #new
        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        assert isinstance(src_len, torch.Tensor), "This is a seq2seq packed class, src len cannot be None"
        
        if trg == None: # this bit has been added again to be able to use the model for inference seamlessly
            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            # 25 seq len randonly chosen. This gives trg shape of [batchsize, 25]. it is filled with value = index of vocab for <sos> = 0
            # in our vocab ( the tutorial was 2).Why SOS ??!!! as the network decoder to get started still needs H,C from encoder ( which is fine), but still need trg sentence first token of SOS to be fed into it
            assert teacher_forcing == 0, "Must be zero during inference"
            
        # src and trg are in batch first mode - [batchsize, seq len] torch.Size([128, 23]) torch.Size([128, 31])
        trg_len = trg.shape[1] # the second dim has the seqlen = 31
        bs = trg.shape[0] # dynamically get batch size - usefult chnage for using same mode for inference.
        
#        outputs = torch.zeros(trg_len, self.bs, self.trg_vocab_len).to(self.device) # create a outputs tensor which will store for every batch torch.Size([31, 128, 10837])
        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)

        encoder_outputs, (hidden,cell) = self.encoder(src, src_len) 
            # encoder output bilstm[ batch size,src len, hid dim * num directions], hc after passed through linear which transfroms from hid_dim*2 to hid_dim [batch size, hid dim]. Note it is not [D*layers, batchs, hidden_dim] as at various place we are taking only final layer which squeezes out the 1 in 0th dim

        context = encoder_outputs
        mask = self.create_mask(src) # [batch size, src len]


        inp = trg[:,0] # [batch_size] as we are taking the single tokens and here the first element of every batch which is sos

        for t in range(1,trg_len): # for the entire sentence or stepping through each time step in sequence skipping sos
                
            output, (hidden, cell), _ = self.decoder(inp, (hidden, cell), context, mask) # the _ is because attention vector is returned
                # but we dont want it during training 
     
                # output is the prediction from decoder output.shape = [batchsize, trg_vocab_len] [128, 10837], 
                #hidden shape [1,batch size, dec hid dim]


            #place predictions in a tensor holding predictions for each token. and prediction are for each token in vocab, this is where 
            # the cross entropy loss is calculated for each batch with this outputs and another matrix of same dims with actual outputs
            outputs[t] = output 

            
             # in the first iteration inp = sos of trg english and hidden cell are context vectors of entire encoded 23 sequence length german statement. GIven this we find the first english token output which is basically the trg[:,1] and insert it in outputs[1]
                # Again to restate the decoder loop which populates outputs starts at 1
                # so this means the 0th element in outputs remains 0. Bascially things will look like this
                #trg =[sos, y1, y2..,<eos>]
                #outputs = [0, yhat1, yhat2..,<eos>]
                #to calculate loss , we cut off the first element from both these matrices.

            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing
            
            #get the vocabulary index ( from 0-10836) whihch holds the hiughest value which corresponds to the predicted token - one for each same in batch - which is why the shape returned is batchsize
            

            top1 = output.argmax(1) # [batchsize] [128]
      
            
            #if teacher forcing, use actual next token as next input - so t[:,1] in the first iteration
            #if not, use predicted token
            inp = trg[:,t] if teacher_force else top1 # either case needs to be [batchsize] =128 as one for each batch sample
            # since trg is [batchsize, seq len], we have to use trg[:,1], trg[:,2], trg[:,3]

        
        return outputs

class Seq2SeqAttnGRU(nn.Module):      
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}, src_pad_idx=0):
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.encoder = encoder 
        self.decoder = decoder

        self.src_pad_idx= src_pad_idx

        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
        for p in params:
#            if 'batch_size' in p:
#                self.bs = params[p]
            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        
#        assert self.bs> 0, "No batch size"
        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"

    
    def forward(self, src, trg = None, teacher_forcing= 0.5): #new
        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        if trg == None: # this bit has been added again to be able to use the model for inference seamlessly
            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            # 25 seq len randonly chosen. This gives trg shape of [batchsize, 25]. it is filled with value = index of vocab for <sos> = 0
            # in our vocab ( the tutorial was 2).Why SOS ??!!! as the network decoder to get started still needs H,C from encoder ( which is fine), but still need trg sentence first token of SOS to be fed into it
            assert teacher_forcing == 0, "Must be zero during inference"
            
        # src and trg are in batch first mode - [batchsize, seq len] torch.Size([128, 23]) torch.Size([128, 31])
        trg_len = trg.shape[1] # the second dim has the seqlen = 31
        bs = trg.shape[0] # dynamically get batch size - usefult chnage for using same mode for inference.
        
#        outputs = torch.zeros(trg_len, self.bs, self.trg_vocab_len).to(self.device) # create a outputs tensor which will store for every batch torch.Size([31, 128, 10837])
        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)


        encoder_outputs, hidden = self.encoder(src)
            # encoder output bilstm[ batch size,src len, hid dim * num directions], hc after passed through linear which transfroms from hid_dim*2 to hid_dim [batch size, hid dim]. Note it is not [D*layers, batchs, hidden_dim] as at various place we are taking only final layer which squeezes out the 1 in 0th dims

            
        context = encoder_outputs



        inp = trg[:,0] # [batch_size] as we are taking the single tokens and here the first element of every batch which is sos
        
        for t in range(1,trg_len): # for the entire sentence or stepping through each time step in sequence skipping sos
            

                
            output, hidden,_ = self.decoder(inp, hidden, context) # the _ is because attention vector is returned
                # but we dont want it during training 
                
                # output is the prediction from decoder output.shape = [batchsize, trg_vocab_len] [128, 10837], 
                #hidden shape [1,batch size, dec hid dim]



            #place predictions in a tensor holding predictions for each token. and prediction are for each token in vocab, this is where 
            # the cross entropy loss is calculated for each batch with this outputs and another matrix of same dims with actual outputs
            outputs[t] = output 

            
             # in the first iteration inp = sos of trg english and hidden cell are context vectors of entire encoded 23 sequence length german statement. GIven this we find the first english token output which is basically the trg[:,1] and insert it in outputs[1]
                # Again to restate the decoder loop which populates outputs starts at 1
                # so this means the 0th element in outputs remains 0. Bascially things will look like this
                #trg =[sos, y1, y2..,<eos>]
                #outputs = [0, yhat1, yhat2..,<eos>]
                #to calculate loss , we cut off the first element from both these matrices.

            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing
            
            #get the vocabulary index ( from 0-10836) whihch holds the hiughest value which corresponds to the predicted token - one for each same in batch - which is why the shape returned is batchsize
            

            top1 = output.argmax(1) # [batchsize] [128]
      
            
            #if teacher forcing, use actual next token as next input - so t[:,1] in the first iteration
            #if not, use predicted token
            inp = trg[:,t] if teacher_force else top1 # either case needs to be [batchsize] =128 as one for each batch sample
            # since trg is [batchsize, seq len], we have to use trg[:,1], trg[:,2], trg[:,3]

        
        return outputs    
    
class Seq2SeqAttnGRUPacked(nn.Module):      
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}, src_pad_idx=0):
        
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.encoder = encoder 
        self.decoder = decoder

        self.src_pad_idx= src_pad_idx

        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
        for p in params:
#            if 'batch_size' in p:
#                self.bs = params[p]
            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        
#        assert self.bs> 0, "No batch size"
        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"
        
    def create_mask(self, src): # so any value pad idx as long as its specified here.
        mask = (src != self.src_pad_idx) # [batch size, src len]
        return mask
    
    
    def forward(self,src, src_len =None, trg = None, teacher_forcing= 0.5): #new
        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        assert isinstance(src_len, torch.Tensor), "This is a seq2seq packed class, src len cannot be None"

        if trg == None: # this bit has been added again to be able to use the model for inference seamlessly
            print ("This is Inference")
            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            # 25 seq len randonly chosen. This gives trg shape of [batchsize, 25]. it is filled with value = index of vocab for <sos> = 0
            # in our vocab ( the tutorial was 2).Why SOS ??!!! as the network decoder to get started still needs H,C from encoder ( which is fine), but still need trg sentence first token of SOS to be fed into it
            assert teacher_forcing == 0, "Must be zero during inference"
            
        # src and trg are in batch first mode - [batchsize, seq len] torch.Size([128, 23]) torch.Size([128, 31])
        trg_len = trg.shape[1] # the second dim has the seqlen = 31
        bs = trg.shape[0] # dynamically get batch size - usefult chnage for using same mode for inference.
        
#        outputs = torch.zeros(trg_len, self.bs, self.trg_vocab_len).to(self.device) # create a outputs tensor which will store for every batch torch.Size([31, 128, 10837])
        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)


        encoder_outputs, hidden = self.encoder(src, src_len)
            # encoder output bilstm[ batch size,src len, hid dim * num directions], hc after passed through linear which transfroms from hid_dim*2 to hid_dim [batch size, hid dim]. Note it is not [D*layers, batchs, hidden_dim] as at various place we are taking only final layer which squeezes out the 1 in 0th dims

            
        context = encoder_outputs

        mask = self.create_mask(src) # [batch size, src len]



        inp = trg[:,0] # [batch_size] as we are taking the single tokens and here the first element of every batch which is sos
        
        for t in range(1,trg_len): # for the entire sentence or stepping through each time step in sequence skipping sos
            

                
            output, hidden,_ = self.decoder(inp, hidden, context, mask) # the _ is because attention vector is returned
                # but we dont want it during training 
                
                # output is the prediction from decoder output.shape = [batchsize, trg_vocab_len] [128, 10837], 
                #hidden shape [1,batch size, dec hid dim]



            #place predictions in a tensor holding predictions for each token. and prediction are for each token in vocab, this is where 
            # the cross entropy loss is calculated for each batch with this outputs and another matrix of same dims with actual outputs
            outputs[t] = output 

            
             # in the first iteration inp = sos of trg english and hidden cell are context vectors of entire encoded 23 sequence length german statement. GIven this we find the first english token output which is basically the trg[:,1] and insert it in outputs[1]
                # Again to restate the decoder loop which populates outputs starts at 1
                # so this means the 0th element in outputs remains 0. Bascially things will look like this
                #trg =[sos, y1, y2..,<eos>]
                #outputs = [0, yhat1, yhat2..,<eos>]
                #to calculate loss , we cut off the first element from both these matrices.

            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing
            
            #get the vocabulary index ( from 0-10836) whihch holds the hiughest value which corresponds to the predicted token - one for each same in batch - which is why the shape returned is batchsize
            

            top1 = output.argmax(1) # [batchsize] [128]
      
            
            #if teacher forcing, use actual next token as next input - so t[:,1] in the first iteration
            #if not, use predicted token
            inp = trg[:,t] if teacher_force else top1 # either case needs to be [batchsize] =128 as one for each batch sample
            # since trg is [batchsize, seq len], we have to use trg[:,1], trg[:,2], trg[:,3]

        
        return outputs
    
class MultiNet(nn.Module):  
    """
    this is for two model two input. 
    Change for three model three input and so on..
    """
    
    def __init__(self,net1, net2, net3):
        super().__init__() # init is called when class is instantiated from Net() class as self.net
        self.model1 = net1
        self.model2 = net2
        self.model3 = net3
    """     
    def forward(self,x1,x2): # below are examples of lstm nets used for mnist images which hasve been fed as 28x28 and one x2 has been 
        # permuted rows and cols
        # in this they x1 and x2 are in shapes of [ batch size, sentence length, embedding dim]
 #       print (x1.shape, x2.shape) # torch.Size([2, 28, 28]) torch.Size([2, 28, 28])
        
        x1 = self.model1(x1)
 #       print (x1.shape) # torch.Size([2, 512]) # last layer last hidden [batchsize , hidden dim *2] for bi
       
        x2 = self.model2(x2)
 #       print (x2.shape) # torch.Size([2, 512])
        x = torch.cat((x1, x2), dim=1)
 #       print (x.shape) # torch.Size([2, 1024]) # concat on features or colummns
        x = self.model3(x)
 #       print (x.shape) # torch.Size([2, 10])
 
        return x
    """
    def forward(self,x): 
        x1 = self.model1(x)
        
        x2= permuteTensor(0,2,1)(x) # permute dims of input to feed that to a second lstm model
        x2 = self.model2(x2)
        
        x = torch.cat((x1, x2), dim=1) 

        x = self.model3(x)

 
        return x    

class SamplerSimilarLengthHFDataset(Sampler):
    """
    give batch size , drop last , shuffle and keyname of src in HF dataset. Dataset is assumed to be pytorch tensor
    """
    
    def __init__(self, dataset,batch_size, shuffle=True, drop_last= True, keyname= ''):
        
        assert keyname, "No keyname provided for dataset"
        self.keyname = keyname
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset # {trg:[many tensors], src:[many tensors]}
        self.drop_last = drop_last
        
        self.get_indices()
        self.create_pooled_indices()
        self.create_batches()

    def get_indices(self):

        # get the indices and length
        self.indices = [(i, s.shape[0]) for i, s in enumerate(self.dataset[self.keyname])] # assuming first elemet of datset is text
        # self.indices creates a list of tuples, where first element is index in dataset and second is 
        # number of tokens in text for below datasetexample example self.indices=  [(0, 4), (1, 3)]
            
    def create_pooled_indices(self):
        if self.shuffle:
            random.shuffle(self.indices) # randomly shuffling this list  [(0, 4), (1, 3)]

        pooled_indices = []
        for i in range(0, len(self.indices), self.batch_size*100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size*100], key=lambda x: x[1]))
        #this is pooled_indices = [(85, 46), (74, 63), (12, 70)...]
        self.pooled_indices = [x[0] for x in pooled_indices]
        # print (self.pooled_indices[:100]) # [85, 74, 12, 24..]
        
    def create_batches(self):
        
        self.batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]
        
        if self.drop_last:
            if len(self.dataset[self.keyname]) % self.batch_size == 0: # self.dataset is just train_dp_list or valid_dp_list
                pass
            else:
                self.batches.pop()

        if self.shuffle:
            random.shuffle(self.batches)  
            
    def __iter__(self):
        for batch in self.batches:          
            yield batch
        
    
class BatchSamplerSimilarLength(Sampler):
    """
    dataloader gets fed the dataset - like train_data. The same dataset gets fed to this batchsampler along with batchsize and tokensizer
    to group inputs of similar seq lens.
    
    Returns indexes of dataset ( train_data for example) , where number of indexes = batchsize.
    The dataloader then uses these indexes and creates a batch of those entries from the dataset and sends to collate.
    
    """
    
    def __init__(self, dataset, batch_size,indices=None, shuffle=True, tokenizer = None, drop_last= True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.indices = indices
        self.tokenizer = tokenizer
        self.drop_last = drop_last
        
        self.get_indices()
        self.create_pooled_indices()
        self.create_batches()

    def get_indices(self):

        # if indices are passed, then use only the ones passed 
        if self.indices is None:
            # get the indices and length
            self.indices = [(i, len(self.tokenizer(s[0]))) for i, s in enumerate(self.dataset)] # assuming first elemet of datset is text
            # self.indices creates a list of tuples, where first element is index in dataset and second is 
            # number of tokens in text for below datasetexample example self.indices=  [(0, 4), (1, 3)]
            
    def create_pooled_indices(self):
        
        if self.shuffle:
            random.shuffle(self.indices) # randomly shuffling this list  [(0, 4), (1, 3)]

        pooled_indices = []
        # create pool of indices with similar lengths
        # since in test below are only using dataset of 200 texts - it will create only one pool as evidenced by value
        # of i =0, in this pool all self.indices are sorted by their second tuple - the length in increasing order
        # example [(85, 46), (74, 63), (12, 70)...] where 85 is the index in dataset and 46 is number of tokens.
        # this is pooled_indices = [(85, 46), (74, 63), (12, 70)...]
        
        for i in range(0, len(self.indices), self.batch_size*100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size*100], key=lambda x: x[1]))
#        print ("pooled_indices" ,pooled_indices[:100])    
        
        # here were simply taking this sorted pools , and creating a new list with just the index numbers
        # this result is a list with index numbers that represent sorted text lens
        self.pooled_indices = [x[0] for x in pooled_indices]
#        print (self.pooled_indices[:100]) # [85, 74, 12, 24..]
        
    def create_batches(self):
        
                # yield indices for current batch
        self.batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]
        
#        print ("batches", batches[0]) # this creates a list of lists, each element is a list of size  = batch, containing 
        # the list of indexes that should all correspond to closely matching token lengths. Again these are indexes of
        # list in original dataset that was passed to this class.
        
        ######################## drop last if not size of batch##############
        
        if self.drop_last:
            if len(self.dataset) % self.batch_size == 0: # self.dataset is just train_dp_list or valid_dp_list
                pass
            else:
                self.batches.pop()

        if self.shuffle:
            random.shuffle(self.batches)
        
    def __iter__(self):

        for batch in self.batches:          
            yield batch # return a list of indices that of size batchsize and these indices should be corredponding to 
            # indexes of datatset that return similar sized texts

            

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size

class Net(object):
    
    def __init__(self, logfile='/home/arnab/notebooks/ml/lregressions/runlogs',print_console =True,logtodisk=False):
        
        self.regress = False
        self.multiclass = False
        self.biclass = False
        self.savebest=False
        self.logger = log_setup('INFO',logtodisk,logfile)
        self.prntconsole=print_console
        self.chkptepoch = False
        self.savestart = False
        self.saveend =  False
        self.register_forward_callbacks = [] # append to this only after self.net has been instantiated
        self.register_backward_callbacks = []
       
            
    def setup_save(self, savedir = '', savestart = False, saveend = False):
        self.savestart = savestart
        self.saveend = saveend
        self.savedir =  savedir
        assert len(self.savedir) >0 , "No save directory entered...exiting"
        
    def saveModel(self, model = None,filename =''):
        if model:
            torch.save(model,self.savedir+filename+str(datetime.datetime.now())) 
        else:
            self.savedmodel ={}
            self.savedmodel['net'] = self.net.state_dict()
            self.savedmodel['opt'] = self.optimizer.state_dict()
            torch.save(self.savedmodel,self.savedir+filename+str(datetime.datetime.now()))
            
    def HFsavemodel(self,model,filedir=""):
        """
        model here needs to be a HF model , only then it has attribute of save_pretrained
        saves weights
        """
        assert filedir, "Enter file directory" 
        model.save_pretrained(filedir)
        
    def HFloadmodel(self,filedir=""):
        """
        mostly a example here
        loads with weights
        """
        model = AutoModelForMaskedLM.from_pretrained(filedir)
        # DistilBertModel.from_pretrained("")
        return model
   
    def HFloadpretrainedmodel_from_dict(self, cfgdir='',dictdir =''):
        
        """
        THis is using the example where i loaded pretrained urlBERT.
        Note this does not load tokenizer
        
        """
        
        config_kwargs = {
                            "cache_dir": None,
                            "revision": 'main',
                            "use_auth_token": None,
                            "hidden_dropout_prob": 0.2,
                            "vocab_size": 5000,
                        }

        config = AutoConfig.from_pretrained("/content/drive/MyDrive/Colab Notebooks/My NN stuff/Modelsaves/urlBERTconfig.json",
                                            **config_kwargs)
  

        bert_model = AutoModelForMaskedLM.from_config(config=config)

 
        bert_model.resize_token_embeddings(config_kwargs["vocab_size"])
        # loads the dict file saved by torch.save command

        bert_dict = torch.load("/content/drive/MyDrive/Colab Notebooks/My NN stuff/Modelsaves/urlBERT.pt", map_location='cpu')

        bert_model.load_state_dict(bert_dict)
        return bert_model
        
        
    def setupCuda(self):
        print ("Is GPU available ? : ", torch.cuda.is_available())  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        
    def configureNetwork(self,confobj =None, networktest = True, vae = None, RNN= False, rnnhc =False, conv1D = False, packed = True):
        """
        confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        """
        if confobj and RNN:
            if rnnhc:
                self.net = RNNhc(confobj =confobj)
            elif packed:                
                #self.net = RNN_classification(confobj =confobj)
                self.net = RNNpacked(confobj =confobj)
            else:
                self.net = RNN_classification(confobj =confobj)
            
        elif confobj and conv1D:
            self.net = CNN1D(confobj =confobj)

        elif vae:
            self.net = VAE(encoder=vae['encoder'], decoder = vae['decoder'],fcin=vae['fcin'],zdims=vae['zdims'])
        else:
            self.net = ANN(confobj =confobj) # anything in ANN class can now be accessed with this self.net
            
        if networktest:
            if vae:
                self.network_test(VAE=True)
            else:
                self.network_test()
            

    def network_test(self, randinput = None, verbose = False, VAE=False, RNN = False):
        
        if randinput:
            tmpx = randinput
        else:
            tmpx, _ = next(iter(self.train_loader))
#            tmpx = self.train_data[:2] # taking two images , for example (2,3,64,64) - 2 otherwise error on batchnorm
            if not RNN:
                tmpx = tmpx[:2]

        y = self.net(tmpx)
        print ("#####Network######")
        print (self.net)
        print ("#######Running Network Test#######")
        print ('Shape of input', tmpx.detach().shape)
        if verbose:
            print ('Network input', tmpx.detach())
        if VAE:
            print ('Shape of output', y[3].detach().shape)
        else:
            print ('Shape of output', y.detach().shape)
        if verbose:
            if VAE:
                print ('Network output', y[3].detach())
            else:
                print ('Network output', y.detach())
        # do not need to reinit starting weights of net since random forward pass did not update starting weights 
        
    def memory_estimate(self):
        #https://medium.com/@baicenxiao/some-basic-knowledge-of-llm-parameters-and-memory-estimation-b25c713c3bd8
        # inference or simply loading up a trained model  = number of model params * size of each param
        # for training other factors come into play =  back prop, adam , batch size, memory is expected to be 4 times just the model
        self.param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        
    
    
    def forwardcallbacks(self):
        if self.register_forward_callbacks:
            for x in self.register_forward_callbacks:
                x()
        else:
            return None
        
    def backwardcallbacks(self):
        if self.register_backward_callbacks:
            for x in self.register_backward_callbacks:
                x()
        else:
            return None        
        
    def setup_checkpoint(self,epoch=50,loss=None, acc=None, 
                         checkpointpath='/home/arnab/notebooks/ml/lregressions/checkpointsaves/'):
        
        """
        chkptepoch is interval to save checkpoints
        checkpointpath is the path
        crieteria  = chkptloss or chkptacc boolean value to decide
        three checkpoint dictionaries which overwrite checkpt1, checkpt2, checkpt3
        """
        
        self.chkptepoch =  epoch # 1 will do every epoch, 2 will do 2, 4 and so on..
        self.checkpointpath = checkpointpath
        
        if loss:
            self.chkptloss = True
            self.chkptacc = False
            self.chkptlossval = loss
            
            self.checkpt1 = {'epoch': None,'net': None,'opt': None,'loss': loss +10}
            self.checkpt2 = {'epoch': None,'net': None,'opt': None,'loss': loss +10}
            self.checkpt3 = {'epoch': None,'net': None,'opt': None,'loss': loss +10}
        elif acc:
            self.chkptloss = False
            self.chkptacc = True
            self.chkptaccval = acc  
            self.checkpt1 = {'epoch': None,'net': None,'opt': None,'acc': acc -10}
            self.checkpt2 = {'epoch': None,'net': None,'opt': None,'acc': acc -10}
            self.checkpt3 = {'epoch': None,'net': None,'opt': None,'acc': acc -10}
        else:
            self.chkptloss = False
            self.chkptacc = False  
            self.chkptepoch =  False
        

    def loadModelfromdisk (self,modeldir='',optimizerdir=''):

        self.net.load_state_dict(torch.load(modeldir))
        optimizer.load_state_dict(torch.load(optimizerdir))
    
    def checkpoint(self,epoch=0):
        
        """
        Max of three based on loss/acc. min every epoch.If want just every epoch, then have loss or acc at values that def accepted.
        """
        if self.chkptloss:
            
            tmp = [self.checkpt1['loss'],self.checkpt2['loss'], self.checkpt3['loss']]
            maxis= tmp.index(max(tmp))
        
            if maxis == 0:

                if np.mean(self.batchLoss) < self.checkpt1['loss']:
                    self.checkpt1['loss'] = np.mean(self.batchLoss)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))

            elif maxis == 1:

                if np.mean(self.batchLoss) < self.checkpt2['loss']:
                    self.checkpt2['loss'] = np.mean(self.batchLoss)
                    self.checkpt2['epoch'] = epoch
                    self.checkpt2['net'] = self.net.state_dict()
                    self.checkpt2['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt2,self.checkpointpath+"checkpt2"+str(datetime.datetime.now()))

            elif maxis == 2:

                if np.mean(self.batchLoss) < self.checkpt3['loss']:
                    self.checkpt3['loss'] = np.mean(self.batchLoss)
                    self.checkpt3['epoch'] = epoch
                    self.checkpt3['net'] = self.net.state_dict()
                    self.checkpt3['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt3,self.checkpointpath+"checkpt3"+str(datetime.datetime.now()))             
            
        elif self.chkptacc:
            
            tmp = [self.checkpt1['acc'],self.checkpt2['acc'], self.checkpt3['acc']]
            minis= tmp.index(min(tmp))
            
            if minis == 0:

                if np.mean(self.batchAcc) > self.checkpt1['acc']:
                    self.checkpt1['acc'] = np.mean(self.batchAcc)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))

            elif minis == 1:

                if np.mean(self.batchAcc) > self.checkpt1['acc']:
                    self.checkpt1['acc'] = np.mean(self.batchAcc)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))


            elif minis == 2:

                if np.mean(self.batchAcc) > self.checkpt1['acc']:
                    self.checkpt1['acc'] = np.mean(self.batchAcc)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))
            
        else:
            print("No checkpoint criteria selected\n")
            return None
                
    
    def configureTraining(self,epochs=500,lossfun=nn.CrossEntropyLoss(),optimizer='adam',lr=0.01, 
                          weight_decay=0,momentum=0.9, prntsummary = False, gpu= False):
        self.gpu = gpu
        self.lossfun = lossfun
        self.lr=lr
        self.epochs=epochs
        self.weight_decay=weight_decay # typically very low. recommended to be tested as metaparam from 0.001-0.1. can go lower too.
        self.momentum=momentum # ideal value 0.9
        if optimizer =='sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        elif optimizer =='rmsprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        elif optimizer=='adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        if prntsummary:
            self.prnt_trainparams()
            
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        
        if self.gpu:
            print ("Is GPU available ? : ", torch.cuda.is_available())
            if torch.cuda.is_available():
                print ("#####Printing GPU Device Info######", '\n')
                print('ID of current CUDA device: ', torch.cuda.current_device())
                print('Name of current CUDA device is: ', torch.cuda.get_device_name(torch.cuda.current_device()))
                print('Amount of GPU memory allocated: ', torch.cuda.memory_allocated(torch.cuda.current_device()))
                print('Amount of GPU memory reserved: ', torch.cuda.memory_reserved(torch.cuda.current_device()))
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            else:
                self.gpu = False
        print("Processor device configured is: ", self.device)  

  
            
    def saveBest(self, accthreshold = 80.0, l1lossthresh=1.0, epochthreshold = 0, lossthreshold = 500, startlossval =  50000):
        
        self.savebest=True
        self.bestaccthresh = accthreshold
        self.bestl1thresh = l1lossthresh
        self.lossthreshold =  lossthreshold

        self.epochthresh = epochthreshold # starts checing to save at epoch 0 by default
        self.bestTrain = {'acc':0.0, 'net':None, 'epoch':None,'opt':None,'loss':startlossval} # net state dict is saved
        
              

    def dataIterables(self, sentences, targets, weights = {"train": 0.8, "valid": 0.1, "test": 0.1}):
        """
        sentences are a list of texts(reviews), targets are list of lists for one hot encode for multi labels or in case binary or multi class- then sholud be just a list of labels.
        in case of multilabel multiclass, the hot encoded values 0 and 1s are long ints
        """
        datapipe = []
        for i in range(len(sentences)):
            datapipe.append([sentences[i],targets[i]])
            
        N_ROWS = len(datapipe)
####### any iterable can be wrapped into iterablewrapper to create datapipe. Wrapping our list here#####
        self.datapipe = IterableWrapper(datapipe)
    
 # <class 'torch.utils.data.datapipes.iter.utils.IterableWrapperIterDataPipe'>
#        for sample in datapipe: sample is a list of size 2 , review, sentiment

        # Split into training and val datapipes early on. Will build vocabulary from training datapipe only.
        # torchdata.datapipes.iter.RandomSplitter()
        self.train_dp, self.valid_dp, self.test_dp = self.datapipe.random_split(total_length=N_ROWS, 
                                                    weights=weights, seed = 0)  
        
        
    def dataIterables_to_list(self):
        
        self.train_dp_list = list(self.train_dp)
        self.valid_dp_list = list(self.valid_dp)
        self.test_dp_list = list(self.test_dp)

                          
    def makeVocabulary(self,dp= None, tokenizer = 'spacy', specials =  ["<UNK>", "<PAD>"], max_tokens =20000, default_index = 0,min_freq =2):
        
        # find better tokensizers
        
        def yield_tokens(data_iter):
            for text, _ in data_iter:
                yield self.tokenizer(text)
                
        if tokenizer:
            self.tokenizer = get_tokenizer(tokenizer)

        if dp:
             self.vocabulary = build_vocab_from_iterator(yield_tokens(dp), specials=specials,special_first= True, 
                                      max_tokens=max_tokens, min_freq = min_freq)
        else:
            self.vocabulary = build_vocab_from_iterator(yield_tokens(self.datapipe), specials=specials,special_first= True, 
                                      max_tokens=max_tokens,  min_freq = min_freq)
        self.vocabulary.set_default_index(default_index)
        self.PADDING_VALUE=self.vocabulary['<PAD>']

    def saveVocab(self, path):
        torch.save(self.vocabulary, path)
        
    def loadVocab(self, path):
        self.vocabulary = torch.load(path)
        
    def makeLoadersAdv(self, batch_size =64, collate_fn=None, iterables_to_list = False):
        
        """
        Batchsampler is called once for each loader, as it passed the entire dataset in a single call.
        So since we are creating train, valid and test loader, it is called three time in total
        
        NOTE : collate fn is not called at all during this makeloader function. Only batchsampler is called passing entire 
        datasets and that is what is taking long. Collate fun is only called when we iter the dataloader objects of self.train_loader
        and the other two. Means its using yield a single batch at a time to ram. So running a for loop over train_loader will call
        collate each time it loops and puts its on CPU ram, which then in traing code gets sent to gpu.
        So technically can send to GPU directly from collate - which may need to do when using more workers.
        """
        
        if iterables_to_list:
            self.dataIterables_to_list()       
        if collate_fn:
            self.collate_fn = collate_fn
        else:
            self.collate_fn =  self.collate_batch
        
        self.batch_size = batch_size
        
        self.train_loader = DataLoader(self.train_dp_list, 
                          batch_sampler=BatchSamplerSimilarLength(dataset = self.train_dp_list, 
                                                                  batch_size=self.batch_size, tokenizer = self.tokenizer),
                          collate_fn=self.collate_fn)

        self.valid_loader = DataLoader(self.valid_dp_list, 
                          batch_sampler=BatchSamplerSimilarLength(dataset = self.valid_dp_list, 
                                                                  batch_size=self.batch_size,
                                                                  shuffle=False, tokenizer = self.tokenizer),
                          collate_fn=self.collate_fn)
        
        self.test_loader = DataLoader(self.test_dp_list, 
                          batch_sampler=BatchSamplerSimilarLength(dataset = self.test_dp_list, 
                                                                  batch_size=self.batch_size,
                                                                  shuffle=False, tokenizer = self.tokenizer),
                          collate_fn=self.collate_fn)
        
        
    def makeLoaders(self, data,labels,train_size=.8,shuffle=True, batch_size=32,drop_last=True,testset=True):

    #note: assumed to already be in pytorch tensors

        self.batch_size=batch_size

        self.train_data,self.test_data, self.train_labels,self.test_labels = \
                                  train_test_split(data, labels, train_size=train_size)
        if testset:
            train_data = torch.utils.data.TensorDataset(self.train_data,self.train_labels)
            test_data  = torch.utils.data.TensorDataset(self.test_data,self.test_labels)
        else:
            train_data = torch.utils.data.TensorDataset(torch.cat((self.train_data,self.test_data),0),torch.cat((self.train_labels,self.test_labels),0))
            test_data  = torch.utils.data.TensorDataset(self.test_data,self.test_labels)            

        self.train_loader = DataLoader(train_data,shuffle=shuffle,batch_size=self.batch_size,drop_last=drop_last)
        self.test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

        return self.train_loader, self.test_loader
    
    def collateseq2seqHFDatasetPack(self,batch):

        encoder_list, decoder_list = [], []
        x_lens, y_lens = [], []

        # what is received here is a batch size of a list with each element of list is a dictionary representing 
        #one example [{"trg" :one tensor, "src" :one tensor},{"trg" :one tensor, "src" :one tensor}....]

        encoder_list = [example["src"] for example in batch] # results in a list of tensors of src
        x_lens = [example["src"].shape[0] for example in batch]
        decoder_list = [example["trg"] for example in batch] # results in a list of tensors of trg
        y_lens = [example["trg"].shape[0] for example in batch]
        x_lens = torch.tensor(np.array(x_lens))
        y_lens = torch.tensor(np.array(y_lens))

        batchedsrc = pad_sequence(encoder_list, batch_first=True,padding_value=self.PADDING_VALUE)
        batchedtrg = pad_sequence(decoder_list, batch_first=True,padding_value=self.PADDING_VALUE)

        collated = {"src" : batchedsrc, "trg" : batchedtrg, "x_lens" : x_lens, "y_lens" : y_lens}

        return collated
    
    def collate_batch(self,batch):
        text_list, label_list = [], []
        for _text, _label in batch:
            processed_text = torch.tensor(self.vocabulary.lookup_indices(self.tokenizer(_text)))
            text_list.append(processed_text) 
            label_list.append(_label)
        
#        if labeltypefloat:
        label_torch = torch.tensor(np.array(label_list)).float()
#        else:
#            label_torch = torch.tensor(np.array(label_list)).long()
            
#        if minsize:    
#            text_list[0] = torch.nn.ConstantPad1d((0,minsize-text_list[0].shape[0]), self.PADDING_VALUE)(text_list[0]) 
    # above line is implementing a min size.. to cope with downsampling in conv. 53 since see above note. also 1 is the
    # the padded value. taking first element of every batch and making it of min size
        return pad_sequence(text_list, batch_first=True,padding_value=self.PADDING_VALUE), label_torch
        
        
    def embedvec_to_vocabtokn(self, vecmat,toknvec,vocabulary):   
        """vecmat = embeddings matrix
        tokenvec = vector for token
        give a embeded vector for a token , and get the token back
        """
        tokenls = []
        
        for i in torch.where((toknvec == vecmat).all(dim=1))[0].detach().numpy().tolist():
            tokenls.append(self.vocabulary.get_itos()[i])
            
        
        return tokenls
    
    def text_to_vocabidx(self, sentences):
        """
        here sentences is a list of numpy array of text
        """
        
        return [self.vocabulary.lookup_indices(self.tokenizer(i)) for i in sentences]
        
    
    def trainVAE(self,batchlogs=True,testset=True,verbose=False, lossreduction = False):
        """
        Note that self.net is returing a tuple and not just yhat
        """

        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
            
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        if self.savestart:
            self.saveModel(filename='start')
            
            
        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,_) in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = X.to(self.device) # moved to gpu

                z,z_mean,z_log_var,yHat = self.net(X)
                
                # total loss = reconstruction loss + KL divergence
                kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) # sum over latent dimension
                batchsize = kl_div.size(0) # from seba
#                print ('kldiv before mean', kl_div.shape)
                kl_div = kl_div.mean()
    
#                print ('kldiv after mean', kl_div.shape,kl_div )
                
                if lossreduction:
                    reconstructloss = self.lossfun(yHat,X)
#                    print("reconstructloss", reconstructloss.shape, reconstructloss)
                    self.loss = reconstructloss + kl_div
#                    print("selfloss", self.loss.shape, self.loss)
                else:               
                    pixelwise = self.lossfun(yHat,X)
    #                print ('pixelwise', pixelwise.shape)
                    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
    #                print ('pixelwise', pixelwise.shape, pixelwise)
                    pixelwise = pixelwise.mean() # average over batch dimension
    #                print ('pixelwise', pixelwise.shape, pixelwise)
                    self.loss = pixelwise + kl_div
    #                print ("loss", self.loss.shape, self.loss)

                ###########################

                    
                self.forwardcallbacks() 
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                X = X.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                                        ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy(abs error) is %f and loss is %f"%(epochi,tmpmeanbatchacc,tmpmeanbatchloss))
                                                                                                     
                if self.prntconsole:
                    
                    print("##Epoch %d averaged batch training accuracy(abs error) is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                
                
            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            """
            if testset:
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to 
                    
                    X = X.to(self.device)
                    z,z_mean,z_log_var,predictions = self.net(X)
                    
                    predictions = predictions.cpu()
                    X = X.cpu()   
                    
                    tmptestloss = self.lossfun(predictions,X).item()
                    tmptestacc = torch.mean(torch.abs(predictions.detach()-X.detach())).item()

                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)

                    
                    if batchlogs: 
                        
                        self.logger.info("##Epoch %d Test accuracy(abs error) is %f and loss is %f"%( epochi,tmptestacc,tmptestloss))
                        if self.prntconsole:
                            print("##Epoch %d Test accuracy(abs error) is %f and loss is %f"%( epochi,tmptestacc, tmptestloss))
            """
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        if self.saveend:
            self.saveModel(filename='End')                           
            if self.savebest:          
                self.saveModel(model = self.bestTrain, filename='bestTrain')           
                           
        return self.trainAcc, self.testAcc,  self.losses,self.testloss                    
                    
    def trainVAEHFds(self,batchlogs=True,testset=True,verbose=False, Xkey = "hidden_state",lossreduction = False):
        """
        Note that self.net is returing a tuple and not just yhat
        """

        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
            
        self.trainAcc = []
        self.reconstructionloss = []
        self.kldiv = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        if self.savestart:
            self.saveModel(filename='start')
            
            
        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            self.batchkl = []
            self.reconstruct = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = batch[Xkey]
                X = X.to(self.device) # moved to gpu

                z,z_mean,z_log_var,yHat = self.net(X)
                
                # total loss = reconstruction loss + KL divergence
                kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) # sum over latent dimension
                batchsize = kl_div.size(0) # from seba
#                print ('kldiv before mean', kl_div.shape)
                kl_div = kl_div.mean()
    
#                print ('kldiv after mean', kl_div.shape,kl_div )
                
                if lossreduction:
                    reconstructloss = self.lossfun(yHat,X)
#                    print("reconstructloss", reconstructloss.shape, reconstructloss)
                    self.loss = reconstructloss + kl_div
#                    print("selfloss", self.loss.shape, self.loss)
                else:               
                    pixelwise = self.lossfun(yHat,X)
    #                print ('pixelwise', pixelwise.shape)
                    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
    #                print ('pixelwise', pixelwise.shape, pixelwise)
                    pixelwise = pixelwise.mean() # average over batch dimension
    #                print ('pixelwise', pixelwise.shape, pixelwise)
                    self.loss = pixelwise + kl_div
    #                print ("loss", self.loss.shape, self.loss)

                ###########################

                    
                self.forwardcallbacks() 
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                X = X.cpu()
                
                self.loss = self.loss.cpu()
                reconstructloss = reconstructloss.cpu()
                kl_div =  kl_div.cpu()
                
                
                tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                self.batchkl.append(kl_div.item()) 
                self.reconstruct.append(reconstructloss.item())
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                    
                    self.logger.info("Reconstruction loss is %f and KL div is %f "% (reconstructloss,kl_div))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")
                        
                        print ("Reconstruction loss is %f and KL div is %f "% (reconstructloss,kl_div))
                                        ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            self.reconstructionloss.append(np.mean(self.reconstruct))
            self.kldiv.append(np.mean(self.batchkl))
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy(abs error) is %f and loss is %f"%(epochi,tmpmeanbatchacc,tmpmeanbatchloss))
                                                                                                     
                if self.prntconsole:
                    
                    print("##Epoch %d averaged batch training accuracy(abs error) is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))

            if testset:
                
                self.net.eval()
                with torch.no_grad():
                    
                    self.batchtestloss = []
                    self.batchtestAcc = []
                    
                    for batchidx, batch in enumerate(self.test_loader):
                        X = batch[Xkey]
                        X = X.to(self.device) # moved to gpu

                        z,z_mean,z_log_var,yHat = self.net(X)

                        # total loss = reconstruction loss + KL divergence
                        kl_div = -0.5 * torch.sum(1 + z_log_var 
                                              - z_mean**2 
                                              - torch.exp(z_log_var), 
                                              axis=1) # sum over latent dimension
                        batchsize = kl_div.size(0) # from seba
                        kl_div = kl_div.mean()

                        if lossreduction:
                            reconstructloss = self.lossfun(yHat,X)
        #                    print("reconstructloss", reconstructloss.shape, reconstructloss)
                            self.loss = reconstructloss + kl_div
#                    print("selfloss", self.loss.shape, self.loss)
                        else:               
                            pixelwise = self.lossfun(yHat,X)
            #                print ('pixelwise', pixelwise.shape)
                            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            #                print ('pixelwise', pixelwise.shape, pixelwise)
                            pixelwise = pixelwise.mean() # average over batch dimension
            #                print ('pixelwise', pixelwise.shape, pixelwise)
                            self.loss = pixelwise + kl_div
    #                print ("loss", self.loss.shape, self.loss)

                        yHat = yHat.cpu()
                        X = X.cpu()
                        self.loss = self.loss.cpu()
                    
                        tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                
                        self.batchtestAcc.append(tmpacc)
                        self.batchtestloss.append(self.loss.item())

                        if batchlogs:

                            self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                             "TEST batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))

                            if self.prntconsole:
                                print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "TEST batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                                                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchtestAcc)
            tmpmeanbatchloss = np.mean(self.batchtestloss)

            self.testAcc.append(tmpmeanbatchacc)
            self.testloss.append(tmpmeanbatchloss)
                
                
                
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        if self.saveend:
            self.saveModel(filename='End')                           
            if self.savebest:          
                self.saveModel(model = self.bestTrain, filename='bestTrain')           
                           
        return self.trainAcc, self.testAcc,  self.losses, self.testloss, self.reconstructionloss, self.kldiv                    
                    


    
    def trainAE(self,batchlogs=True,testset=True,verbose=False):       

        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
            
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        if self.savestart:
            self.saveModel(filename='start')

            
        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = X.to(self.device) # moved to gpu

                yHat = self.net(X)
                self.loss = self.lossfun(yHat,X)
                    
                self.forwardcallbacks() 
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                X = X.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                                        ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy(abs error) is %f and loss is %f"%(epochi,tmpmeanbatchacc,tmpmeanbatchloss))
                                                                                                     
                if self.prntconsole:
                    
                    print("##Epoch %d averaged batch training accuracy(abs error) is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                
                
            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to 
                    
                    X = X.to(self.device)
                    predictions = self.net(X)
                    
                    predictions = predictions.cpu()
                    X = X.cpu()   
                    
                    tmptestloss = self.lossfun(predictions,X).item()
                    tmptestacc = torch.mean(torch.abs(predictions.detach()-X.detach())).item()

                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)

                    
                    if batchlogs: 
                        
                        self.logger.info("##Epoch %d Test accuracy(abs error) is %f and loss is %f"%( epochi,tmptestacc,tmptestloss))
                        if self.prntconsole:
                            print("##Epoch %d Test accuracy(abs error) is %f and loss is %f"%( epochi,tmptestacc, tmptestloss))
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')        
        return self.trainAcc, self.testAcc,  self.losses,self.testloss 
    
    def trainseq2seqHFdatasetpacked(self,teacher_forcing = 1,clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
    
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in tqdm.tqdm(range(self.epochs)):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                X = batch["src"]  
                y = batch["trg"]
                x_lens = batch["x_lens"]
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                
                yHat = self.net(X,x_lens,y, teacher_forcing =  teacher_forcing)
                
                #y/trg = [ batch size, trg len]
                # after permute y/trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]                
                
                y = y.permute(1,0) # after permute trg = [trg len, batch size]
                yHat_dim = yHat.shape[-1] # yhat_dim = output_dim = number of classes of size of trg/english vocab        
                yHat = yHat[1:].view(-1, yHat_dim) # [(trg len - 1) * batch size, output dim]. Also :1 is to ignore the zero on first index
                y = y[1:].flatten() # [(trg len - 1) * batch size] , :1 is to ignore sos in first index
                self.loss = self.lossfun(yHat,y)
                #N batch size
                #C number of classes
                # cross entropy 
                #input shape (C), (N,C)
                # target (), (N)
                #loss is scalar when reduction is present.

                
                    
                self.forwardcallbacks() # not sure how these will work since they refer to net params
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for mutlinomial classification problem
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                # detach from yhat ?
                self.batchLoss.append(self.loss.item())
                
                #Explanation and how crossentropy is working here - IMPORTANT:
                # additional note : first looking back at worked out examples, there is no softmax - logits ( outputs from fcn ) are 
                # directly input to cross entropy loss, again logits dont need to be positive or sum to 1.
                # logits are basically values for each of the classes. So if we have 3 classes, and batch =1, logits fed to crossentropy
                # will be of size [1,3], the 3 colums signifying the class index of each class and we typically want to take the 
                # max value and check its index, and then verify is that index matches the target. So the target( or y) will be the class
                # index and in this example of size ([1]). So aboveself.lossfun(yHat,y) is cross_entropy(size[1,3], size[1]) and
                # it works on class indices!!
                
                
                # what the crossentropyloss seems to be doing is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
                
                #For whats happening in accuracy calculation
                # consider iris where output is 3. first yhat for a single input row is three o/p , each standing for the three classes, and 
                # the values are the logits for each
                #second since this is a batch not a row of input, yhat will be a matrix of same rows of batch and three cols
                #so accuracy compute goes as follows - first you find the col index number which has max value for each row in yhat. This                   #correspond to which col( class) has max value.Second you check if this col index matches ( 0-2 in case of iris) the ground
                #truth y ( whose labels were also made 0-2). if it does then it returns true which is converted to float 1 ( false is 0).                   #Third you average out the number of matches ( 1's) over this batch by taking mean and then multipley by 100. append this                   #value in batchac list as that batches averaged accuracy
                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
#                    self.testloss = []
#                    self.testAcc = []

                    tmpbatchloss = [] # to hold values for each batch
                    tmpbatchacc = []
                    for batchidx, batch in enumerate(self.test_loader):
                        X = batch["src"]
                        y = batch["trg"]
                        x_lens = batch["x_lens"]

                        X = X.to(self.device) # moved to gpu
                        y = y.to(self.device)

                        pred = self.net(X,x_lens,y,teacher_forcing = 0) # no teacher forcing during validation/test


                        y = y.permute(1,0)
                        pred_dim = pred.shape[-1] # yhat_dim = number of classes or size of english vocab        
                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() # [(trg len - 1) * batch size] , :1 is to ignore sos in first index

                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    # comparing two tensors predlabel(yhat) and y of say size 30 ( 30 rows/test samples in X)
                    #which are basically the labels of these each row of X testdata -size 30
                    # mean counts the number of times predlabel and y are equal and divdes them by number of samples 30
                    tmpmeanbatchloss = np.mean(tmpbatchloss) # take mean of all batches
                    tmpmeanbatchacc = np.mean(tmpbatchacc) # take mean of all batches
                    
                    self.testloss.append(tmpmeanbatchloss) # add mean value for this epoch
                    self.testAcc.append(tmpmeanbatchacc) # add mean value for this epoch
                    
                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                        
                        if self.prntconsole:
        
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
          
        return self.trainAcc,self.testAcc, self.losses
        # for accuracy per label/class see metaparams_multioutput notebook

    def trainseq2seqHFdataset(self,teacher_forcing = 1,clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in tqdm.tqdm(range(self.epochs)):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                X = batch["src"] # TODO give generic label 
                y = batch["trg"] # TODO give generic label 
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                
                yHat = self.net(X,y, teacher_forcing =  teacher_forcing)
                
                #y/trg = [ batch size, trg len]
                # after permute y/trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]                
                
                y = y.permute(1,0) # after permute trg = [trg len, batch size]
                yHat_dim = yHat.shape[-1] # yhat_dim = output_dim = number of classes of size of trg/english vocab        
                yHat = yHat[1:].view(-1, yHat_dim) # [(trg len - 1) * batch size, output dim]. Also :1 is to ignore the zero on first index
                y = y[1:].flatten() # [(trg len - 1) * batch size] , :1 is to ignore sos in first index
                self.loss = self.lossfun(yHat,y)
                #N batch size
                #C number of classes
                # cross entropy 
                #input shape (C), (N,C)
                # target (), (N)
                #loss is scalar when reduction is present.

                
                    
                self.forwardcallbacks() # not sure how these will work since they refer to net params
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for mutlinomial classification problem
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                # detach from yhat ?
                self.batchLoss.append(self.loss.item())
                
                #Explanation and how crossentropy is working here - IMPORTANT:
                # additional note : first looking back at worked out examples, there is no softmax - logits ( outputs from fcn ) are 
                # directly input to cross entropy loss, again logits dont need to be positive or sum to 1.
                # logits are basically values for each of the classes. So if we have 3 classes, and batch =1, logits fed to crossentropy
                # will be of size [1,3], the 3 colums signifying the class index of each class and we typically want to take the 
                # max value and check its index, and then verify is that index matches the target. So the target( or y) will be the class
                # index and in this example of size ([1]). So aboveself.lossfun(yHat,y) is cross_entropy(size[1,3], size[1]) and
                # it works on class indices!!
                
                
                # what the crossentropyloss seems to be doing is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
                
                #For whats happening in accuracy calculation
                # consider iris where output is 3. first yhat for a single input row is three o/p , each standing for the three classes, and 
                # the values are the logits for each
                #second since this is a batch not a row of input, yhat will be a matrix of same rows of batch and three cols
                #so accuracy compute goes as follows - first you find the col index number which has max value for each row in yhat. This                   #correspond to which col( class) has max value.Second you check if this col index matches ( 0-2 in case of iris) the ground
                #truth y ( whose labels were also made 0-2). if it does then it returns true which is converted to float 1 ( false is 0).                   #Third you average out the number of matches ( 1's) over this batch by taking mean and then multipley by 100. append this                   #value in batchac list as that batches averaged accuracy
                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
#                    self.testloss = []
#                    self.testAcc = []

                    tmpbatchloss = [] # to hold values for each batch
                    tmpbatchacc = []
                    for batchidx, batch in enumerate(self.test_loader):
                        X = batch["src"]
                        y = batch["trg"]

                        X = X.to(self.device) # moved to gpu
                        y = y.to(self.device)

                        pred = self.net(X,y,teacher_forcing = 0) # no teacher forcing during validation/test


                        y = y.permute(1,0)
                        pred_dim = pred.shape[-1] # yhat_dim = number of classes or size of english vocab        
                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() # [(trg len - 1) * batch size] , :1 is to ignore sos in first index

                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    # comparing two tensors predlabel(yhat) and y of say size 30 ( 30 rows/test samples in X)
                    #which are basically the labels of these each row of X testdata -size 30
                    # mean counts the number of times predlabel and y are equal and divdes them by number of samples 30
                    tmpmeanbatchloss = np.mean(tmpbatchloss) # take mean of all batches
                    tmpmeanbatchacc = np.mean(tmpbatchacc) # take mean of all batches
                    
                    self.testloss.append(tmpmeanbatchloss) # add mean value for this epoch
                    self.testAcc.append(tmpmeanbatchacc) # add mean value for this epoch
                    
                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                        
                        if self.prntconsole:
        
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
          
        return self.trainAcc,self.testAcc, self.losses
        # for accuracy per label/class see metaparams_multioutput notebook

        
    def trainseq2seqSelfAttnHFdataset(self, clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in tqdm.tqdm(range(self.epochs)):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                
                # forward pass and loss
                X = batch["src"] # TODO give generic label 
                y = batch["trg"] # TODO give generic label 
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                
                #y/trg = [ batch size, trg len]
                #output/yhat = [ batch size, seq len, output dim(vocab)]   
                
                yHat, _ = self.net(X, y[:,:-1])
                 # i guess the way the network and masks are setup we want to remove the eos from y before sending to net, so our model predicts eos(hopefully), but does not take eos as an input for next token prediction. So if y = [sos, x1, x2...xn, eos], y[:-1] = [sos,x1,...xn]
                    
                # yhat created here will not have sos and will look like hopefully with eos yhat = [y1, y2..yn,eos]
                # then to caluclate loss we need to take sos out of y so y[1:] = [x1, x2..xn, eos]
                
                yHat_dim = yHat.shape[-1] # output dim(vocab)
                
                yHat = yHat.contiguous().view(-1, yHat_dim) # [seq len * batch size, output dim]
                y = y[:,1:].contiguous().view(-1) # [(trg len - 1) * batch size] the 1: is to ignore sos in first index
                

                self.loss = self.lossfun(yHat,y)
                #N batch size
                #C number of classes
                # cross entropy 
                #input shape (C), (N,C)
                # target (), (N)
                #loss is scalar when reduction is present.

                    
                self.forwardcallbacks() # not sure how these will work since they refer to net params
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for mutlinomial classification problem
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                # detach from yhat ?
                self.batchLoss.append(self.loss.item())
                
                #Explanation and how crossentropy is working here - IMPORTANT:
                # additional note : first looking back at worked out examples, there is no softmax - logits ( outputs from fcn ) are 
                # directly input to cross entropy loss, again logits dont need to be positive or sum to 1.
                # logits are basically values for each of the classes. So if we have 3 classes, and batch =1, logits fed to crossentropy
                # will be of size [1,3], the 3 colums signifying the class index of each class and we typically want to take the 
                # max value and check its index, and then verify is that index matches the target. So the target( or y) will be the class
                # index and in this example of size ([1]). So aboveself.lossfun(yHat,y) is cross_entropy(size[1,3], size[1]) and
                # it works on class indices!!
                
                
                # what the crossentropyloss seems to be doing is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
                
                #For whats happening in accuracy calculation
                # consider iris where output is 3. first yhat for a single input row is three o/p , each standing for the three classes, and 
                # the values are the logits for each
                #second since this is a batch not a row of input, yhat will be a matrix of same rows of batch and three cols
                #so accuracy compute goes as follows - first you find the col index number which has max value for each row in yhat. This                   #correspond to which col( class) has max value.Second you check if this col index matches ( 0-2 in case of iris) the ground
                #truth y ( whose labels were also made 0-2). if it does then it returns true which is converted to float 1 ( false is 0).                   #Third you average out the number of matches ( 1's) over this batch by taking mean and then multipley by 100. append this                   #value in batchac list as that batches averaged accuracy
                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
#                    self.testloss = []
#                    self.testAcc = []

                    tmpbatchloss = [] # to hold values for each batch
                    tmpbatchacc = []
                    for batchidx, batch in enumerate(self.test_loader):
                
                        X = batch["src"]
                        y = batch["trg"]

                        X = X.to(self.device) # moved to gpu
                        y = y.to(self.device)
                        
                        

                        pred,_ = self.net(X,y) # no teacher forcing during validation/test


                   
                        pred_dim = pred.shape[-1] # yhat_dim = number of classes or size of english vocab        
                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() # [(trg len - 1) * batch size] , :1 is to ignore sos in first index

                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    # comparing two tensors predlabel(yhat) and y of say size 30 ( 30 rows/test samples in X)
                    #which are basically the labels of these each row of X testdata -size 30
                    # mean counts the number of times predlabel and y are equal and divdes them by number of samples 30
                    tmpmeanbatchloss = np.mean(tmpbatchloss) # take mean of all batches
                    tmpmeanbatchacc = np.mean(tmpbatchacc) # take mean of all batches
                    
                    self.testloss.append(tmpmeanbatchloss) # add mean value for this epoch
                    self.testAcc.append(tmpmeanbatchacc) # add mean value for this epoch
                    
                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                        
                        if self.prntconsole:
        
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
          
        return self.trainAcc,self.testAcc, self.losses
        # for accuracy per label/class see metaparams_multioutput notebook




    def trainseq2seq(self,teacher_forcing = 1,clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,x_lens,y,y_lens) in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                
                yHat = self.net(X,x_lens,y, teacher_forcing =  teacher_forcing)
                
                #y/trg = [ batch size, trg len]
                # after permute y/trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]                
                
                y = y.permute(1,0) # after permute trg = [trg len, batch size]
                yHat_dim = yHat.shape[-1] # yhat_dim = output_dim = number of classes of size of trg/english vocab        
                yHat = yHat[1:].view(-1, yHat_dim) # [(trg len - 1) * batch size, output dim]. Also :1 is to ignore the zero on first index
                y = y[1:].flatten() # [(trg len - 1) * batch size] , :1 is to ignore sos in first index
                self.loss = self.lossfun(yHat,y)
                #N batch size
                #C number of classes
                # cross entropy 
                #input shape (C), (N,C)
                # target (), (N)
                #loss is scalar when reduction is present.

                
                    
                self.forwardcallbacks() # not sure how these will work since they refer to net params
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for mutlinomial classification problem
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                # detach from yhat ?
                self.batchLoss.append(self.loss.item())
                
                #Explanation and how crossentropy is working here - IMPORTANT:
                # additional note : first looking back at worked out examples, there is no softmax - logits ( outputs from fcn ) are 
                # directly input to cross entropy loss, again logits dont need to be positive or sum to 1.
                # logits are basically values for each of the classes. So if we have 3 classes, and batch =1, logits fed to crossentropy
                # will be of size [1,3], the 3 colums signifying the class index of each class and we typically want to take the 
                # max value and check its index, and then verify is that index matches the target. So the target( or y) will be the class
                # index and in this example of size ([1]). So aboveself.lossfun(yHat,y) is cross_entropy(size[1,3], size[1]) and
                # it works on class indices!!
                
                
                # what the crossentropyloss seems to be doing is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
                
                #For whats happening in accuracy calculation
                # consider iris where output is 3. first yhat for a single input row is three o/p , each standing for the three classes, and 
                # the values are the logits for each
                #second since this is a batch not a row of input, yhat will be a matrix of same rows of batch and three cols
                #so accuracy compute goes as follows - first you find the col index number which has max value for each row in yhat. This                   #correspond to which col( class) has max value.Second you check if this col index matches ( 0-2 in case of iris) the ground
                #truth y ( whose labels were also made 0-2). if it does then it returns true which is converted to float 1 ( false is 0).                   #Third you average out the number of matches ( 1's) over this batch by taking mean and then multipley by 100. append this                   #value in batchac list as that batches averaged accuracy
                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
#                    self.testloss = []
#                    self.testAcc = []

                    tmpbatchloss = [] # to hold values for each batch
                    tmpbatchacc = []
                    for batchidx, (X,x_lens,y,y_lens) in enumerate(self.test_loader):

                        X = X.to(self.device) # moved to gpu
                        y = y.to(self.device)

                        pred = self.net(X,x_lens,y,teacher_forcing = 0) # no teacher forcing during validation/test


                        y = y.permute(1,0)
                        pred_dim = pred.shape[-1] # yhat_dim = number of classes or size of english vocab        
                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() # [(trg len - 1) * batch size] , :1 is to ignore sos in first index

                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    # comparing two tensors predlabel(yhat) and y of say size 30 ( 30 rows/test samples in X)
                    #which are basically the labels of these each row of X testdata -size 30
                    # mean counts the number of times predlabel and y are equal and divdes them by number of samples 30
                    tmpmeanbatchloss = np.mean(tmpbatchloss) # take mean of all batches
                    tmpmeanbatchacc = np.mean(tmpbatchacc) # take mean of all batches
                    
                    self.testloss.append(tmpmeanbatchloss) # add mean value for this epoch
                    self.testAcc.append(tmpmeanbatchacc) # add mean value for this epoch
                    
                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                        
                        if self.prntconsole:
        
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc, tmpmeanbatchloss))
                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
          
        return self.trainAcc,self.testAcc, self.losses
        # for accuracy per label/class see metaparams_multioutput notebook
        
        
        
    def trainDistilbertMask(clipping =0, Xkey='input_ids',attnkey = 'attention_mask', ykey='labels'):

        self.trainperplex = []
        self.losses   = []

        starttime = time.time()

        self.net.to(self.device) # moved to gpu

        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in range(self.epochs):

            self.net.train() #flag to turn on train

            batchAcc = []
            batchLoss = []

            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

            for batchidx, batch in enumerate(self.train_loader):

                self.net.train()
                # forward pass and loss

                X = batch[Xkey]  

                attn_mask = batch[attnkey] 

                y = batch[ykey] 

                X = X.to(self.device) 
                attn_mask = attn_mask.to(self.device)
                y = y.to(self.device) 

                outputs = net(X,attention_mask= attn_mask,labels = y)

                self.loss = outputs.loss # HF model output for masked LM returns loss as an attribute

                self.optimizer.zero_grad()

                self.loss.backward()

                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)

                self.optimizer.step()

                self.loss = loss.cpu()

                batchLoss.append(self.loss.item())

                print ('At Batchidx %d in epoch %d: '%(batchidx,epochi), "loss is %f "% (loss.item()))



                ##### end of batch loop######


            tmpmeanbatchloss = np.mean(batchLoss)
            try:
                perplexity = math.exp(tmpmeanbatchloss)
            except OverflowError:
                perplexity = float("inf")

            self.losses.append(tmpmeanbatchloss)
            self.trainperplex.append(perplexity)



            print("##Epoch %d averaged batch training perplexity is %f and loss is %f"%(epochi,perplexity,
                                                                                              tmpmeanbatchloss))


        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done

        return self.trainperplex, self.losses

        
        
        
    def trainTransformerFTmulticlass(self,clipping =0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False,
                                Xkey='input_ids',attnkey = 'attention_mask', ykey='label'):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss

                    
                X = batch[Xkey] # TODO give generic label 
                attn_mask = batch[attnkey] 
                y = batch[ykey] # TODO give generic label 
                
                X = X.to(self.device) # moved to gpu
                attn_mask = attn_mask.to(self.device)
                y = y.to(self.device) # moved to gpu
                
                yHat = self.net(X,attn_mask)
                
                self.loss = self.lossfun(yHat,y)
                    
                self.forwardcallbacks() # not sure how these will work since they refer to net params
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for mutlinomial classification problem
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                # detach from yhat ?
                self.batchLoss.append(self.loss.item())
                
                #Explanation and how crossentropy is working here - IMPORTANT:
                # additional note : first looking back at worked out examples, there is no softmax - logits ( outputs from fcn ) are 
                # directly input to cross entropy loss, again logits dont need to be positive or sum to 1.
                # logits are basically values for each of the classes. So if we have 3 classes, and batch =1, logits fed to crossentropy
                # will be of size [1,3], the 3 colums signifying the class index of each class and we typically want to take the 
                # max value and check its index, and then verify is that index matches the target. So the target( or y) will be the class
                # index and in this example of size ([1]). So aboveself.lossfun(yHat,y) is cross_entropy(size[1,3], size[1]) and
                # it works on class indices!!
                
                
                # what the crossentropyloss seems to be doing is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
                
                #For whats happening in accuracy calculation
                # consider iris where output is 3. first yhat for a single input row is three o/p , each standing for the three classes, and 
                # the values are the logits for each
                #second since this is a batch not a row of input, yhat will be a matrix of same rows of batch and three cols
                #so accuracy compute goes as follows - first you find the col index number which has max value for each row in yhat. This                   #correspond to which col( class) has max value.Second you check if this col index matches ( 0-2 in case of iris) the ground
                #truth y ( whose labels were also made 0-2). if it does then it returns true which is converted to float 1 ( false is 0).                   #Third you average out the number of matches ( 1's) over this batch by taking mean and then multipley by 100. append this                   #value in batchac list as that batches averaged accuracy
                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
                    batch = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to

                    
                    X = batch[Xkey] # TODO give generic label 
                    attn_mask = batch[attnkey] 
                    y = batch[ykey] # TODO give generic label 
                
                    X = X.to(self.device) # moved to gpu
                    attn_mask = attn_mask.to(self.device)
                    y = y.to(self.device) # moved to gpu
                

                    
                    pred = self.net(X,attn_mask)
                    
                    pred = pred.detach().cpu() # move to cpu
                    y = y.cpu() # move to cpu
                    
                    predlabels = torch.argmax( pred,axis=1 )
                    
                    tmptestloss = self.lossfun(pred,y.detach()).item()
                    tmptestacc = 100*torch.mean((predlabels == y.detach()).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append( tmptestacc)
                    # comparing two tensors predlabel(yhat) and y of say size 30 ( 30 rows/test samples in X)
                    #which are basically the labels of these each row of X testdata -size 30
                    # mean counts the number of times predlabel and y are equal and divdes them by number of samples 30
                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmptestacc, tmptestloss))
                        
                        if self.prntconsole:
        
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmptestacc, tmptestloss))
                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
        if testset:
            
            self.net.eval() # flag to turn off train                    
            self.misclassifiedTest = np.where(predlabels != y.detach())[0]
    #        to find missclassified rows in testset - > print X[self.misclassified
            if prnt_misclassified:
                print ('###Misclassfied samples in Testset####\n',"Row indices in Testset: ",self.misclassifiedTest)
                if verbose :
                    print ("Rows in Testset: ",X[self.misclassifiedTest].detach())

        
        self.net.eval() # run misclassfied for train regardless
        
        with torch.no_grad():
            batch = next(iter(self.train_loader))
            
            X = batch[Xkey] # TODO give generic label 
            attn_mask = batch[attnkey] 
            y = batch[ykey] # TODO give generic label 
            
            X = X.to(self.device) # moved to gpu
            attn_mask = attn_mask.to(self.device)
            y = y.to(self.device)
            
            pred = self.net(X,attn_mask)
            
            pred = pred.detach().cpu() # move to cpu
            y = y.cpu() # move to cpu
            
            predlabels = torch.argmax(pred,axis=1 )
            self.misclassifiedTrain = np.where(predlabels != y.detach())[0]
            if prnt_misclassified:
                print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",self.misclassifiedTrain)
                if verbose:
                    print("Rows in Trainset: ",X[self.misclassifiedTrain].detach())
         
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')   
                
        return self.trainAcc,self.testAcc, self.losses,  self.misclassifiedTrain, self.misclassifiedTest
        # for accuracy per label/class see metaparams_multioutput notebook
        
        
        
        
    def trainmulticlassHFdataset(self,clipping =0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False,
                                Xkey  = 'X', ykey='y'):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss

                    
                X = batch[Xkey] # TODO give generic label 
                y = batch[ykey] # TODO give generic label 
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                    
                self.forwardcallbacks() # not sure how these will work since they refer to net params
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for mutlinomial classification problem
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                # detach from yhat ?
                self.batchLoss.append(self.loss.item())
                
                #Explanation and how crossentropy is working here - IMPORTANT:
                # additional note : first looking back at worked out examples, there is no softmax - logits ( outputs from fcn ) are 
                # directly input to cross entropy loss, again logits dont need to be positive or sum to 1.
                # logits are basically values for each of the classes. So if we have 3 classes, and batch =1, logits fed to crossentropy
                # will be of size [1,3], the 3 colums signifying the class index of each class and we typically want to take the 
                # max value and check its index, and then verify is that index matches the target. So the target( or y) will be the class
                # index and in this example of size ([1]). So aboveself.lossfun(yHat,y) is cross_entropy(size[1,3], size[1]) and
                # it works on class indices!!
                
                
                # what the crossentropyloss seems to be doing is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
                
                #For whats happening in accuracy calculation
                # consider iris where output is 3. first yhat for a single input row is three o/p , each standing for the three classes, and 
                # the values are the logits for each
                #second since this is a batch not a row of input, yhat will be a matrix of same rows of batch and three cols
                #so accuracy compute goes as follows - first you find the col index number which has max value for each row in yhat. This                   #correspond to which col( class) has max value.Second you check if this col index matches ( 0-2 in case of iris) the ground
                #truth y ( whose labels were also made 0-2). if it does then it returns true which is converted to float 1 ( false is 0).                   #Third you average out the number of matches ( 1's) over this batch by taking mean and then multipley by 100. append this                   #value in batchac list as that batches averaged accuracy
                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
                    batch = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to
                    
                    X = batch[Xkey] # TODO give generic label 
                    y = batch[ykey] # TODO give generic label 
                    
                    X = X.to(self.device) # moved to gpu
                    y = y.to(self.device)
                    
                    pred = self.net(X)
                    
                    pred = pred.detach().cpu() # move to cpu
                    y = y.cpu() # move to cpu
                    
                    predlabels = torch.argmax( pred,axis=1 )
                    
                    tmptestloss = self.lossfun(pred,y.detach()).item()
                    tmptestacc = 100*torch.mean((predlabels == y.detach()).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append( tmptestacc)
                    # comparing two tensors predlabel(yhat) and y of say size 30 ( 30 rows/test samples in X)
                    #which are basically the labels of these each row of X testdata -size 30
                    # mean counts the number of times predlabel and y are equal and divdes them by number of samples 30
                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmptestacc, tmptestloss))
                        
                        if self.prntconsole:
        
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmptestacc, tmptestloss))
                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
        if testset:
            
            self.net.eval() # flag to turn off train                    
            self.misclassifiedTest = np.where(predlabels != y.detach())[0]
    #        to find missclassified rows in testset - > print X[self.misclassified
            if prnt_misclassified:
                print ('###Misclassfied samples in Testset####\n',"Row indices in Testset: ",self.misclassifiedTest)
                if verbose :
                    print ("Rows in Testset: ",X[self.misclassifiedTest].detach())

        
        self.net.eval() # run misclassfied for train regardless
        
        with torch.no_grad():
            batch = next(iter(self.train_loader))
            
            X = batch[Xkey] # TODO give generic label 
            y = batch[ykey] # TODO give generic label 
            
            X = X.to(self.device) # moved to gpu
            y = y.to(self.device)
            
            pred = self.net(X)
            
            pred = pred.detach().cpu() # move to cpu
            y = y.cpu() # move to cpu
            
            predlabels = torch.argmax(pred,axis=1 )
            self.misclassifiedTrain = np.where(predlabels != y.detach())[0]
            if prnt_misclassified:
                print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",self.misclassifiedTrain)
                if verbose:
                    print("Rows in Trainset: ",X[self.misclassifiedTrain].detach())
         
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')   
                
        return self.trainAcc,self.testAcc, self.losses,  self.misclassifiedTrain, self.misclassifiedTest
        # for accuracy per label/class see metaparams_multioutput notebook


        
    def trainmulticlass(self,clipping =0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
        # NOte during this whole thing looks like X and net remain in gpu

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                # here y will be [batchsize] = torch.Size([128])
                #yhat will be [batchsize,numclass] = torch.Size([128, 6])
                    
                self.forwardcallbacks() # not sure how these will work since they refer to net params
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for mutlinomial classification problem
                
                #move to cpu for accuracy calc
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                # detach from yhat ?
                self.batchLoss.append(self.loss.item())
                
                #Explanation and how crossentropy is working here - IMPORTANT:
                # additional note : first looking back at worked out examples, there is no softmax - logits ( outputs from fcn ) are 
                # directly input to cross entropy loss, again logits dont need to be positive or sum to 1.
                # logits are basically values for each of the classes. So if we have 3 classes, and batch =1, logits fed to crossentropy
                # will be of size [1,3], the 3 colums signifying the class index of each class and we typically want to take the 
                # max value and check its index, and then verify is that index matches the target. So the target( or y) will be the class
                # index and in this example of size ([1]). So aboveself.lossfun(yHat,y) is cross_entropy(size[1,3], size[1]) and
                # it works on class indices!!
                
                
                # what the crossentropyloss seems to be doing is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
                
                #For whats happening in accuracy calculation
                # consider iris where output is 3. first yhat for a single input row is three o/p , each standing for the three classes, and 
                # the values are the logits for each
                #second since this is a batch not a row of input, yhat will be a matrix of same rows of batch and three cols
                #so accuracy compute goes as follows - first you find the col index number which has max value for each row in yhat. This                   #correspond to which col( class) has max value.Second you check if this col index matches ( 0-2 in case of iris) the ground
                #truth y ( whose labels were also made 0-2). if it does then it returns true which is converted to float 1 ( false is 0).                   #Third you average out the number of matches ( 1's) over this batch by taking mean and then multipley by 100. append this                   #value in batchac list as that batches averaged accuracy
                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
                    X,y = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to
                    
                    X = X.to(self.device) # moved to gpu
                    y = y.to(self.device)
                    
                    pred = self.net(X)
                    
                    pred = pred.detach().cpu() # move to cpu
                    y = y.cpu() # move to cpu
                    
                    predlabels = torch.argmax( pred,axis=1 )
                    
                    tmptestloss = self.lossfun(pred,y.detach()).item()
                    tmptestacc = 100*torch.mean((predlabels == y.detach()).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append( tmptestacc)
                    # comparing two tensors predlabel(yhat) and y of say size 30 ( 30 rows/test samples in X)
                    #which are basically the labels of these each row of X testdata -size 30
                    # mean counts the number of times predlabel and y are equal and divdes them by number of samples 30
                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmptestacc, tmptestloss))
                        
                        if self.prntconsole:
        
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,tmptestacc, tmptestloss))
                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
        if testset:
            
            self.net.eval() # flag to turn off train                    
            self.misclassifiedTest = np.where(predlabels != y.detach())[0]
    #        to find missclassified rows in testset - > print X[self.misclassified
            if prnt_misclassified:
                print ('###Misclassfied samples in Testset####\n',"Row indices in Testset: ",self.misclassifiedTest)
                if verbose :
                    print ("Rows in Testset: ",X[self.misclassifiedTest].detach())

        
        self.net.eval() # run misclassfied for train regardless
        with torch.no_grad():
            X,y = next(iter(self.train_loader))
            
            X = X.to(self.device) # moved to gpu
            y = y.to(self.device)
            
            pred = self.net(X)
            
            pred = pred.detach().cpu() # move to cpu
            y = y.cpu() # move to cpu
            
            predlabels = torch.argmax(pred,axis=1 )
            self.misclassifiedTrain = np.where(predlabels != y.detach())[0]
            if prnt_misclassified:
                print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",self.misclassifiedTrain)
                if verbose:
                    print("Rows in Trainset: ",X[self.misclassifiedTrain].detach())
         
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')            
        return self.trainAcc,self.testAcc, self.losses,  self.misclassifiedTrain, self.misclassifiedTest
        # for accuracy per label/class see metaparams_multioutput notebook
        
        
    def trainTransformerFTbiclass(self,clipping = 0,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False,
                                 Xkey='input_ids',attnkey = 'attention_mask', ykey='label'):

        self.biclass = True
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")        
        
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 
        
        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = batch[Xkey] # TODO give generic label 
                attn_mask = batch[attnkey] 
                y = batch[ykey] # TODO give generic label 
                
                X = X.to(self.device) # moved to gpu
                attn_mask = attn_mask.to(self.device)
                y = y.to(self.device) # moved to gpu
                                
              
                yHat = self.net(X,attn_mask)
                
                self.loss = self.lossfun(yHat,y)
                
                self.forwardcallbacks()
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)       
                
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for binary classification problem
                
                #move back to cpu for accuracy calcs
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()                
           
                tmpacc = 100*torch.mean(((yHat.detach() > 0 ) == y.detach()).float()).item() # this assumes you are using bcewithlogitloss
            # and no sigmoid at final output activation layer and output layer as one unit
            
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())
                
                #Explanation:
                # yhat here are continuos values like [-4, -1, 4, 3]. Why ? Because we use bcewithlogitloss loss function and not
                # use a sigmoid activtation funct at the final output. Which means the o/p of model does not go through a sigmoid ( sig
                #moid output would be between 0-1). so yhat will be [-4, -1, 4, 3] but loss (yhat,y) will turn yhat into sigmoid o/p
                # and then calculate loss with y
                
                # yhat>0 return a tensor with elemnetwise true/false [0,0,1,1]. It is not >0.5 since we are not using signoid at o/p
                #layer. checking is resulting tensor is elementwise = y. So false and 0 match and true and 1 match and both
                # return a True. This way we a new resulting boolean tensor bbased on match with y.This is converted to 0,1
                # and then mean is taken which will give us accuracy
                
                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:
                        
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc is %f and loss is %f "% (tmpacc,
                                                         self.loss.item()))

                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)
            
            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:
                    
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
 
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())



            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    self.batchtestloss = []
                    self.batchtestAcc = []
                    
                    for batchidx, batch in enumerate(self.test_loader):
#                    batch = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to
                    
                        X = batch[Xkey] # TODO give generic label 
                        attn_mask = batch[attnkey] 
                        y = batch[ykey] # TODO give generic label 

                        X = X.to(self.device) # moved to gpu
                        attn_mask = attn_mask.to(self.device)
                        y = y.to(self.device) # moved to gpu                    

                        predlabels = self.net(X,attn_mask)

                        predlabels = predlabels.detach().cpu() # move to cpu
                        y = y.detach().cpu() # move to cpu                    

                        tmptestloss = self.lossfun(predlabels,y).item()
                        tmptestacc = 100*torch.mean(((predlabels > 0 ) == y).float()).item()
                    
                        self.batchtestloss.append(tmptestloss)
                        self.batchtestAcc.append(tmptestacc)
                        
                        if batchlogs:

                            self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc ,tmptestloss))
                            if self.prntconsole:                          
                                print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc,tmptestloss))
                            
                    tmpmeanbatchloss = np.mean(self.batchtestloss)
                    tmpmeanbatchacc = np.mean(self.batchtestAcc)
            
                    self.testAcc.append(tmpmeanbatchacc)
                    self.testloss.append(tmpmeanbatchloss)
                    
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
        if testset:
            self.net.eval() # flag to turn off train
            self.misclassifiedTest = np.where((predlabels > 0 ) != y.detach())[0]

    #        to find missclassified rows in testset - > print X[self.misclassified
            if prnt_misclassified:
                print ('###Misclassfied samples in Testset####\n',"Row indices in Testset: ",self.misclassifiedTest)
                if verbose:
                    print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) # will probably fails as on gpu
                   

        self.net.eval() # Run missclassified for train regardless
        with torch.no_grad():
            
            batch = next(iter(self.train_loader))
            
            X = batch[Xkey] # TODO give generic label 
            attn_mask = batch[attnkey] 
            y = batch[ykey] # TODO give generic label 
            
            X = X.to(self.device) # moved to gpu
            attn_mask = attn_mask.to(self.device)
            y = y.to(self.device)           
            
            predlabels = self.net(X,attn_mask)
            
            predlabels = predlabels.detach().cpu() # move to cpu
            y = y.cpu() # move to cpu  
            
            self.misclassifiedTrain = np.where((predlabels > 0 ) != y.detach())[0]
            if prnt_misclassified:
                print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",self.misclassifiedTrain)
                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())
            
            
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
                
        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest
                
        

    def trainbinaryclassHFDataset(self,clipping = 0,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False,
                                 Xkey ='hidden_state', ykey='label'):

        self.biclass = True
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 
        
        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx,batch in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = batch[Xkey] # TODO give generic label 
                y = batch[ykey] # TODO give generic label                

                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                                
              
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                
                self.forwardcallbacks()
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)                
                
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for binary classification problem
                
                #move back to cpu for accuracy calcs
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()                
           
                tmpacc = 100*torch.mean(((yHat.detach() > 0 ) == y.detach()).float()).item() # this assumes you are using bcewithlogitloss
            # and no sigmoid at final output activation layer and output layer as one unit
            
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())
                
                #Explanation:
                # yhat here are continuos values like [-4, -1, 4, 3]. Why ? Because we use bcewithlogitloss loss function and not
                # use a sigmoid activtation funct at the final output. Which means the o/p of model does not go through a sigmoid ( sig
                #moid output would be between 0-1). so yhat will be [-4, -1, 4, 3] but loss (yhat,y) will turn yhat into sigmoid o/p
                # and then calculate loss with y
                
                # yhat>0 return a tensor with elemnetwise true/false [0,0,1,1]. It is not >0.5 since we are not using signoid at o/p
                #layer. checking is resulting tensor is elementwise = y. So false and 0 match and true and 1 match and both
                # return a True. This way we a new resulting boolean tensor bbased on match with y.This is converted to 0,1
                # and then mean is taken which will give us accuracy
                
                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:
                        
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc is %f and loss is %f "% (tmpacc,
                                                         self.loss.item()))

                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)
            
            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:
                    
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
 
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())



            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    
                    batch = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to
                    
                    X = batch[Xkey] # TODO give generic label 
                    y = batch[ykey] # TODO give generic label
                    X = X.to(self.device) # moved to gpu
                    y = y.to(self.device)                    
                    
                    predlabels = self.net(X)
                    
                    predlabels = predlabels.detach().cpu() # move to cpu
                    y = y.detach().cpu() # move to cpu                    
                    
                    tmptestloss = self.lossfun(predlabels,y).item()
                    tmptestacc = 100*torch.mean(((predlabels > 0 ) == y).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)
                    if batchlogs:
                        
                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc ,tmptestloss))
                        if self.prntconsole:                          
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc,tmptestloss))
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
        if testset:
            self.net.eval() # flag to turn off train
            self.misclassifiedTest = np.where((predlabels > 0 ) != y.detach())[0]

    #        to find missclassified rows in testset - > print X[self.misclassified
            if prnt_misclassified:
                print ('###Misclassfied samples in Testset####\n',"Row indices in Testset: ",self.misclassifiedTest)
                if verbose:
                    print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) # will probably fails as on gpu
                   

        self.net.eval() # Run missclassified for train regardless
        with torch.no_grad():
            batch = next(iter(self.train_loader))
            X = batch[Xkey] # TODO give generic label 
            y = batch[ykey] # TODO give generic label
            
            X = X.to(self.device) # moved to gpu
            y = y.to(self.device)            
            
            predlabels = self.net(X)
            
            predlabels = predlabels.detach().cpu() # move to cpu
            y = y.cpu() # move to cpu  
            
            self.misclassifiedTrain = np.where((predlabels > 0 ) != y.detach())[0]
            if prnt_misclassified:
                print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",self.misclassifiedTrain)
                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())
            
            
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
                
        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest

### changes fr adding to class module

    def trainbinaryclass(self,clipping = 0,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False):

        self.biclass = True
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 
        
        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                # forward pass and loss
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu
                                
              
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                # here y is [batchsize] and yhat should also be batchsize [batchsize]?
                self.forwardcallbacks()
                
                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)                
                
                self.optimizer.step()
                
                self.backwardcallbacks()

                # compute training accuracy just per batch for binary classification problem
                
                #move back to cpu for accuracy calcs
                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()                
           
                tmpacc = 100*torch.mean(((yHat.detach() > 0 ) == y.detach()).float()).item() # this assumes you are using bcewithlogitloss
            # and no sigmoid at final output activation layer and output layer as one unit
            
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())
                
                #Explanation:
                # yhat here are continuos values like [-4, -1, 4, 3]. Why ? Because we use bcewithlogitloss loss function and not
                # use a sigmoid activtation funct at the final output. Which means the o/p of model does not go through a sigmoid ( sig
                #moid output would be between 0-1). so yhat will be [-4, -1, 4, 3] but loss (yhat,y) will turn yhat into sigmoid o/p
                # and then calculate loss with y
                
                # yhat>0 return a tensor with elemnetwise true/false [0,0,1,1]. It is not >0.5 since we are not using signoid at o/p
                #layer. checking is resulting tensor is elementwise = y. So false and 0 match and true and 1 match and both
                # return a True. This way we a new resulting boolean tensor bbased on match with y.This is converted to 0,1
                # and then mean is taken which will give us accuracy
                
                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:
                        
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc is %f and loss is %f "% (tmpacc,
                                                         self.loss.item()))

                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)
            
            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:
                    
                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))
 
            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())



            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to
                    
                    X = X.to(self.device) # moved to gpu
                    y = y.to(self.device)                    
                    
                    predlabels = self.net(X)
                    
                    predlabels = predlabels.detach().cpu() # move to cpu
                    y = y.detach().cpu() # move to cpu                    
                    
                    tmptestloss = self.lossfun(predlabels,y).item()
                    tmptestacc = 100*torch.mean(((predlabels > 0 ) == y).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)
                    if batchlogs:
                        
                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc ,tmptestloss))
                        if self.prntconsole:                          
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc,tmptestloss))
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
        if testset:
            self.net.eval() # flag to turn off train
            self.misclassifiedTest = np.where((predlabels > 0 ) != y.detach())[0]

    #        to find missclassified rows in testset - > print X[self.misclassified
            if prnt_misclassified:
                print ('###Misclassfied samples in Testset####\n',"Row indices in Testset: ",self.misclassifiedTest)
                if verbose:
                    print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) # will probably fails as on gpu
                   

        self.net.eval() # Run missclassified for train regardless
        with torch.no_grad():
            X,y = next(iter(self.train_loader))
            
            X = X.to(self.device) # moved to gpu
            y = y.to(self.device)            
            
            predlabels = self.net(X)
            
            predlabels = predlabels.detach().cpu() # move to cpu
            y = y.cpu() # move to cpu  
            
            self.misclassifiedTrain = np.where((predlabels > 0 ) != y.detach())[0]
            if prnt_misclassified:
                print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",self.misclassifiedTrain)
                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())
            
            
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
                
        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest

### changes fr adding to class module


    def trainmultilabel(self,clipping = 0,numclasses = 3,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False):

        self.multilabel = True
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []

        onestensor = torch.ones(numclasses).detach()

        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)

        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()

        self.net.to(self.device)

        for epochi in range(self.epochs):

            self.net.train() #flag to turn on train

            self.batchAcc = []
            self.batchLoss = []

            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

            for batchidx, (X,y) in enumerate(self.train_loader):

                self.net.train()
                
                X = X.to(self.device) # moved to gpu
                y = y.to(self.device) # moved to gpu


                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)

                self.forwardcallbacks()

                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)                
                
                self.optimizer.step()

                self.backwardcallbacks()



                # compute training accuracy just per batch for binary classification problem

                #move back to cpu for accuracy calcs

                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()   

                tmpacc = 100*(
                    torch.where(
                        (((yHat.detach() > 0 ) == y.detach()) == onestensor).all(dim=1))[0].shape[0])/y.detach().shape[0]

                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                #Explanation:
                # yhat here are continuos values like [-4, -1, 4, 3]
                # yhat>0 return a tensor with elemnetwise true/false [0,0,1,1]. It is not >0.5 since we are not using signoid at o/p
                #layer. checking is resulting tensor is elementwise = y. So false and 0 match and true and 1 match and both
                # return a True. This way we a new resulting boolean tensor bbased on match with y.This is converted to 0,1
                # and then mean is taken which will give us accuracy

                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                        "batchacc is %f and loss is %f "% (tmpacc,
                                                                          self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                "batchacc is %f and loss is %f "% (tmpacc,
                                                             self.loss.item()))

                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch

            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)

            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)

            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:

                    print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
    #                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
    ### continue editing here

            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to

                    X = X.to(self.device) # moved to gpu
                    y = y.to(self.device)                    

                    predlabels = self.net(X)
                    tmptestloss = self.lossfun(predlabels,y)
                    
                    #### have all loss calculations in gpu as well , it is faster####
                    
                    predlabels = predlabels.detach().cpu() # move to cpu
                    y = y.detach().cpu() # move to cpu 
                    
                    tmptestloss = tmptestloss.cpu().item()
                    tmptestacc = 100*(torch.where((((predlabels > 0 ) == y) == onestensor).all(dim=1))[0].shape[0])/y.shape[0]
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)

                    if batchlogs:

                        self.logger.info("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc ,tmptestloss))
                        if self.prntconsole:                          
                            print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi, tmptestacc,tmptestloss))


        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done
        #once all training epoch are done, find rows/entries of testset which are still misclassified
        if testset:
            self.net.eval() # flag to turn off train
            self.misclassifiedTest = torch.where((((predlabels > 0 ) == y) == onestensor).all(dim=1))[0]
            print ('###Misclassfied samples in Testset####\n',"Row indices in Testset: ",self.misclassifiedTest)
            if verbose:
                print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) # will probably fails as on gpu

        self.net.eval() # Run missclassified for train regardless
        with torch.no_grad():
            X,y = next(iter(self.train_loader))

            X = X.to(self.device) # moved to gpu
            y = y.to(self.device)            

            predlabels = self.net(X)

            predlabels = predlabels.detach().cpu() # move to cpu
            y = y.detach().cpu() # move to cpu  

            self.misclassifiedTrain = torch.where((((predlabels > 0 ) == y) == onestensor).all(dim=1))[0]
            if prnt_misclassified:
                print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",self.misclassifiedTrain)
                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())


        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')

        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest


    
    
    def trainregression(self,clipping =0,batchlogs=True, testset=True):
        
        self.regress = True

        self.trainAcc = []
        self.testAcc  = []
        #### note correlation will give error if data is more than 2d ( for example any cnn related stuff)####
#        self.trainCorr = []
#        self.testCorr = []
        self.losses   = []
        self.testloss = [] # new for regression only

        
        if self.savestart:
            self.saveModel(filename='start')

            
        starttime = time.time()
        
        self.net.to(self.device) # moved to gpu
            
        if batchlogs:
            self.logger.info("##########Starting Batch Logs##########")

        for epochi in range(self.epochs):
            
            self.net.train() #flag to turn on train
            self.batchAcc = []
#            self.batchCorr = []
            self.batchLoss = []
    
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                
                X = X.to(self.device) # moved to gpu
                # forward pass and loss
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                
                self.forwardcallbacks()

                # backprop
                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                yHat = yHat.cpu()
                X = X.cpu()
                self.loss = self.loss.cpu()

                # compute training accuracy for regression. here accuracy is the correlation coefficient between actual
                # and predicted continuos values. correlation coefficient always takes [0,1], but not sure why we are
                # transpoing
                tmpacc = torch.mean(torch.abs(yHat.detach()-y.detach())).item()
                self.batchAcc.append(tmpacc)
#                self.batchCorr.append(np.corrcoef(y.detach().T,yHat.detach().T)[0,1])
                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc L1loss (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc L1loss (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                ##### end of batch loop######

            # average training accuracy and loss for each epoch by averaging all the batchacc/batchloss in that single epoch. 
            #Note batchacc is reset start of every new epoch
            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)
#            self.trainCorr.append(np.mean(self.batchCorr))
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("##Epoch %d averaged batch training accuracy(L1loss) is %f and loss is %f"%(epochi,tmpmeanbatchacc,tmpmeanbatchloss))
                                                                                                     
                if self.prntconsole:
                    
                    print("##Epoch %d averaged batch training accuracy(L1loss) is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: # needs updating
                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
#                if (tmpmeanbatchacc < self.bestl1thresh) and (tmpmeanbatchacc < self.bestTrainacc['acc']):
                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                  
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())                    
                
                
            # test accuracy for each epoch , passing test data through learned weights at the end of each epoch
            if testset:
                self.net.eval() # flag to turn off train
                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) # extract X,y from test dataloader. 
                    #Since it is one batch, it is the full matrix of the testset. X is matrix tensor of some number of rows/samples
                    #y is tensor which corresponds to labels/categories each sameple/row that X belogs to
                    X = X.to(self.device)
                    predictions = self.net(X)
                    
                    predictions = predictions.cpu()
                    X = X.cpu()
                    
                    tmptestloss = self.lossfun(predictions,y).item()
                    tmptestacc = torch.mean(torch.abs(predictions.detach()-y.detach())).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)
#                    self.testCorr.append(np.corrcoef(y.detach().T,predictions.detach().T)[0,1])
                    if batchlogs:
                        
                        self.logger.info("##Epoch %d Test accuracy(L1loss) is %f and loss is %f"%( epochi,tmptestacc,tmptestloss))
                        if self.prntconsole:
                            print("##Epoch %d Test accuracy(L1loss) is %f and loss is %f"%( epochi,tmptestacc, tmptestloss))
                        
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
        return self.trainAcc, self.testAcc, self.losses,self.testloss
                           

    def prnt_trainparams(self,skip_lastbatchsize=True,verbose=False):
        
        print ("## Network ##")
        print (self.net)
        print ("\n")
#        print ("## Train and Test Data Shapes ##")
#        print (self.train_data.shape, self.train_labels.shape, self.test_data.shape,self.test_labels.shape)
#        print ("Size of training data: ", len(self.train_data))
#        print ("## First batch size of Train Loader##")
#        for X,y in self.train_loader:
#            print(X.detach().shape,y.detach().shape)
#            break
#        if skip_lastbatchsize:
#            pass
#        else:
#            print("## Last batch size of Train Loader ##")
#            for X,y in self.train_loader:
#                pass
#            print (X.detach().shape,y.detach().shape)
        if verbose :
            
            print ("## First sample from first batch of Train Loader##")
            for X,y in self.train_loader:
                print(X.detach()[0],y.detach()[0])
                break
#        print ("Number of batches in training data: ", len(self.train_loader))
#        print ("\n")
#        print ("## Batch size of Test Loader##")
#        for X,y in self.test_loader:
#            print(X.detach().shape,y.detach().shape)
#        if verbose:
#            print ("## First sample from first batch of Test Loader##")
#            for X,y in self.test_loader:
#                print(X.detach()[0],y.detach()[0])
#                break
#        print ('Number of batches in test data: ', len(self.test_loader))
#        print("\n")
        print("## trainable params ##")
        print ("The model has " +str(sum(p.numel() for p in self.net.parameters() if p.requires_grad))+" trainable params")
        
        print("## training metaparams ##")
        print("Loss function: ",self.lossfun,"\n", "Learning Rate: ", self.lr,"\n","Epochs :", self.epochs,"\n","Optimizer :", self.optimizer)
                           
        
    def lossplot(self, corrcoeff=False):
        
        # dont use corrcoeff for multi-output
        
        #losses, trainAcc and testAcc are a list of numpy float objects 
        figloss = go.Figure()
        figloss.add_trace(go.Scatter(x=[i for i in range(len(self.losses))], y=self.losses,name="Training Loss"))
        
        figloss.add_trace(go.Scatter(x=[i for i in range(len(self.testloss))], y=self.testloss,name="Test Loss"))
            
        figloss.update_layout(title="Loss Curve" , xaxis_title="Epochs", 
                          yaxis_title="Loss", height=400, width=500)
        figloss.show()
        
        figacc = go.Figure()
                           
        figacc.add_trace(go.Scatter(x=[i for i in range(len(self.trainAcc))], y=self.trainAcc, name="Training Accuracy"))
        figacc.add_trace(go.Scatter(x=[i for i in range(len(self.testAcc))], y=self.testAcc, name="Test Accuracy"))
        
        figacc.update_layout(title="Accuracy", xaxis_title="Epochs", yaxis_title="Training  & Test Accuracy",
                             height=400, width=500)
        figacc.show()
        
        if corrcoeff:
            figcorr = go.Figure()
            figcorr.add_trace(go.Scatter(x=[i for i in range(len(self.trainCorr))], y=self.trainCorr, name="Training Correlation"))
            figcorr.add_trace(go.Scatter(x=[i for i in range(len(self.testCorr))], y=self.testCorr, name="Test Correlation"))
            
            figcorr.update_layout(title="Correlation-Coefficient", xaxis_title="Epochs",
                                  yaxis_title="Training  & Test Corrcoeff",height=400, width=500)
   
        if corrcoeff:
            self.net.eval() # flag to turn off train
            with torch.no_grad():
                
                yhattrain = self.net(self.train_data) # presuming model self.net is trained, passing just train data (no labels) and checking how well it predicts compared to actual train_label
                yhattest = self.net(self.test_data) # same with test data
                
            figpred = go.Figure()
 
            # correlations between predictions and outputs
            corrTrain = np.corrcoef(yhattrain.detach().T,self.train_labels.T)[1,0]
            corrTest  = np.corrcoef(yhattest.detach().T, self.test_labels.T)[1,0]


            figpred.add_trace(go.Scatter(x=torch.flatten(yhattrain.detach()).numpy(), y=torch.flatten(self.train_labels.detach()).numpy(),name="Training predictions r= "+str(corrTrain))) # high corrcoeff means should see linear line through the origin for this
            figpred.add_trace(go.Scatter(x=torch.flatten(yhattest.detach()).numpy(), y=torch.flatten(self.test_labels.detach()).numpy(),name="Test predictions r="+str(corrTest)))
            
            figpred.update_layout(title="Training and Test predictions vs actual readings (all data) ", xaxis_title="Predicted value", 
                          yaxis_title="True value", height=600, width=1000)
            figpred.show()
    
    
    def accuracyPerclass(self, misclass=None ,y=None):
    
        if self.biclass:

            print ("Number of missclassification for label 1 :",(y[misclass==1]==1).sum())
            print ("Accuracy in classifying label 1 :", ((y==1).sum().item() - (y[misclass==1]==1).sum().item()) / ((y==1).sum().item()))
            print ("Number of missclassification for label 0 :",(y[misclass==1]==0).sum())
            print ("Accuracy in classifying label 0 :", ((y==0).sum().item() - (y[misclass==1]==0).sum().item()) / ((y==0).sum().item()))

        elif multi:
            pass

    def aprf(self, y=None,yhat=None):

        # for binaryclass the training class does not have a sigmoid at end as it trains with bcelogitloss.
        # so you will need to do the following to yhat
#yHat = yHat.detach().numpy()
#y = y.detach().numpy()
#yhat =  yHat > 0
#yhat = yhat.astype(float)
        
        self.sprfdict= {'accuracy':0,'precision':0,'recall':0,'f1':0 }

        if self.biclass:


            self.sprfdict['accuracy']  = skm.accuracy_score (y,yhat)
            self.sprfdict['precision'] = skm.precision_score(y,yhat)
            self.sprfdict['recall']    = skm.recall_score(y,yhat)
            self.sprfdict['f1']        = skm.f1_score(y,yhat)
            return self.sprfdict

        elif self.multiclass:

            #yHat = net(train_loader.dataset.tensors[0])
            #train_predictions = torch.argmax(yHat,axis=1)
            #skm.precision_score(train_loader.dataset.tensors[1],train_predictions,average='weighted')

            self.sprfdict['accuracy']  = skm.accuracy_score (y,yhat)
            self.sprfdict['precision'] = skm.precision_score(y,yhat,average = 'weighted')
            self.sprfdict['recall']    = skm.recall_score(y,yhat,average = 'weighted')
            self.sprfdict['f1']        = skm.f1_score(y,yhat,average = 'weighted')
            return self.sprfdict


    def confusionmatrix(self,y=None, yhat=None,labels =[],heatmap = False):
        conf = skm.confusion_matrix(y,yhat)
        conf.astype(int)
        if heatmap and labels:
            
            fig = px.imshow(conf,  aspect="auto", x= labels, y = labels)
            fig.show()
        return conf
                    
                