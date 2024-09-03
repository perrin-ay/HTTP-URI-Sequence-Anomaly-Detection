# -*- coding: utf-8 -*-
"""Transformer Encoder Classifier.ipynb
"""

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
"""**Load training and test dataset**"""

ds_concat = load_from_disk("ds_concat_classif_meanpooled")
ds_concat = ds_concat.class_encode_column('label')


train_test = ds_concat.train_test_split(test_size=0.2, seed =2018, stratify_by_column="label")

train = train_test['train']
test = train_test['test']

ds = datasets.DatasetDict({
    'train': train,
    'test': test})

ds.set_format("torch",columns=["label", "hidden_state"], output_all_columns= True )

"""**Collate function and Dataloader**"""

batch_size = 16
shuffle = True
drop_last =False

def collateseq2seqHFDatasetPack(batch):

    encoder_list = []
    x_lens = []
    label_list = []


    encoder_list = [example["hidden_state"] for example in batch]
    encoder_list = encoder_list[:99] # limiting to positionwise vocab length
    label_list = [example["label"] for example in batch]

    label_torch = torch.tensor(np.array(label_list)).float()
    batchedsrc = pad_sequence(encoder_list, batch_first=True, padding_value=0)


    collated = {"hidden_state" : batchedsrc, "label" : label_torch}

    return collated

train_loader = DataLoader(ds["train"],
                          batch_sampler=SamplerSimilarLengthHFDataset(dataset = ds["train"],
                                                                  batch_size=batch_size, shuffle=shuffle, drop_last= drop_last,
                                                                      keyname= 'hidden_state'), collate_fn=collateseq2seqHFDatasetPack)


test_loader = DataLoader(ds["test"],
                          batch_sampler=SamplerSimilarLengthHFDataset(dataset = ds["test"],
                                                                  batch_size=batch_size, shuffle=False, drop_last= drop_last,
                                                                      keyname= 'hidden_state'), collate_fn=collateseq2seqHFDatasetPack)

"""**Transfer encoder with classification head**

- While training we mean pool embedding output of encoder (embeddings of a http request sequence) and pass to downstream classifier head
"""

class TransformerEncoderclassifier(nn.Module):

    def __init__(self , transformermodel, device, model_dims = 768, num_labels= 1):

        super().__init__()
        self.device = device
        self.net = transformermodel
        self.num_labels = num_labels
        self.model_dims = model_dims

        self.linear1 = nn.Linear(self.model_dims,512)
        self.linear2 = nn.Linear(512,self.num_labels)

    def src_mask(self, src, embed_dims=True):


        src_mask = (src == 0).to(self.device)
        #src_mask = [batch size, src len,embed_dims]

        if embed_dims:
            src_mask = src_mask[:,:,0]
        return src_mask

    def src_mean_mask(self, src, embed_dims=True):

        src_mask = (src != 0).to(self.device)

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

        x = self.net(src, src_key_padding_mask = src_mask) # (batchs, seqln, hidden_dims=768)
        x = self.mean_pooling(x,mean_mask) # (batchs, hidden_dims)

        ############### classifier head ##################
        x = self.linear1(x)
        x = nn.Dropout(p=0.1)(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x= x.squeeze()
        return x

########## initializing transformer encoder with multihead attention ( 6 heads) and 4 layers ##############

encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=6, dim_feedforward=2048, dropout=0.1,
                                           activation ="gelu",batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
classifier = Net()
classifier.setupCuda()
device = classifier.device
transformer_encoder.to(device)
classifier.net = TransformerEncoderclassifier(transformer_encoder, device, model_dims = 768, num_labels=1)
classifier.net.to(device)

# assigning loaders
classifier.train_loader, classifier.test_loader =  train_loader, test_loader

"""**Training Transformer Encoder**"""

optimizer = AdamW(classifier.net.parameters(), lr=5e-5)



epochs = 5
num_training_steps = epochs * len(num_batches)

lr_scheduler = get_scheduler(
    optimizer=optimizer,
    num_warmup_steps=1,
    num_training_steps=num_training_steps,
)

def trainTransformerEncoderbiclass(clipping = 0,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False,
                                 Xkey='hidden_state', ykey='label', net= None, epochs = epochs,
                                   lossfun= nn.BCEWithLogitsLoss(), optimizer =optimizer):

        biclass = True

        trainAcc = []
        testAcc  = []
        losses   = []
        testloss = []

        misclassifiedTrain= np.array(None)
        misclassifiedTest= np.array(None)

        starttime = time.time()

        net.to(device)

        for epochi in range(epochs):

            net.train()

            batchAcc = []
            batchLoss = []

            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

            for batchidx, batch in enumerate(train_loader):

                net.train()


                X = batch[Xkey]
                y = batch[ykey]

                X = X.to(device)
                y = y.to(device)


                yHat = net(X)

                loss = lossfun(yHat,y)

                optimizer.zero_grad()
                loss.backward()

                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clipping)

                optimizer.step()

                lr_scheduler.step()

                yHat = yHat.cpu()
                y = y.cpu()
                loss = loss.cpu()



                tmpacc = 100*torch.mean(((yHat.detach() > 0 ) == y.detach()).float()).item()

                batchAcc.append(tmpacc)
                batchLoss.append(loss.item())




                print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                        "batchacc is %f and loss is %f "% (tmpacc,
                                                         loss.item()))
                print("Learning rate value:",lr_scheduler.get_last_lr())



            tmpmeanbatchloss = np.mean(batchLoss)
            tmpmeanbatchacc = np.mean(batchAcc)

            trainAcc.append(tmpmeanbatchacc)
            losses.append(tmpmeanbatchloss)



            print("##Epoch %d averaged batch training accuracy is %f and loss is %f"%(epochi,tmpmeanbatchacc,
                                                                                              tmpmeanbatchloss))

            if testset:
                net.eval() # flag to turn off train
                with torch.no_grad():
                    batchtestloss = []
                    batchtestAcc = []

                    for batchidx, batch in enumerate(test_loader):


                        X = batch[Xkey]
                        y = batch[ykey]

                        X = X.to(device)
                        y = y.to(device)

                        predlabels = net(X)

                        predlabels = predlabels.detach().cpu()
                        y = y.detach().cpu()

                        tmptestloss = lossfun(predlabels,y).item()
                        tmptestacc = 100*torch.mean(((predlabels > 0 ) == y).float()).item()

                        batchtestloss.append(tmptestloss)
                        batchtestAcc.append(tmptestacc)

                        print("##Epoch %d Batch Test accuracy is %f and loss is %f"%(epochi, tmptestacc,tmptestloss))

                    tmpmeanbatchloss = np.mean(batchtestloss)
                    tmpmeanbatchacc = np.mean(batchtestAcc)


                    print("##Epoch %d averaged Test accuracy is %f and loss is %f"%(epochi,
                                                                                        tmpmeanbatchacc,tmpmeanbatchloss))

                    testAcc.append(tmpmeanbatchacc)
                    testloss.append(tmpmeanbatchloss)


        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

        net.eval()
        with torch.no_grad():

            batch = next(iter(train_loader))

            X = batch[Xkey]
            y = batch[ykey]

            X = X.to(device)
            y = y.to(device)

            predlabels = net(X)

            predlabels = predlabels.detach().cpu() # move to cpu
            y = y.cpu() # move to cpu

            misclassifiedTrain = np.where((predlabels > 0 ) != y.detach())[0]

            print ('###Misclassfied samples in Trainset####\n',"Row indices in Trainset: ",misclassifiedTrain)

        return trainAcc,testAcc, losses,testloss, misclassifiedTrain, misclassifiedTest

result = trainTransformerEncoderbiclass(clipping = 1,batchlogs=True,prnt_misclassified=True,testset=True,verbose=False,
                                 Xkey='hidden_state', ykey='label', net= classifier.net, epochs = epochs,
                                   lossfun= nn.BCEWithLogitsLoss(), optimizer = optimizer)

savedmodel ={}
savedmodel['net'] = classifier.net.state_dict()
savedmodel['opt'] = optimizer.state_dict()
torch.save(savedmodel,"Anomaly_classify_save")

"""**Results**

results = ([97.59524372929484, 99.16883577851397, 99.3877188831046, 99.49124467581638, 99.60068622811168], [99.39744801512288, 99.05482041587902, 99.59829867674858, 99.64555765595463, 99.64555765595463], [0.08133145403123415, 0.0347824873365815, 0.027471243422460257, 0.022356209365447133, 0.017547047132957377], [0.03256244014931331, 0.03834756738002305, 0.018342088693653304, 0.013903011386060475, 0.015386741949415996], array([9]), array(None, dtype=object))

**Final Epoch training prints**

Epoch 4 averaged batch training accuracy is 99.600686 and loss is 0.017547
Epoch 4 averaged Test accuracy is 99.645558 and loss is 0.015387
Time ( in secs) elapsed since start of training :  3810.2276995182037
"""



