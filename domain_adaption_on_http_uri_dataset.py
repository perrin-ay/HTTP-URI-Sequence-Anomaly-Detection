# -*- coding: utf-8 -*-
"""Domain adaption on HTTP URI dataset
"""

!pip install transformers
!pip install datasets

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
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time,sys,os
import math
import collections

"""- Training data consists of legitimate sequence of REST calls (URI's) to an Alteon device.

- This has been uploaded as HF dataset.
https://huggingface.co/bridge4

"""

def pickle_df_todisk(df, filename=''):
    assert filename,"No filename provided"
    df.to_pickle(filename)
    return "DF saved to file "+filename

def load_pickle_todf(filename=''):
    assert filename,"No filename provided"
    df = pd.read_pickle(filename)
    return df


dfuri = load_pickle_todf(filename='URIanomalygood.pickle')
print (dfuri)
dfuri_ds = Dataset.from_pandas(dfuri)
print (dfuri_ds)
print (dfuri_ds[1])

"""- Using distilbert (trained transformer encoder model)  as embeddings model
- Load distilbert and its tokenizer
"""

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

## load tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print (tokenizer.unk_token_id, tokenizer.unk_token) # 100 [UNK]
print (tokenizer.sep_token, tokenizer.sep_token_id,tokenizer.cls_token, tokenizer.cls_token_id) # [SEP] 102 [CLS] 101

"""**Tokenize Data**"""

def tokenize_function(examples):

    result = tokenizer(examples["URI"]) # creates input_id and attention mask vectors

    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))] # word_ids represents positional embeddings
    return result


tokenized_datasets = dfuri_ds.map(tokenize_function, batched=True, batch_size=1000)
tokenized_datasets= tokenized_datasets.remove_columns("URI")

print (tokenized_datasets)
print (tokenized_datasets[1])
print (dfuri_ds['URI'][1])

"""Concat and chunk dataset"""

chunk_size = 128

def concat_chunk(examples):

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split concatanated examples by chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(concat_chunk, batched=True, batch_size=1000)
lm_datasets

print (lm_datasets[1])
print (tokenizer.decode(lm_datasets[1]["input_ids"]))
print (tokenizer.decode([103])) # mask token
print (tokenizer.decode([101])) # cls token
print (tokenizer.decode([102])) # SEP token

"""**Whole word masking Collator**"""


wwm_probability = 0.13


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id # 103 which is the mask token [MASK]
        feature["labels"] = new_labels

    return default_data_collator(features)



samples = [lm_datasets[i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)
print (batch["input_ids"][0], "\n", tokenizer.decode(batch["input_ids"][0]), "\n", batch["labels"][0])

batch_size = 32
epochs = 4

train_loader = DataLoader(lm_datasets,shuffle=True,batch_size=batch_size, collate_fn=whole_word_masking_data_collator)
optimizer = AdamW(model.parameters(), lr=0.00005)
print ("Is GPU available ? : ", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


########### Training Loop ########################################

def trainDistelbertMask(net, optimizer, device, epochs, train_loader,
                        clipping =0, Xkey='input_ids',attnkey = 'attention_mask', ykey='labels'):

    trainperplex = []
    losses   = []

    starttime = time.time()

    net.to(device)
    for epochi in range(epochs):

        net.train() #flag to turn on train

        batchAcc = []
        batchLoss = []

        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

        for batchidx, batch in enumerate(train_loader):

            net.train()


            X = batch[Xkey]

            attn_mask = batch[attnkey]

            y = batch[ykey]

            X = X.to(device)
            attn_mask = attn_mask.to(device)
            y = y.to(device)

            outputs = net(X,attention_mask= attn_mask,labels = y)

            loss = outputs.loss

            optimizer.zero_grad()

            loss.backward()

            if clipping > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping)

            optimizer.step()

            loss = loss.cpu()

            batchLoss.append(loss.item())

            print ('At Batchidx %d in epoch %d: '%(batchidx,epochi), "loss is %f "% (loss.item()))



            ##### end of batch loop######


        tmpmeanbatchloss = np.mean(batchLoss)
        try:
            perplexity = math.exp(tmpmeanbatchloss)
        except OverflowError:
            perplexity = float("inf")

        losses.append(tmpmeanbatchloss)
        trainperplex.append(perplexity)



        print("##Epoch %d averaged batch training perplexity is %f and loss is %f"%(epochi,perplexity,
                                                                                          tmpmeanbatchloss))


    print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done

    return trainperplex, losses


trainperplex, losses = trainDistelbertMask(model, optimizer, device, epochs, train_loader)

"""**Testing inference on trained model**"""



# Loading saved model that was trained previous cell. The trained model had been uploaded to huggingface https://huggingface.co/bridge4

model = AutoModelForMaskedLM.from_pretrained("4epochdistilbert_uri_domainadapt")
urltext = "/[MASK]"
mask_filler = pipeline(
    "fill-mask", model=model,
    tokenizer = tokenizer
)
preds = mask_filler(urltext)
print (preds)
for pred in preds:
    print(f">>> {pred['sequence']}")

"""
>>> / system
>>> / network
>>> / html
>>> / reporter
"""
