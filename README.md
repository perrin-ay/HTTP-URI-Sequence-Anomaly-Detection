# HTTP-URI-Sequence-Anomaly-Detection

---

## Motivation

To provide HTTP request sequence analytics and attack/abuse detection

- Here a HTTP request sequence is any time-ordered sequence of URI calls while browsing a website or interacting with an API.
- For example below is one instance of legitimate and illegitimate sequence of HTTP calls to Alteon REST API

**Legitimate**
```
1) /reporter/system
2) /reporter/virtualServer
3) /
4) /webui/default.html
5) /
5) /webui/default.html
6) /
7) /webui/default.html
8) /
9) /webui/default.html
```
**Offending**
```
1) /reporter/network
2) /reporter/system
3) /reporter/system
4) /reporter/virtualServer
5) /reporter/system
6) /config/hwDRAMSize
7) /
9) /reporter/network
10) /reporter/system 
```

- When attackers or malicious bots attempt to access websites or API's with offending requests they rarely follow expected patterened sequence of requests appropriate for the resource being accessed.

## Objective

- Learn http request sequences and provide a sequence analytics - a view into what legitimate sequences look like for a particular resource being accessed.
- Identify and detect offending and malicious sequences in an effort to then apply mitigation strategies.
- Transformer and LSTM VAE networks are used for learning and detecting legitimate and malicious sequences of HTTP request URI's
 

## Hierarchical design

- Continue pretraining of distilBERT on legitimate URIs with masked language modeling for domain adaptation ( create a URI2vec model)
- Use this trained URI2vec to get embeddings of HTTP request URI.
- Two approaches hereafter:
  
   - **Supervised learning** - Use the embedding and train a seperate transformer encoder with classifier head
   - **Unsupervised learning** - Use the embeddings and train a LSTM VAE network

#### Domain adaptation of distilBERT to create a URI2vec model

- Masked language modeling using Huggingface
- Using legitimate URIs, we continue pre-training for masked language modeling and domain adapt the model as an URI2vec for better URI embeddings.
- Domain adapted model can be found here: https://huggingface.co/bridge4/distilbert_SequenceURI
- Dataset used can be found here: https://huggingface.co/datasets/bridge4/URI_APIsequence_dataset

![image](https://github.com/user-attachments/assets/f74091d4-6b20-4eb3-98e7-c222a1f42b29)

#### Create dataset for supervised learning
- create good/bad sequence dataset
- bad is created from the good sequences by permuting the order
- see notebook for details https://github.com/perrin-ay/HTTP-URI-Sequence-Anomaly-Detection/blob/main/Legitimate_and_anomalous_HTTP_request_sequence_dataset.ipynb
- dataset can be found : https://huggingface.co/datasets/bridge4/URI_API_SEQ_GOOD_BAD_withEmbeddings_Dataset
  
![image](https://github.com/user-attachments/assets/2fa115a3-75ae-454c-857f-5479ba43a4b6)

#### Train transformer encoder
- see file for training details : https://github.com/perrin-ay/HTTP-URI-Sequence-Anomaly-Detection/blob/main/transformer_encoder_classifier.py

**Results**
```
results = ([97.59524372929484, 99.16883577851397, 99.3877188831046, 99.49124467581638, 99.60068622811168], [99.39744801512288, 99.05482041587902, 99.59829867674858, 99.64555765595463, 99.64555765595463], [0.08133145403123415, 0.0347824873365815, 0.027471243422460257, 0.022356209365447133, 0.017547047132957377], [0.03256244014931331, 0.03834756738002305, 0.018342088693653304, 0.013903011386060475, 0.015386741949415996], array([9]), array(None, dtype=object))
```

**Final epoch training prints**
```
Epoch 4 averaged batch training accuracy is 99.600686 and loss is 0.017547
Epoch 4 averaged Test accuracy is 99.645558 and loss is 0.015387
Time ( in secs) elapsed since start of training :  3810.2276995182037
```
