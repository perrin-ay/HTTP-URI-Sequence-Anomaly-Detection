# Sequence-Anomaly-Detection-in-HTTP-URIs

---

## Motivation

To provide HTTP request sequence analytics and attack/abuse detection

- Here a HTTP request sequence is any time-ordered sequence of URI calls while browsing a website or interacting with an API.
- For example below is one instance of legitimate and illegitimate sequence of HTTP calls looks the Alteon REST API

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
 

## Hierarchical design

- Transformer and LSTM VAE networks are used for learning and detecting legitimate and malicious sequences of HTTP request URI's  

#### Domain adaptation of distilBERT to create a HTTP2vec model

- Masked language modeling using Huggingface
- Using legitimate HTTP requests from CSIC 2010 dataset, we continue pre-training for masked language modeling and domain adapt the model as an HTTP2vec for better HTTP embeddings.
  
**Input**
```
[CLS] get / tienda1 / index. jsp http / 1. 1 user - agent :  Mozilla / 5. 0 ( compatible ; konqueror / 3. 5 ; linux ) khtml / 3. 5. 8 ( like gecko ) pragma : no - cache cache - control : no - cache accept : text / xml, application / xml, application / xhtml + xml, text / html ; q = 0. 9, text / plain ; q = 0. 8, image / png , * / * ; q = 0. 5 accept - encoding : x - g'
```
**Target**
```
[CLS] get / tienda1 / index. jsp http / 1. 1 user - agent : [MASK] [MASK] [MASK] / 5. 0 ( compatible ; konqueror / 3. 5 ; linux ) khtml / 3. 5. 8 ( like gecko ) pragma : no - cache cache - control [MASK] no [MASK] [MASK] accept : text / xml, application [MASK] xml, application / xhtml + xml, text / html ; q = 0. 9, text [MASK] plain ; [MASK] = 0. 8, image / png [MASK] * / * ; q = 0. 5 accept - encoding : x - g'
```

- Domain adapted model can be found here: https://huggingface.co/bridge4/distilbert_HTTPtoVec_CSIC2010
- Dataset used can be found here: https://huggingface.co/datasets/bridge4/CSIC2010_dataset_domain_adaptation
