# Sequence-Anomaly-Detection-in-HTTP-URIs

---

## Motivation


## Hierarchical design

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
