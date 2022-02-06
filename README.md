# Debiasing Pre-trained Contextualised Embeddings

[Masahiro Kaneko](https://sites.google.com/view/masahirokaneko/english?authuser=0), [Danushka Bollegala](http://danushka.net/)


Code and debiased word embeddings for the paper: "Debiasing Pre-trained Contextualised Embeddings" (In EACL 2021). If you use any part of this work, make sure you include the following citation:

```
@inproceedings{kaneko-bollegala-2021-context,
    title={Debiasing Pre-trained Contextualised Embeddings},
    author={Masahiro Kaneko and Danushka Bollegala},
    booktitle = {Proc. of the 16th European Chapter of the Association for Computational Linguistics (EACL)},
    year={2021}
}
```


### Requirements
- python==3.7.3
- torch==1.5.0
- nltk==3.5
- transformers==2.8.0
- tensorboard==2.0.2

### Installation
```
cd transformers
pip install .
```


### To debias your contextualised embeddings
```
curl -o data/news-commentary-v15.en.gz -OL https://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz
gunzip data/news-commentary-v15.en.gz
cd script
./preprocess.sh [bert/roberta/albert/dbert/electra] ../data/news-commentary-v15.en
./debias.sh [bert/roberta/albert/dbert/electra] gpu_id

```

### Our debiased conttextualised embeddings

You can directly download our ``all-token`` debiased [contextualised embeddings](https://drive.google.com/drive/folders/1a99jISCUfTp2E5BNQtIHEelQT-Pf8ayB?usp=sharing).

### License
See the LICENSE file.
