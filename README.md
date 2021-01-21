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


### Our experiment settings
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
cd script
./preprocess.sh [bert/roberta/albert/dbert/electra] /path/to/your/data
./debias.sh [bert/roberta/albert/dbert] gpu_id

```
Output is a debiased binary word embeddings saved in `--save-prefix`


### License
See the LICENSE file.
