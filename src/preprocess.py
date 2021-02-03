import argparse
import regex as re
import nltk

import torch
from torch.utils.data import Dataset

from transformers import *

def parse_args():
    parser = argparse.ArgumentParser()
    tp = lambda x:list(x.split(','))

    parser.add_argument('--input', type=str, required=True,
                        help='Data')
    parser.add_argument('--stereotypes', type=str)
    parser.add_argument('--attributes', type=tp, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'roberta', 'electra', 'albert', 'dbert'])

    args = parser.parse_args()

    return args

def prepare_transformer(args):
    if args.model_type == 'bert':
        pretrained_weights = 'bert-base-uncased'
        model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'roberta':
        pretrained_weights = 'roberta-base'
        model = RobertaModel.from_pretrained(pretrained_weights)
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'albert':
        pretrained_weights = 'albert-base-v2'
        model = AlbertModel.from_pretrained(pretrained_weights)
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'dbert':
        pretrained_weights = 'distilbert-base-uncased'
        model = DistilBertModel.from_pretrained(pretrained_weights)
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'xlnet':
        pretrained_weights = 'xlnet-base-cased'
        model = XLNetModel.from_pretrained(pretrained_weights)
        tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'electra':
        pretrained_weights = 'google/electra-small-discriminator'
        model = ElectraModel.from_pretrained(pretrained_weights)
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'gpt':
        pretrained_weights = 'openai-gpt'
        model = OpenAIGPTModel.from_pretrained(pretrained_weights)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'gpt2':
        pretrained_weights = 'gpt2'
        model = GPT2Model.from_pretrained(pretrained_weights)
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'xl':
        pretrained_weights = 'transfo-xl-wt103'
        model = TransfoXLModel.from_pretrained(pretrained_weights)
        tokenizer = TransfoXLTokenizer.from_pretrained(pretrained_weights)

    return model, tokenizer

def encode_to_is(tokenizer, data, add_special_tokens):
    if type(data) == list:
        data = [tuple(tokenizer.encode(sentence, add_special_tokens=add_special_tokens)) for sentence in data]
    elif type(data) == dict:
        data = {tuple(tokenizer.encode(key, add_special_tokens=add_special_tokens)): tokenizer.encode(value, add_special_tokens=add_special_tokens)
                for key, value in data.items()}

    return data

def split_data(input, dev_rate, max_train_data_size):
    if max_train_data_size > 0:
        train = input[:max_train_data_size]
        dev = input[max_train_data_size:]
    else:
        train = input[int(dev_rate*len(input)):]
        dev = input[:int(dev_rate*len(input))]

    return train, dev

def main(args):
    data = [l.strip() for l in open(args.input)]
    if args.stereotypes:
        stereotypes = [word.strip() for word in open(args.stereotypes)]
        stereotype_set = set(stereotypes)

    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    attributes_l = []
    all_attributes_set = set()
    for attribute in args.attributes:
        l = [word.strip() for word in open(attribute)]
        attributes_l.append(set(l))
        all_attributes_set |= set(l)

    model, tokenizer = prepare_transformer(args)

    if args.stereotypes:
        tok_stereotypes = encode_to_is(tokenizer, stereotypes, add_special_tokens=False)

    neutral_examples = []
    if args.stereotypes:
        neutral_labels = []
    attributes_examples = [[] for _ in range(len(attributes_l))]
    attributes_labels = [[] for _ in range(len(attributes_l))]

    other_num = 0

    for line in data:
        neutral_flag = True
        line = line.strip()
        if len(line) < 1:
            continue
        leng = len(line.split())
        if leng > args.block_size or leng <= 1:
            continue
        tokens_orig = [token.strip() for token in re.findall(pat, line)]
        tokens_lower = [token.lower() for token in tokens_orig]
        token_set = set(tokens_lower)

        attribute_other_l = []
        for i, _ in enumerate(attributes_l):
            a_set = set()
            for j, attribute in enumerate(attributes_l):
                if i != j:
                    a_set |= attribute
            attribute_other_l.append(a_set)

        for i, (attribute_set, other_set) in enumerate(zip(attributes_l, attribute_other_l)):
            if attribute_set & token_set:
                neutral_flag = False
                if not other_set & token_set:
                    orig_line = line
                    line = tokenizer.encode(line, add_special_tokens=True)
                    labels = attribute_set & token_set
                    for label in list(labels):
                        idx = tokens_lower.index(label)
                    label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                    line_ngram = list(nltk.ngrams(line, len(label)))
                    if label not in line_ngram:
                        label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=False))
                        line_ngram = list(nltk.ngrams(line, len(label)))
                        if label not in line_ngram:
                            label = tuple(tokenizer.encode(f'a {tokens_orig[idx]} a'))[1:-1]
                            line_ngram = list(nltk.ngrams(line, len(label)))
                            if label not in line_ngram:
                                label = tuple([tokenizer.encode(f'{tokens_orig[idx]}2')[0]])
                                line_ngram = list(nltk.ngrams(line, len(label)))
                    idx = line_ngram.index(label)
                    attributes_examples[i].append(line)
                    attributes_labels[i].append([idx + j for j in range(len(label))])
                break

        if neutral_flag:
            if args.stereotypes:
                if stereotype_set & token_set:
                    orig_line = line
                    line = tokenizer.encode(line, add_special_tokens=True)
                    labels = stereotype_set & token_set
                    for label in list(labels):
                        idx = tokens_lower.index(label)
                        label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                        line_ngram = list(nltk.ngrams(line, len(label)))
                        if label not in line_ngram:
                            label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=False))
                            line_ngram = list(nltk.ngrams(line, len(label)))
                            if label not in line_ngram:
                                label = tuple(tokenizer.encode(f'a {tokens_orig[idx]} a'))[1:-1]
                                line_ngram = list(nltk.ngrams(line, len(label)))
                                if label not in line_ngram:
                                    label = tuple([tokenizer.encode(f'{tokens_orig[idx]}2')[0]])
                                    line_ngram = list(nltk.ngrams(line, len(label)))
                        idx = line_ngram.index(label)
                        neutral_examples.append(line)
                        neutral_labels.append([idx + i for i in range(len(label))])
            else:
                neutral_examples.append(tokenizer.encode(line, add_special_tokens=True))

    print('neutral:', len(neutral_examples))
    for i, examples in enumerate(attributes_examples):
        print(f'attributes{i}:', len(examples))

    data = {'attributes_examples': attributes_examples,
            'attributes_labels': attributes_labels,
            'neutral_examples': neutral_examples}


    if args.stereotypes:
        data['neutral_labels'] = neutral_labels

    torch.save(data, args.output + '/data.bin')


if __name__ == "__main__":
    args = parse_args()
    main(args)
