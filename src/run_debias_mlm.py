# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import copy
from typing import Dict, List, Tuple

import nltk
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from multiprocessing import Pool
import multiprocessing as multi

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, examples: list, labels: list):
        self.examples = examples
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.labels:
            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)
        else:
            return torch.tensor(self.examples[i], dtype=torch.long)


def create_dataset(data, dataset):
    d = dict()
    for key in data['example'].keys():
        if key not in data['label']:
            d[key] = dataset(data['example'][key], None)
        else:
            d[key] = dataset(data['example'][key], data['label'][key])

    return d


def load_and_cache_examples(data, args, tokenizer):
    if args.line_by_line:
        train_dataset = create_dataset(data['train'], LineByLineTextDataset)
        dev_dataset = create_dataset(data['dev'], LineByLineTextDataset)
        return {'train': train_dataset, 'dev': dev_dataset}
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)

def split_data(attributes_examples, attributes_labels, neutral_examples, neutral_labels, args):
    data = {'train': {'example': {}, 'label': {}}, 'dev': {'example': {}, 'label': {}}}

    for i, (examples, labels) in enumerate(zip(attributes_examples, attributes_labels)):
        idx_l = list(range(len(examples)))
        random.shuffle(idx_l)
        examples = [examples[idx] for idx in idx_l]
        labels = [labels[idx] for idx in idx_l]
        data['train']['example'][f'attribute{i}'] = examples[args.dev_data_size:]
        data['train']['label'][f'attribute{i}'] = labels[args.dev_data_size:]
        data['dev']['example'][f'attribute{i}'] = examples[:args.dev_data_size]
        data['dev']['label'][f'attribute{i}'] = labels[:args.dev_data_size]

    idx_l = list(range(len(neutral_examples)))
    random.shuffle(idx_l)
    neutral_examples = [neutral_examples[idx] for idx in idx_l]
    data['train']['example']['neutral'] = neutral_examples[args.dev_data_size:]
    data['dev']['example']['neutral'] = neutral_examples[:args.dev_data_size]
    if neutral_labels is not None:
        neutral_labels = [neutral_labels[idx] for idx in idx_l]
        data['train']['label']['neutral'] = neutral_labels[args.dev_data_size:]
        data['dev']['label']['neutral'] = neutral_labels[:args.dev_data_size]

    return data


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def create_dataloader(args, datasets, tokenizer, train=False):
    def collate(batch: List[torch.Tensor]):
        if type(batch[0]) == tuple:
            examples, labels = list(zip(*batch))
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id), \
                   pad_sequence(labels, batch_first=True, padding_value=0)
        else:
            return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

    dataloaders = {}
    example_num = 0
    data_distribution = []

    max_size = max([len(value) for key, value in datasets.items() if key != 'neutral'])
    min_size = min([len(value) for key, value in datasets.items() if key != 'neutral'])

    for key, dataset in datasets.items():
        example_num += len(dataset)
        if train:
            dataloaders[key] = iter(DataLoader(dataset, batch_size=args.train_batch_size, collate_fn=collate, shuffle=True))
            data_distribution += [key for _ in range(int(min_size / args.train_batch_size))]
        else:
            dataloaders[key] = iter(DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=collate , shuffle=False))
            data_distribution += [key for _ in range(int(min_size / args.eval_batch_size))]

    return dataloaders, example_num, data_distribution


def train(args, data, datasets, model: PreTrainedModel, original_model, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    train_datasets = datasets['train']
    dev_datasets = datasets['dev']

    train_dataloaders, train_example_num, train_distribution = create_dataloader(args, train_datasets, tokenizer, train=True)
    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)

    train_iter_num = sum([len(dataloader) for dataloader in train_dataloaders.values()])
    dev_iter_num = sum([len(dataloader) for dataloader in dev_dataloaders.values()])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_iter_num // args.gradient_accumulation_steps) + 1
    else:
        t_total = train_iter_num // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    original_model = original_model.module if hasattr(original_model, "module") else original_model  # Take care of distributed/parallel training
    original_model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        original_model = torch.nn.DataParallel(original_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        original_model = torch.nn.parallel.DistributedDataParallel(
            original_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_example_num)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_loss = float('inf')
    best_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (train_iter_num // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (train_iter_num // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")


    model.zero_grad()
    original_model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    def inner_product(x, y):
        return torch.mean(torch.sum(y * x, 3))

    def mean_square(x, y, idx):
        return torch.mean(torch.mean((y - x) ** 2, idx))
        #return torch.mean(torch.sum((y - x) ** 2, 3))

    def save_best_model(best_loss, best_step, dev_dataloaders):
        if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
            eval_loss = evaluate(model, attributes_hiddens, dev_dataloaders)
            #eval_loss = evaluate(args, model, original_model, dev_dataloaders, dev_example_num, dev_distribution, criterion_mse, criterion_ip, feminine_hiddens, masculine_hiddens, gender_hiddens)
            logger.info(" global_step = %s, evaluate loss = %s", global_step, eval_loss)
            tb_writer.add_scalar("eval_loss", eval_loss, global_step)
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_step = global_step
            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            #_rotate_checkpoints(args, checkpoint_prefix)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info(" best_step = %s, best loss = %s", best_step, best_loss)

        return best_loss, best_step

    def get_hiddens_of_model(input):
        model.zero_grad()
        if args.model_type == 'roberta':
            _, _, hiddens = model.roberta(input)
        elif args.model_type == 'bert':
            _, _, hiddens = model.bert(input)
        elif args.model_type == 'albert':
            _, _, hiddens = model.albert(input)
        elif args.model_type == 'dbert':
            _, hiddens = model.distilbert(input)
        elif args.model_type == 'electra':
            _, hiddens = model.electra(input)
        elif args.model_type == 'gpt2':
            _, _, hiddens = model.transformer(input)
        elif args.model_type == 'gpt':
            _, hiddens = model.transformer(input)

        return hiddens

    def attribute_vector_example():
        attributes_hiddens = {f'attribute{i}': [] for i in range(2)}

        dataloaders, _, distribution = create_dataloader(args, train_datasets, tokenizer, train=True)
        for key in distribution:
            if key != 'neutral':
                inputs, labels = next(dataloaders[key])
                inputs = inputs.to(args.device)
                hiddens = get_hiddens_of_model(inputs)
                hiddens = torch.stack(hiddens, 2)
                if labels.size(1) > 1:
                    onehot = torch.eye(hiddens.size(1))
                    zeros = torch.zeros(1, onehot.size(0))
                    onehot = torch.cat((zeros, onehot), 0)
                    onehot = onehot[labels]
                    onehot = torch.sum(onehot, 1)
                    onehot = onehot.view(hiddens.size(0), -1, 1, 1)
                else:
                    onehot = torch.eye(hiddens.size(1))[labels].view(hiddens.size(0), -1, 1, 1)
                onehot = onehot.to(args.device)
                attributes_hiddens[key].append(torch.sum(hiddens * onehot, 1) / labels.size(1))

        # neutralも含まれている
        attribute_size = len(data['train']['example'])
        for i in range(attribute_size - 1):
            attributes_hiddens[f'attribute{i}'] = torch.mean(torch.cat(attributes_hiddens[f'attribute{i}'], 0), 0).detach().unsqueeze(0)

        return attributes_hiddens

    def forward(attributes_hiddens, dataloaders, key):
        inputs = next(dataloaders[key])
        if len(inputs) == 2:
            inputs, labels = inputs
            labels = labels.to(args.device)
        else:
            labels = None
        inputs = inputs.to(args.device)
        if args.model_type == 'roberta':
            final_layer_hiddens, first_token_hidden, all_layer_hiddens = model.roberta(inputs)
            if 'neutral' != key:
                with torch.no_grad():
                    final_layer_original_hiddens, _, all_layer_original_hiddens = original_model.roberta(inputs)
                if args.token_loss:
                    token_predicts = model.lm_head(final_layer_hiddens)
                    token_original = original_model.lm_head(final_layer_original_hiddens)
        elif args.model_type == 'bert':
            final_layer_hiddens, first_token_hidden, all_layer_hiddens = model.bert(inputs)
            if 'neutral' != key:
                with torch.no_grad():
                    final_layer_original_hiddens, _, all_layer_original_hiddens = original_model.bert(inputs)
                if args.token_loss:
                    token_predicts = model.cls(final_layer_hiddens)
                    token_original = original_model.cls(final_layer_original_hiddens)
        elif args.model_type == 'albert':
            final_layer_hiddens, first_token_hidden, all_layer_hiddens = model.albert(inputs)
            if 'neutral' != key:
                with torch.no_grad():
                    final_layer_original_hiddens, _, all_layer_original_hiddens = original_model.albert(inputs)
                if args.token_loss:
                    token_predicts = model.classifier(final_layer_hiddens)
                    token_original = original_model.classifier(final_layer_original_hiddens)
        elif args.model_type == 'dbert':
            final_layer_hiddens, all_layer_hiddens = model.distilbert(inputs)
            if 'neutral' != key:
                with torch.no_grad():
                    final_layer_original_hiddens, all_layer_original_hiddens = original_model.distilbert(inputs)
                if args.token_loss:
                    token_predicts = model.classifier(final_layer_hiddens)
                    token_original = original_model.classifier(final_layer_original_hiddens)
        elif args.model_type == 'electra':
            final_layer_hiddens, all_layer_hiddens = model.electra(inputs)
            if 'neutral' != key:
                with torch.no_grad():
                    final_layer_original_hiddens, all_layer_original_hiddens = original_model.electra(inputs)
                if args.token_loss:
                    hiddens = model.generator_predictions(final_layer_hiddens)
                    token_predicts = model.generator_lm_head(hiddens)
                    original_hiddens = original_model.generator_predictions(final_layer_original_hiddens)
                    token_original = original_model.generator_lm_head(original_hiddens)
        elif args.model_type == 'gpt2':
            final_layer_hiddens, first_token_hidden, all_layer_hiddens = model.transformer(inputs)
            if 'neutral' != key:
                with torch.no_grad():
                   final_layer_original_hiddens, _, all_layer_original_hiddens = original_model.transformer(inputs)
                if args.token_loss:
                    token_predicts = model.lm_head(final_layer_hiddens)
                    token_original = original_model.lm_head(final_layer_original_hiddens)
        elif args.model_type == 'gpt':
            final_layer_hiddens, all_layer_hiddens = model.transformer(inputs)
            if 'neutral' != key:
                with torch.no_grad():
                    final_layer_original_hiddens, all_layer_original_hiddens = original_model.transformer(inputs)
                if args.token_loss:
                    token_predicts = model.lm_head(final_layer_hiddens)
                    token_original = original_model.lm_head(final_layer_original_hiddens)

        all_layer_hiddens = torch.stack(all_layer_hiddens, 2)
        if 'neutral' != key:
            all_original_hiddens =  torch.stack(all_layer_original_hiddens, 2)
            all_original_hiddens = all_original_hiddens.detach()
            if args.token_loss:
                original_hiddens - original_hiddens.detach()
                token_original = token_original.detach()
        if args.debias_layer == 'all':
            target_layer_hiddens = all_layer_hiddens
            target_original_hiddens = all_layer_hiddens
        else:
            if args.debias_layer == 'first':
                idx = 0
            elif args.debias_layer == 'last':
                idx = -1
            target_layer_hiddens = all_layer_hiddens[:,:,idx]
            target_layer_hiddens = target_layer_hiddens.unsqueeze(2)
            if 'neutral' != key:
                target_original_hiddens = all_original_hiddens[:,:,idx]
                target_original_hiddens = target_original_hiddens.unsqueeze(2)
            else:
                attributes_hiddens = {key: value[:,idx,:].unsqueeze(1) for key, value in attributes_hiddens.items()}

        if args.loss_target == 'sentence' or labels is None:
            attributes_hiddens = {key: value.unsqueeze(1) for key, value in attributes_hiddens.items()}
        #elif args.loss_target == 'token' and key == 'neutral':
        elif args.loss_target == 'token':
            if labels.size(1) > 1:
                onehot = torch.eye(target_layer_hiddens.size(1))
                zeros = torch.zeros(1, onehot.size(0))
                onehot = torch.cat((zeros, onehot), 0)
                onehot = onehot[labels]
                onehot = torch.sum(onehot, 1)
                onehot = onehot.view(target_layer_hiddens.size(0), -1, 1, 1)
            else:
                onehot = torch.eye(target_layer_hiddens.size(1))[labels].view(target_layer_hiddens.size(0), -1, 1, 1)
            onehot = onehot.to(args.device)
            target_layer_hiddens = torch.sum(target_layer_hiddens * onehot, 1).unsqueeze(1) / labels.size(1)
            if 'neutral' != key:
                target_original_hiddens = torch.sum(target_original_hiddens * onehot, 1).unsqueeze(1) / labels.size(1)
            else:
                attributes_hiddens = {key: value.expand(target_layer_hiddens.size(0),
                                                        1,
                                                        value.size(1),
                                                        value.size(2))
                                      for key, value in attributes_hiddens.items()}

        if 'neutral' == key:
            loss = 0
            for attribute_hiddens in attributes_hiddens.values():
                tmp_loss = criterion_ip(target_layer_hiddens, attribute_hiddens)
                if args.square_loss:
                    tmp_loss = tmp_loss ** 2
                tmp_loss *= alpha
                loss += tmp_loss
        else:
            #loss = criterion_ms(target_layer_hiddens, target_original_hiddens)
            loss = criterion_ms(all_layer_hiddens, all_original_hiddens, 3)
            if args.token_loss:
                loss += criterion_ms(token_predicts, token_original, 2)
                #loss += criterion_ms(hiddens, original_hiddens, 2)
            loss *= beta

        return loss

    #def evaluate(args, model: PreTrainedModel, original_model, dev_dataloaders, dev_example_num, dev_distribution, criterion_mse, criterion_ip, feminine_hiddens, masculine_hiddens, gender_hiddens, prefix="") -> Dict:
    def evaluate(model, attributes_hiddens, dev_dataloaders, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = args.output_dir

        if args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", dev_example_num)
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        model.eval()
        #criterion.eval()

        for key in tqdm(dev_distribution):
            with torch.no_grad():
                loss = forward(attributes_hiddens, dev_dataloaders, key)

                eval_loss += loss.item()

                model.zero_grad()
                original_model.zero_grad()

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        '''
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            logger.info("  Loss = %s", eval_loss)
            writer.write("Loss = %s\n" % (eval_loss))
        '''

        return eval_loss

    #criterion_ms = torch.nn.MSELoss()
    criterion_ms = mean_square
    #criterion.train()
    criterion_ip = inner_product
    original_model.eval()

    alpha, beta = args.weighted_loss
    alpha = float(alpha)
    beta = float(beta)

    train_loss = 0.0

    for _ in train_iterator:

        random.shuffle(train_distribution)
        epoch_iterator = tqdm(train_distribution, desc="Iteration", disable=args.local_rank not in [-1, 0])

        model.eval()
        with torch.no_grad():
            attributes_hiddens = attribute_vector_example()

        for step, key in enumerate(epoch_iterator):
            model.train()

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            loss = forward(attributes_hiddens, train_dataloaders, key)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                original_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info(" global_step = %s, train loss = %s", global_step, train_loss)
                    train_loss = 0.0
                    # Log metrics
                    best_loss, best_step = save_best_model(best_loss, best_step, dev_dataloaders)
                    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            train_dataloaders, train_example_num, train_distribution = create_dataloader(args, train_datasets, tokenizer, train=True)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    dev_dataloaders, dev_example_num, dev_distribution = create_dataloader(args, dev_datasets, tokenizer, train=False)
    best_loss, best_step = save_best_model(best_loss, best_step, dev_dataloaders)

    if args.local_rank in [-1, 0]:
        tb_writer.close()



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_file", default=None, type=str, required=True, help="The input data file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--loss_target", type=str, help=""
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weighted_loss", default=[1.0, 1.0], nargs='*', help="")
    parser.add_argument("--debias_layer", default=all, type=str, choices=['all', 'first', 'last'], help="")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--square_loss", action='store_true', help="")
    parser.add_argument("--token_loss", action='store_true', help="")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--dev_data_size", type=int, default=1000, help="")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    tp = lambda x:list(x.split(','))
    parser.add_argument('--exclusion_list', type=tp, default=[], help='')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    '''
    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    '''

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    config.output_hidden_states = 'true' # 全層の隠れ層を取得する
    #config.output_attentions = 'true' # 全層のアテンションを取得する

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        original_model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)
        original_model = AutoModelWithLMHead.from_config(config)

    # GPT-2 and GPT do not have pad.
    if tokenizer._pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        original_model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    original_model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    data = torch.load(args.data_file)

    attributes_examples = data['attributes_examples']
    attributes_labels = data['attributes_labels']
    neutral_examples = data['neutral_examples']

    if 'neutral_labels' in data:
        neutral_labels = data['neutral_labels']
        splited_data = split_data(attributes_examples, attributes_labels, neutral_examples, neutral_labels, args)
    else:
        splited_data = split_data(attributes_examples, attributes_labels, neutral_examples, None, args)

    datasets = load_and_cache_examples(splited_data, args, tokenizer)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.local_rank == 0:
        torch.distributed.barrier()

    train(args, splited_data, datasets, model, original_model, tokenizer)


if __name__ == "__main__":
    main()
