#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number

"""
Author: Wesley (liwanshui12138@gmail.com)
Date: 2022-6-10
"""
from logging import log
import os
import sys
import argparse
import time
import math
import pickle
import random
import csv
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from util import NLIProcessor, adjust_learning_rate, accuracy, warmup_learning_rate, load_and_cache_examples, \
    save_model, AverageMeter, ProgressMeter
from bert_model import BertForCL, LinearClassifier, PairSupConBert


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model dataset
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--model', type=str, default='BERT')
    parser.add_argument('--dataset', type=str, default='SNLI',
                        choices=['SNLI', 'MNLI', 'EXIM', 'SKILL', 'SKILL_Q'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='./datasets/preprocessed', help='path to custom dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    # distribute
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    # ckpt
    parser.add_argument('--ckpt_bert', type=str, default='', help="path to pre-trained model")
    parser.add_argument('--ckpt_classifier', type=str, default='', help="path to pre-trained model")
    parser.add_argument('--label_num', type=int, help="num of label")
    args = parser.parse_args()

    return args


def test(val_loader, model, classifier, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top = AverageMeter('Accuracy', ':.2f')

    # switch to validate mode
    model.eval()
    classifier.eval()
    res = {}
    res["label"] = []
    res["fc"] = []
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            bsz = batch[0].size(0)
            if args.gpu is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

            # compute loss
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            features = model(**inputs)

            logits = classifier(features)
            # update metric
            _, pred = logits.topk(1, 1, True, True)
            res["label"] += pred.t().cpu().numpy().tolist()[0]
            res["fc"] += features[1].cpu().numpy().tolist()
            acc1 = accuracy(logits, batch[3])
            top.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    np.save(os.path.join('/'.join(args.ckpt_bert.split('/')[:-1]), 'fc.npy'), np.array(res["fc"]))
    np.save(os.path.join('/'.join(args.ckpt_bert.split('/')[:-1]), 'label.npy'), np.array(res["label"]))
    return top.avg


def test_mnli(val_loader, model, classifier, args):
    # switch to validate mode
    model.eval()
    classifier.eval()
    res = {}
    res["id"] = []
    res["label"] = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if args.gpu is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(args.gpu, non_blocking=True)

            # compute loss
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            features = model.encoder(**inputs)
            logits = classifier(features[1])
            # update metric
            _, pred = logits.topk(1, 1, True, True)
            res["id"] += batch[4].cpu().numpy().tolist()
            res["label"] += pred.t().cpu().numpy().tolist()[0]
    return res


def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.label_num == 3:
        label_map = ["contradiction", "entailment", "neutral"]
    elif args.label_num == 2:
        label_map = ['explicit', 'implicit']
    elif args.label_num == 7:
        label_map = ['action', 'causal relationship', 'character', 'feeling', 'outcome resolution', 'prediction',
                     'setting']
    else:
        raise ValueError('wrong label num')

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = PairSupConBert(BertForCL.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=128,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ), is_train=False)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    classifier = LinearClassifier(BertForCL.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.label_num,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    ), num_classes=args.label_num)

    ckpt_bert = torch.load(args.ckpt_bert, map_location='cpu')
    ckpt_classifier = torch.load(args.ckpt_classifier, map_location='cpu')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            classifier.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

    model.load_state_dict(ckpt_bert['model'])
    classifier.load_state_dict(ckpt_classifier['model'])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(classifier, torch.nn.DataParallel):
        classifier = classifier.module

    cudnn.benchmark = True

    # construct data loader
    if args.dataset in ['SNLI', 'EXIM', 'SKILL', 'SKILL_Q']:
        test_file = os.path.join(args.data_folder, args.dataset, "test_data.pkl")
        print("load dataset")
        with open(test_file, "rb") as pkl:
            test_processor = NLIProcessor(pickle.load(pkl))
        test_dataset = load_and_cache_examples(args, test_processor, tokenizer, "test", args.dataset)
        
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)
        acc = test(test_loader, model, classifier, args)
        print("Accuracy: {:.2f}".format(acc))

    elif args.dataset == 'MNLI':
        test_match = os.path.join(args.data_folder, args.dataset, "matched_test_data.pkl")
        test_mismatch = os.path.join(args.data_folder, args.dataset, "mismatched_test_data.pkl")

        print("load dataset")
        with open(test_match, "rb") as pkl:
            pkls = pickle.load(pkl)
            match_processor = NLIProcessor(pkls)

        match_dataset = load_and_cache_examples(args, match_processor, tokenizer, "test_match", args.dataset)
        match_sampler = None

        with open(test_mismatch, "rb") as pkl:
            mismatch_processor = NLIProcessor(pickle.load(pkl))
        mismatch_dataset = load_and_cache_examples(args, mismatch_processor, tokenizer, "test_mismatch", args.dataset)
        mismatch_sampler = None

        match_loader = torch.utils.data.DataLoader(
            match_dataset, batch_size=args.batch_size, shuffle=(match_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=match_sampler)
        mismatch_loader = torch.utils.data.DataLoader(
            mismatch_dataset, batch_size=args.batch_size, shuffle=(mismatch_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=mismatch_sampler)
        res1 = test_mnli(match_loader, model, classifier, args)
        res2 = test_mnli(mismatch_loader, model, classifier, args)
        csvFile1 = open('./matched_test_submission.csv', 'w', newline='')
        writer1 = csv.writer(csvFile1)
        writer1.writerow(["pairID", "gold_label"])
        for i in range(len(res1["id"])):
            writer1.writerow([res1["id"][i], label_map[res1["label"][i]]])
        csvFile1.close()
        csvFile2 = open('./mismatched_test_submission.csv', 'w', newline='')
        writer2 = csv.writer(csvFile2)
        writer2.writerow(["pairID", "gold_label"])
        for i in range(len(res2["id"])):
            writer2.writerow([res2["id"][i], label_map[res2["label"][i]]])
        csvFile2.close()
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))


if __name__ == '__main__':
    main()
