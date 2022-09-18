# from run import train, baseline
import os
import argparse
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from run import train


def parse_option():
    parser = argparse.ArgumentParser()
    
    word_emb_file = "./dataset/word_emb.json"
    train_data_file = "./dataset/bridge_train_data.json"
    dev_data_file = "./dataset/bridge_dev_data.json"
    debug_data_file = "./dataset/train_debug.json"
    
    word2idx_file = "./dataset/word2idx.json"
    idx2word_file = './dataset/idx2word.json'
    
    # default settings
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save', type=str, default='HOTPOT')
    parser.add_argument('--fullwiki', action='store_true')
    parser.add_argument('--prediction_file', type=str)

    # data files
    parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
    parser.add_argument('--train_data_file', type=str, default=train_data_file)
    parser.add_argument('--dev_data_file', type=str, default=dev_data_file)
    parser.add_argument('--debug_data_file', type=str, default=debug_data_file)
    parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
    parser.add_argument('--idx2word_file', type=str, default=idx2word_file)
    parser.add_argument('--glove_word_size', type=int, default=int(2.2e6))
    parser.add_argument('--glove_dim', type=int, default=300)

    # training
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--num_epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--init_lr', type=float, default=0.00001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='5,8',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--checkpoint', type=int, default=1000)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--patience', type=int, default=1)
    
    # pretrain model settings
    parser.add_argument('--roberta_config', type=str, default="roberta-base")
    parser.add_argument('--roberta_dim', type=int, default=768)
    parser.add_argument('--context_max_length', type=int, default=256)
    parser.add_argument('--q_max_length', type=int, default=50)
    parser.add_argument('--sub_q_max_length', type=int, default=30)

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--num_types', type=int, default=2)
    
    config = parser.parse_args()
    return config


def main():
    config = parse_option()

    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can '
                      'slow down your training considerably! You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(train, nprocs=ngpus_per_node)
    else:
        # Simply call main_worker function
        train(config, ngpus_per_node)


if __name__ == '__main__':
    main()

