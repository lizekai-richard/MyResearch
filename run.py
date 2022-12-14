from tqdm import tqdm
import os
from torch import optim, nn
from util import *
import time
import shutil
import random
import torch
import torch.nn.functional as F
from model import BaseLineModel, NewModel
from sp_model import SupportingFactBaseline
from ques_type_classification.type_classification import TypeClassifier
from data import *
from transformers import RobertaTokenizer, BertTokenizer
import torch.distributed as dist


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def select_top_k(logits, k=5):
    # logits shape: [bsz, 2]
    softmax_logits = F.softmax(logits, dim=-1)
    scores = torch.max(softmax_logits, dim=-1)[0]
    sorted, indices = torch.sort(scores, dim=-1, descending=True)
    return indices[:k]


def predict_sp(model, max_length, hotpot_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for example in tqdm(hotpot_data):
        sp_data = prepare_sp_data(example)
        pred_sp = []
        ques_input_ids, ques_attn_mask, ctx_input_ids, ctx_attn_mask = prepare_input_data(
            sp_data,
            tokenizer,
            max_length
        )

        logits = model(ques_input_ids, ques_attn_mask, ctx_input_ids,
                       ctx_attn_mask)
        topk_indices = select_top_k(logits).cpu().tolist()
        for i in topk_indices:
            pred_sp.append(sp_data[i]['context'])

        example['sp_sentences'] = pred_sp

    return hotpot_data


def train(config, ngpus_per_node):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py'])

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))



    print("loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(config.roberta_config)
    print("Done!")

    print("Building DataLoader...")
    train_ds = HotpotQADataset(config, is_train=True)
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None
    train_loader = generate_dataloader(train_ds, config.batch_size, data_sampler=train_sampler)

    dev_ds = HotpotQADataset(config, is_train=False)
    if config.distributed:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)
    else:
        dev_sampler = None
    dev_loader = generate_dataloader(dev_ds, config.batch_size, is_train=False, data_sampler=dev_sampler)
    print("Done!")

    print("Building model...")
    qa_model = NewModel(config)
    logging('nparams {}'.format(sum([p.nelement() for p in qa_model.parameters() if p.requires_grad])))

    q_type_model = TypeClassifier(config)
    sp_model = SupportingFactBaseline(config)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            qa_model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            qa_model = nn.parallel.DistributedDataParallel(qa_model, device_ids=[config.gpu],
                                                           find_unused_parameters=True)
        else:
            qa_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            qa_model = nn.parallel.DistributedDataParallel(qa_model, find_unused_parameters=True)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        qa_model = qa_model.cuda(config.gpu)
        sp_model = sp_model.cuda(config.gpu)
        q_type_model = q_type_model.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(qa_model).cuda()

    for param in sp_model.parameters():
        param.requires_grad_(False)

    for param in q_type_model.parameters():
        param.requires_grad_(False)
    print("Done!")

    lr = config.init_lr
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, qa_model.parameters()), lr=config.init_lr,
                            weight_decay=config.weight_decay)
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    qa_model.train()

    for epoch in range(config.num_epochs):
        for batch in train_loader:
            ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_mask, sub_q1_input_ids, sub_q1_attn_masks, \
            sub_q2_input_ids, sub_q2_attn_masks, sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, \
            sub_q2_features, q_features, sub_q1_mask, sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, \
            answer_types, ques_types, qa_ids = batch

            ctx_input_ids, ctx_attn_mask = ctx_input_ids.cuda(), ctx_attn_masks.cuda()
            sub_q1_input_ids, sub_q1_attn_masks = sub_q1_input_ids.cuda(), sub_q1_attn_masks.cuda()
            sub_q2_input_ids, sub_q2_attn_masks = sub_q2_input_ids.cuda(), sub_q2_attn_masks.cuda()
            answer_starts, answer_ends, answer_types, ques_types = answer_starts.cuda(), answer_ends.cuda(), \
                                                                   answer_types.cuda(), ques_types.cuda()

            type_logits = q_type_model(ques_input_ids, ques_attn_mask, q_edge_list, q_features, q_mask)
            type_pred = type_logits.argmax(dim=-1).cpu().tolist()

            logit_start, logit_end, logit_answer_type = qa_model(ctx_input_ids, ctx_attn_mask, sub_q1_input_ids,
                                                                 sub_q1_attn_masks, sub_q2_input_ids, sub_q2_attn_masks,
                                                                 type_pred)

            loss = criterion(logit_start, answer_starts) + criterion(logit_end, answer_ends) + criterion(
                logit_answer_type, answer_types)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch,
                                                                                                                  global_step,
                                                                                                                  lr,
                                                                                                                  elapsed * 1000 / config.period,
                                                                                                                  cur_loss))
                total_loss = 0
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                metrics = evaluate_batch(dev_loader, model, criterion, tokenizer)
                model.train()

                logging('-' * 89)
                logging(
                    '| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(
                        global_step // config.checkpoint,
                        epoch, time.time() - eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0
        if stop_train:
            break
    logging('best_dev_F1 {}'.format(best_dev_F1))


def evaluate_batch(data_iter, model, criterion, tokenizer):
    answer_list = []
    pred_list = []
    total_loss, step_cnt = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(data_iter):
            ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_mask, sub_q1_input_ids, sub_q1_attn_masks, \
            sub_q2_input_ids, sub_q2_attn_masks, sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, \
            sub_q2_features, q_features, sub_q1_mask, sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, \
            answer_types, ques_types, qa_ids = batch

            ctx_input_ids, ctx_attn_mask = ctx_input_ids.cuda(), ctx_attn_masks.cuda()
            sub_q1_input_ids, sub_q1_attn_masks = sub_q1_input_ids.cuda(), sub_q1_attn_masks.cuda()
            sub_q2_input_ids, sub_q2_attn_masks = sub_q2_input_ids.cuda(), sub_q2_attn_masks.cuda()
            answer_starts, answer_ends, answer_typess = answer_starts.cuda(), answer_ends.cuda(), answer_types.cuda()

            logit_start, logit_end, logit_answer_type = model(ctx_input_ids, ctx_attn_mask, sub_q1_input_ids,
                                                              sub_q1_attn_masks, sub_q2_input_ids, sub_q2_attn_masks)

            loss = criterion(logit_start, answer_starts) + criterion(logit_end, answer_ends) + criterion(
                logit_answer_type, answer_types)

            start_pred = torch.argmax(logit_start, dim=-1)
            end_pred = torch.argmax(logit_end, dim=-1)
            answer_type_pred = torch.argmax(logit_answer_type, dim=-1)

            predicted_answers = [
                convert_ids_to_tokens(ctx_input_ids[i], tokenizer, start_pred[i], end_pred[i], answer_type_pred[i])
                for i in range(len(qa_ids))]

            answer_list.extend(answer_texts)
            pred_list.extend(predicted_answers)

            total_loss += loss.item()
            step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(answer_list, pred_list)
    metrics['loss'] = loss

    return metrics


def predict(data_iter, model, tokenizer, config, prediction_file):
    pred_dict = {}
    sp_dict = {}
    sp_th = config.sp_threshold
    for step, batch in enumerate(tqdm(data_iter)):
        ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_mask, sub_q1_input_ids, sub_q1_attn_masks, \
        sub_q2_input_ids, sub_q2_attn_masks, sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, \
        sub_q2_features, q_features, sub_q1_mask, sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, \
        answer_types, ques_types, qa_ids = batch

        ctx_input_ids, ctx_attn_mask = ctx_input_ids.cuda(), ctx_attn_masks.cuda()
        ques_input_ids, ques_attn_mask = ques_input_ids.cuda(), ques_attn_mask.cuda()
        sub_q1_input_ids, sub_q1_attn_masks = sub_q1_input_ids.cuda(), sub_q1_attn_masks.cuda()
        sub_q2_input_ids, sub_q2_attn_masks = sub_q2_input_ids.cuda(), sub_q2_attn_masks.cuda()
        answer_starts, answer_ends, answer_types, ques_types = answer_starts.cuda(), answer_ends.cuda(), \
                                                               answer_types.cuda(), ques_types.cuda()

        logit_start, logit_end, logit_answer_type, logit_ques_types = model(ctx_input_ids, ctx_attn_mask,
                                                                            ques_input_ids, ques_attn_mask,
                                                                            sub_q1_input_ids, sub_q1_attn_masks,
                                                                            sub_q2_input_ids, sub_q2_attn_masks)

        start_pred = torch.argmax(logit_start, dim=-1)
        end_pred = torch.argmax(logit_end, dim=-1)
        answer_type_pred = torch.argmax(logit_answer_type, dim=-1)

        predicted_answers = [
            convert_ids_to_tokens(ctx_input_ids[i], tokenizer, start_pred[i], end_pred[i], answer_type_pred[i])
            for i in range(len(qa_ids))]
        predicted_answers = {idx: text for (idx, text) in zip(qa_ids, predicted_answers)}
        pred_dict.update(predicted_answers)

    prediction = {'answer': pred_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)


def baseline(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.distributed and config.local_rank != -1:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(config.local_rank)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py'])

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    print("loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(config.roberta_config)
    print("Done!")

    print("Building DataLoader...")
    train_ds = BaselineDataset(config, is_train=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader = generate_baseline_loader(train_ds, config.batch_size, data_sampler=train_sampler)

    dev_ds = BaselineDataset(config, is_train=False)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)
    dev_loader = generate_baseline_loader(dev_ds, config.batch_size, data_sampler=dev_sampler, is_train=False)
    print("Done!")

    print("Building model...")
    model = BaseLineModel(config)
    if config.distributed and config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[config.local_rank],
                                                          output_device=config.local_rank)
    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    print("Done!")

    lr = config.init_lr
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()

    for epoch in range(config.num_epochs):
        for batch in train_loader:

            context_input_ids, context_attention_mask, sub_q1_input_ids, sub_q1_attention_mask, sub_q2_input_ids, \
            sub_q2_attention_mask, answer_starts, answer_ends, answer_types, answer_texts = batch

            context_input_ids = context_input_ids.cuda(non_blocking=True)
            context_attention_mask = context_attention_mask.cuda(non_blocking=True)
            sub_q1_input_ids = sub_q1_input_ids.cuda(non_blocking=True)
            sub_q1_attention_mask = sub_q1_attention_mask.cuda(non_blocking=True)
            sub_q2_input_ids = sub_q2_input_ids.cuda(non_blocking=True)
            sub_q2_attention_mask = sub_q2_attention_mask.cuda(non_blocking=True)

            answer_starts = answer_starts.cuda(non_blocking=True)
            answer_ends = answer_ends.cuda(non_blocking=True)
            answer_types = answer_types.cuda(non_blocking=True)

            logit_start, logit_end, logit_type = model(context_input_ids, context_attention_mask, sub_q1_input_ids,
                                                       sub_q1_attention_mask, sub_q2_input_ids, sub_q2_attention_mask)

            loss = criterion(logit_start, answer_starts) + criterion(logit_end, answer_ends) + criterion(logit_type,
                                                                                                         answer_types)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch,
                                                                                                                  global_step,
                                                                                                                  lr,
                                                                                                                  elapsed * 1000 / config.period,
                                                                                                                  cur_loss))
                total_loss = 0
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                metrics = baseline_evaluate_batch(dev_loader, model, criterion, tokenizer)
                model.train()

                logging('-' * 89)
                logging(
                    '| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(
                        global_step // config.checkpoint,
                        epoch, time.time() - eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0
        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))


def baseline_evaluate_batch(data_iter, model, criterion, tokenizer, is_debug=False):
    answer_list = []
    pred_list = []
    total_loss, step_cnt = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(data_iter):
            context_input_ids, context_attention_mask, sub_q1_input_ids, sub_q1_attention_mask, sub_q2_input_ids, \
            sub_q2_attention_mask, answer_starts, answer_ends, answer_types, answer_texts = batch

            context_input_ids = context_input_ids.cuda(non_blocking=True)
            context_attention_mask = context_attention_mask.cuda(non_blocking=True)
            sub_q1_input_ids = sub_q1_input_ids.cuda(non_blocking=True)
            sub_q1_attention_mask = sub_q1_attention_mask.cuda(non_blocking=True)
            sub_q2_input_ids = sub_q2_input_ids.cuda(non_blocking=True)
            sub_q2_attention_mask = sub_q2_attention_mask.cuda(non_blocking=True)

            answer_starts = answer_starts.cuda(non_blocking=True)
            answer_ends = answer_ends.cuda(non_blocking=True)
            answer_types = answer_types.cuda(non_blocking=True)

            logit_start, logit_end, logit_type = model(context_input_ids, context_attention_mask, sub_q1_input_ids,
                                                       sub_q1_attention_mask, sub_q2_input_ids, sub_q2_attention_mask)

            loss = criterion(logit_start, answer_starts) + criterion(logit_end, answer_ends) + criterion(logit_type,
                                                                                                         answer_types)

            start_pred = torch.argmax(logit_start, dim=-1)
            end_pred = torch.argmax(logit_end, dim=-1)
            type_pred = torch.argmax(logit_type, dim=-1)

            predicted_answers = [convert_ids_to_tokens(context_input_ids[i], tokenizer, start_pred[i], end_pred[i],
                                                       type_pred[i]) for i in range(len(answer_texts))]

            answer_list.extend(answer_texts)
            pred_list.extend(predicted_answers)

            total_loss += loss.item()
            step_cnt += 1
            save_eval_data(answer_texts, predicted_answers, step)
    loss = total_loss / step_cnt
    metrics = evaluate(answer_list, pred_list)
    metrics['loss'] = loss

    return metrics

# def test(config):
#     with open(config.word_emb_file, "r") as fh:
#         word_mat = np.array(json.load(fh), dtype=np.float32)
#     with open(config.char_emb_file, "r") as fh:
#         char_mat = np.array(json.load(fh), dtype=np.float32)
#     if config.data_split == 'dev':
#         with open(config.dev_eval_file, "r") as fh:
#             dev_eval_file = json.load(fh)
#     else:
#         with open(config.test_eval_file, 'r') as fh:
#             dev_eval_file = json.load(fh)
#     with open(config.idx2word_file, 'r') as fh:
#         idx2word_dict = json.load(fh)
#
#     random.seed(config.seed)
#     np.random.seed(config.seed)
#     torch.manual_seed(config.seed)
#     torch.cuda.manual_seed_all(config.seed)
#
#     def logging(s, print_=True, log_=True):
#         if print_:
#             print(s)
#         if log_:
#             with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
#                 f_log.write(s + '\n')
#
#     if config.data_split == 'dev':
#         dev_buckets = get_buckets(config.dev_record_file)
#         para_limit = config.para_limit
#         ques_limit = config.ques_limit
#     elif config.data_split == 'test':
#         para_limit = None
#         ques_limit = None
#         dev_buckets = get_buckets(config.test_record_file)
#
#     def build_dev_iterator():
#         return DataIterator(dev_buckets, config.batch_size, para_limit,
#                             ques_limit, config.char_limit, False, config.sent_limit)
#
#     if config.sp_lambda > 0:
#         model = SPModel(config, word_mat, char_mat)
#     else:
#         model = Model(config, word_mat, char_mat)
#     ori_model = model.cuda()
#     ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
#     model = nn.DataParallel(ori_model)
#
#     model.eval()
#     predict(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file)
