import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from util import *


class HotpotQADataset(Dataset):
    def __init__(self, config, is_train=True):
        super(HotpotQADataset, self).__init__()
        self.config = config
        if is_train:
            self.data = read_data_file(config.train_data_file)
        else:
            self.data = read_data_file(config.dev_data_file)
        self.word2idx, self.idx2word, self.word_emb = read_features(config.word2idx_file, config.idx2word_file,
                                                                    config.word_emb_file)
        self.tokenizer = AutoTokenizer.from_pretrained(config.roberta_config)
        self.nlp = spacy.load('en_core_web_sm')

    def __getitem__(self, index):
        data_example = self.data[index]
        processed_example = {}

        sub_q1_edge_list, sub_q2_edge_list = build_sub_question_graph(self.nlp, data_example)
        q_edge_list = build_question_graph(self.nlp, data_example)
        q_features, sub_q1_features, sub_q2_features = build_node_features(self.nlp, data_example, self.word_emb,
                                                                           self.word2idx)
        context_sentences = extract_context(data_example)
        context_input_ids, context_attn_mask, context_offset = construct_roberta_features(self.tokenizer,
                                                                                          context_sentences,
                                                                                          self.config.context_max_length)
        ques_input_ids, ques_attn_mask, _ = construct_roberta_features(self.tokenizer, data_example['question'],
                                                                       self.config.q_max_length)
        sub_q1_text = data_example['sub_questions'][0]
        sub_q1_input_ids, sub_q1_attn_mask, _ = construct_roberta_features(self.tokenizer, sub_q1_text,
                                                                           self.config.sub_q_max_length)
        sub_q2_text = data_example['sub_questions'][1]
        sub_q2_input_ids, sub_q2_attn_mask, _ = construct_roberta_features(self.tokenizer, sub_q2_text,
                                                                           self.config.sub_q_max_length)

        answer_start_char = get_answer_starts(context_sentences, data_example['answer'])

        answer_start_index, answer_end_index = get_answer_mapping(context_input_ids, data_example['answer'],
                                                                  answer_start_char, context_offset)

        answer_type = get_answer_type(data_example['answer'])
        ques_type = get_ques_type(data_example['type'])

        processed_example['q_edge_list'] = q_edge_list
        processed_example['sub_q1_edge_list'], processed_example[
            'sub_q2_edge_list'] = sub_q1_edge_list, sub_q2_edge_list
        processed_example['q_features'] = q_features
        processed_example['sub_q1_features'], processed_example['sub_q2_features'] = sub_q1_features, sub_q2_features
        processed_example['context_input_ids'] = context_input_ids
        processed_example['context_mask'] = context_attn_mask
        processed_example['ques_input_ids'] = ques_input_ids
        processed_example['ques_attn_mask'] = ques_attn_mask
        processed_example['sub_q1_input_ids'] = sub_q1_input_ids
        processed_example['sub_q1_mask'] = sub_q1_attn_mask
        processed_example['sub_q2_input_ids'] = sub_q2_input_ids
        processed_example['sub_q2_mask'] = sub_q2_attn_mask
        processed_example['answer_text'] = data_example['answer']
        processed_example['answer_start_index'] = answer_start_index
        processed_example['answer_end_index'] = answer_end_index
        processed_example['answer_type'] = answer_type
        processed_example['ques_type'] = ques_type
        processed_example['id'] = data_example['_id']

        return processed_example

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    ctx_input_ids, ctx_attn_masks = [], []
    ques_input_ids, ques_attn_masks = [], []
    sub_q1_input_ids, sub_q1_attn_masks = [], []
    sub_q2_input_ids, sub_q2_attn_masks = [], []
    sub_q1_edge_list, sub_q2_edge_list, q_edge_list = [], [], []
    sub_q1_features, sub_q2_features, q_features = [], [], []
    sub_q1_mask, sub_q2_mask, q_mask = [], [], []
    answer_texts, answer_starts, answer_ends, answer_types = [], [], [], []
    ques_types = []
    qa_ids = []

    for example in batch:
        ctx_input_ids.append(example['context_input_ids'])
        ctx_attn_masks.append(example['context_mask'])

        ques_input_ids.append(example['ques_input_ids'])
        ques_attn_masks.append(example['ques_attn_mask'])

        sub_q1_input_ids.append(example['sub_q1_input_ids'])
        sub_q1_attn_masks.append(example['sub_q1_mask'])

        sub_q2_input_ids.append(example['sub_q2_input_ids'])
        sub_q2_attn_masks.append(example['sub_q2_mask'])

        sub_q1_edge_list.append(example['sub_q1_edge_list'])
        sub_q2_edge_list.append(example['sub_q2_edge_list'])
        q_edge_list.append(example['q_edge_list'])

        q_features.append(example['q_features'])
        sub_q1_features.append(example['sub_q1_features'])
        sub_q2_features.append(example['sub_q2_features'])

        q_mask.append(example['q_features'].size(0))
        sub_q1_mask.append(example['sub_q1_features'].size(0))
        sub_q2_mask.append(example['sub_q2_features'].size(0))

        answer_starts.append(example['answer_start_index'])
        answer_ends.append(example['answer_end_index'])
        answer_texts.append(example['answer_text'])
        answer_types.append(example['answer_type'])

        ques_types.append(example['ques_type'])
        qa_ids.append(example['id'])

    ctx_input_ids = torch.stack(ctx_input_ids, dim=0)
    ctx_attn_masks = torch.stack(ctx_attn_masks, dim=0)
    ques_input_ids = torch.stack(ques_input_ids, dim=0)
    ques_attn_masks = torch.stack(ques_attn_masks, dim=0)
    sub_q1_input_ids = torch.stack(sub_q1_input_ids, dim=0)
    sub_q1_attn_masks = torch.stack(sub_q1_attn_masks, dim=0)
    sub_q2_input_ids = torch.stack(sub_q2_input_ids, dim=0)
    sub_q2_attn_masks = torch.stack(sub_q2_attn_masks, dim=0)

    sub_q1_mask = torch.tensor(sub_q1_mask, dtype=torch.long)
    sub_q2_mask = torch.tensor(sub_q2_mask, dtype=torch.long)
    q_mask = torch.tensor(q_mask, dtype=torch.long)

    answer_starts = torch.tensor(answer_starts, dtype=torch.long)
    answer_ends = torch.tensor(answer_ends, dtype=torch.long)
    answer_types = torch.tensor(answer_types, dtype=torch.long)
    ques_types = torch.tensor(ques_types, dtype=torch.long)

    return ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_masks, sub_q1_input_ids, sub_q1_attn_masks, \
           sub_q2_input_ids, sub_q2_attn_masks, sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, \
           sub_q2_features, q_features, sub_q1_mask, sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, \
           answer_types, ques_types, qa_ids


def generate_dataloader(dataset, batch_size, data_sampler=None, is_train=True):
    if data_sampler is not None:
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if is_train else False,
                                 collate_fn=collate_fn)
    return data_loader


def generate_samples(config, batch_size, device='cuda'):
    train_ds = HotpotQADataset(config)
    train_loader = generate_dataloader(train_ds, batch_size)
    ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_masks, sub_q1_input_ids, sub_q1_attn_masks, \
    sub_q2_input_ids, sub_q2_attn_masks, sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, \
    sub_q2_features, q_features, sub_q1_mask, sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, \
    answer_types, ques_types, qa_ids = next(iter(train_loader))
    if device == 'cuda':
        sub_q1_edge_list = [edge_list.cuda() for edge_list in sub_q1_edge_list]
        sub_q2_edge_list = [edge_list.cuda() for edge_list in sub_q2_edge_list]
        q_edge_list = [edge_list.cuda() for edge_list in q_edge_list]
        return ctx_input_ids.cuda(), ctx_attn_masks.cuda(), ques_input_ids.cuda(), ques_attn_masks.cuda(),\
               sub_q1_input_ids.cuda(), sub_q1_attn_masks.cuda(), sub_q2_input_ids.cuda(), sub_q2_attn_masks.cuda(), \
               sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, sub_q2_features, q_features, \
               sub_q1_mask.cuda(), sub_q2_mask.cuda(), q_mask.cuda(), answer_starts.cuda(), answer_ends.cuda(), \
               answer_texts, answer_types.cuda(), ques_types.cuda(), qa_ids
    else:
        return ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_masks, sub_q1_input_ids, sub_q1_attn_masks, \
               sub_q2_input_ids, sub_q2_attn_masks, sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, \
               sub_q2_features, q_features, sub_q1_mask, sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, \
               answer_types, ques_types, qa_ids


class BaselineDataset(Dataset):
    def __init__(self, config, is_train=True):
        super(BaselineDataset, self).__init__()
        self.config = config
        if is_train:
            self.data = read_data_file(config.train_data_file)
        else:
            self.data = read_data_file(config.dev_data_file)
        self.word2idx, self.idx2word, self.word_emb = read_features(config.word2idx_file, config.idx2word_file,
                                                                    config.word_emb_file)
        self.tokenizer = AutoTokenizer.from_pretrained(config.roberta_config)

    def __getitem__(self, index):
        data_example = self.data[index]
        processed_example = {}

        context_sentences = extract_context(data_example)
        context_features, context_mask, context_offset = construct_roberta_features(self.tokenizer, context_sentences,
                                                                                    self.config.context_max_length)
        sub_q1_text = data_example['sub_questions'][0]
        sub_q1_features, sub_q1_mask, _ = construct_roberta_features(self.tokenizer, sub_q1_text,
                                                                     self.config.q_max_length)

        sub_q2_text = data_example['sub_questions'][1]
        sub_q2_features, sub_q2_mask, _ = construct_roberta_features(self.tokenizer, sub_q2_text,
                                                                     self.config.q_max_length)

        answer_start_char = get_answer_starts(context_sentences, data_example['answer'])

        answer_start_index, answer_end_index = get_answer_mapping(context_features, data_example['answer'],
                                                                  answer_start_char, context_offset)

        answer_type = get_answer_type(data_example['answer'])

        processed_example['sub_q1_features'] = sub_q1_features
        processed_example['sub_q2_features'] = sub_q2_features
        processed_example['context_features'] = context_features
        processed_example['context_mask'] = context_mask
        processed_example['sub_q1_mask'] = sub_q1_mask
        processed_example['sub_q2_mask'] = sub_q2_mask
        processed_example['answer_text'] = data_example['answer']
        processed_example['answer_start_index'] = answer_start_index
        processed_example['answer_end_index'] = answer_end_index
        processed_example['answer_type'] = answer_type

        return processed_example

    def __len__(self):
        return len(self.data)


def baseline_collate_fn(batch):
    context_features, context_masks = [], []
    sub_q1_features, sub_q2_features = [], []
    sub_q1_mask, sub_q2_mask = [], []
    answer_texts, answer_starts, answer_ends, answer_types = [], [], [], []

    for example in batch:
        context_features.append(example['context_features'])
        context_masks.append(example['context_mask'])

        sub_q1_features.append(example['sub_q1_features'])
        sub_q2_features.append(example['sub_q2_features'])

        sub_q1_mask.append(example['sub_q1_mask'])
        sub_q2_mask.append(example['sub_q2_mask'])

        answer_starts.append(example['answer_start_index'])
        answer_ends.append(example['answer_end_index'])
        answer_texts.append(example['answer_text'])
        answer_types.append(example['answer_type'])

    context_features = torch.stack(context_features, dim=0)
    sub_q1_features = torch.stack(sub_q1_features, dim=0)
    sub_q2_features = torch.stack(sub_q2_features, dim=0)

    context_masks = torch.stack(context_masks, dim=0)
    sub_q1_mask = torch.stack(sub_q1_mask, dim=0)
    sub_q2_mask = torch.stack(sub_q2_mask, dim=0)

    answer_starts = torch.tensor(answer_starts, dtype=torch.long)
    answer_ends = torch.tensor(answer_ends, dtype=torch.long)
    answer_types = torch.tensor(answer_types, dtype=torch.long)

    return context_features, context_masks, sub_q1_features, sub_q1_mask, sub_q2_features, sub_q2_mask, answer_starts, \
           answer_ends, answer_types, answer_texts


def generate_baseline_loader(dataset, batch_size, data_sampler=None, is_train=True):
    if is_train:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(data_sampler is None), sampler=data_sampler,
                                 collate_fn=baseline_collate_fn)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, collate_fn=baseline_collate_fn)

    return data_loader
