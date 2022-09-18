import numpy as np
import torch
import json
import re
from collections import Counter
import string
from torch_geometric.data import Data, Batch

IGNORE_INDEX = -1


def read_data_file(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def read_record_file(record_file):
    record = torch.load(record_file)
    return record


def read_features(word2idx_file, idx2word_file, word_emb_file):
    with open(word2idx_file, "r") as f:
        word2idx_dict = json.load(f)
    with open(idx2word_file, "r") as f:
        idx2word_dict = json.load(f)
    with open(word_emb_file, "r") as f:
        word_emb = json.load(f)

    return word2idx_dict, idx2word_dict, word_emb


def build_sub_question_graph(nlp, example):
    edge_list1 = []
    edge_list2 = []
    sub_q1 = nlp(example['sub_questions'][0])
    sub_q2 = nlp(example['sub_questions'][1])
    for node in sub_q1:
        edge = [node.head.i, node.i]
        edge_list1.append(edge)
    for node in sub_q2:
        edge = [node.head.i, node.i]
        edge_list2.append(edge)
    edge_list1 = torch.from_numpy(np.array(edge_list1)).permute(1, 0)
    edge_list2 = torch.from_numpy(np.array(edge_list2)).permute(1, 0)
    return edge_list1, edge_list2


def build_question_graph(nlp, example):
    edge_list = []
    q = nlp(example['question'])
    for node in q:
        edge = [node.head.i, node.i]
        edge_list.append(edge)
    edge_list = torch.from_numpy(np.array(edge_list)).permute(1, 0)
    return edge_list


def build_node_features(nlp, example, word_emb, word2idx_dict):
    q = nlp(example['question'])
    sub_q1 = nlp(example['sub_questions'][0])
    sub_q2 = nlp(example['sub_questions'][1])

    q_features = []
    sub_q1_features = []
    sub_q2_features = []

    for token in q:
        token = token.text
        emb = get_word_embedding(token, word_emb, word2idx_dict)
        q_features.append(emb)

    for token in sub_q1:
        token = token.text
        emb = get_word_embedding(token, word_emb, word2idx_dict)
        sub_q1_features.append(emb)

    for token in sub_q2:
        token = token.text
        emb = get_word_embedding(token, word_emb, word2idx_dict)
        sub_q2_features.append(emb)

    q_features = torch.tensor(q_features, dtype=torch.float32)
    sub_q1_features = torch.tensor(sub_q1_features, dtype=torch.float32)
    sub_q2_features = torch.tensor(sub_q2_features, dtype=torch.float32)

    return q_features, sub_q1_features, sub_q2_features


def process_answer(nlp, answer, word2idx_dict):
    tokenized_answer = nlp(answer)
    answer_ids = torch.tensor([word2idx(token, word2idx_dict) for token in tokenized_answer], dtype=torch.long)
    return answer_ids


def word2idx(token, word2idx_dict):
    return word2idx_dict.get(token, word2idx_dict['--OOV--'])


def idx2word(idx, idx2word_dict):
    return idx2word_dict[idx]


def get_word_embedding(token, word_emb, word2idx_dict):
    word_idx = word2idx(token, word2idx_dict)
    return word_emb[word_idx]


def graph_padding(node_features, max_node_num, device='cuda'):
    num_nodes, feature_dim = node_features.size(0), node_features.size(1)
    if device == 'cuda':
        padded_node_features = torch.cat((node_features, torch.zeros(max_node_num - num_nodes, feature_dim).cuda()),
                                         dim=0)
    else:
        padded_node_features = torch.cat((node_features, torch.zeros(max_node_num - num_nodes, feature_dim)), dim=0)
    graph_attention_mask = torch.from_numpy(np.array([1 for _ in range(num_nodes)] + [0 for _ in
                                                                                      range(
                                                                                          max_node_num - max_node_num)]))
    return padded_node_features, graph_attention_mask


def extract_context(example):
    contexts = example['context']
    sup_facts = example['supporting_facts']
    sup_sents = ""
    for sup_fact in sup_facts:
        title = sup_fact[0]
        for context in contexts:
            if context[0] == title:
                article = ""
                for sent in context[1]:
                    article += sent
                sup_sents += article
    return sup_sents


def evaluate(ground_truth, predicted_answer):
    f1 = exact_match = total = 0
    for t, p in zip(ground_truth, predicted_answer):
        total += 1
        cur_EM = exact_match_score(p, t)
        cur_f1, _, _ = f1_score(p, t)
        exact_match += cur_EM
        f1 += cur_f1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_redundant_whitespace(text):
        return text.strip()

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ''.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_redundant_whitespace(remove_articles(remove_punc(lower(s)))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def create_pyg_data_object(node_features, edge_list):
    return Data(x=node_features, edge_index=edge_list)


def create_pyg_batch_object(graph_list):
    graph_batch = Batch()
    graph_batch = graph_batch.from_data_list(graph_list)
    return graph_batch


def split_graph_features(total_graph_emb, mask_list, device='cuda'):
    graph_emb_list = []
    max_num_nodes = mask_list.max().item()
    accumulated_num = 0
    for num in mask_list:
        graph_emb_list.append(total_graph_emb[accumulated_num: accumulated_num + num])
        accumulated_num += num
    graph_emb_list = [graph_padding(features, max_num_nodes, device)[0] for features in graph_emb_list]
    graph_mask_list = [graph_padding(features, max_num_nodes, device)[1] for features in graph_emb_list]
    graph_emb = torch.stack(graph_emb_list)
    graph_mask = torch.stack(graph_mask_list)
    if device == 'cuda':
        return graph_emb.cuda(), graph_mask.cuda()
    else:
        return graph_emb, graph_mask


def construct_roberta_features(tokenizer, texts, max_length):
    tokenized_texts = tokenizer(
        texts,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    input_ids = tokenized_texts['input_ids'][0]
    attention_mask = tokenized_texts['attention_mask'][0]
    offset_mapping = tokenized_texts['offset_mapping'][0]
    return input_ids, attention_mask, offset_mapping


def get_answer_starts(context, answer):
    if answer in ['yes', 'no']:
        return -1
    else:
        start_positions = [i for i in range(len(context)) if context[i:].startswith(answer)]
        return start_positions[0]


def get_answer_mapping(context_features, answer, start_char, offsets):
    end_char = start_char + len(answer)

    # locate the context span
    token_start_idx = 0
    token_end_idx = len(context_features) - 1

    while offsets[token_end_idx][0] == 0 and offsets[token_end_idx][1] == 0:
        token_end_idx -= 1

    # verify if the answer is inside the context span. if yes, locate
    # it. otherwise, print "answer out of span"
    if offsets[token_start_idx][0] <= start_char and offsets[token_end_idx][1] >= end_char:
        while token_start_idx < len(offsets) and start_char >= offsets[token_start_idx][0]:
            token_start_idx += 1
        token_start_idx -= 1
        while end_char <= offsets[token_end_idx][1]:
            token_end_idx -= 1
        token_end_idx += 1
        return token_start_idx, token_end_idx
    else:
        return -1, -1


def convert_ids_to_tokens(input_ids, tokenizer, start, end, answer_type):
    if answer_type == 1:
        return "yes"
    elif answer_type == 2:
        return "no"
    else:
        return tokenizer.decode(input_ids[start: end + 1])


def get_answer_type(answer_text):
    if answer_text.lower() == 'yes':
        return 1
    elif answer_text.lower() == 'no':
        return 2
    else:
        return 0


def get_ques_type(type_text):
    if type_text == 'bridge':
        return 0
    else:
        return 1


def save_eval_data(answers, predictions, idx):
    with open('./eval_files/eval_data_{}.txt'.format(idx), 'w', encoding='utf-8') as f:
        f.write("----".join(answers) + "\n")
        f.write("----".join(predictions) + "\n")


def prepare_sp_data(example):
    sp_data = []
    question = example['question']
    contexts = example['context']
    id = example['_id']

    for i, context in enumerate(contexts):
        sentences = context[1]
        for j, sentence in enumerate(sentences):
            sp_data.append({
                'id': id + str(i) + str(j),
                'question': question,
                'context': sentence,
                'gold_label': 'irrelevant'
            })

    return sp_data


def prepare_input_data(sp_data, tokenizer, max_length):
    ques_input_ids = []
    ques_attn_mask = []
    ctx_input_ids = []
    ctx_attn_mask = []

    for sp_pair in sp_data:
        question = sp_pair['question']
        context = sp_pair['context']
        ques_features = tokenizer(question, max_length=max_length, padding='max_length',
                                  truncation=True, return_tensor='pt')
        ctx_features = tokenizer(context, max_length=max_length, padding='max_length',
                                 truncation=True, return_tensor='pt')
        ques_input_ids.append(ques_features['input_ids'])
        ques_attn_mask.append(ques_features['attention_mask'])
        ctx_input_ids.append(ctx_features['input_ids'])
        ctx_attn_mask.append(ctx_features['attention_mask'])

    ques_input_ids = torch.stack(ques_input_ids, dim=-1)
    ques_attn_mask = torch.stack(ques_attn_mask, dim=-1)
    ctx_input_ids = torch.stack(ctx_input_ids, dim=-1)
    ctx_attn_mask = torch.stack(ctx_attn_mask, dim=-1)

    return ques_input_ids.cuda(), ques_attn_mask.cuda(), ctx_input_ids.cuda(), ctx_attn_mask.cuda()



