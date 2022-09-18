import argparse
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import GraphModule, RobertaEncoder
from data import HotpotQADataset
from util import *
from transformers import AutoTokenizer


class TypeDataset(Dataset):

    def __init__(self, args, is_train=True):
        super(TypeDataset, self).__init__()
        self.args = args
        if is_train:
            self.data = read_data_file(args.train_data_file)
        else:
            self.data = read_data_file(args.dev_data_file)
        self.tokenizer = AutoTokenizer.from_pretrained(args.roberta_config)
        self.word2idx, self.idx2word, self.word_emb = read_features(args.word2idx_file, args.idx2word_file,
                                                                    args.word_emb_file)
        self.nlp = spacy.load('en_core_web_sm')

    def __getitem__(self, idx):
        data_example = self.data[idx]
        processed_example = {}

        q_edge_list = build_question_graph(self.nlp, data_example)
        q_features, _, _ = build_node_features(self.nlp, data_example, self.word_emb, self.word2idx)

        ques_input_ids, ques_attn_mask, _ = construct_roberta_features(self.tokenizer, data_example['question'],
                                                                       self.args.q_max_length)
        processed_example['ques_input_ids'] = ques_input_ids
        processed_example['ques_attn_mask'] = ques_attn_mask
        processed_example['ques_edge_list'] = q_edge_list
        processed_example['ques_features'] = q_features
        processed_example['type'] = get_ques_type(data_example['type'])

        return processed_example

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    ques_input_ids, ques_attn_mask = [], []
    ques_edge_list, ques_graph_features, ques_mask = [], [], []
    ques_types = []

    for example in batch:
        ques_input_ids.append(example['ques_input_ids'])
        ques_attn_mask.append(example['ques_attn_mask'])
        ques_edge_list.append(example['ques_edge_list'])
        ques_graph_features.append(example['ques_features'])
        ques_mask.append(example['ques_features'].size(0))
        ques_types.append(example['type'])

    ques_input_ids = torch.stack(ques_input_ids, dim=0)
    ques_attn_mask = torch.stack(ques_attn_mask, dim=0)
    ques_mask = torch.tensor(ques_mask, dtype=torch.long)
    ques_types = torch.tensor(ques_types, dtype=torch.long)

    return ques_input_ids, ques_attn_mask, ques_edge_list, ques_graph_features, ques_mask, ques_types


class TypeClassifier(nn.Module):
    def __init__(self, args):
        super(TypeClassifier, self).__init__()
        self.ques_graph_encoder = GraphModule(args.model_name, args.layer_num, args.glove_dim, args.hidden_dim,
                                              args.hidden_dim)
        self.ques_text_encoder = RobertaEncoder(model_checkpoint=args.roberta_config, roberta_dim=args.roberta_dim,
                                                hidden_dim=args.hidden_dim, output_dim=args.hidden_dim,
                                                dropout_p=args.dropout_p)
        self.linear = nn.Linear(args.hidden_dim, 128)
        self.classifier = nn.Linear(128, args.num_types)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_p)

    def forward(self, ques_input_ids, ques_attention_mask, ques_edge_list, ques_features, ques_graph_mask):
        ques_graph_list = [create_pyg_data_object(features, edge_list) for (features, edge_list)
                           in zip(ques_features, ques_edge_list)]
        ques_graph_batch = create_pyg_batch_object(ques_graph_list)
        # [tot_num_nodes, hidden_dim]
        ques_graph_emb = self.ques_graph_encoder(ques_graph_batch.x.cuda(), ques_graph_batch.edge_index.long().cuda())
        # [bsz, q_len, hidden_dim]
        ques_graph_emb, ques_graph_mask = split_graph_features(ques_graph_emb, ques_graph_mask)
        # [bsz, q_len, hidden_dim]
        graph_output = torch.sigmoid(ques_graph_emb)
        # [bsz, c_len, hidden_dim]
        text_output, _ = self.ques_text_encoder(ques_input_ids, ques_attention_mask)
        output = torch.cat([text_output, graph_output], dim=1)
        mask = torch.cat([ques_attention_mask, ques_graph_mask], dim=-1)
        output = output - 1e10 * (1 - mask[:, :, None])
        output = output.max(dim=1)[0]
        output = self.dropout(self.relu(self.linear(output)))
        output = self.classifier(output)
        return output


def write_log(log_str):
    with open('./log.txt', 'a+') as f:
        f.write(log_str + '\n')


def train(epoch, period, checkpoint, train_loader, dev_loader, model, optimizer, criterion):
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        model.train()
        total_correct = 0.0
        total_examples = 0.0
        ques_input_ids, ques_attn_mask, ques_edge_list, ques_features, ques_mask, ques_types = batch
        ques_input_ids, ques_attn_mask, ques_mask, ques_types = ques_input_ids.cuda(), ques_attn_mask.cuda(), \
                                                                ques_mask.cuda(), ques_types.cuda()

        optimizer.zero_grad()
        logits = model(ques_input_ids, ques_attn_mask, ques_edge_list, ques_features, ques_mask)
        loss = criterion(logits, ques_types)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss = criterion(logits, ques_types)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=-1)
            total_correct += (pred == ques_types).float().sum().item()
            total_examples += pred.size(0)

        train_acc = total_correct / total_examples
        if (batch_idx + 1) % period == 0:
            print('| epoch {:3d} | step {:6d} | train loss {:8.3f} | train acc {:8.1f}%'.format(epoch, batch_idx + 1,
                                                                                                loss.item(),
                                                                                                100 * train_acc))
            write_log(
                '| epoch {:3d} | step {:6d} | train loss {:8.3f} | train acc {:8.1f}%'.format(epoch, batch_idx + 1,
                                                                                              loss.item(),
                                                                                              100 * train_acc))
        if (batch_idx + 1) % checkpoint == 0:
            validate(epoch, model, dev_loader, criterion)

    return total_loss / len(train_loader)


def validate(epoch, model, dev_loader, criterion):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_examples = 0.0
        for batch_idx, batch in enumerate(dev_loader):
            ques_input_ids, ques_attn_mask, ques_edge_list, ques_features, ques_mask, ques_types = batch
            ques_input_ids, ques_attn_mask, ques_mask, ques_types = ques_input_ids.cuda(), ques_attn_mask.cuda(), \
                                                                    ques_mask.cuda(), ques_types.cuda()

            logits = model(ques_input_ids, ques_attn_mask, ques_edge_list, ques_features, ques_mask)
            loss = criterion(logits, ques_types)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=-1)
            total_correct += (pred == ques_types).float().sum().item()
            total_examples += pred.size(0)

        val_avg_loss = total_loss / len(dev_loader)
        val_avg_acc = total_correct / total_examples

        print("| eval in epoch {:3d} | val loss: {:8.3f} | val acc: {:8.1f}%".format(epoch, val_avg_loss,
                                                                                     100 * val_avg_acc))

        write_log(
            "| eval in epoch {:3d} | val loss: {:8.3f} | val acc: {:8.1f}%".format(epoch, val_avg_loss,
                                                                                   100 * val_avg_acc))


def main():
    parser = argparse.ArgumentParser()

    word_emb_file = "./dataset/word_emb.json"
    train_data_file = "./dataset/decomposed_hotpot_qa_train.json"
    dev_data_file = "./dataset/decomposed_hotpot_qa_dev.json"
    debug_data_file = "./dataset/train_debug.json"

    word2idx_file = "./dataset/word2idx.json"
    idx2word_file = './dataset/idx2word.json'
    model_path = "./ques_type_best.pth"

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save', type=str, default='HOTPOT')

    parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
    parser.add_argument('--train_data_file', type=str, default=train_data_file)
    parser.add_argument('--dev_data_file', type=str, default=dev_data_file)
    parser.add_argument('--debug_data_file', type=str, default=debug_data_file)

    parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
    parser.add_argument('--idx2word_file', type=str, default=idx2word_file)
    parser.add_argument('--model_path', type=str, default=model_path)

    parser.add_argument('--glove_dim', type=int, default=300)
    parser.add_argument('--roberta_config', type=str, default='roberta-base')
    parser.add_argument('--roberta_dim', type=int, default=768)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=int, default=1000)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.8)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=13)

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--context_max_length', type=int, default=512)
    parser.add_argument('--q_max_length', type=int, default=30)
    parser.add_argument('--num_epochs', type=int, default=5)

    parser.add_argument('--model_name', type=str, default='GIN')
    parser.add_argument('--layer_num', type=int, default=1)
    parser.add_argument('--num_types', type=int, default=2)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Use CPU for training")
        device = torch.device('cpu')
    else:
        print("Use GPU for training")
        device = torch.device('cuda:0')

    print("loading dataset")
    train_ds = TypeDataset(args, is_train=True)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)

    dev_ds = TypeDataset(args, is_train=False)
    dev_loader = DataLoader(dev_ds, args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("building model")
    model = TypeClassifier(args).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)

    print("start training")
    for epoch in range(args.num_epochs):
        train_avg_loss = train(epoch, args.period, args.checkpoint, train_loader, dev_loader, model,
                               optimizer, criterion)
        lr_scheduler.step(train_avg_loss)

    torch.save(model.state_dict(), args.model_path)


if __name__ == '__main__':
    main()
