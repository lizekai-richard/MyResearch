import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import GraphModule
from data import HotpotQADataset, generate_dataloader
from util import create_pyg_batch_object, split_graph_features, create_pyg_data_object
from transformers import RobertaModel


class GraphClassifier(nn.Module):
    def __init__(self, args):
        super(GraphClassifier, self).__init__()
        self.ques_encoder = GraphModule(args.model_name, args.layer_num, args.glove_dim, args.hidden_dim,
                                        args.hidden_dim, args.num_heads)
        self.linear = nn.Linear(args.hidden_dim, 128)
        self.classifier = nn.Linear(128, args.num_types)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_p)

    def forward(self, ques_features, ques_edge_list, ques_graph_mask):
        ques_graph_list = [create_pyg_data_object(features, edge_list) for (features, edge_list)
                           in zip(ques_features, ques_edge_list)]
        ques_graph_batch = create_pyg_batch_object(ques_graph_list)
        # [tot_num_nodes, hidden_dim]
        ques_graph_emb = self.ques_encoder(ques_graph_batch.x, ques_graph_batch.edge_index)
        # [bsz, q_len, hidden_dim]
        ques_graph_emb, _ = split_graph_features(ques_graph_emb, ques_graph_mask, device='cpu')
        # [bsz, hidden_dim]
        output = ques_graph_emb.max(dim=1)[0]
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
        ctx_input_ids, ctx_attn_masks, sub_q1_input_ids, sub_q1_attn_masks, sub_q2_input_ids, sub_q2_attn_masks, \
        sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, sub_q2_features, q_features, sub_q1_mask, \
        sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, answer_types, ques_types, qa_ids = batch

        optimizer.zero_grad()
        logits = model(q_features, q_edge_list, q_mask)
        print(logits.size())
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
        if batch_idx % period == 0:
            print('| epoch {:3d} | step {:6d} | train loss {:8.3f} | train acc {:8.3f}'.format(epoch, batch_idx,
                                                                                               loss.item(),
                                                                                               train_acc))
            write_log('| epoch {:3d} | step {:6d} | train loss {:8.3f} | train acc {:8.3f}'.format(epoch, batch_idx,
                                                                                                   loss.item(),
                                                                                                   train_acc))
        if batch_idx % checkpoint == 0:
            validate(epoch, model, dev_loader, criterion)

    return total_loss / len(train_loader)


def validate(epoch, model, dev_loader, criterion):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_examples = 0.0
        for batch_idx, batch in enumerate(dev_loader):
            ctx_input_ids, ctx_attn_masks, sub_q1_input_ids, sub_q1_attn_masks, sub_q2_input_ids, sub_q2_attn_masks, \
            sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, sub_q2_features, q_features, sub_q1_mask, \
            sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, answer_types, ques_types, qa_ids = batch

            logits = model(q_features, q_edge_list, q_mask)
            loss = criterion(logits, ques_types)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=-1)
            total_correct += (pred == ques_types).float().sum().item()
            total_examples += pred.size(0)

        val_avg_loss = total_loss / len(dev_loader)
        val_avg_acc = total_correct / total_examples

        print("| eval in epoch {:3d} | val loss: {:8.3f} | val acc: {:8.3f}".format(epoch, val_avg_loss, val_avg_acc))

        write_log(
            "| eval in epoch {:3d} | val loss: {:8.3f} | val acc: {:8.3f}".format(epoch, val_avg_loss, val_avg_acc))


def main():
    parser = argparse.ArgumentParser()

    word_emb_file = "./dataset/word_emb.json"
    train_data_file = "./dataset/decomposed_hotpot_qa_train.json"
    dev_data_file = "./dataset/decomposed_hotpot_qa_dev.json"
    debug_data_file = "./dataset/train_debug.json"

    word2idx_file = "./dataset/word2idx.json"
    idx2word_file = './dataset/idx2word.json'

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save', type=str, default='HOTPOT')

    parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
    parser.add_argument('--train_data_file', type=str, default=train_data_file)
    parser.add_argument('--dev_data_file', type=str, default=dev_data_file)
    parser.add_argument('--debug_data_file', type=str, default=debug_data_file)

    parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
    parser.add_argument('--idx2word_file', type=str, default=idx2word_file)

    parser.add_argument('--glove_dim', type=int, default=300)
    parser.add_argument('--roberta_config', type=str, default='roberta-base')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=int, default=1000)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.8)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=13)

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--context_max_length', type=int, default=512)
    parser.add_argument('--q_max_length', type=int, default=30)
    parser.add_argument('--num_epochs', type=int, default=20)

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
    train_ds = HotpotQADataset(args, is_train=True)
    train_loader = generate_dataloader(train_ds, args.batch_size, is_train=True)

    dev_ds = HotpotQADataset(args, is_train=False)
    dev_loader = generate_dataloader(dev_ds, args.batch_size, is_train=False)

    print("building model")
    model = GraphClassifier(args).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)

    print("start training")
    for epoch in range(args.num_epochs):
        train_avg_loss = train(epoch, args.period, args.checkpoint, train_loader, dev_loader, model,
                               optimizer, criterion)
        lr_scheduler.step(train_avg_loss)


if __name__ == '__main__':
    main()

