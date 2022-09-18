import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from PairSCL.losses import SupConLoss


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                       .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses


class SupportingFactBaseline(nn.Module):

    def __init__(self, bert_config, bert_dim=768, max_length=256, hidden_dim=300, dropout_p=0.5):
        super(SupportingFactBaseline, self).__init__()
        self.context_encoder = BertModel.from_pretrained(bert_config)
        self.question_encoder = BertModel.from_pretrained(bert_config)
        self.attn = SoftmaxAttention()
        self.ques_proj = nn.Sequential(nn.Linear(4 * bert_dim, hidden_dim),
                                       nn.LeakyReLU())
        self.ctx_proj = nn.Sequential(nn.Linear(4 * bert_dim, hidden_dim),
                                      nn.LeakyReLU())
        self.classifier = nn.Sequential(nn.Linear(4 * hidden_dim, 2 * hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(2 * hidden_dim, hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, 2),
                                        nn.LeakyReLU())
        self.pooler = nn.MaxPool1d(kernel_size=max_length)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask):
        ctx_emb = self.context_encoder(ctx_input_ids, ctx_attn_mask).last_hidden_state
        ques_emb = self.question_encoder(ques_input_ids, ques_attn_mask).last_hidden_state
        ques_attn, ctx_attn = self.attn(ques_emb, ques_attn_mask, ctx_emb, ctx_attn_mask)
        enhanced_ques_emb = torch.cat([ques_emb, ques_attn, ques_emb - ques_attn, ques_emb * ques_attn], dim=-1)
        enhanced_ctx_emb = torch.cat([ctx_emb, ctx_attn, ctx_emb - ctx_attn, ctx_emb * ctx_attn], dim=-1)
        ques_projection = self.dropout(self.ques_proj(enhanced_ques_emb))
        ctx_projection = self.dropout(self.ctx_proj(enhanced_ctx_emb))
        output = torch.cat([ctx_projection, ques_projection, ctx_projection - ques_projection,
                            ctx_projection * ques_projection], dim=-1)
        output = self.pooler(output.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        output = self.classifier(output)
        return output.squeeze(dim=1)


class SupportingFactDataset(Dataset):

    def __init__(self, bert_config, data, ctx_max_length, ques_max_length):
        super(SupportingFactDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_config)
        self.data = data
        self.ctx_max_length = ctx_max_length
        self.ques_max_length = ques_max_length

    def __getitem__(self, idx):
        example = self.data[idx]
        processed_example = {}
        question = example['question']
        context = example['context']
        label = example['gold_label']

        ctx_features = self.tokenizer(context, padding='max_length', max_length=self.ctx_max_length,
                                      truncation=True, return_tensors='pt')
        ques_features = self.tokenizer(question, padding='max_length', max_length=self.ques_max_length,
                                       truncation=True, return_tensors='pt')
        processed_example['ctx_input_ids'] = ctx_features['input_ids']
        processed_example['ctx_attn_mask'] = ctx_features['attention_mask']
        processed_example['ques_input_ids'] = ques_features['input_ids']
        processed_example['ques_attn_mask'] = ques_features['attention_mask']
        processed_example['label'] = 1 if label == 'relevant' else 0

        return processed_example

    def __len__(self):
        return len(self.data)


def collate(batch):
    ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_mask, labels = [], [], [], [], []
    for example in batch:
        ctx_input_ids.append(example['ctx_input_ids'])
        ctx_attn_masks.append(example['ctx_attn_mask'])
        ques_input_ids.append(example['ques_input_ids'])
        ques_attn_mask.append(example['ques_attn_mask'])
        labels.append(example['label'])

    ctx_input_ids = torch.cat(ctx_input_ids, dim=0)
    ctx_attn_masks = torch.cat(ctx_attn_masks, dim=0)
    ques_input_ids = torch.cat(ques_input_ids, dim=0)
    ques_attn_mask = torch.cat(ques_attn_mask, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_mask, labels


def logger(log):
    with open("./sp_log.txt", "a+") as f:
        f.write(log)


def train(epoch, train_iter, dev_iter, model, loss_fn, contra_loss_fn, optimizer, lr):
    model.train()
    best_dev_acc = None
    patience = 0
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_iter):
        ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask, labels = batch
        ctx_input_ids, ctx_attn_mask = ctx_input_ids.cuda(), ctx_attn_mask.cuda()
        ques_input_ids, ques_attn_mask = ques_input_ids.cuda(), ques_attn_mask.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        output = model(ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask)
        loss = loss_fn(output, labels) + contra_loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            train_log = 'train | epoch {:3d} | step {:6d} | lr {:05.5f} | train loss {:8.6f}\n'.format(epoch, batch_idx,
                                                                                                       lr,
                                                                                                       train_loss / batch_idx)
            print(train_log)
            logger(train_log)

        if (batch_idx + 1) % 1000 == 0:
            dev_loss, dev_acc = evaluate(dev_iter, model, loss_fn, contra_loss_fn)
            if best_dev_acc is None:
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), './sp_model.pth')
            elif best_dev_acc > dev_acc:
                best_dev_acc = dev_acc
                patience = 0
                torch.save(model.state_dict(), './sp_model.pth')
            else:
                patience += 1
                if patience >= 2:
                    lr /= 2.0
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            model.train()
            dev_log = 'evaluate | epoch {:3d} | step {:6d} | lr {:05.5f} | dev loss {:8.6f} | dev acc {:8.3f}\n'.format(
                epoch, batch_idx, lr, dev_loss, dev_acc)
            print(dev_log)
            logger(dev_log)


def evaluate(dev_iter, model, loss_fn, contra_loss_fn):
    model.eval()
    dev_loss = 0.0
    pred_cnt = 0.0
    correct_cnt = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_iter):
            ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask, labels = batch
            ctx_input_ids, ctx_attn_mask = ctx_input_ids.cuda(), ctx_attn_mask.cuda()
            ques_input_ids, ques_attn_mask = ques_input_ids.cuda(), ques_attn_mask.cuda()
            labels = labels.cuda()
            output = model(ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask)
            loss = loss_fn(output, labels) + contra_loss_fn(output, labels)
            dev_loss += loss.item()

            pred = torch.argmax(output, dim=-1)
            correct_pred = torch.eq(pred, labels).float().sum().item()
            pred_cnt += pred.size(0)
            correct_cnt += correct_pred

    dev_acc = correct_cnt / pred_cnt
    return dev_loss, dev_acc


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with open("./dataset/hotpot_train.json", "r") as f:
        train_data = json.load(f)

    with open("./dataset/hotpot_dev.json", "r") as f:
        dev_data = json.load(f)

    print("Building datasets")
    train_dataset = SupportingFactDataset("bert-base-uncased", train_data, 256, 256)
    dev_dataset = SupportingFactDataset("bert-base-uncased", dev_data, 256, 256)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=collate)
    print("Done!")

    print("Building model")
    model = SupportingFactBaseline("bert-base-uncased").to(device)
    if os.path.exists("sp_model.pth"):
        model_state_dict = torch.load("./sp_model.pth", "cuda")
        model.load_state_dict(model_state_dict)
    else:
        model.apply(init_weights)
    print("Done")

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    contrastive_loss_fn = SupConLoss(temperature=0.05).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-5, weight_decay=0.8)

    print("Start training")
    for epoch in range(3):
        train(epoch, train_loader, dev_loader, model, criterion, optimizer, contrastive_loss_fn, 1e-5)

    torch.save(model.state_dict(), './sp_model.pth')
    # pred_cnt = 0.0
    # correct_cnt = 0.0
    #
    # with torch.no_grad():
    #     for batch_idx, batch in enumerate(dev_loader):
    #         ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask, labels = batch
    #         ctx_input_ids, ctx_attn_mask = ctx_input_ids.cuda(), ctx_attn_mask.cuda()
    #         ques_input_ids, ques_attn_mask = ques_input_ids.cuda(), ques_attn_mask.cuda()
    #         labels = labels.cuda()
    #         output = model(ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask)
    #
    #         pred = torch.argmax(output, dim=-1)
    #         correct_pred = torch.eq(pred, labels).float().sum().item()
    #         pred_cnt += pred.size(0)
    #         correct_cnt += correct_pred
    #
    # acc = pred_cnt / correct_cnt
    # print(acc)



if __name__ == '__main__':
    main()