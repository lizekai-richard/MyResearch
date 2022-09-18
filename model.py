import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch_geometric.nn as pyg_nn
from transformers import RobertaModel
from data import *
# from main import parse_option


class GraphModule(nn.Module):
    def __init__(self, model_name, layer_num, input_dim, hidden_dim, out_channels):
        super().__init__()
        if model_name == 'GAT':
            self.graph_layers = nn.ModuleList([pyg_nn.GATConv(input_dim, hidden_dim, False)
                                               for _ in range(layer_num)])
        elif model_name == 'GCN':
            self.graph_layers = nn.ModuleList([pyg_nn.GCNConv(input_dim, hidden_dim, False)
                                               for _ in range(layer_num)])
        elif model_name == 'SAGE':
            self.graph_layers = nn.ModuleList([pyg_nn.SAGEConv(input_dim, hidden_dim, False)
                                               for _ in range(layer_num)])
        elif model_name == 'GIN':
            mlp = nn.Linear(input_dim, hidden_dim)
            self.graph_layers = nn.ModuleList([pyg_nn.GINConv(mlp) for _ in range(layer_num)])
        self.linear = nn.Linear(hidden_dim, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        out = x
        for graph_layer in self.graph_layers:
            out = self.relu(graph_layer(out, edge_index))
        out = self.relu(self.linear(out))
        return out


class RobertaEncoder(nn.Module):
    def __init__(self, model_checkpoint, roberta_dim, hidden_dim, output_dim, dropout_p):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_checkpoint)
        self.rnn = nn.LSTM(roberta_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids, attention_mask).last_hidden_state
        output, (h_n, c_n) = self.rnn(output)
        output = self.relu(self.dropout(self.linear(output)))
        return output, attention_mask


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class BiAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.input_linear = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear = nn.Linear(input_dim, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_dim).uniform_(1.0 / (input_dim ** 0.5)))
        self.output_linear = nn.Linear(4 * input_dim, output_dim, bias=False)

    def forward(self, input, memory, mask):
        """
        
        :param input: context encoding [batch_size, c_len, hidden_dim]
        :param memory: question encoding [batch_size, q_len, hidden_dim]
        :param mask: question mask [batch_size, q_len]
        :return: 
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)  # [batch_size, input_len, 1]
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)  # [batch_size, 1, memory_len]
        # [batch_size, input_len, hidden_dim] [batch_size, hidden_dim, memory_len]
        # -> [batch_size, input_len, memory_len]
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        # [batch_size, input_len, memory_len]
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        # [batch_size, input_len, memory_len]
        weight_one = F.softmax(att, dim=-1)
        # [batch_size, input_len, hidden_dim]
        output_one = torch.bmm(weight_one, memory)
        # [batch_size, 1, input_len]
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        # [batch_size, 1, hidden_dim]
        output_two = torch.bmm(weight_two, input)
        # [batch_size, input_len, 4 * hidden_dim]
        output = torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1)
        output = self.dropout(self.output_linear(output))
        return output


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class OutputLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p, num_label):
        super(OutputLayer, self).__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            LayerNorm(2 * hidden_dim, eps=1e-12),
            nn.Dropout(dropout_p),
            nn.Linear(2 * hidden_dim, num_label),
        )

    def forward(self, x, output_mask=None):
        output = self.output_layer(x).squeeze(dim=-1)
        if output_mask is not None:
            output = output - 1e9 * (1 - output_mask)
        return output


class GatedAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, output_dim, dropout, gate_method='gate_att_up'):
        super(GatedAttention, self).__init__()
        self.gate_method = gate_method
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_1 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.input_linear_2 = nn.Linear(input_dim + memory_dim, output_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input_dot = F.relu(self.input_linear_1(input))  # N x Ld x d
        memory_dot = F.relu(self.memory_linear_1(memory))  # N x Lm x d

        # N * Ld * Lm
        att = torch.bmm(input_dot, memory_dot.permute(0, 2, 1).contiguous()) / self.dot_scale

        att = att - 1e30 * (1 - mask[:, None])
        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)

        if self.gate_method == 'no_gate':
            output = torch.cat([input, output_one], dim=-1)
            output = F.relu(self.input_linear_2(output))
        elif self.gate_method == 'gate_att_or':
            output = torch.cat([input, input - output_one], dim=-1)
            output = F.relu(self.input_linear_2(output))
        elif self.gate_method == 'gate_att_up':
            output = torch.cat([input, output_one], dim=-1)
            gate_sg = torch.sigmoid(self.input_linear_2(output))
            gate_th = torch.tanh(self.input_linear_2(output))
            output = gate_sg * gate_th
        else:
            raise ValueError("Not support gate method: {}".format(self.gate_method))

        return output


class TypeClassifier(nn.Module):
    def __init__(self, args):
        super(TypeClassifier, self).__init__()
        self.ques_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
                                           config.hidden_dim, config.dropout_p)
        self.linear = nn.Linear(args.hidden_dim, 128)
        self.classifier = nn.Linear(128, args.num_types)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_p)

    def forward(self, ques_input_ids, ques_attn_mask):
        # [bsz, q_len, hidden_dim]
        output, _ = self.ques_encoder(ques_input_ids, ques_attn_mask)
        # [bsz, hidden_dim]
        output = output.max(dim=1)[0]
        output = self.dropout(self.relu(self.linear(output)))
        output = self.classifier(output)
        return output


class BridgeModel(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_p):
        super(BridgeModel, self).__init__()
        self.ctx_attn1 = BiAttention(input_dim, output_dim, dropout_p)
        self.ctx_attn2 = BiAttention(input_dim, output_dim, dropout_p)
        self.linear_mapping = nn.Linear(output_dim, 2 * output_dim)

    def forward(self, ctx_features, sub_q1_features, sub_q1_attn_mask, sub_q2_features, sub_q2_attn_mask):
        # [bsz, c_len, 2 * hidden_dim]
        output = self.ctx_attn1(ctx_features, sub_q1_features, sub_q1_attn_mask)
        # [bsz, c_len, 2 * hidden_dim]
        output = self.ctx_attn2(output, sub_q2_features, sub_q2_attn_mask)
        return self.linear_mapping(output)


class ComparisonModel(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_p):
        super(ComparisonModel, self).__init__()
        self.ctx_attn1 = BiAttention(input_dim, output_dim, dropout_p)
        self.ctx_attn2 = BiAttention(input_dim, output_dim, dropout_p)
        self.ctx_attn3 = BiAttention(input_dim, output_dim, dropout_p)
        self.linear = nn.Linear(4 * input_dim, input_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, ctx_features, ques_features, ques_attn_mask, sub_q1_features, sub_q1_attn_mask,
                sub_q2_features, sub_q2_attn_mask):
        """

        :param ques_attn_mask: [bsz, q_len]
        :param ques_features: [bsz, q_len, hidden_dim]
        :param ctx_features: [bsz, c_len, hidden_dim]
        :param sub_q1_features: [bsz, q_len, hidden_dim]
        :param sub_q1_attn_mask: [bsz, q_len]
        :param sub_q2_features: [bsz, q_len, hidden_dim]
        :param sub_q2_attn_mask: [bsz, q_len]
        :return:
        """
        # [bsz, c_len, h_dim]
        attn1 = self.ctx_attn1(ctx_features, sub_q1_features, sub_q1_attn_mask)
        attn2 = self.ctx_attn2(ctx_features, sub_q2_features, sub_q2_attn_mask)
        # [bsz, c_len, 4 * h_dim]
        updated_ctx_features = torch.cat([attn1, attn2, attn1 - attn2, attn1 * attn2], dim=-1)
        # [bsz, c_len, h_dim]
        updated_ctx_features = self.dropout(self.relu(self.linear(updated_ctx_features)))
        # [bsz, c_len, h_dim]
        output = self.ctx_attn3(updated_ctx_features, ques_features, ques_attn_mask.long())
        return output


class BaseLineModel(nn.Module):

    def __init__(self, config):
        super(BaseLineModel, self).__init__()
        self.context_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
                                              config.hidden_dim, config.dropout_p)
        self.sub_q1_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
                                             config.hidden_dim, config.dropout_p)
        self.sub_q2_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
                                             config.hidden_dim, config.dropout_p)

        self.bridge_model = BridgeModel(config.hidden_dim, config.hidden_dim, config.dropout_p)

        self.start_rnn = nn.LSTM(2 * config.hidden_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        self.end_rnn = nn.LSTM(4 * config.hidden_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        self.type_rnn = nn.LSTM(4 * config.hidden_dim, config.hidden_dim, bidirectional=True, batch_first=True)

        self.start_prediction = OutputLayer(2 * config.hidden_dim, config.hidden_dim, config.dropout_p, 1)
        self.end_prediction = OutputLayer(2 * config.hidden_dim, config.hidden_dim, config.dropout_p, 1)
        self.type_prediction = OutputLayer(2 * config.hidden_dim, config.hidden_dim, config.dropout_p, 3)

    def forward(self, context_input_ids, context_attention_mask, sub_q1_input_ids, sub_q1_attention_mask,
                sub_q2_input_ids, sub_q2_attention_mask):
        context_encoding, context_mask = self.context_encoder(context_input_ids, context_attention_mask)
        sub_q1_encoding, sub_q1_mask = self.sub_q1_encoder(sub_q1_input_ids, sub_q1_attention_mask)
        sub_q2_encoding, sub_q2_mask = self.sub_q2_encoder(sub_q2_input_ids, sub_q2_attention_mask)

        output = self.bridge_model(context_encoding, sub_q1_encoding, sub_q1_mask, sub_q2_encoding, sub_q2_mask)

        start_logit, (h_n, c_n) = self.start_rnn(output)
        end_rnn_input = torch.cat([output, start_logit], dim=-1)
        end_logit, (h_n, c_n) = self.end_rnn(end_rnn_input)
        type_rnn_input = torch.cat([output, end_logit], dim=-1)
        type_logit, (h_n, c_n) = self.type_rnn(type_rnn_input)
        type_logit = type_logit[:, -1, :]

        start_pred = self.start_prediction(start_logit, context_mask)
        end_pred = self.end_prediction(end_logit, context_mask)
        type_pred = self.type_prediction(type_logit)

        return start_pred, end_pred, type_pred


class NewModel(nn.Module):

    def __init__(self, config):
        super(NewModel, self).__init__()
        self.ctx_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
                                          config.hidden_dim, config.dropout_p)
        # self.sub_q1_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
        #                                      config.hidden_dim, config.dropout_p)
        # self.sub_q2_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
        #                                      config.hidden_dim, config.dropout_p)
        self.ques_encoder = RobertaEncoder(config.roberta_config, config.roberta_dim, config.hidden_dim,
                                           config.hidden_dim, config.dropout_p)
        self.type_classifier = TypeClassifier(config)
        self.bridge_model = BridgeModel(config.hidden_dim, config.hidden_dim, config.dropout_p)
        self.comp_model = ComparisonModel(config.hidden_dim, config.hidden_dim, config.dropout_p)

        self.start_rnn = nn.LSTM(2 * config.hidden_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        self.end_rnn = nn.LSTM(4 * config.hidden_dim, config.hidden_dim, bidirectional=True, batch_first=True)
        self.type_rnn = nn.LSTM(4 * config.hidden_dim, config.hidden_dim, bidirectional=True, batch_first=True)

        self.start_prediction = OutputLayer(2 * config.hidden_dim, config.hidden_dim, config.dropout_p, 1)
        self.end_prediction = OutputLayer(2 * config.hidden_dim, config.hidden_dim, config.dropout_p, 1)
        self.answer_type_prediction = OutputLayer(2 * config.hidden_dim, config.hidden_dim, config.dropout_p, 3)

    def forward(self, ctx_input_ids, ctx_attn_mask, ques_input_ids, ques_attn_mask, sub_q1_input_ids, sub_q1_attn_mask,
                sub_q2_input_ids, sub_q2_attn_mask, type_pred):
        # [bsz, c_len, hidden_dim]
        ctx_features, ctx_attn_mask = self.ctx_encoder(ctx_input_ids, ctx_attn_mask)
        # [bsz, q_len, hidden_dim]
        ques_features, ques_attn_mask = self.ques_encoder(ques_input_ids, ques_attn_mask)
        sub_q1_features, sub_q1_attn_mask = self.ques_encoder(sub_q1_input_ids, sub_q1_attn_mask)
        sub_q2_features, sub_q2_attn_mask = self.ques_encoder(sub_q2_input_ids, sub_q2_attn_mask)

        brig_ctx_features, brig_sub_q1_features, brig_sub_q1_attn_mask, brig_sub_q2_features, brig_sub_q2_attn_mask = \
            [], [], [], [], []
        comp_ctx_features, comp_sub_q1_features, comp_sub_q1_attn_mask, comp_sub_q2_features, comp_sub_q2_attn_mask = \
            [], [], [], [], []
        for i in range(type_pred.size(0)):
            if type_pred[i] == 0:
                # [brig_bsz, c_len, hidden_dim]
                brig_ctx_features.append(ctx_features[i])
                brig_sub_q1_features.append(sub_q1_features[i])
                brig_sub_q1_attn_mask.append(sub_q1_attn_mask[i])
                brig_sub_q2_features.append(sub_q2_features[i])
                brig_sub_q2_attn_mask.append(sub_q2_attn_mask[i])
            if type_pred[i] == 1:
                # [brig_bsz, c_len, hidden_dim]
                comp_ctx_features.append(ctx_features[i])
                comp_sub_q1_features.append(sub_q1_features[i])
                comp_sub_q1_attn_mask.append(sub_q1_attn_mask[i])
                comp_sub_q2_features.append(sub_q2_features[i])
                comp_sub_q2_attn_mask.append(sub_q2_attn_mask[i])

        # [brig_bsz, c_len, 2 * hidden_dim]
        brig_ctx_features, brig_sub_q1_features, brig_sub_q1_attn_mask, brig_sub_q2_features, brig_sub_q2_attn_mask = \
            torch.stack(brig_ctx_features, dim=0), torch.stack(brig_sub_q1_features, dim=0), \
            torch.stack(brig_sub_q1_attn_mask, dim=0), torch.stack(brig_sub_q2_features, dim=0), \
            torch.stack(brig_sub_q2_attn_mask, dim=0)
        brig_output = self.bridge_model(brig_ctx_features, brig_sub_q1_features, brig_sub_q1_attn_mask,
                                        brig_sub_q2_features, brig_sub_q2_attn_mask)
        # [comp_bsz, c_len, 2 * hidden_dim]
        comp_ctx_features, comp_sub_q1_features, comp_sub_q1_attn_mask, comp_sub_q2_features, comp_sub_q2_attn_mask = \
            torch.stack(comp_ctx_features, dim=0), torch.stack(comp_sub_q1_features, dim=0), \
            torch.stack(comp_sub_q1_attn_mask, dim=0), torch.stack(comp_sub_q2_features, dim=0), \
            torch.stack(comp_sub_q2_attn_mask, dim=0)
        comp_output = self.comp_model(comp_ctx_features, ques_features, ques_attn_mask, comp_sub_q1_features,
                                      comp_sub_q1_attn_mask, comp_sub_q2_features, comp_sub_q2_attn_mask)
        # [bsz, c_len, 2 * hidden_dim]
        updated_ctx_features = torch.concat([brig_output, comp_output], dim=0)
        updated_ctx_features = updated_ctx_features[torch.randperm(updated_ctx_features.size(0))]

        # [bsz, c_len, 2 * hidden_dim]
        start_logit, (h_n, c_n) = self.start_rnn(updated_ctx_features)
        # [bsz, c_len, 4 * hidden_dim]
        end_rnn_input = torch.cat([updated_ctx_features, start_logit], dim=-1)
        # [bsz, c_len, 2 * hidden_dim]
        end_logit, (h_n, c_n) = self.end_rnn(end_rnn_input)
        # [bsz, c_len, 4 * hidden_dim]
        answer_type_rnn_input = torch.cat([updated_ctx_features, end_logit], dim=-1)
        # [bsz, c_len, 2 * hidden_dim]
        answer_type_logit, (h_n, c_n) = self.type_rnn(answer_type_rnn_input)
        # [bsz, 2 * hidden_dim]
        answer_type_logit = answer_type_logit[:, -1, :]

        start_pred = self.start_prediction(start_logit, ctx_attn_mask)
        end_pred = self.end_prediction(end_logit, ctx_attn_mask)
        answer_type_pred = self.answer_type_prediction(answer_type_logit)

        return start_pred, end_pred, answer_type_pred


if __name__ == '__main__':
    config = parse_option()

    ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_masks, sub_q1_input_ids, sub_q1_attn_masks, \
    sub_q2_input_ids, sub_q2_attn_masks, sub_q1_edge_list, sub_q2_edge_list, q_edge_list, sub_q1_features, \
    sub_q2_features, q_features, sub_q1_mask, sub_q2_mask, q_mask, answer_starts, answer_ends, answer_texts, \
    answer_types, ques_types, qa_ids = generate_samples(config, 2, device='cpu')

    model = BaseLineModel(config)
    start_logit, end_logit, answer_type_logit = model(ctx_input_ids, ctx_attn_masks, ques_input_ids, ques_attn_masks,
                                                      sub_q1_input_ids, sub_q1_attn_masks, sub_q2_input_ids,
                                                      sub_q2_attn_masks)
