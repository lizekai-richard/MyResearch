# from run import train, baseline
import argparse

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

parser.add_argument('--glove_word_size', type=int, default=int(2.2e6))
parser.add_argument('--glove_dim', type=int, default=300)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint', type=int, default=1000)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--init_lr', type=float, default=0.5)
parser.add_argument('--dropout_p', type=float, default=0.2)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--patience', type=int, default=1)
parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--fullwiki', action='store_true')
parser.add_argument('--prediction_file', type=str)

parser.add_argument('--roberta_config', type=str, default="roberta-base")
parser.add_argument('--roberta_dim', type=int, default=768)

parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--context_max_length', type=int, default=512)
parser.add_argument('--q_max_length', type=int, default=50)
parser.add_argument('--sub_q_max_length', type=int, default=30)
parser.add_argument('--num_epochs', type=int, default=10)

parser.add_argument('--distributed', type=bool, default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)

parser.add_argument('--graph_model_name', type=str, default='GIN')
parser.add_argument('--graph_layer_num', type=int, default=1)
parser.add_argument('--num_types', type=int, default=2)

config = parser.parse_args()


if __name__ == '__main__':
    if config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        pass
    elif config.mode == 'baseline':
        baseline(config)
    # elif config.mode == 'debug':
    #     debug(config)
