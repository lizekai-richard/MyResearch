import json
import random


with open("./PairSCL/Hotpot/hotpot_train.json", "r") as f:
    sp_data = json.load(f)

with open("./dataset/decomposed_hotpot_qa_train.json", "r") as f:
    hotpot_data = json.load(f)

sample = random.choice(sp_data)
print(sample['question'])
print(sample['context'])
print(sample['gold_label'])

question = "What movie was released in 2016 and stared Joo Won?"

# for example in hotpot_data:
#     ques = example['question']
#     sp = example['supporting_facts']
#     contexts = example['context']
#     if ques == question:
#         print(sp)
#         print(contexts)
#         break