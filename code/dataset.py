import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import create_mlm_instances

WORD_PADDING_INDEX = 103

class E2EDataset(Dataset):
    def __init__(self, pretrain_input_file_dir, word_mask_index, word_vocab_size, entity_vocab_size,
                 relation_vocab_size, random_seed, tokenizer, neg_ratio=1, dataset_name='GDELT'):
        """
        :param pretrain_input_file_dir: training_data.json (text + tuple), train.txt, val.txt, test.txt
        :param word_mask_index: 103, if use bert tokenizer
        :param word_vocab_size: vocab_size in bert
        :param entity_vocab_size: all entities + [MASK]
        :param relation_vocab_size: all relations + [MASK]
        :param random_seed: random seed
        :param tokenizer: bert tokenizer
        """
        self.pretrain_input_file_dir = pretrain_input_file_dir
        self.word_mask_index = word_mask_index
        self.word_vocab_size = word_vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.relation_vocab_size = relation_vocab_size
        self.random_seed = random_seed
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        if self.dataset_name == 'GDELT' or self.dataset_name =='Wiki':
            # read the training json file
            # i split the data into 10 files
            self.current_file_idx = 0
            self.data = self.read_training_json_file(
                os.path.join(pretrain_input_file_dir, f'training_data_{self.current_file_idx}.json'))
            self.num_samples_per_file = len(self.data)
            if self.dataset_name == 'GDELT':
                self.last_file_idx = 9
            elif self.dataset_name == 'Wiki':
                self.last_file_idx = 2
            self.last_file_data = self.read_training_json_file(os.path.join(pretrain_input_file_dir,
                                                            f'training_data_{self.last_file_idx}.json'))
            self.num_samples = self.num_samples_per_file * self.last_file_idx + len(self.last_file_data)
        elif self.dataset_name == 'DuEE':
            self.data = self.read_training_json_file(os.path.join(pretrain_input_file_dir, 'training_data.json'))
            self.num_samples_per_file = len(self.data)
            self.num_samples = len(self.data)
        print(f'#num samples = {self.num_samples}')

        self.neg_ratio = neg_ratio

        # read the train, val and test txt data
        self.tkg_data = dict()
        self.tkg_data['train'] = self.read_val_test_file('train.txt')
        self.tkg_data['val'] = self.read_val_test_file('val.txt')
        self.tkg_data['test'] = self.read_val_test_file('test.txt')
        self.num_ent = entity_vocab_size - 1
        self.num_rel = relation_vocab_size - 1
        self.all_data_as_tuples = set([tuple(d) for d in self.tkg_data['train'] + self.tkg_data['val'] + self.tkg_data['test']])
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.all_data_as_tuples)
        for spl in ['train', 'val', 'test']:
            self.tkg_data[spl] = np.array(self.tkg_data[spl])

    @staticmethod
    def get_true_head_and_tail(all_tuples):
        true_head = dict()
        true_tail = dict()
        for head, rel, tail, timestamp in all_tuples:
            if (rel, tail, timestamp) not in true_head:
                true_head[(rel, tail, timestamp)] = []
            true_head[(rel, tail, timestamp)].append(head)
            if (head, rel, timestamp) not in true_tail:
                true_tail[(head, rel, timestamp)] = []
            true_tail[(head, rel, timestamp)].append(tail)
        return true_head, true_tail

    def read_training_json_file(self, training_file):
        data = []
        with open(training_file, 'r') as f:
            for x in f:
                # text (in token ids) + training_tuple_index
                # {'token_ids': [...], 'tuple': [s, p, o, t]}
                instance = json.loads(x)
                data.append(instance)
        return data

    def read_val_test_file(self, filename):
        facts = []
        with open(os.path.join(self.pretrain_input_file_dir, filename), 'r') as freader:
            data = freader.readlines()

        for line in data:
            line = line.strip().split()
            head_id = int(line[0])
            rel_id = int(line[1])
            tail_id = int(line[2])
            # use the discrete integer number as timestamp
            timestamp = float(line[3])
            facts.append([head_id, rel_id, tail_id, timestamp])
        return facts

    def __getitem__(self, item):
        # sample = self.data[item]
        if self.num_samples_per_file != self.num_samples:
            file_idx = item // self.num_samples_per_file
            if file_idx != self.current_file_idx:
                self.data = self.read_training_json_file(os.path.join(self.pretrain_input_file_dir, f'training_data_{file_idx}.json'))
                self.current_file_idx = file_idx
            sample = self.data[item - file_idx*self.num_samples_per_file]
        else:
            self.data = self.read_training_json_file(os.path.join(self.pretrain_input_file_dir, f'training_data.json'))
            sample = self.data[item]

        return sample

    def __len__(self):
        return self.num_samples

    def add_neg_facts(self, pos_fact, mode='head'):
        pos_neg_group_size = 1 + self.neg_ratio
        head, rel, tail, timestamp = pos_fact[0], pos_fact[1], pos_fact[2], pos_fact[3]
        pos_fact = np.array(pos_fact)
        pos_neg_facts = np.repeat(np.copy(pos_fact.reshape(1, -1)), pos_neg_group_size, axis=0)

        neg_samples_list = []
        neg_sample_size = 0
        while neg_sample_size < self.neg_ratio:
            neg_sample = np.random.randint(self.num_ent, size=self.neg_ratio)
            if mode == 'head':
                mask = np.in1d(neg_sample, self.true_head[(rel, tail, timestamp)], assume_unique=True, invert=True)
            else:
                mask = np.in1d(neg_sample, self.true_tail[(head, rel, timestamp)], assume_unique=True, invert=True)
            neg_sample = neg_sample[mask]
            neg_samples_list.append(neg_sample)
            neg_sample_size += len(neg_sample)
        res = np.concatenate(neg_samples_list[:self.neg_ratio])

        # debug... why this could happen???????????
        """
        if pos_neg_facts.shape[0] != 1+res.shape[0]:
            print(pos_neg_facts.shape)
            print(res.shape)
            print(pos_neg_group_size)
        """

        if mode == 'head':
            pos_neg_facts[1:, 0] = res[:self.neg_ratio]
        else:
            pos_neg_facts[1:, 2] = res[:self.neg_ratio]

        return pos_neg_facts

    def apply_mask_strategy(self, token_ids):
        # [CLS], word tokens in text, [SEP], HEAD, TAIL, REL
        # create the mlm training instance
        # mask the word tokens in the text

        words, words_mlm_labels = create_mlm_instances(token_ids[0: -3], self.word_mask_index,
                                                            self.word_vocab_size, self.tokenizer)
        mask_i = random.randint(1,2)
        if mask_i == 1:
            # mask the head/tail entities
            entities, entities_mlm_labels = create_mlm_instances(token_ids[-3: -1], self.entity_vocab_size - 1,
                                                            self.entity_vocab_size)
            relation = [token_ids[-1]]
            relation_mlm_labels = np.linspace(-100, -100, len(relation)).astype(int).tolist()
        elif mask_i == 2:
            # mask the relation
            relation, relation_mlm_labels = create_mlm_instances([token_ids[-1]], self.relation_vocab_size - 1,
                                                                self.relation_vocab_size)
            entities = token_ids[-3: -1]
            entities_mlm_labels = np.linspace(-100, -100, len(entities)).astype(int).tolist()
    
        return words, words_mlm_labels, entities, entities_mlm_labels, relation, relation_mlm_labels

    def collate_fn(self, batch):
        # MLM part and TKG part
        data = []
        for instance in batch:
            token_ids = instance['token_ids']
            words, words_mlm_labels, entities, entities_mlm_labels, relation, relation_mlm_labels = \
                self.apply_mask_strategy(token_ids)

            input_ids = words + entities + relation  # + token_ids[-1] # add the last [SEP] token
            num_tokens = len(input_ids)
            attention_mask = [1] * num_tokens
            # word tokens: 0, entity: 1, relation: 2
            token_type_ids = [0] * (num_tokens - 3) + [1] * 2 + [2]
            data.append({
                'input_ids': input_ids,
                'num_tokens': num_tokens,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'word_masked_lm_labels': words_mlm_labels,
                'entity_masked_lm_labels': entities_mlm_labels,
                'relation_masked_lm_labels': relation_mlm_labels,
                'tkg_tuple': instance['tuple']
            })

        input_keys = ['input_ids', 'num_tokens', 'attention_mask', 'token_type_ids', 'word_masked_lm_labels',
                      'entity_masked_lm_labels', 'relation_masked_lm_labels', 'tkg_tuple', 'tuple_labels']
        output_keys = ['word_masked_lm_labels', 'entity_masked_lm_labels', 'relation_masked_lm_labels']
        max_tokens_len = 0
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in output_keys}
        batch_num_tokens = []

        for sample in data:  # batch:
            for k, v in sample.items():
                if k in input_keys:
                    batch_x[k].append(v)
                if k in output_keys:
                    batch_y[k].append(v)
            num_tokens = sample['num_tokens']
            batch_num_tokens.append(num_tokens)
            max_tokens_len = max(max_tokens_len, num_tokens)
        # tuple_labels
        batch_x['tuple_labels'] = [[] for i in range(len(batch))]
        # pad the instance, add the negative examples
        for i in range(len(batch_num_tokens)):
            pad_tokens = max_tokens_len - batch_num_tokens[i]
            num_word_tokens = batch_num_tokens[i] - 3
            batch_x['input_ids'][i] = batch_x['input_ids'][i][0: num_word_tokens] + \
                                      [WORD_PADDING_INDEX] * pad_tokens + batch_x['input_ids'][i][num_word_tokens:]
            batch_x['attention_mask'][i] = batch_x['attention_mask'][i][0: num_word_tokens] + \
                                           [0] * pad_tokens + batch_x['attention_mask'][i][num_word_tokens:]
            batch_x['token_type_ids'][i] = batch_x['token_type_ids'][i] + [0] * pad_tokens
            batch_x['word_masked_lm_labels'][i] = batch_x['word_masked_lm_labels'][i][0: num_word_tokens] + \
                                                  [-100] * (pad_tokens + 3)
            batch_y['word_masked_lm_labels'][i] = batch_x['word_masked_lm_labels'][i][0: -3]

            batch_x['entity_masked_lm_labels'][i] = [-100] * (max_tokens_len - 3) + batch_x['entity_masked_lm_labels'][
                i] + [-100]

            batch_x['relation_masked_lm_labels'][i] = [-100] * (max_tokens_len - 1) + \
                                                      batch_x['relation_masked_lm_labels'][i]

            batch_x['num_tokens'][i] = max(max_tokens_len, batch_num_tokens[i])

            # generate the negative examples
            # head
            pos_neg_head = self.add_neg_facts(batch_x['tkg_tuple'][i], mode='head')
            # tail
            pos_neg_tail = self.add_neg_facts(batch_x['tkg_tuple'][i], mode='tail')
            pos_negs = np.concatenate((pos_neg_head, pos_neg_tail), axis=0)
            batch_x['tkg_tuple'][i] = pos_negs
            batch_x['tuple_labels'][i] = [0, 0]

        # convert to tensor
        for k, v in batch_x.items():
            batch_x[k] = torch.tensor(np.array(v))
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(np.array(v))
        return batch_x, batch_y

class E2EEvalDataset:
    def __init__(self, data_dir, ent_vocab_size, rel_vocab_size):
        """
        :param data_dir: where val.txt and test.txt locate
        :param ent_vocab_size: entity vocabulary size (including [MASK])
        :param rel_vocab_size: relation vocabulary size (including [MASK])
        """
        self.data_dir = data_dir
        self.num_ent = ent_vocab_size - 1
        self.num_rel = rel_vocab_size - 1
        self.data = dict()
        self.data['train'] = self.readfile('train.txt')
        self.data['val'] = self.readfile('val.txt')
        self.data['test'] = self.readfile('test.txt')
        self.all_data_as_tuples = set([tuple(d) for d in self.data['train'] + self.data['val'] + self.data['test']])
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.all_data_as_tuples)

        self.startbatch = 0
        for spl in ['train', 'val', 'test']:
            self.data[spl] = np.array(self.data[spl])

    @staticmethod
    def get_true_head_and_tail(all_tuples):
        true_head = dict()
        true_tail = dict()
        for head, rel, tail, timestamp in all_tuples:
            if (rel, tail, timestamp) not in true_head:
                true_head[(rel, tail, timestamp)] = []
            true_head[(rel, tail, timestamp)].append(head)
            if (head, rel, timestamp) not in true_tail:
                true_tail[(head, rel, timestamp)] = []
            true_tail[(head, rel, timestamp)].append(tail)
        return true_head, true_tail

    def readfile(self, filename):
        facts = []
        with open(os.path.join(self.data_dir, filename), 'r') as freader:
            data = freader.readlines()

        for line in data:
            line = line.strip().split()
            head_id = int(line[0])
            rel_id = int(line[1])
            tail_id = int(line[2])
            # use the discrete integer number as timestamp
            timestamp = float(line[3])
            facts.append([head_id, rel_id, tail_id, timestamp])
        return facts

    def next_positive_batch(self, batch_size):
        if self.startbatch + batch_size > len(self.data['train']):
            pos_facts = self.data['train'][self.startbatch:]
            self.startbatch = 0
        else:
            pos_facts = self.data['train'][self.startbatch: self.startbatch + batch_size]
            self.startbatch += batch_size
        return pos_facts

    def add_neg_facts(self, pos_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        # generate negative samples for each tuple (head, rel, tail, time)
        # like (?, rel, tail, time) and (head, rel, ?, time)
        facts1 = np.repeat(np.copy(pos_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)

        rand_facts1 = np.random.randint(low=1, high=self.num_ent(), size=facts1.shape[0])
        rand_facts2 = np.random.randint(low=1, high=self.num_ent(), size=facts2.shape[0])

        # set the added index to positive sample to 0 (stay unchanged)
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_facts1[i * pos_neg_group_size] = 0
            rand_facts2[i * pos_neg_group_size] = 0

        # replace the head entity
        facts1[:, 0] = (facts1[:, 0] + rand_facts1) % self.num_ent()
        # replace the tail entity
        facts2[:, 2] = (facts2[:, 2] + rand_facts2) % self.num_ent()
        return np.concatenate((facts1, facts2), axis=0)

    def add_neg_facts_2(self, pos_facts, neg_ratio, mode='head'):
        pos_neg_group_size = 1 + neg_ratio
        pos_neg_facts = np.repeat(np.copy(pos_facts), pos_neg_group_size, axis=0)
        res = []
        num_ent = self.num_ent()
        for head, rel, tail, timestamp in pos_facts:
            neg_samples_list = []
            neg_samples_size = 0
            while neg_samples_size < neg_ratio:
                neg_sample = np.random.randint(num_ent, size=neg_ratio)
                if mode == 'head':
                    mask = np.in1d(neg_sample, self.true_head[(rel, tail, timestamp)], assume_unique=True, invert=True)
                else:
                    # tail mode
                    mask = np.in1d(neg_sample, self.true_tail[(head, rel, timestamp)], assume_unique=True, invert=True)
                neg_sample = neg_sample[mask]
                neg_samples_list.append(neg_sample)
                neg_samples_size += len(neg_sample)
            res.append(np.concatenate(neg_samples_list)[:neg_ratio])
        # res = np.concatenate(res)
        # res = res.reshape(len(pos_facts), -1)

        for i in range(pos_neg_facts.shape[0] // pos_neg_group_size):
            if mode == 'head':
                pos_neg_facts[i * pos_neg_group_size + 1:(i + 1) * pos_neg_group_size, 0] = res[i]
            else:
                pos_neg_facts[i * pos_neg_group_size + 1:(i + 1) * pos_neg_group_size, 2] = res[i]
        return pos_neg_facts

    def next_batch(self, batch_size, neg_ratio=1, device=None):
        pos_facts = self.next_positive_batch(batch_size)
        # pos_neg_batch0 = self.add_neg_facts(pos_facts, neg_ratio)

        # filter the existing true tuples
        pos_neg_head_batch = self.add_neg_facts_2(pos_facts, neg_ratio, mode='head')
        pos_neg_tail_batch = self.add_neg_facts_2(pos_facts, neg_ratio, mode='tail')
        pos_neg_batch = np.concatenate((pos_neg_head_batch, pos_neg_tail_batch), axis=0)

        # convert numpy to tensor and move to correct device
        heads = torch.tensor(pos_neg_batch[:, 0]).long().to(device)
        rels = torch.tensor(pos_neg_batch[:, 1]).long().to(device)
        tails = torch.tensor(pos_neg_batch[:, 2]).long().to(device)
        if type(timestamps[0]) == str:
            timestamps = torch.tensor(pos_neg_batch[:, 3]).str().to(device)
            days = pos_neg_batch[:, 3] // 10000 # year
            days = torch.tensor(days).float().to(device)
            hours = (pos_neg_batch[:, 3] % 10000) // 100 # month
            hours = torch.tensor(hours).float().to(device)
            mins = (pos_neg_batch[:, 3] % 10000) % 100 # day
            mins = torch.tensor(mins).float().to(device)
        else:
            timestamps = torch.tensor(pos_neg_batch[:, 3]).float().to(device)
            # convert to days, hour, minites
            days = (pos_neg_batch[:, 3] / 15) // 96 + 1
            days = torch.tensor(days).float().to(device)
            hours = (pos_neg_batch[:, 3] % 1440) // 60
            mins = ((pos_neg_batch[:, 3] % 1440) % 60) // 15
            hours = torch.tensor(hours).float().to(device)
            mins = torch.tensor(mins).float().to(device)


        return heads, rels, tails, timestamps, days, hours, mins

    def is_last_batch(self):
        return self.startbatch == 0