import os
import random
import collections
import numpy as np
import argparse
import torch
import logging
import socket
from torch.optim.lr_scheduler import LambdaLR


MLMInstance = collections.namedtuple("MLMInstance", ["index", "label"])

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def create_mlm_instances(tokens, mask_index, vocab_size, tokenizer=None,
                         random_seed=12345, masked_lm_prob=0.15, max_predictions_per_seq=20):
    rng = random.Random(random_seed)
    cand_indexes = []
    if mask_index == 103: # [MASK] id in bert tokenizer
        # mask the words in the sentence
        special_tokens = [0, 101, 102] # [PAD], [CLS], [SEP]
        for (i, token) in enumerate(tokens):
            if token in special_tokens:
                continue
            if len(cand_indexes)>=1 and tokenizer.decode([token]).startswith('##'):
                cand_indexes[-1].append(i)
            cand_indexes.append([i])
    else:
        # mask entity or relation as an unit
        special_tokens = [vocab_size-1] # [MASK]
        for (i, token) in enumerate(tokens):
            if token in special_tokens:
                continue
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    output_tokens = list(tokens)

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            if rng.random() < 0.8:
                # mask out the token
                masked_token = mask_index
            else:
                if rng.random() < 0.5:
                    # keep the original token
                    masked_token = tokens[index]
                else:
                    # a random selected token
                    masked_token = rng.randint(0, vocab_size-1)

            output_tokens[index] = masked_token
            masked_lms.append(MLMInstance(index=index, label=tokens[index]))
    masked_lms = sorted(masked_lms, key=lambda x : x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for m in masked_lms:
        masked_lm_positions.append(m.index)
        masked_lm_labels.append(m.label)
    masked_lms = np.ones(len(tokens), dtype=int) * -100 # -100 will be ignored in CrossEntropy loss function
    masked_lms[masked_lm_positions] = masked_lm_labels
    masked_lms = list(masked_lms)
    return output_tokens, masked_lms