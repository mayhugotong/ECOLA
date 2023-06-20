import os
import sys
import numpy as np
import argparse
import torch
import random
import time
import datetime
import json
import math
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from fastNLP import TorchLoaderIter
import logging
import torch.distributed
import transformers
from transformers.trainer_utils import is_main_process
from models import E2EBertTKG
from dataset import E2EDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
import socket


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=f'e2e2_{datetime.datetime.now()}', help='set your experiment_name')
    parser.add_argument('--data_dir', type=str, default='../data/DuEE/',
                        help='training data directory path')
    parser.add_argument('--dataset', type=str, default='DuEE', help='name of loaded dataset', choices=['GDELT', 'DuEE', 'Wiki'])
    parser.add_argument('--tkg_type', type=str, default='DE', help='name of used tkg_embs', choices=['DE', 'UTEE','DyERNIE'])
    parser.add_argument('--model_save_dir', type=str, default='../model/ckpts_e2e_2/',
                        help='checkpoint and predictions saving path')
    parser.add_argument('--model_to_load', type=str, default=None,
                        help='the pretrained bert model using the plain text data')
    parser.add_argument('--kg_model_chkpnt', type=str,
                        default=None,
                        help='The tkg trained model, if exists')
    parser.add_argument('--log_file_dir', type=str, default='../logs/e2e_2', help='the logger output path')

    parser.add_argument('--entity_dic_file', type=str, default='../data/DuEE/entities2id.txt',
                        help='each line is of form: entity in text, entity index')
    parser.add_argument('--relation_dic_file', type=str, default='../data/DuEE/relations2id.txt',
                        help='each line is of form: relation id in gdelt, relation index, relation in plain text')
    parser.add_argument('--num_samples_per_file', type=int, default=80000)
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--log_batch_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument('--warm_up', type=float, default=0.1, help="warmup steps")
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--num_train_epochs', type=int, default=500)
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank for distributed training on gpus')
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization and dataset shuffling")
    parser.add_argument('--se_prop', type=float, default=0.68, help='static dimensions in de-simple')
    parser.add_argument('--neg_ratio', type=int, default=100)
    parser.add_argument('--drop_out', type=float, default=0.4)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--print_out_loss_steps', type=int, default=1000)

    parser.add_argument('--loss_lambda', type=float, default=0.3, help='Regularization lambda for MLM loss')
    parser.add_argument('--no_cuda', action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].g"
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()
    return args


def setup_args_gpu(args, logger):
    """
    setup arguments for CUDA
    """
    if args.local_rank == -1 or args.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    ws = os.environ.get('WORLD_SIZE')
    args.distributed_world_size = int(ws) if ws else 1
    logger.info(
        'Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d', socket.gethostname(),
        args.local_rank, device,
        args.n_gpu,
        args.distributed_world_size)
    logger.info("16-bits training: %s ", args.fp16)


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def print_args(args, logger):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")


def setup_for_distributed_mode(model, optimizer, device, n_gpu: int=1, local_rank: int=-1,
                               fp16: bool=False, fp16_opt_level: str="O1"):
    model.to(device)

    if fp16:
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    return model, optimizer


def get_linear_schedule(optimizer, warmup_steps, training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
        linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def parse_batch_x(batch_x:dict, device):
    # 'input_ids', 'num_tokens', 'attention_mask', 'token_type_ids', 'word_masked_lm_labels',
    #  'entity_masked_lm_labels', 'relation_masked_lm_labels', 'tkg_tuple', 'tuple_labels'
    input_ids = batch_x['input_ids'].to(device)
    num_tokens = batch_x['num_tokens'].to(device)
    attention_mask = batch_x['attention_mask'].to(device)
    token_type_ids = batch_x['token_type_ids'].to(device)
    word_masked_lm_labels = batch_x['word_masked_lm_labels'].to(device)
    entity_masked_lm_labels = batch_x['entity_masked_lm_labels'].to(device)
    relation_masked_lm_labels = batch_x['relation_masked_lm_labels'].to(device)
    tkg_tuple = batch_x['tkg_tuple'].to(device)
    tuple_labels = batch_x['tuple_labels'].to(device)
    return input_ids, num_tokens, attention_mask, token_type_ids, word_masked_lm_labels, entity_masked_lm_labels, \
           relation_masked_lm_labels, tkg_tuple, tuple_labels


def parse_batch_y(batch_y: dict, device):
    # 'word_masked_lm_labels', 'entity_masked_lm_labels', 'relation_masked_lm_labels'
    word_masked_lm_labels = batch_y['word_masked_lm_labels'].to(device)
    entity_masked_lm_labels = batch_y['entity_masked_lm_labels'].to(device)
    relation_masked_lm_labels = batch_y['relation_masked_lm_labels'].to(device)
    return word_masked_lm_labels, entity_masked_lm_labels, relation_masked_lm_labels


def calculate_accuracy(pred, target, seq_len=None):
    masks = target!=-100 # -100 will be neglected
    if pred.dim() == target.dim():
        pass
    elif pred.dim() == target.dim() + 1:
        pred = pred.argmax(dim=-1)
    else:
        raise RuntimeError(f'Dimension of prediction does not match the target')

    accuracy_count = torch.sum(torch.eq(pred, target).masked_fill(masks.eq(False), 0)).item()
    total = torch.sum(masks).item()
    return round(float(accuracy_count) / (total + 1e-12), 6)


def replace_batch_data(facts, head_or_tail, num_ent, device, tkg_type, dataset):
    # generate replaced batch data
    # the first one is the ground truth
    batch_size = facts.shape[0]
    replaced_data = np.repeat(np.copy(facts), num_ent+1, axis=0)
    if head_or_tail == 'heads':
        for i in range(batch_size):
            # replaced_data[i*num_ent+1: (i+1)*num_ent+1, 0] = np.arange(0, num_ent)
            replaced_data[i*(1+num_ent)+1: i*(1+num_ent)+num_ent+1, 0] = np.arange(0, num_ent)
    else:
        for i in range(batch_size):
            replaced_data[i*(1+num_ent)+1: i*(1+num_ent)+num_ent+1, 2] = np.arange(0, num_ent)

    # convert to tensor and move to the correct device
    heads = torch.tensor(replaced_data[:, 0]).long().to(device)
    rels = torch.tensor(replaced_data[:, 1]).long().to(device)
    tails = torch.tensor(replaced_data[:, 2]).long().to(device)
    if tkg_type == 'DE':
        # convert to days, hour, minites
        if dataset == 'DuEE': #YYYYMMDD
            days = replaced_data[:, 3] // 10000 # year
            days = torch.tensor(days).float().to(device)
            hours = (replaced_data[:, 3] % 10000) // 100 # month
            hours = torch.tensor(hours).float().to(device)
            mins = (replaced_data[:, 3] % 10000) % 100 # day
            mins = torch.tensor(mins).float().to(device)
        elif dataset == 'GDELT': # Gdelt
            days = (replaced_data[:, 3] / 15) // 96 + 1
            days = torch.tensor(days).float().to(device)
            hours = (replaced_data[:, 3] % 1440) // 60
            mins = ((replaced_data[:, 3] % 1440) % 60) // 15
            hours = torch.tensor(hours).float().to(device)
            mins = torch.tensor(mins).float().to(device)
        elif dataset == 'Wiki': # Wiki
            days = torch.tensor(replaced_data[:, 3]).float().to(device)
            hours, mins = None, None
        return heads, rels, tails, days, hours, mins, replaced_data
    elif tkg_type == 'UTEE' or tkg_type == 'DyERNIE':
        timestamps = replaced_data[:, 3]
        timestamps = torch.tensor(timestamps).float().to(device)
        return heads, rels, tails, timestamps, replaced_data


def validate_batch_and_save(model, dataset, args, val_or_test, logger):
    metrics = {'hit1': 0.0, 'hit3': 0.0, 'hit10': 0.0, 'mrr': 0.0, 'mr': 0}
    start_batch = 0
    end_batch = 0
    num_ent = dataset.num_ent
    l = len(dataset.tkg_data[val_or_test])
    print(f'total val samples: {l}')

    step = 0
    while end_batch < l:
        if start_batch + args.val_batch_size > l:
            end_batch = l
        else:
            end_batch += args.val_batch_size
        batch_facts = dataset.tkg_data[val_or_test][start_batch: end_batch]
        for head_or_tail in ['heads', 'tails']:
            if args.tkg_type == 'DE':
                heads, rels, tails, days, hours, mins, replaced_data = \
                    replace_batch_data(batch_facts, head_or_tail, num_ent, args.device, args.tkg_type, args.dataset)
                scores = model.val_or_test(heads, rels, tails, days, hours, mins)
            elif args.tkg_type == 'UTEE':
                heads, rels, tails, timestamps, replaced_data = \
                    replace_batch_data(batch_facts, head_or_tail, num_ent, args.device, args.tkg_type, args.dataset)
                scores = model.val_or_test_utee(heads, rels, tails, timestamps)
            elif args.tkg_type == 'DyERNIE':
                heads, rels, tails, timestamps, replaced_data = \
                    replace_batch_data(batch_facts, head_or_tail, num_ent, args.device, args.tkg_type, args.dataset)
                scores = model.val_or_test_dyernie(heads, rels, tails, timestamps)

            replaced_data = replaced_data.tolist()
            for i in range(len(replaced_data)//(1+num_ent)):
                checked_data = replaced_data[i*(1+num_ent)+1: i*(1+num_ent)+num_ent+1]
                for idx, cd in enumerate(checked_data):
                    if tuple(cd) in dataset.all_data_as_tuples:
                        scores[i*(1+num_ent)+1+idx] = float('-inf')
            # reshape the scores
            ranks = torch.ones(end_batch - start_batch)
            scores = scores.reshape(-1, 1+num_ent)
            targets = scores[:, 0].unsqueeze(1)
            targets = targets.repeat(1, 1+num_ent)
            ranks += torch.sum((scores > targets).float(), dim=1).cpu()

            metrics['mr'] += torch.sum(ranks)
            metrics['mrr'] += torch.sum(1.0 / ranks)
            metrics['hit1'] += torch.sum((ranks == 1.0).float())
            metrics['hit3'] += torch.sum((ranks <= 3.0).float())
            metrics['hit10'] += torch.sum((ranks <= 10.0).float())
        start_batch = end_batch
        step += 1
        if step % args.print_out_loss_steps == 0:
            logger.info(f'current {step} step, already validated {end_batch} samples')

    # normalize
    for k, v in metrics.items():
        metrics[k] /= (2 * l)
        metrics[k] = v.item()
    logger.info(f'{val_or_test} result:\n')
    logger.info(f'\tHit@1 = {metrics["hit1"]}')
    logger.info(f'\tHit@3 = {metrics["hit3"]}')
    logger.info(f'\tHit@10 = {metrics["hit10"]}')
    logger.info(f'\tMR = {metrics["mr"]}')
    logger.info(f'\tMRR = {metrics["mrr"]}')
    return metrics

def randomize_training_instances(args, logger, dataset_name):
    logger.info('Shuffling the data......')
    # read all data from existing shards files
    # i have 10 files
    all_processed_data = []
    if dataset_name == 'GDELT':
        for i in range(10):
            with open(os.path.join(args.data_dir, f'training_data_{i}.json'), 'r') as f:
                for x in f:
                    instance = json.loads(x)
                    all_processed_data.append(instance)
    elif dataset_name == 'DuEE':
        with open(os.path.join(args.data_dir, f'training_data.json'), 'r') as f:
            for x in f:
                instance = json.loads(x)
                all_processed_data.append(instance)
    elif dataset_name == 'Wiki':
        for i in range(3):
            with open(os.path.join(args.data_dir, f'training_data_{i}.json'), 'r') as f:
                for x in f:
                    instance = json.loads(x)
                    all_processed_data.append(instance)
    # shuffle the data
    random.seed(args.seed)
    random.shuffle(all_processed_data)
    # split the new data
    logger.info('Start to split and save all data......')
    l = len(all_processed_data)
    if dataset_name == 'GDELT' or dataset_name == 'Wiki':
        num_files = int(math.ceil(l / args.num_samples_per_file))
        for n in range(num_files - 1):
            cur_data = all_processed_data[n * args.num_samples_per_file: (n + 1) * args.num_samples_per_file]
            logger.info(f'Writing #{n} file...')
            with open(os.path.join(args.data_dir, f'training_data_{n}.json'), 'w') as fw:
                for d in cur_data:
                    fw.write(json.dumps(d) + '\n')
        # the last shard
        cur_data = all_processed_data[(num_files - 1) * args.num_samples_per_file:]
        logger.info(f'Writing #{num_files - 1} file...')
        with open(os.path.join(args.data_dir, f'training_data_{num_files - 1}.json'), 'w') as fw:
            for d in cur_data:
                fw.write(json.dumps(d) + '\n')
    elif dataset_name == 'DuEE':
        num_files = 1
        cur_data = all_processed_data
        logger.info(f'Writing #{num_files} file...')
        with open(os.path.join(args.data_dir, f'training_data.json'), 'w') as fw:
            for d in cur_data:
                fw.write(json.dumps(d) + '\n')

    logger.info('Done!\n')

def main():
    args = parse_args()
    
    if args.model_save_dir is not None:
        args.model_save_dir = os.path.join(args.model_save_dir, f'{datetime.datetime.now()}')
        os.makedirs(args.model_save_dir, exist_ok=True)

    if args.log_file_dir is not None:
        os.makedirs(args.log_file_dir, exist_ok=True)

    # set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = f'e2e2_{datetime.datetime.now()}.log'
    if (logger.hasHandlers()):
        logger.handlers.clear()
    console = logging.StreamHandler()
    logger.addHandler(console)
    fileHandler = logging.FileHandler(os.path.join(args.log_file_dir, log_file), mode='w')
    logger.addHandler(fileHandler)

    # gpu setting
    setup_args_gpu(args, logger)
    # random seed setting
    set_seed(args)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps // args.distributed_world_size

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    print_args(args, logger)

    # load the entity and relation vocab length
    with open(args.entity_dic_file, 'r') as freader:
        lines = freader.readlines()
    ent_num = len(lines)

    with open(args.relation_dic_file, 'r') as freader:
        lines = freader.readlines()
    rel_num = len(lines)

    logger.info(f'#entities: {ent_num - 1}, #relations: {rel_num - 1}')

    tokenizer = BertTokenizer.from_pretrained('../../FTKE_Bert/model/bert_origin/bert-base-uncased')
    config = BertConfig.from_pretrained('../../FTKE_Bert/model/bert_origin/bert-base-uncased')
    word_mask_index = tokenizer.mask_token_id

    # prepare the training data
    train_data = E2EDataset(args.data_dir, word_mask_index, config.vocab_size, ent_num, rel_num,
                            args.seed, tokenizer, args.neg_ratio, args.dataset)
    if args.local_rank == -1:
        # train_sampler = RandomSampler(train_data)
        train_sampler = SequentialSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_iterator = TorchLoaderIter(train_data, collate_fn=train_data.collate_fn, batch_size=train_batch_size,
                                     sampler=train_sampler, num_workers=4)

    if args.local_rank in [-1, 0]:
        logger.info(f'#training data samples: {len(train_data)}')

    # prepare components for training
    logger.info('****** Initializing the component for training ******')
    if args.kg_model_chkpnt is not None:
        logger.info('Loading the pretrained kg-models......')
        logger.info(args.kg_model_chkpnt)
        tkg_model = torch.load(args.kg_model_chkpnt, map_location=torch.device('cpu')).module
    else:
        tkg_model = None
    model = E2EBertTKG.from_pretrained('../../FTKE_Bert/model/bert_origin/bert-base-uncased', ent_num=ent_num, rel_num=rel_num, se_prop=args.se_prop,
                                       drop_out=args.drop_out, tkg_type=args.tkg_type, dataset=args.dataset, \
                                        loss_lambda=args.loss_lambda)

    # load the pre-trained checkpoint if exists
    if args.model_to_load is not None:
        pretrained_dict = torch.load(args.model_to_load, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        # filter out the unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite the parameters
        model_dict.update(pretrained_dict)
        # load the new state dict
        model.load_state_dict(model_dict)

    model.extend_type_embeddings(token_type=3)
    
    # load the pre-trained tkg checkpoint if exists
    if args.kg_model_chkpnt is not None:
        pretrained_model = torch.load(args.kg_model_chkpnt, map_location=torch.device('cpu')).module
        pretrained_model.eval()
        pretrained_dict = pretrained_model.state_dict()
        target_keys = ['ent_embeddingsh_static.weight', 'ent_embeddingst_static.weight', 'rel_embeddings_f.weight', 'rel_embeddings_i.weight']
        new_keys = ['ent_embs_h.weight','ent_embs_t.weight','rel_embs_f.weight', 'rel_embs_i.weight']
        for key,n_key in zip(target_keys, new_keys):
            pretrained_dict[n_key] = pretrained_dict.pop(key)
        # print(pretrained_dict.keys())
        model_dict = model.state_dict()
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        # print(model_dict.keys())
        model.load_state_dict(model_dict)


    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_eps, weight_decay=args.weight_decay)

    # also move the model to the current device
    setup_for_distributed_mode(model, optimizer, args.device, args.n_gpu, args.local_rank,
                               args.fp16, args.fp16_opt_level)
    num_training_steps = int(
        len(train_data) / train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    train_steps_per_process = num_training_steps
    if args.local_rank != -1:
        train_steps_per_process = train_steps_per_process // torch.distributed.get_world_size()
    warm_up_steps = int(args.warm_up * num_training_steps)
        
    # initial evaluation of model
    ini_eval = False
    if ini_eval:
        logger.info('****** Start initial evaluation of model ******')
        model.eval()
        metrics = validate_batch_and_save(model, train_data, args, 'test', logger)
        init_mrr = metrics['mrr']
        logger.info(f'init metrics of the model = {init_mrr}')
        
    # start to train
    if args.do_train:
        logger.info('****** Start to train ******')
        best_mrr = -1.0
        best_model_index = '0'
        logger.info(f'\t#training samples = {len(train_data)}')
        logger.info(f'\t#traning epochs = {args.num_train_epochs}')
        logger.info(f'\t#training steps = {num_training_steps}')

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=num_training_steps)
        fout = open(os.path.join(args.model_save_dir, f'loss.{datetime.datetime.now()}'), 'w')
        total_steps_per_epoch = train_iterator.num_batches

        for epoch in range(args.num_train_epochs):
            model.train()
            train_loss = 0.0
            t1 = time.time()
            # total_steps_per_epoch = train_iterator.num_batches

            for step, batch in enumerate(train_iterator):
                
                batch_x, batch_y = batch
                # batch_x and batch_y are dictionaries, get the tensor and move them to the current device
                input_ids, num_tokens, attention_mask, token_type_ids, word_masked_lm_labels, entity_masked_lm_labels, \
                relation_masked_lm_labels, tkg_tuple, tuple_labels = parse_batch_x(batch_x, args.device)

                word_masked_lm_labels_y, entity_masked_lm_labels_y, relation_masked_lm_labels_y \
                    = parse_batch_y(batch_y, args.device)

                # loss, word_predict, entity_predict, relation_predict are keys in output_dic
                output_dic = model(input_ids=input_ids,
                                   num_tokens=num_tokens,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   word_masked_lm_labels=word_masked_lm_labels,
                                   entity_masked_lm_labels=entity_masked_lm_labels,
                                   relation_masked_lm_labels=relation_masked_lm_labels,
                                   tkg_tuple=tkg_tuple,
                                   tuple_labels=tuple_labels)
                if args.n_gpu > 1:
                    loss = output_dic['total_loss'].mean()
                    mlm_loss = output_dic['mlm_loss'].mean()
                    tkg_loss = output_dic['tkg_loss'].mean()
                else:
                    loss = output_dic['total_loss']
                    mlm_loss = output_dic['mlm_loss']
                    tkg_loss = output_dic['tkg_loss']
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    from apex import amp
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                train_loss += loss.item()
                # print(f'loss: {loss.item()}')

                fout.write(f'total_loss: {loss.item() * args.gradient_accumulation_steps}, '
                           f'mlm_loss: {mlm_loss.item() * args.gradient_accumulation_steps}, '
                           f'tkg_loss: {tkg_loss.item() * args.gradient_accumulation_steps}\n')

                # calculate the accuracy

                word_acc = calculate_accuracy(output_dic['word_pred'], word_masked_lm_labels_y)
                entity_acc = calculate_accuracy(output_dic['ent_pred'], entity_masked_lm_labels_y)
                relation_acc = calculate_accuracy(output_dic['rel_pred'], relation_masked_lm_labels_y)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % args.log_batch_steps == 0:
                    lr = optimizer.param_groups[0]['lr']
                    t2 = time.time()
                    logger.info(f'Epoch: {epoch}, Step: {step+1}/{total_steps_per_epoch}, LR: {lr}, Time: {t2 - t1}s, '
                                f'Word_acc: {word_acc}, Ent_acc: {entity_acc}, Rel_acc: {relation_acc}, \t'
                                f'Total_loss: {loss.item()}, MLM_loss: {mlm_loss.item()}, TKG_loss: {tkg_loss.item()}\n')
                    t1 = t2

            train_loss = train_loss / train_steps_per_process * args.num_train_epochs
            logger.info(f'Average loss of epoch {epoch}: {train_loss} \n\n')
            fout.write(f'Average loss of epoch {epoch}: {train_loss} \n\n')

            # evaluate the model
            logger.info('****** Start validation ******')
            model.eval()
            if args.dataset == 'Wiki' or args.dataset == 'DuEE':
                metrics = validate_batch_and_save(model, train_data, args, 'test', logger)
            else:
                metrics = validate_batch_and_save(model, train_data, args, 'val', logger)
            mrr = metrics['mrr']
            if mrr > best_mrr:
                best_mrr = mrr
                best_model_index = str(epoch)

            # save the model
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_name = os.path.join(args.model_save_dir, f'model_{epoch}.bin')
            torch.save(model_to_save.state_dict(), output_model_name)

            # shuffle the data by myself...... (text + tuple)
            randomize_training_instances(args, logger, args.dataset)
            # need to load the data again???
            # prepare the training data
            train_data = E2EDataset(args.data_dir, word_mask_index, config.vocab_size, ent_num, rel_num,
                                    args.seed, tokenizer, args.neg_ratio, args.dataset)
            if args.local_rank == -1:
                train_sampler = SequentialSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_iterator = TorchLoaderIter(train_data, collate_fn=train_data.collate_fn, batch_size=train_batch_size,
                                             sampler=train_sampler, num_workers=4)

        logger.info(f'The best mrr is {best_mrr}, the best model index is {best_model_index}')


        # start to test
        model_path = os.path.join(args.model_save_dir, f'model_{best_model_index}.bin')
        plm_state_dict = torch.load(model_path)
        pretrain_config = BertConfig.from_pretrained('../../FTKE_Bert/model/bert_origin/bert-base-uncased', type_vocab_size=3)
        plm_model = E2EBertTKG(pretrain_config, ent_num=ent_num, rel_num=rel_num, se_prop=args.se_prop,
                                       drop_out=args.drop_out, tkg_type=args.tkg_type, dataset=args.dataset, \
                                        loss_lambda=args.loss_lambda)
        plm_model.load_state_dict(plm_state_dict, strict=True)
        setup_for_distributed_mode(plm_model, optimizer, args.device, args.n_gpu, args.local_rank,
                               args.fp16, args.fp16_opt_level)
        plm_model.eval()
        test_metrics = validate_batch_and_save(plm_model, train_data, args, 'test', logger)
        test_metrics = {f'test_{k}': v for k, v in test_metrics.items()}
        logger.info(f'All training and val/test has been finished, the best test mrr is {test_metrics}')
        fout.close()


if __name__ == '__main__':
    main()