import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from utils import *

class E2EBertTKG(BertForMaskedLM):
    config_class = BertConfig
    def __init__(self, config, ent_num, rel_num, se_prop=0.68, drop_out=0.4, tkg_type='DE', dataset='GDELT', loss_lambda=0.1):
        # the last item in ent_emb and rel_emb is [MASK]
        super().__init__(config)
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.drop_out = drop_out
        self.loss_lambda = loss_lambda
        self.ent_lm_head = EntityMLMHead(config, ent_num)
        self.rel_lm_head = RelationMLMHead(config, rel_num)
        self.bert_hidden_size = config.hidden_size
        self.dataset = dataset
        self.tkg_type = tkg_type

        self.static_emb_dim = int(se_prop * config.hidden_size)
        self.temporal_emb_dim = config.hidden_size - self.static_emb_dim

        if self.tkg_type == 'DyERNIE':
            self.create_dyernie_embeds()
        else:
            ###############   DE or UTEE   ###############
            self.ent_embs_h = nn.Embedding(self.ent_num - 1, self.static_emb_dim)  
            self.ent_embs_t = nn.Embedding(self.ent_num - 1, self.static_emb_dim)  
            self.rel_embs_f = nn.Embedding(self.rel_num - 1, config.hidden_size)  
            self.rel_embs_i = nn.Embedding(self.rel_num - 1, config.hidden_size)  
            
            nn.init.xavier_uniform_(self.ent_embs_h.weight)
            nn.init.xavier_uniform_(self.ent_embs_t.weight)
            nn.init.xavier_uniform_(self.rel_embs_f.weight)
            nn.init.xavier_uniform_(self.rel_embs_i.weight)
            # temporal embeddings for entities
            if self.tkg_type == 'DE':
                self.create_time_embeds()
                self.time_nonlinear = torch.sin
            elif self.tkg_type == 'UTEE':
                self.create_utee_time_embeds()
            # initialize weights and apply finial processing
            self.init_weights()


    def create_dyernie_embeds(self):

        self.name = 'Euclidean'
        self.score_func_choice = 'MuRP'
        dim = 768
        self.dyn_part = 0.68
        self.use_seasonal_part = False
        self.use_cosh = False
        self.dropout = 0

        self.curvature = to_device(torch.tensor(0., dtype=torch.float, requires_grad=False))
        self.cur_train = False
        self.only_static_part = False

        #############Must-use parameters
        r = 6 / np.sqrt(dim)
        self.Wu = nn.Parameter(to_device(torch.tensor(np.random.uniform(-r, r, (self.rel_num, dim)), dtype=torch.float, requires_grad=True)))
        self.bs = nn.Parameter(to_device(torch.zeros(self.ent_num, dtype=torch.float, requires_grad=True)))
        self.bo = nn.Parameter(to_device(torch.zeros(self.ent_num, dtype=torch.float, requires_grad=True)))

        dyn_dims = int(self.dyn_part * dim)
        self.initial_Eh_euc = nn.Embedding(self.ent_num, dyn_dims, padding_idx=0)
        self.initial_Eh_euc.weight.data = to_device(torch.tensor(np.random.uniform(-r, r, (self.ent_num, dyn_dims)), dtype=torch.float))
        # Parameters for learn changing term of time embedding
        if not self.only_static_part:
            self.time_emb_v =nn.Parameter(to_device(torch.tensor(np.random.uniform(-r, r, (self.ent_num, dyn_dims)), dtype=torch.float, requires_grad=True)))
        if self.dyn_part < 1:
            self.initial_Eh_euc_static = nn.Embedding(self.ent_num, dim - dyn_dims, padding_idx=0)
            self.initial_Eh_euc_static.weight.data = to_device(torch.tensor(np.random.uniform(-r, r, (self.ent_num, dim - dyn_dims)), dtype=torch.float))
        #########optional paramters
        ####relation vector
        self.rvh_euc = nn.Embedding(self.rel_num, dim, padding_idx=0)
        self.rvh_euc.weight.data = to_device(1e-3 * torch.randn((self.rel_num, dim), dtype=torch.float))

        ####SimplE parameter
        self.Wu_inverse = nn.Parameter(to_device(torch.tensor(np.random.uniform(-r, r, (self.rel_num, dim)), dtype=torch.float, requires_grad=True)))
        self.time_emb_v_tail =nn.Parameter(to_device(torch.tensor(np.random.uniform(-r, r, (self.ent_num, dyn_dims)), dtype=torch.float, requires_grad=True)))
        self.initial_Eh_euc_tail = nn.Embedding(self.ent_num, dyn_dims, padding_idx=0)
        self.initial_Eh_euc_tail.weight.data =to_device(torch.tensor(np.random.uniform(-r, r, (self.ent_num, dyn_dims)), dtype=torch.float))
        if self.dyn_part < 1:
            self.initial_Eh_euc_tail_static = nn.Embedding(self.ent_num, dim - dyn_dims, padding_idx=0)
            self.initial_Eh_euc_tail_static.weight.data = to_device(torch.tensor(np.random.uniform(-r, r, (self.ent_num, dim - dyn_dims)), dtype=torch.float))
        

    def emb_evolving_vanilla(self, e_idx, times, use_tail_embd = False, use_dropout = False):
        init_embd_e = self.initial_Eh_euc_tail.weight[e_idx] if use_tail_embd else self.initial_Eh_euc.weight[e_idx]
        if self.only_static_part:
            return init_embd_e
        linear_velocities = self.time_emb_v_tail[e_idx] if use_tail_embd else  self.time_emb_v[e_idx]

        # #########all velocity vectors are defined in tangent space, update the embedding vector
        emd_linear_temp = linear_velocities * times.unsqueeze(1) #[:, :, None] # batch*nneg*dim

        # ##################Drift in tangent space
        new_embds_e = init_embd_e + emd_linear_temp
        if self.dyn_part < 1:
            static_embd_e = self.initial_Eh_euc_tail_static.weight[e_idx] if use_tail_embd else self.initial_Eh_euc_static.weight[e_idx]
            final_embd_e = torch.cat((new_embds_e, static_embd_e), dim=-1)
            if use_dropout:
                final_embd_e = F.dropout(final_embd_e, p=self.dropout, training=self.training)
            return final_embd_e
        if use_dropout:
            new_embds_e = F.dropout(new_embds_e, p=self.dropout, training=self.training)

        return new_embds_e

    def get_dyernie_tkg_embeddings(self, u_idx, r_idx, v_idx, t, batch_comp=True):
        if self.dataset == 'GDELT':
            time_rescale = 45000
        elif self.dataset == 'DuEE':              
            time_rescale = 20220301
        elif self.dataset == 'Wiki':
            time_rescale = 83
        t = t / time_rescale
        # SIMPLE
        Wu = self.Wu[r_idx]
        Wu_inverse = self.Wu_inverse[r_idx]
        v_head = self.emb_evolving_vanilla(v_idx, t, use_dropout=False)
        u_head = self.emb_evolving_vanilla(u_idx, t, use_dropout=False)
        u_tail = self.emb_evolving_vanilla(u_idx, t, use_tail_embd=True, use_dropout=False)
        v_tail = self.emb_evolving_vanilla(v_idx, t, use_tail_embd=True, use_dropout=False)

        return u_head, Wu, v_tail, u_tail, Wu_inverse, v_head # h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2
        
    def create_utee_time_embeds(self, ):
        dim = self.temporal_emb_dim
        if self.dataset == 'GDELT':
            t_min, t_max = 0, 45000
        elif self.dataset == 'DuEE':
            t_min, t_max = 20180101, 20220301
        elif self.dataset == 'Wiki':
            t_min, t_max = 0, 83
        self.freq = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.amps = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        self.phas = nn.Parameter(data=torch.zeros([1, dim]), requires_grad=True)
        torch.nn.init.uniform_(self.freq.data, a=t_min, b=t_max)
        torch.nn.init.xavier_uniform_(self.amps.data)
        torch.nn.init.uniform_(self.phas.data, a=0, b=t_max)

    def extend_type_embeddings(self, token_type=3):
        self.bert.embeddings.token_type_embeddings = nn.Embedding(token_type, self.config.hidden_size,
                                                                  _weight=torch.zeros(
                                                                      (token_type, self.config.hidden_size)))

    def create_time_embeds(self):
        # frequency embeddings
        if self.dataset == 'Wiki':
            self.freq_h = nn.Embedding(self.ent_num-1, self.temporal_emb_dim)
            self.freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.freq_h.weight)
            nn.init.xavier_uniform_(self.freq_t.weight)
            nn.init.xavier_uniform_(self.phi_h.weight)
            nn.init.xavier_uniform_(self.phi_t.weight)
            nn.init.xavier_uniform_(self.amp_h.weight)
            nn.init.xavier_uniform_(self.amp_t.weight)

        else:
            self.day_freq_h = nn.Embedding(self.ent_num-1, self.temporal_emb_dim)
            self.day_freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_freq_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_freq_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_freq_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.day_freq_h.weight)
            nn.init.xavier_uniform_(self.day_freq_t.weight)
            nn.init.xavier_uniform_(self.hour_freq_h.weight)
            nn.init.xavier_uniform_(self.hour_freq_t.weight)
            nn.init.xavier_uniform_(self.min_freq_h.weight)
            nn.init.xavier_uniform_(self.min_freq_t.weight)

            # phi embeddings
            self.day_phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.day_phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_phi_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_phi_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.day_phi_h.weight)
            nn.init.xavier_uniform_(self.day_phi_t.weight)
            nn.init.xavier_uniform_(self.hour_phi_h.weight)
            nn.init.xavier_uniform_(self.hour_phi_t.weight)
            nn.init.xavier_uniform_(self.min_phi_h.weight)
            nn.init.xavier_uniform_(self.min_phi_t.weight)

            # amplitude embeddings
            self.day_amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.day_amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.hour_amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_amp_h = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            self.min_amp_t = nn.Embedding(self.ent_num - 1, self.temporal_emb_dim)
            nn.init.xavier_uniform_(self.day_amp_h.weight)
            nn.init.xavier_uniform_(self.day_amp_t.weight)
            nn.init.xavier_uniform_(self.hour_amp_h.weight)
            nn.init.xavier_uniform_(self.hour_amp_t.weight)
            nn.init.xavier_uniform_(self.min_amp_h.weight)
            nn.init.xavier_uniform_(self.min_amp_t.weight)
    
    def get_utee_time_embedd(self, timestamps):
        timestamps = timestamps.squeeze().unsqueeze(-1)
        assert timestamps.dim() == 2 and timestamps.size(1) == 1, f"timestamp {timestamps.size()}"

        omega = 1 / self.freq
        feat = self.amps * torch.sin(timestamps * omega + self.phas)
        return feat

    def get_time_embedd(self, entities, days, hours, mins, head_or_tail):
        # dataset gdelt/wiki
        if self.dataset == 'GDELT' or self.dataset == 'DuEE': # dataset Gdelt/DuEE
            if head_or_tail == 'head':
                d = self.day_amp_h(entities) * self.time_nonlinear(self.day_freq_h(entities) * days + self.day_phi_h(entities))
                h = self.hour_amp_h(entities) * self.time_nonlinear(self.hour_freq_h(entities) * hours + self.hour_phi_h(entities))
                m = self.min_amp_h(entities) * self.time_nonlinear(self.min_freq_h(entities) * mins + self.min_phi_h(entities))
            else:
                d = self.day_amp_t(entities) * self.time_nonlinear(self.day_freq_t(entities) * days + self.day_phi_t(entities))
                h = self.hour_amp_t(entities) * self.time_nonlinear(self.hour_freq_t(entities) * hours + self.hour_phi_t(entities))
                m = self.min_amp_t(entities) * self.time_nonlinear(self.min_freq_t(entities) * mins + self.min_phi_t(entities))
            return d + h + m
        
        elif self.dataset == 'Wiki' :
            if head_or_tail == 'head':
                d = self.amp_h(entities) * self.time_nonlinear(self.freq_h(entities) * days + self.phi_h(entities))
            else:
                d = self.amp_t(entities) * self.time_nonlinear(self.freq_t(entities) * days + self.phi_t(entities))
            return d

    def get_tkg_added_ent_embs(self, ent_ids, h_embs1, t_embs1):
        entity_embeddings = torch.cat((h_embs1.unsqueeze(1), t_embs1.unsqueeze(1)), dim=1)
        return entity_embeddings

    def get_tkg_added_rel_embs(self, rel_ids, r_embs1, r_embs2):
        return (r_embs1 + r_embs2) / 2

    def get_tkg_static_Embeddings(self, heads, rels, tails, intervals = None):
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def get_tkg_Embeddings(self, heads, rels, tails, years, months, days, intervals = None):
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        
        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)
        
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2
 
    def get_utee_tkg_Embeddings(self, heads, rels, tails, timestamps):
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_static_Embeddings(heads, rels, tails)
        h_embs1 = torch.cat((h_embs1, self.get_utee_time_embedd(timestamps)), 1)
        t_embs1 = torch.cat((t_embs1, self.get_utee_time_embedd(timestamps)), 1)
        h_embs2 = torch.cat((h_embs2, self.get_utee_time_embedd(timestamps)), 1)
        t_embs2 = torch.cat((t_embs2, self.get_utee_time_embedd(timestamps)), 1)
        
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def get_tuple_time(self, tuple_t):
        if self.dataset == 'DuEE' : #YYYYMMDD
            days = tuple_t // 10000 # year
            days = days.float()
            hours = (tuple_t % 10000) // 100 # month
            hours = hours.float()
            mins = (tuple_t % 10000) % 100 # day
            mins = mins.float()
            
        elif self.dataset == 'GDELT' :
            days = (tuple_t / 15) // 96 + 1
            days = days.float()
            hours = (tuple_t % 1440) // 60
            mins = ((tuple_t % 1440) % 60) // 15
            hours = hours.float()
            mins = mins.float()
            
        elif self.dataset == 'Wiki':
            days = tuple_t.float()
            hours, mins = None, None
        
        days = days.view(-1, 1)
        if hours != None and mins != None:
            hours = hours.view(-1, 1)
            mins = mins.view(-1, 1)
        return days, hours, mins

    def forward(self, input_ids=None, num_tokens=None, attention_mask=None, token_type_ids=None, inputs_embeds=None,
                word_masked_lm_labels=None, entity_masked_lm_labels=None, relation_masked_lm_labels=None, tkg_tuple=None,
                tuple_labels=None):
        loss_fct = CrossEntropyLoss()
        batch_size = input_ids.shape[0]
        #### TKG part ####
        # get heads, rels, tails, timestamps(convert to min, hour, day) from tkg_tuple
        tkg_tuple = tkg_tuple.view(-1, 4)
        heads = tkg_tuple[:, 0].long()
        rels = tkg_tuple[:, 1].long()
        tails = tkg_tuple[:, 2].long()
        days, hours, mins = self.get_tuple_time(tkg_tuple[:, 3])
        # get embeddings
        if self.tkg_type == 'DE':
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_Embeddings(heads, rels, tails, days, hours, mins)
        elif self.tkg_type == 'UTEE':
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_utee_tkg_Embeddings(heads, rels, tails, tkg_tuple[:, 3].long())
        elif self.tkg_type == 'DyERNIE':
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_dyernie_tkg_embeddings(heads, rels, tails, tkg_tuple[:, 3].long())
        # get tkg scores
        tkg_scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        tkg_scores = F.dropout(tkg_scores, p=self.drop_out, training=self.training)
        tkg_scores = torch.sum(tkg_scores, dim=1)
        # get tkg loss
        num_examples = tuple_labels.shape[0]
        tkg_scores = tkg_scores.reshape(2*num_examples, -1)
        tuple_labels = tuple_labels.reshape(-1).long()
        tkg_loss = loss_fct(tkg_scores, tuple_labels)


        #### bert MLM part ####
        num_word_tokens = num_tokens[0] - 3
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids[:, :num_word_tokens])
        entity_embeddings = self.get_tkg_added_ent_embs(input_ids[:, num_word_tokens: num_word_tokens+2],\
            h_embs1.view(batch_size,-1,self.bert_hidden_size)[:,0], t_embs1.view(batch_size,-1,self.bert_hidden_size)[:,0])
        relation_embeddings = self.get_tkg_added_rel_embs(input_ids[:, -1],\
            r_embs1.view(batch_size,-1,self.bert_hidden_size)[:,0], r_embs2.view(batch_size,-1,self.bert_hidden_size)[:,0]).unsqueeze(1)

        # MLM loss
        # concat the 3 parts of embeddings
        inputs_embeds = torch.cat([word_embeddings, entity_embeddings, relation_embeddings], dim=1)
        outputs_mlm = self.bert(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                inputs_embeds=inputs_embeds)
        sequence_output = outputs_mlm[0]
            
        word_prediction_scores = self.cls(sequence_output)
        word_prediction = torch.argmax(word_prediction_scores[:, :num_word_tokens, :], dim=-1)
        word_lm_loss = loss_fct(word_prediction_scores.view(-1, self.config.vocab_size), word_masked_lm_labels.view(-1))

        entity_prediction_scores = self.ent_lm_head(sequence_output)
        entity_prediction = torch.argmax(entity_prediction_scores[:, num_word_tokens:num_word_tokens+2, :], dim=-1)
        entity_lm_loss = loss_fct(entity_prediction_scores.view(-1, self.ent_num), entity_masked_lm_labels.view(-1))

        relation_prediction_scores = self.rel_lm_head(sequence_output)
        relation_prediction = torch.argmax(relation_prediction_scores[:, -1, :], dim=-1).unsqueeze(1)
        relation_lm_loss = loss_fct(relation_prediction_scores.view(-1, self.rel_num),
                                    relation_masked_lm_labels.view(-1))

        mlm_loss = word_lm_loss + entity_lm_loss + relation_lm_loss

        total_loss = self.loss_lambda * mlm_loss + tkg_loss
        return {'total_loss': total_loss, 'mlm_loss': mlm_loss, 'tkg_loss': tkg_loss,
                'word_pred': word_prediction, 'ent_pred': entity_prediction, 'rel_pred': relation_prediction}


    def val_or_test(self, heads, rels, tails, days, hours, mins):
        # in this function, just use the tkg model to do test, and just compute the score
        # DE-SimplE part
        days = days.view(-1, 1)
        if hours != None and mins != None:
            hours = hours.view(-1, 1)
            mins = mins.view(-1, 1)
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_tkg_Embeddings(heads, rels, tails, days, hours, mins)
        tkg_scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        tkg_scores = F.dropout(tkg_scores, p=self.drop_out, training=self.training)
        tkg_scores = torch.sum(tkg_scores, dim=1)
        return tkg_scores
    
    def val_or_test_utee(self, heads, rels, tails, timestamps):
        # in this function, just use the tkg model to do test, and just compute the score
        # UTEE part
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_utee_tkg_Embeddings(heads, rels, tails, timestamps)
        tkg_scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        tkg_scores = F.dropout(tkg_scores, p=self.drop_out, training=self.training)
        tkg_scores = torch.sum(tkg_scores, dim=1)
        return tkg_scores

    def val_or_test_dyernie(self, heads, rels, tails, timestamps):
        # in this function, just use the tkg model to do test, and just compute the score
        # DyERNIE part
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.get_dyernie_tkg_embeddings(heads, rels, tails, timestamps)
        tkg_scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        tkg_scores = F.dropout(tkg_scores, p=self.drop_out, training=self.training)
        tkg_scores = torch.sum(tkg_scores, dim=1)
        return tkg_scores


class EntityMLMHead(nn.Module):
    def __init__(self, config, ent_num):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, ent_num, bias=False)
        self.bias = nn.Parameter(torch.zeros(ent_num), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        h = self.dense(hidden_states)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        h = self.decoder(h)
        return h


class RelationMLMHead(nn.Module):
    def __init__(self, config, rel_num):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, rel_num, bias=False)
        self.bias = nn.Parameter(torch.zeros(rel_num), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        h = self.dense(hidden_states)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        h = self.decoder(h)
        return h
