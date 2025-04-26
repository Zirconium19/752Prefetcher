import math
import copy
from abc import ABC, abstractmethod
from collections import defaultdict
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import lzma


class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass

class CacheSimulator(object):

    def __init__(self, sets, ways, block_size, eviction_hook=None, name=None) -> None:
        super().__init__()
        self.ways = ways
        self.name = name
        self.way_shift = int(math.log2(ways))
        self.sets = sets
        self.block_size = block_size
        self.block_shift = int(math.log2(block_size))
        self.storage = defaultdict(list)
        self.label_storage = defaultdict(list)
        self.eviction_hook = eviction_hook

    def parse_address(self, address):
        block = address >> self.block_shift
        way = block % self.ways
        tag = block >> self.way_shift
        return way, tag

    def load(self, address, label=None, overwrite=False):
        way, tag = self.parse_address(address)
        hit, l = self.check(address)
        if not hit:
            self.storage[way].append(tag)
            self.label_storage[way].append(label)
            if len(self.storage[way]) > self.sets:
                evicted_tag = self.storage[way].pop(0)
                evicted_label = self.label_storage[way].pop(0)
                if self.eviction_hook:
                    self.eviction_hook(self.name, address, evicted_tag, evicted_label)
        else:
            current_index = self.storage[way].index(tag)
            _t, _l = self.storage[way].pop(current_index), self.label_storage[way].pop(current_index)
            self.storage[way].append(_t)
            self.label_storage[way].append(_l)
        if overwrite:
            self.label_storage[way][self.storage[way].index(tag)] = label
        return hit, l

    def check(self, address):
        way, tag = self.parse_address(address)
        if tag in self.storage[way]:
            return True, self.label_storage[way][self.storage[way].index(tag)]
        else:
            return False, None


# Configuration class for TransforMAP
class TransforMAPConfig:
    def __init__(self):
        # Core configuration parameters
        self.BLOCK_BITS = 6
        self.PAGE_BITS = 12
        self.TOTAL_BITS = 64
        self.BLOCK_NUM_BITS = self.TOTAL_BITS - self.BLOCK_BITS
        self.SPLIT_BITS = 6
        self.LOOK_BACK = 5
        self.PRED_FORWARD = 2
        self.BITMAP_SIZE = 2**(self.PAGE_BITS - self.BLOCK_BITS)

        # Model tokens
        self.PAD_ID = 0
        self.START_ID = 2
        self.END_ID = 3
        self.VOCAB_SIZE = 2**self.SPLIT_BITS + 3  # binary + 3
        self.OFFSET = 4

        # Transformer model configuration
        self.d_model = 128
        self.n_heads = 4
        self.n_layers = 4
        self.d_k = 32
        self.d_v = 32
        self.d_ff = 128
        self.dropout = 0.1
        self.padding_idx = 0
        self.bos_idx = 2
        self.eos_idx = 3
        self.src_vocab_size = 2**self.SPLIT_BITS + 4
        self.tgt_vocab_size = 2**self.SPLIT_BITS + 4
        self.batch_size = 128
        self.max_len = self.PRED_FORWARD * 1 + 2 - 1
        self.beam_size = 2
        self.use_smoothing = False
        self.use_noamopt = True
        self.lr = 3e-4

        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Modules needed for the TransforMAP model
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        if device:
            pe = pe.to(device)
        
        position = torch.arange(0., max_len).unsqueeze(1)
        if device:
            position = position.to(device)
            
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        if device:
            div_term = div_term.to(device)
            
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# Main Transformer model for TransforMAP
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class BatchWrapper:
    def __init__(self, src, trg=None, pad=0, device=None):
        if device:
            src = src.to(device)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            if device:
                trg = trg.to(device)
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# Dataset class for TransforMAP
class MAPDataset(Dataset):
    def __init__(self, df):
        self.past = list(df["past"].values)
        self.future = list(df["future"].values)

    def __getitem__(self, idx):
        past = self.past[idx]
        future = self.future[idx]
        return [past, future]

    def __len__(self):
        return len(self.past)

    def collate_fn(self, batch, pad_id=0, device=None):
        past_b = [x[0] for x in batch]
        future_b = [x[1] for x in batch]
        
        batch_input = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(np.array(l_)) for l_ in past_b],
                                                     batch_first=True, padding_value=pad_id)
        batch_target = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(np.array(l_)) for l_ in future_b],
                                                      batch_first=True, padding_value=pad_id)
        return BatchWrapper(batch_input, batch_target, pad_id, device)


# Helper function to create the Transformer model
def make_transformer_model(config):
    import copy
    c = copy.deepcopy
    
    device = config.device
    attn = MultiHeadedAttention(config.n_heads, config.d_model).to(device)
    ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout).to(device)
    position = PositionalEncoding(config.d_model, config.dropout, device=device).to(device)
    
    model = Transformer(
        Encoder(EncoderLayer(config.d_model, c(attn), c(ff), config.dropout).to(device), config.n_layers).to(device),
        Decoder(DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout).to(device), config.n_layers).to(device),
        nn.Sequential(Embeddings(config.d_model, config.src_vocab_size).to(device), c(position)),
        nn.Sequential(Embeddings(config.d_model, config.tgt_vocab_size).to(device), c(position)),
        Generator(config.d_model, config.tgt_vocab_size).to(device)
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# Beam search for TransforMAP prediction
class Beam:
    def __init__(self, size, pad, bos, eos, device=False):
        self.size = size
        self._done = False
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []
        self.prev_ks = []
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_logprob):
        num_words = word_logprob.size(1)
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            beam_lk = word_logprob[0]

        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)
        return dec_seq

    def get_hypothesis(self, k):
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))


def beam_search(model, src, src_mask, max_len, pad, bos, eos, beam_size, device):
    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)
        return beamed_tensor

    def collate_active_info(src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)
        return active_src_enc, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
            assert enc_output.shape[0] == dec_seq.shape[0] == src_mask.shape[0]
            out = model.decode(enc_output, src_mask, dec_seq, subsequent_mask(dec_seq.size(1)).type_as(src.data))
            word_logprob = model.generator(out[:, -1])
            word_logprob = word_logprob.view(n_active_inst, n_bm, -1)
            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]
            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        word_logprob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_logprob, inst_idx_to_position_map)
        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        src_enc = model.encode(src, src_mask)
        batch_size, sent_len, h_dim = src_enc.size()
        src_enc = src_enc.repeat(1, beam_size, 1).view(batch_size * beam_size, sent_len, h_dim)
        src_mask = src_mask.repeat(1, beam_size, 1).view(batch_size * beam_size, 1, src_mask.shape[-1])
        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        for len_dec_seq in range(1, max_len + 1):
            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)

            if not active_inst_idx_list:
                break
            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, beam_size)
    return batch_hyp, batch_scores


# Implementation of TransforMAP
class TransforMAP(MLPrefetchModel):
    def __init__(self):
        super().__init__()
        self.config = TransforMAPConfig()
        self.model = None
        self.cache_simulator = CacheSimulator(16, 2048, 64, name="TransforMAP-LLC")
        
    def load(self, path):
        print('Loading TransforMAP model from ' + path)
        self.model = make_transformer_model(self.config)
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.config.device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save(self, path):
        print('Saving TransforMAP model to ' + path)
        if self.model:
            torch.save(self.model.state_dict(), path)

    def train(self, data):
        print('Training TransforMAP')
        # Process data for training
        df_train = self._preprocess_data(data)
        
        # Create dataset and dataloader
        train_dataset = MAPDataset(df_train)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=lambda batch: train_dataset.collate_fn(batch, self.config.PAD_ID, self.config.device)
        )
        
        # Initialize model if not already loaded
        if not self.model:
            self.model = make_transformer_model(self.config)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        criterion = criterion.to(self.config.device)
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # Train the model
        self.model.train()
        for epoch in range(10):  # Reduced number of epochs for quick training
            total_loss = 0
            total_correct = 0
            total_tokens = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                out = self.model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
                
                # Calculate loss
                x = self.model.generator(out)
                loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                                 batch.trg_y.contiguous().view(-1)) / batch.ntokens
                
                # Calculate accuracy
                _, predicted = torch.max(x, dim=-1)
                valid_mask = (batch.trg_y != self.config.PAD_ID)
                correct = ((predicted == batch.trg_y) & valid_mask).sum().item()
                tokens = valid_mask.sum().item()
                
                total_loss += loss.item() * batch.ntokens
                total_correct += correct
                total_tokens += tokens
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Calculate average loss and accuracy
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0
            
            print(f"Epoch: {epoch+1}, Loss: {avg_loss:.6f}, Accuracy Score: {accuracy:.2f}%")
    
    # def train(self, data):
    #     print('Training TransforMAP')
    #     # Process data for training
    #     df_train = self._preprocess_data(data)
        
    #     # Create dataset and dataloader
    #     train_dataset = MAPDataset(df_train)
    #     train_dataloader = DataLoader(
    #         train_dataset, 
    #         batch_size=self.config.batch_size, 
    #         shuffle=True, 
    #         collate_fn=lambda batch: train_dataset.collate_fn(batch, self.config.PAD_ID, self.config.device)
    #     )
        
    #     # Initialize model if not already loaded
    #     if not self.model:
    #         self.model = make_transformer_model(self.config)
        
    #     # Define loss function
    #     criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    #     criterion = criterion.to(self.config.device)
        
    #     # Define optimizer
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
    #     # Train the model
    #     self.model.train()
    #     for epoch in range(10):  # Reduced number of epochs for quick training
    #         total_loss = 0
    #         for batch in train_dataloader:
    #             optimizer.zero_grad()
    #             out = self.model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
    #             loss = self._compute_loss(out, batch.trg_y, batch.ntokens, criterion)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
            
    #         print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
    
    def _compute_loss(self, x, y, ntokens, criterion):
        """Compute the loss for training"""
        x = self.model.generator(x)
        loss = criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / ntokens
        return loss
    
    def batch_greedy_decode(self, src, src_mask, max_len=None, start_symbol=None, end_symbol=None):
        """Greedy decode function for prediction"""
        if max_len is None:
            max_len = self.config.max_len
        if start_symbol is None:
            start_symbol = self.config.bos_idx
        if end_symbol is None:
            end_symbol = self.config.eos_idx
            
        batch_size, src_seq_len = src.size()
        results = [[] for _ in range(batch_size)]
        stop_flag = [False for _ in range(batch_size)]

        memory = self.model.encode(src, src_mask)
        tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

        for s in range(max_len):
            tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(src.data)
            out = self.model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))
            prob = self.model.generator(out[:, -1, :])
            pred = torch.argmax(prob, dim=-1)
            tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
            pred = pred.cpu().numpy()
            
            for i in range(batch_size):
                results[i].append(pred[i].item())

        return results
    
    def translate(self, src, use_beam=False):
        """Translate input addresses to predicted next addresses"""
        with torch.no_grad():
            self.model.eval()
            src_tensor = torch.LongTensor(np.array(src)).to(self.config.device)
            src_mask = (src_tensor != 0).unsqueeze(-2)
            
            if use_beam:
                decode_result, _ = beam_search(
                    self.model, src_tensor, src_mask, self.config.max_len,
                    self.config.padding_idx, self.config.bos_idx, self.config.eos_idx,
                    self.config.beam_size, self.config.device
                )
                decode_result = [h[0] for h in decode_result]
            else:
                decode_result = self.batch_greedy_decode(
                    src_tensor, src_mask, 
                    self.config.max_len, 
                    self.config.bos_idx, 
                    self.config.eos_idx
                )
                
            # Remove ending symbol - return only actual predictions
            return np.array(decode_result)[:, :-1]
    
    def _preprocess_data(self, data):
        """Preprocess the raw data for TransforMAP training"""
        # Create dataframe from raw data
        df = pd.DataFrame(data)
        df.columns = ["id", "cycle", "addr", "ip", "hit"]
        
        # Extract page and offset information
        df['raw'] = df['addr']
        df['page_address'] = [x >> self.config.PAGE_BITS for x in df['raw']]
        df['page_offset'] = [x - (x >> self.config.PAGE_BITS << self.config.PAGE_BITS) for x in df['raw']]
        df['cache_line_index'] = [int(x >> self.config.BLOCK_BITS) for x in df['page_offset']]
        df['page_cache_index'] = [x >> self.config.BLOCK_BITS for x in df['raw']]
        
        # Convert to binary representation
        df["page_cache_index_bin"] = df.apply(
            lambda x: self._convert_to_binary(x['page_cache_index'], self.config.BLOCK_NUM_BITS), 
            axis=1
        )
        
        # Generate past sequences (input)
        for i in range(self.config.LOOK_BACK):
            df[f'page_cache_index_bin_past_{i+1}'] = df['page_cache_index_bin'].shift(periods=(i+1))
            
        for i in range(self.config.LOOK_BACK):
            if i == 0:
                df["past"] = df[f'page_cache_index_bin_past_{i+1}']
            else:   
                df["past"] += df[f'page_cache_index_bin_past_{i+1}']
        
        # Generate future sequences (labels)
        df = df.sort_values(by=["page_address","cycle"])
        for i in range(self.config.PRED_FORWARD):
            df[f'cache_line_index_future_{i+1}'] = df['cache_line_index'].shift(periods=-(i+1))
        
        for i in range(self.config.PRED_FORWARD):
            if i == 0:
                df["future_idx"] = df[f'cache_line_index_future_{i+1}']
            else:   
                df["future_idx"] = df[['future_idx', f'cache_line_index_future_{i+1}']].values.tolist()
        
        # Restore original ordering
        df = df.sort_values(by=["id"])
        df = df.dropna()
        
        # Add offset and start/end tokens
        df["future"] = (np.stack(df["future_idx"]) + self.config.OFFSET).tolist()
        df["future"] = df.apply(
            lambda x: self._add_start_end(x['future'], self.config.START_ID, self.config.END_ID), 
            axis=1
        )
        df["past"] = df.apply(
            lambda x: self._add_start_end(x['past'], self.config.START_ID, self.config.END_ID), 
            axis=1
        )
        
        return df[["future", "past"]]
    
    def _convert_to_binary(self, data, bit_size):
        """Convert data to binary representation"""
        get_bin = lambda x, n: format(x, 'b').zfill(n)
        res = get_bin(data, bit_size)
        return [int(char) + self.config.OFFSET for char in res]
    
    def _add_start_end(self, column_list, start_id, end_id):
        """Add start and end tokens to sequence"""
        return [start_id] + column_list + [end_id]
    
    def _words_back_address(self, values):
        """Convert predicted tokens back to addresses"""
        multiplier = 2**self.config.SPLIT_BITS
        res = []
        period = self.config.BLOCK_NUM_BITS // self.config.SPLIT_BITS + 1 + 1  # 11
        for i in range(self.config.PRED_FORWARD):
            res_1 = 0
            for j in range(period - 1):
                res_1 += (multiplier**j) * (values[i*period + j + 1] - 4)
            res.append(res_1)
        return res
    
    def _add_and_convert(self, pred_2, page_address):
        """Convert predicted block index to raw hex address"""
        res = int(((int(page_address) << self.config.BLOCK_BITS) + int(pred_2)) << self.config.BLOCK_BITS)
        res2 = res.to_bytes(((res.bit_length() + 7) // 8), "big").hex().lstrip('0')
        return res2
    
    def generate(self, data):
        """Generate prefetch predictions for the trace data"""
        print('Generating prefetches with TransforMAP')
        
        if not self.model:
            raise ValueError("Model not loaded. Call load() first.")
        
        prefetches = []
        prefetch_buffer = set()
        max_buffer_size = 64  # Limit the buffer size
        
        # Process data in batches
        batch_size = 128
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch_data = data[start_idx:end_idx]
            
            # Preprocess the batch
            df_batch = self._preprocess_data(batch_data)
            
            # Skip empty batches
            if df_batch.empty or len(df_batch) < 2:
                continue
            
            # Generate predictions for this batch
            for i, row in df_batch.iterrows():
                instr_id, cycle_count, load_addr, load_ip, hit = batch_data[i - df_batch.index[0]]
                
                # Check if address is already in the cache
                cache_hit, _ = self.cache_simulator.load(load_addr, False)
                
                if not cache_hit:
                    # Get page and cache line information
                    page_addr = load_addr >> self.config.PAGE_BITS
                    
                    # Generate predictions using the model
                    past_sequence = [row['past']]
                    predictions = self.translate(past_sequence, use_beam=True)[0]
                    
                    # Convert predictions to addresses
                    pred_offsets = [(pred - self.config.OFFSET) for pred in predictions if pred > self.config.OFFSET]
                    pred_offsets = pred_offsets[:self.config.PRED_FORWARD]  # Limit to PRED_FORWARD prefetches
                    
                    # Generate prefetch addresses
                    for offset in pred_offsets:
                        if len(offset) > 0:  # Skip empty predictions
                            prefetch_addr = (page_addr << self.config.PAGE_BITS) + (offset << self.config.BLOCK_BITS)
                            
                            # Skip if already in buffer or in cache
                            if prefetch_addr in prefetch_buffer:
                                continue
                                
                            prefetch_buffer.add(prefetch_addr)
                            prefetches.append((instr_id, prefetch_addr))
                            
                            # Update prefetch buffer size
                            if len(prefetch_buffer) > max_buffer_size:
                                prefetch_buffer.pop()
                            
                            # Limit to 2 prefetches per instruction
                            if len([p for p in prefetches if p[0] == instr_id]) >= 2:
                                break
        
        return prefetches


class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass


class BestOffset(MLPrefetchModel):
    offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14,
               -14, 15, -15, 16, -16, 18, -18, 20, -20, 24, -24, 30, -30, 32, -32, 36, -36, 40, -40]
    scores = [0 for _ in range(len(offsets))]
    round = 0
    best_index = 0
    second_best_index = 0
    best_index_score = 0
    temp_best_index = 0
    score_scale = eval(os.environ.get('BO_SCORE_SCALE', '1'))
    bad_score = int(10 * score_scale)
    low_score = int(20 * score_scale)
    max_score = int(31 * score_scale)
    max_round = int(100 * score_scale)
    llc = CacheSimulator(16, 2048, 64)
    rrl = {}
    rrr = {}
    dq = []
    acc = []
    acc_alt = []
    active_offsets = set()
    p = 0
    memory_latency = 200
    rr_latency = 60
    fuzzy = eval(os.environ.get('FUZZY_BO', 'False'))

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for BestOffset')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for BestOffset')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training BestOffset')

    def rr_hash(self, address):
        return ((address >> 6) + address) % 64

    def rr_add(self, cycles, address):
        self.dq.append((cycles, address))

    def rr_add_immediate(self, address, side='l'):
        if side == 'l':
            self.rrl[self.rr_hash(address)] = address
        elif side == 'r':
            self.rrr[self.rr_hash(address)] = address
        else:
            assert False

    def rr_pop(self, current_cycles):
        while self.dq:
            cycles, address = self.dq[0]
            if cycles < current_cycles - self.rr_latency:
                self.rr_add_immediate(address, side='r')
                self.dq.pop(0)
            else:
                break

    def rr_hit(self, address):
        return self.rrr.get(self.rr_hash(address)) == address or self.rrl.get(self.rr_hash(address)) == address

    def reset_bo(self):
        self.temp_best_index = -1
        self.scores = [0 for _ in range(len(self.offsets))]
        self.p = 0
        self.round = 0
        # self.acc.clear()
        # self.acc_alt.clear()

    def train_bo(self, address):
        testoffset = self.offsets[self.p]
        testlineaddr = address - testoffset

        if address >> 6 == testlineaddr >> 6 and self.rr_hit(testlineaddr):
            self.scores[self.p] += 1
            if self.scores[self.p] >= self.scores[self.temp_best_index]:
                self.temp_best_index = self.p

        if self.p == len(self.scores) - 1:
            self.round += 1
            if self.scores[self.temp_best_index] == self.max_score or self.round == self.max_round:
                self.best_index = self.temp_best_index if self.temp_best_index != -1 else 1
                self.second_best_index = sorted([(s, i) for i, s in enumerate(self.scores)])[-2][1]
                self.best_index_score = self.scores[self.best_index]
                if self.best_index_score <= self.bad_score:
                    self.best_index = -1
                self.active_offsets.add(self.best_index)
                self.reset_bo()
                return
        self.p += 1
        self.p %= len(self.scores)

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for BestOffset')
        prefetches = []
        prefetch_requests = []
        percent = len(data) // 100
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # Prefetch the next two blocks
            hit, prefetched = self.llc.load(load_addr, False)
            while prefetch_requests and prefetch_requests[0][0] + self.memory_latency < cycle_count:
                fill_addr = prefetch_requests[0][1]
                h, p = self.llc.load(fill_addr, True)
                if not h:
                    if self.best_index == -1:
                        fill_line_addr = fill_addr >> 6
                        if self.best_index != -1:
                            offset = self.offsets[self.best_index]
                        else:
                            offset = 0
                        self.rr_add_immediate(fill_line_addr - offset)
                prefetch_requests.pop(0)
            self.rr_pop(cycle_count)
            if not hit or prefetched:
                line_addr = (load_addr >> 6)
                self.train_bo(line_addr)
                self.rr_add(cycle_count, line_addr)
                if self.best_index != -1 and self.best_index_score > self.low_score:
                    addr_1 = (line_addr + 1 * self.offsets[self.best_index]) << 6
                    addr_2 = (line_addr + 2 * self.offsets[self.best_index]) << 6
                    addr_2_alt = (line_addr + 1 * self.offsets[self.second_best_index]) << 6
                    acc = len({addr_2 >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc.append(acc)
                    acc_alt = len({addr_2_alt >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc_alt.append(acc_alt)
                    # if acc_alt > acc:
                    #     addr_2 = addr_2_alt
                    prefetches.append((instr_id, addr_1))
                    prefetches.append((instr_id, addr_2))
                    prefetch_requests.append((cycle_count, addr_1))
                    prefetch_requests.append((cycle_count, addr_2))
            else:
                pass
            if i % percent == 0:
                print(i // percent, self.active_offsets, self.best_index_score,
                      sum(self.acc) / 2 / (len(self.acc) + 1),
                      sum(self.acc_alt) / 2 / (len(self.acc_alt) + 1))
                self.acc.clear()
                self.acc_alt.clear()
                self.active_offsets.clear()
        return prefetches


class AccessPatternClassifier(MLPrefetchModel):

    def __init__(self, window_size=11, delta_threshold=0.4):
        """
            window_size: size of instruction sequence
            delta_threshold: threshold for delta pattern classification
        """
        self.window_size = window_size
        self.delta_threshold = delta_threshold
        self.classified_patterns = defaultdict(int)
        
    def classify(self, address_sequence):
        """
        classify the emory access pattern
        
        parameters:
            address_sequence: sequence of memory addresses of load and store
            
        returns:
            0: sequential
            1: non-sequential
        """
        # Too short to classify
        if len(address_sequence) < 3:
            return 1  # default to non-sequential
        
        block_addresses = [addr >> 6 for addr in address_sequence]

        # Check if the address sequence is already classified
        # Compare the beginning addr and first delta 
        seq_key = self._get_sequence_key(address_sequence)
        if seq_key in self.classified_patterns:
            return self.classified_patterns[seq_key]
        

        # Computing deltas
        deltas = [block_addresses[i+1] - block_addresses[i] for i in range(len(block_addresses)-1)]
                  
        # Check for sequential pattern
        if self._is_sequential_pattern(deltas):
            self.classified_patterns[seq_key] = 0
            return 0
            
        # Classify as non-sequential 
        self.classified_patterns[seq_key] = 1
        return 1
        
    def _get_sequence_key(self, sequence):
        """
        key for recoginzing the same sequence
        """
        # Using first address and first delta as key
        start_addr = sequence[0]
        if len(sequence) > 3:
            first_delta = sequence[1] - sequence[0]
            return (start_addr, first_delta)
        return start_addr
        
    def _is_sequential_pattern(self, deltas):
        """
        Check the delta patterns
        """
        if not deltas:
            return False
            
        # Sure to be delta patterns (Including constant)
        if all(delta == deltas[0] for delta in deltas):
            return True
            
        # Average and standard deviation for delta
        mean = sum(deltas) / len(deltas)
        abs_dev = [abs(delta - mean) for delta in deltas]
        mean_dev = sum(abs_dev) / len(deltas)

        return mean_dev <= self.delta_threshold
    
    # Implement MLPrefetchModel abstract methods
    def load(self, path):
        pass
        
    def save(self, path):
        pass
        
    def train(self, data):
        pass
        
    def generate(self, data):
        return []

class Hybrid(MLPrefetchModel):
    prefetcher_classes = (BestOffset, TransforMAP)

    def __init__(self) -> None:
        super().__init__()
        self.prefetchers = [prefetcher_class() for prefetcher_class in self.prefetcher_classes]
        # Use the existing AccessPatternClassifier
        self.pattern_classifier = AccessPatternClassifier()
        # Store instruction pointers and their access patterns
        self.ip_patterns = defaultdict(list)
        self.ip_classifications = {}

    def load(self, path):
        for i, prefetcher in enumerate(self.prefetchers):
            prefetcher.load(f"{path}.{i}")

    def save(self, path):
        for i, prefetcher in enumerate(self.prefetchers):
            prefetcher.save(f"{path}.{i}")

    def train(self, data):
        # First, collect access patterns by instruction pointer
        self._collect_access_patterns(data)
        
        # Classify all instruction pointers
        self._classify_instruction_pointers()
        
        # Train BO on all data (unchanged)
        self.prefetchers[0].train(data)

        # Train TransforMAP only on non-sequential data
        non_sequential_data = self._filter_non_sequential_data(data)
        if non_sequential_data:
            self.prefetchers[1].train(non_sequential_data)

    def _collect_access_patterns(self, data):
        """Collect memory access patterns for each instruction pointer."""
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            if not llc_hit:
            # Store addresses by IP for pattern recognition
                self.ip_patterns[load_ip].append(load_addr)
            # Limit the pattern length to avoid excessive memory usage
                if len(self.ip_patterns[load_ip]) > 100:
                    self.ip_patterns[load_ip].pop(0)

    def _classify_instruction_pointers(self):
        """Classify all instruction pointers based on their access patterns."""
        for ip, addresses in self.ip_patterns.items():
            if len(addresses) >= 3:  # Need at least 3 addresses to classify
                self.ip_classifications[ip] = self.pattern_classifier.classify(addresses)

    def _filter_non_sequential_data(self, data):
        """Filter data to get only those with non-sequential access patterns."""
        non_sequential_data = []
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # If IP is classified as non-sequential (1) or not classified, include it
            classification = self.ip_classifications.get(load_ip, 1)  # Default to non-sequential if not classified
            if classification == 1:
                non_sequential_data.append((instr_id, cycle_count, load_addr, load_ip, llc_hit))
        return non_sequential_data

    def generate(self, data):
        prefetch_sets = defaultdict(lambda: defaultdict(list))
        for p, prefetcher in enumerate(self.prefetchers):
            prefetches = prefetcher.generate(data)
            for iid, addr in prefetches:
                prefetch_sets[p][iid].append((iid, addr))
        total_prefetches = []

        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            instr_prefetches = []

            for d in range(2):
                for p in range(len(self.prefetchers)):
                    if prefetch_sets[p][instr_id]:
                        instr_prefetches.append(prefetch_sets[p][instr_id].pop(0))
            instr_prefetches = instr_prefetches[:2]
            total_prefetches.extend(instr_prefetches)
        return total_prefetches
    
ml_model_name = os.environ.get('ML_MODEL_NAME', 'Hybrid')
Model = eval(ml_model_name)