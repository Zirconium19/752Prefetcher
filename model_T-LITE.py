import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter, deque
import os
import pickle
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod

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

class TLiteModel(nn.Module):
    """
    Implementation of T-LITE neural model for data prefetching.
    
    T-LITE uses behavioral clustering and frequency-based candidate selection
    to create a compact and efficient prefetcher.
    """
    def __init__(self, n_candidates=4, history_len=3, dpf_history_len=1, 
                 pc_dim=64, cluster_dim=25, offset_expert_count=100,
                 n_clusters=1024):
        super(TLiteModel, self).__init__()
        self.n_candidates = n_candidates  # Reduced from Twilight's 20 to 4
        self.history_len = history_len    # Length of history to consider
        self.dpf_history_len = dpf_history_len  # Reduced from 3 to 1
        self.n_clusters = n_clusters      # Number of behavioral clusters
        
        # Embedding dimensions
        self.pc_dim = pc_dim
        self.cluster_dim = cluster_dim
        self.offset_expert_count = offset_expert_count
        
        # Embeddings for clusters, PCs, and offsets
        self.cluster_embedding = nn.Embedding(n_clusters, cluster_dim)
        self.pc_embedding = nn.Embedding(4096, pc_dim)
        self.offset_embedding = nn.Embedding(64, offset_expert_count * cluster_dim)
        
        # Context-aware offset embedding
        self.offset_context_attention = nn.Linear(cluster_dim + pc_dim, offset_expert_count)
        
        # Prediction layers
        input_dim = (history_len * cluster_dim) + (history_len * cluster_dim) + \
                    pc_dim + (history_len * n_candidates * dpf_history_len)
                    
        # Candidate and offset prediction layers
        self.candidate_fc = nn.Linear(input_dim, n_candidates + 1)  # +1 for "no prefetch"
        self.offset_fc = nn.Linear(input_dim, 64)
        
        
    def forward(self, cluster_history, offset_history, pc, dpf_vectors):
        """
        Forward pass for the T-LITE model.
        
        Args:
            cluster_history: Tensor of cluster IDs [batch_size, history_len]
            offset_history: Tensor of offset IDs [batch_size, history_len]
            pc: Tensor of program counter IDs [batch_size]
            dpf_vectors: Tensor of DPF distribution vectors 
                       [batch_size, history_len, dpf_history_len, n_candidates]
        
        Returns:
            candidate_logits: Logits for each candidate [batch_size, n_candidates+1]
            offset_logits: Logits for each offset [batch_size, 64]
        """
        batch_size = cluster_history.size(0)
        
        # Get embeddings for input features
        cluster_embeds = self.cluster_embedding(cluster_history)
        pc_embed = self.pc_embedding(pc)
        
        # Get context-aware offset embeddings
        context_aware_offset_embeds = self.get_context_aware_offset_embedding(
            offset_history, cluster_embeds, pc_embed)
        
        # Flatten DPF vectors
        dpf_flat = dpf_vectors.view(batch_size, -1)
        
        # Concatenate all features
        combined = torch.cat([
            cluster_embeds.view(batch_size, -1),
            context_aware_offset_embeds.view(batch_size, -1),
            pc_embed,
            dpf_flat
        ], dim=1)
        
        # Generate predictions
        candidate_logits = self.candidate_fc(combined)
        offset_logits = self.offset_fc(combined)
        
        return candidate_logits, offset_logits
    
    def get_context_aware_offset_embedding(self, offset_ids, cluster_embeds, pc_embed):
        """Generate context-aware offset embeddings using mixture of experts approach"""
        batch_size = offset_ids.size(0)
    
    # Get raw offset embeddings [batch_size, history_len, offset_expert_count * cluster_dim]
        offset_embeds = self.offset_embedding(offset_ids)
    
    # Reshape to separate experts
        offset_embeds = offset_embeds.view(batch_size, self.history_len, 
                                     self.offset_expert_count, self.cluster_dim)
    
    # Make sure pc_embed is properly shaped as [batch_size, pc_dim]
        if len(pc_embed.shape) == 1:
            pc_embed = pc_embed.unsqueeze(0)  # Add batch dimension if missing
    
    # Calculate attention weights for each expert
        context = torch.cat([
            cluster_embeds.view(batch_size, -1),
            pc_embed
            ], dim=1)
    
        # [batch_size, offset_expert_count]
        expert_weights = F.softmax(self.offset_context_attention(context), dim=1)
    
        # Apply attention weights to combine experts
        expert_weights = expert_weights.view(batch_size, 1, self.offset_expert_count, 1)
    
        # Sum across experts
        context_aware_offset_embeds = torch.sum(offset_embeds * expert_weights, dim=2)
    
        return context_aware_offset_embeds

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


class TLITE(MLPrefetchModel):
    """
    Implementation of the T-LITE prefetcher that combines behavioral clustering
    and frequency-based candidate selection with a neural model.
    
    T-LITE is a slimmed-down version of Twilight optimized for practical deployment.
    """
    def __init__(self):
        # Hyperparameters
        self.n_candidates = 4                # Reduced from 20 to 4
        self.history_len = 3                 # Length of history
        self.dpf_history_len = 1             # Reduced from 3 to 1
        self.pc_dim = 64                     # PC embedding dimension
        self.cluster_dim = 25                # Cluster embedding dimension
        self.offset_expert_count = 100       # Number of offset experts
        self.n_clusters = 1024               # Number of behavioral clusters
        self.batch_size = 256                # Training batch size
        self.epochs = 30                     # Number of training epochs
        self.learning_rate = 0.002           # Learning rate
        self.lr_decay = 0.5                  # Learning rate decay
        self.threshold = 0.5                 # Confidence threshold for prefetching
        self.dpf_cache_size = 65536          # 64KB of DPF metadata
        
        # Environment variable overrides
        self.n_candidates = int(os.environ.get('TLITE_N_CANDIDATES', self.n_candidates))
        self.history_len = int(os.environ.get('TLITE_HISTORY', self.history_len))
        self.dpf_history_len = int(os.environ.get('TLITE_DPF_HISTORY', self.dpf_history_len))
        self.n_clusters = int(os.environ.get('TLITE_N_CLUSTERS', self.n_clusters))
        
        # Create the neural model
        self.model = TLiteModel(
            n_candidates=self.n_candidates,
            history_len=self.history_len,
            dpf_history_len=self.dpf_history_len,
            pc_dim=self.pc_dim,
            cluster_dim=self.cluster_dim,
            offset_expert_count=self.offset_expert_count,
            n_clusters=self.n_clusters
        )
        
        # Initialize DPF tracking structures (limited to 64KB)
        self.dpf_tables = [defaultdict(Counter) for _ in range(self.dpf_history_len)]
        self.dpf_lru = {}  # LRU tracking for DPF entries
        
        # History tracking
        self.page_history = deque(maxlen=self.history_len + self.dpf_history_len)
        self.offset_history = deque(maxlen=self.history_len)
        
        # Mappings
        self.pc_map = {}
        self.page_to_cluster = {}
        self.candidate_tables = defaultdict(list)
        
        # Clustering
        self.kmeans = None
        self.cluster_offset_matrices = [np.zeros((self.n_clusters, 64, 64)) for _ in range(2)]
        
        # Top-k PCs tracking
        self.pc_counter = Counter()
        self.max_pcs = 4096
        
        # Offset transition tracking for new pages
        self.page_offset_transitions = defaultdict(lambda: np.zeros((64, 64)))
        
    def load(self, path):
        """Load the model and metadata from a file"""
        print(f'Loading T-LITE model from {path}')
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.kmeans = checkpoint.get('kmeans')
        self.page_to_cluster = checkpoint.get('page_to_cluster', {})
        self.cluster_offset_matrices = checkpoint.get('cluster_offset_matrices', 
                                              [np.zeros((self.n_clusters, 64, 64)) for _ in range(2)])
        self.pc_map = checkpoint.get('pc_map', {})
        
        # Convert to 8-bit quantized model (if not already)
        if not checkpoint.get('quantized', False):
            self.quantize_model()
            
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.model.eval()

    def save(self, path):
        """Save the model and metadata to a file"""
        print(f'Saving T-LITE model to {path}')
        
        # Create the checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'kmeans': self.kmeans,
            'page_to_cluster': self.page_to_cluster,
            'cluster_offset_matrices': self.cluster_offset_matrices,
            'pc_map': self.pc_map,
            'quantized': True  # Mark as quantized
        }
        
        torch.save(checkpoint, path)
    
    def quantize_model(self):
        """Quantize model weights to 8-bit integers"""
        print("Quantizing model to 8-bit...")
        
        # Simple quantization for demonstration
        # In a real implementation, you'd use a proper quantization framework
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Compute scale factor
                max_val = torch.max(torch.abs(param.data))
                scale = 127.0 / max_val
                
                # Quantize
                param.data = torch.round(param.data * scale) / scale
    
    def update_dpf(self, access):
        """Update DPF tables with a new memory access"""
        _, _, load_addr, _, _ = access
        page = load_addr >> 12
        
        # Update DPF tables with LRU management
        for i, past_page in enumerate(list(self.page_history)[-self.dpf_history_len:]):
            if i < len(self.dpf_tables):
                # Check if we need to evict an entry
                if len(self.dpf_tables[i]) >= self.dpf_cache_size:
                    # Find least recently used entry
                    lru_page = min(self.dpf_lru.items(), key=lambda x: x[1])[0]
                    del self.dpf_tables[i][lru_page]
                    del self.dpf_lru[lru_page]
                
                # Update counter and LRU timestamp
                self.dpf_tables[i][past_page][page] += 1
                self.dpf_lru[past_page] = len(self.dpf_lru)
    
    def update_offset_transitions(self, prev_addr, curr_addr):
        """Update offset transition matrix for page clustering"""
        if prev_addr is None or curr_addr is None:
            return
            
        prev_page = prev_addr >> 12
        curr_page = curr_addr >> 12
        prev_offset = (prev_addr >> 6) & 0x3F
        curr_offset = (curr_addr >> 6) & 0x3F
        
        # Update transition matrix for the page
        self.page_offset_transitions[prev_page][prev_offset, curr_offset] += 1
        
        # If we have a transition matrix for this page, update cluster matrices
        if prev_page in self.page_to_cluster:
            cluster = self.page_to_cluster[prev_page]
            # Update transition matrix for current access
            self.cluster_offset_matrices[0][cluster][prev_offset, curr_offset] += 1
            # Update transition matrix for future behaviors
            if curr_page in self.page_to_cluster and curr_page != prev_page:
                curr_cluster = self.page_to_cluster[curr_page]
                self.cluster_offset_matrices[1][curr_cluster][prev_offset, curr_offset] += 1
    
    def assign_cluster(self, page):
        """Assign a cluster to a page based on its offset transition patterns"""
        if page in self.page_to_cluster:
            return self.page_to_cluster[page]
            
        # If page has no transitions yet, assign to random cluster
        if page not in self.page_offset_transitions or np.sum(self.page_offset_transitions[page]) < 10:
            cluster = hash(page) % self.n_clusters
            self.page_to_cluster[page] = cluster
            return cluster
            
        # Normalize transition matrix
        transitions = self.page_offset_transitions[page]
        normalized = transitions / (np.sum(transitions) + 1e-10)
        flattened = normalized.flatten()
        
        # If we have a kmeans model, use it
        if self.kmeans is not None:
            cluster = self.kmeans.predict([flattened])[0]
            self.page_to_cluster[page] = cluster
            return cluster
            
        # Otherwise, find closest cluster based on transition matrix similarity
        best_cluster = 0
        best_similarity = -1
        
        for i in range(self.n_clusters):
            cluster_transitions = self.cluster_offset_matrices[0][i]
            if np.sum(cluster_transitions) == 0:
                continue
                
            cluster_normalized = cluster_transitions / (np.sum(cluster_transitions) + 1e-10)
            cluster_flattened = cluster_normalized.flatten()
            
            # Compute cosine similarity
            similarity = np.dot(flattened, cluster_flattened) / (
                np.linalg.norm(flattened) * np.linalg.norm(cluster_flattened) + 1e-10)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = i
        
        self.page_to_cluster[page] = best_cluster
        return best_cluster
    
    def get_candidate_pages(self, history):
        """Get candidate pages for a given history based on DPF"""
        if len(history) < self.dpf_history_len:
            return []
            
        # Initialize candidate counters
        candidates = Counter()
        
        # Combine DPF distributions
        for i, page in enumerate(history[-self.dpf_history_len:]):
            if i < len(self.dpf_tables):
                dpf_counter = self.dpf_tables[i][page]
                if dpf_counter:
                    for candidate, freq in dpf_counter.items():
                        candidates[candidate] += freq
        
        # Get top-N candidates
        return [cand for cand, _ in candidates.most_common(self.n_candidates)]
    
    def get_dpf_vectors(self, history, candidates):
        """
        Generate DPF distribution vectors for the given history and candidates.
        Returns normalized distribution vectors.
        """
        dpf_vectors = torch.zeros(self.history_len, self.dpf_history_len, self.n_candidates)
        
        # Fill DPF vectors for each position in history
        for i, page in enumerate(history[-self.history_len:]):
            for j in range(self.dpf_history_len):
                if j < len(self.dpf_tables) and i + j < len(history):
                    hist_page = history[-(i+j+1)]
                    dpf_counter = self.dpf_tables[j][hist_page]
                    
                    # Total frequency for normalization
                    total = sum(dpf_counter.values())
                    
                    if total > 0:
                        # Fill in frequencies for each candidate
                        for k, candidate in enumerate(candidates):
                            if k < self.n_candidates:
                                freq = dpf_counter.get(candidate, 0)
                                dpf_vectors[i, j, k] = freq / total
        
        return dpf_vectors
    
    def prepare_batch(self, data):
        """Prepare batches of training data for the neural model"""
        cluster_histories = []
        offset_histories = []
        pcs = []
        dpf_vectors_list = []
        targets_candidate = []
        targets_offset = []
        
        for i in range(0, len(data) - self.history_len - 1):
            # Extract history
            history = data[i:i+self.history_len]
            target = data[i+self.history_len]
            
            # Skip if not enough history
            if len(history) < self.history_len:
                continue
                
            # Extract features
            page_history = [h[2] >> 12 for h in history]
            offset_history = [(h[2] >> 6) & 0x3F for h in history]
            pc = history[-1][3]
            
            # Track PC for embedding table limitation
            self.pc_counter[pc] += 1
            
            # Extract target
            target_page = target[2] >> 12
            target_offset = (target[2] >> 6) & 0x3F
            
            # Get candidate pages
            candidates = self.get_candidate_pages(page_history)
            if not candidates:
                continue
                
            # Find target's position in candidates
            try:
                target_candidate = candidates.index(target_page)
            except ValueError:
                # Target not in candidates, set to "no prefetch"
                target_candidate = self.n_candidates
            
            # Get DPF vectors
            dpf_vectors = self.get_dpf_vectors(page_history, candidates)
            
            # Convert pages to clusters
            cluster_history = [self.page_to_cluster.get(page, hash(page) % self.n_clusters) 
                              for page in page_history]
            
            # Map PC to index for embedding table
            if pc not in self.pc_map:
                self.pc_map[pc] = len(self.pc_map) % self.max_pcs
            pc_idx = self.pc_map[pc]
            
            # Add to batch
            cluster_histories.append(cluster_history)
            offset_histories.append(offset_history)
            pcs.append(pc_idx)
            dpf_vectors_list.append(dpf_vectors)
            targets_candidate.append(target_candidate)
            targets_offset.append(target_offset)
            
            # Return batch when full
            if len(cluster_histories) == self.batch_size:
                yield (
                    torch.tensor(cluster_histories,dtype=torch.long),
                    torch.tensor(offset_histories,dtype=torch.long),
                    torch.tensor(pcs,dtype=torch.long),
                    torch.stack(dpf_vectors_list),
                    torch.tensor(targets_candidate,dtype=torch.long),
                    torch.tensor(targets_offset,dtype=torch.long)
                )
                
                # Clear batch
                cluster_histories = []
                offset_histories = []
                pcs = []
                dpf_vectors_list = []
                targets_candidate = []
                targets_offset = []
        
        # Return final partial batch
        if cluster_histories:
            yield (
                torch.tensor(cluster_histories,dtype=torch.long),
                torch.tensor(offset_histories,dtype=torch.long),
                torch.tensor(pcs,dtype=torch.long),
                torch.stack(dpf_vectors_list),
                torch.tensor(targets_candidate,dtype=torch.long),
                torch.tensor(targets_offset)
            )
    
    def create_page_embeddings(self, data):
        """Create page embeddings for clustering"""
        print("Creating page embeddings for clustering...")
        
        # Reset transition matrices
        self.page_offset_transitions.clear()
        
        # First pass: collect offset transitions
        prev_addr = None
        for access in data:
            _, _, load_addr, _, _ = access
            
            # Update transition matrix
            self.update_offset_transitions(prev_addr, load_addr)
            
            # Update previous address
            prev_addr = load_addr
        
        # Extract features for clustering
        pages = list(self.page_offset_transitions.keys())
        features = []
        
        for page in pages:
            transitions = self.page_offset_transitions[page]
            if np.sum(transitions) > 0:
                normalized = transitions / (np.sum(transitions) + 1e-10)
                features.append(normalized.flatten())
            else:
                features.append(np.zeros(64*64))
        
        return pages, np.array(features)
    
    def perform_clustering(self, pages, features):
        """Perform k-means clustering on page features"""
        print(f"Clustering {len(pages)} pages into {self.n_clusters} clusters...")
        
        # Limit the number of clusters to the number of pages
        n_clusters = min(self.n_clusters, len(pages))
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        # Map pages to clusters
        self.page_to_cluster = {page: cluster for page, cluster in zip(pages, clusters)}
        self.kmeans = kmeans
        
        # Initialize cluster offset matrices
        self.cluster_offset_matrices = [np.zeros((self.n_clusters, 64, 64)) for _ in range(2)]
        
        # Populate cluster offset matrices
        for page, cluster in self.page_to_cluster.items():
            if page in self.page_offset_transitions:
                self.cluster_offset_matrices[0][cluster] += self.page_offset_transitions[page]
    
    def train(self, data):
        """Train the T-LITE model on the given data"""
        print('Training T-LITE prefetcher')
        
        # Reset tracking structures
        self.dpf_tables = [defaultdict(Counter) for _ in range(self.dpf_history_len)]
        self.dpf_lru = {}
        self.page_history.clear()
        self.offset_history.clear()
        self.pc_counter.clear()
        
        # First pass: collect DPF statistics
        print('Collecting DPF statistics and offset transitions...')
        prev_addr = None
        for access in data:
            _, _, load_addr, _, _ = access
            page = load_addr >> 12
            offset = (load_addr >> 6) & 0x3F
            
            # Update transition matrix
            self.update_offset_transitions(prev_addr, load_addr)
            
            # Update page history
            self.page_history.append(page)
            self.offset_history.append(offset)
            
            # Update DPF tables
            if len(self.page_history) > self.dpf_history_len:
                self.update_dpf(access)
                
            # Update previous address
            prev_addr = load_addr
        
        # Create page embeddings and perform clustering
        pages, features = self.create_page_embeddings(data)
        self.perform_clustering(pages, features)
        
        # Select top PCs for embedding table
        top_pcs = [pc for pc, _ in self.pc_counter.most_common(self.max_pcs)]
        self.pc_map = {pc: i for i, pc in enumerate(top_pcs)}
        
        # Prepare optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.lr_decay)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Training loop
        print('Training neural model...')
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0
            
            # Prepare batches
            for batch in self.prepare_batch(data):
                cluster_history, offset_history, pc, dpf_vectors, target_candidate, target_offset = batch
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    cluster_history = cluster_history.cuda()
                    offset_history = offset_history.cuda()
                    pc = pc.cuda()
                    dpf_vectors = dpf_vectors.cuda()
                    target_candidate = target_candidate.cuda()
                    target_offset = target_offset.cuda()
                
                # Forward pass
                candidate_logits, offset_logits = self.model(cluster_history, offset_history, pc, dpf_vectors)
                
                # Calculate loss
                candidate_loss = F.cross_entropy(candidate_logits, target_candidate)
                offset_loss = F.cross_entropy(offset_logits, target_offset)
                loss = candidate_loss + offset_loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # Step the scheduler
            scheduler.step()
            
            # Print epoch statistics
            avg_loss = total_loss / max(1, batch_count)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}')
        
        # Quantize model to 8-bit
        self.quantize_model()
    
    def generate(self, data):
        """Generate prefetch addresses for the given data"""
        print('Generating prefetches with T-LITE')
        prefetches = []
        
        # Reset history tracking
        self.page_history.clear()
        self.offset_history.clear()
        
        # Track issued prefetches to avoid duplicates
        issued_prefetches = set()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Process each access
        prev_addr = None
        for i, access in enumerate(data):
            instr_id, _, load_addr, load_pc, llc_hit = access
            page = load_addr >> 12
            offset = (load_addr >> 6) & 0x3F
            
            # Update offset transitions for new pages
            self.update_offset_transitions(prev_addr, load_addr)
            prev_addr = load_addr
            
            # Assign cluster to page if needed
            if page not in self.page_to_cluster:
                self.assign_cluster(page)
            
            # Update history
            self.page_history.append(page)
            self.offset_history.append(offset)
            
            # Update DPF tables for online learning
            if len(self.page_history) > self.dpf_history_len:
                self.update_dpf(access)
            
            # Skip if history is too short
            if len(self.page_history) < self.history_len + self.dpf_history_len:
                continue
                
            # Only prefetch on LLC misses
            if llc_hit:
                continue
                
            # Get candidate pages
            page_history = list(self.page_history)[-self.history_len:]
            candidates = self.get_candidate_pages(page_history)
            if not candidates:
                continue
            
            # Store candidates for this access
            self.candidate_tables[instr_id] = candidates
                
            # Map pages to clusters
            cluster_history = [self.page_to_cluster.get(p, hash(p) % self.n_clusters) for p in page_history]
                
            # Map PC to index
            if load_pc not in self.pc_map:
                # For unseen PCs, use a default mapping
                pc_idx = len(self.pc_map) % self.max_pcs
                self.pc_map[load_pc] = pc_idx
            else:
                pc_idx = self.pc_map[load_pc]
            
            # Prepare input for the model
            cluster_history_tensor = torch.tensor([cluster_history])
            offset_history_tensor = torch.tensor([[(h >> 6) & 0x3F for h in self.offset_history]])
            pc_tensor = torch.tensor([pc_idx])
            
            # Get DPF vectors
            dpf_vectors = self.get_dpf_vectors(page_history, candidates)
            dpf_vectors_tensor = torch.tensor([dpf_vectors],dtype=torch.float)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                cluster_history_tensor = cluster_history_tensor.cuda()
                offset_history_tensor = offset_history_tensor.cuda()
                pc_tensor = pc_tensor.cuda()
                dpf_vectors_tensor = dpf_vectors_tensor.cuda()
            
            # Get predictions
            with torch.no_grad():
                candidate_logits, offset_logits = self.model(
                    cluster_history_tensor, offset_history_tensor, pc_tensor, dpf_vectors_tensor)
                
                # Get candidate page prediction
                candidate_probs = F.softmax(candidate_logits, dim=1)[0]
                candidate_idx = torch.argmax(candidate_probs).item()
                candidate_conf = candidate_probs[candidate_idx].item()
                
                # Skip if model predicts "no prefetch" or confidence is too low
                if candidate_idx == self.n_candidates or candidate_conf < self.threshold:
                    continue
                
                # Get predicted page
                if candidate_idx < len(candidates):
                    predicted_page = candidates[candidate_idx]
                else:
                    continue
                
                # Get offset prediction
                offset_probs = F.softmax(offset_logits, dim=1)[0]
                predicted_offset = torch.argmax(offset_probs).item()
                
                # Compute prefetch address
                prefetch_addr = (predicted_page << 12) | (predicted_offset << 6)
                
                # Skip if this prefetch has already been issued for this instruction
                prefetch_key = (instr_id, prefetch_addr)
                if prefetch_key in issued_prefetches:
                    continue
                    
                # Add prefetch
                prefetches.append((instr_id, prefetch_addr))
                issued_prefetches.add(prefetch_key)
                
                # Limit to 2 prefetches per instruction
                instr_prefetch_count = sum(1 for p in prefetches if p[0] == instr_id)
                if instr_prefetch_count >= 2:
                    continue
        
        return prefetches
    

class Hybrid(MLPrefetchModel):
    prefetcher_classes = (BestOffset, TLITE)

    def __init__(self) -> None:
        super().__init__()
        self.prefetchers = [prefetcher_class() for prefetcher_class in self.prefetcher_classes]
        # Use the existing AccessPatternClassifier
        self.pattern_classifier = AccessPatternClassifier()
        # Store instruction pointers and their access patterns
        self.ip_patterns = defaultdict(list)
        self.ip_classifications = {}

    def load(self, path):
        for prefetcher in self.prefetchers:
            prefetcher.load(path)

    def save(self, path):
        for prefetcher in self.prefetchers:
            prefetcher.save(path)

    def train(self, data):
        # First, collect access patterns by instruction pointer
        self._collect_access_patterns(data)
        
        # Classify all instruction pointers
        self._classify_instruction_pointers()
        
        # Train BO on all data (unchanged)
        self.prefetchers[0].train(data)

        # Train TerribleMLModel only on non-sequential data
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

# Replace the Model with T-LITE
ml_model_name = os.environ.get('ML_MODEL_NAME', 'Hybrid')
Model = eval(ml_model_name)
