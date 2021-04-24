# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset
from Memorymodules.memory import Memory
from Memorymodules.message_aggregator import get_message_aggregator
from Memorymodules.message_function import get_message_function
from Memorymodules.memory_updater import get_memory_updater
from Memorymodules.embedding_module import get_embedding_module

from model.time_encoding import TimeEncode

from collections import defaultdict

class DE_SimplE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_SimplE, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        self.create_time_embedds()

        self.time_nl = torch.sin

        ###################
        '''
        specify parameters for each module
        '''

        #Time_diff Encoder
        self.n_nodes = dataset.numEnt()
        self.time_encoder = TimeEncode(dimension=100)
        #Memorymodules
        self.memory_dimension = 100
        self.n_edge_features = 100
        #self.time_encoder.dimension = 100

        device = torch.device('cuda:0')
        aggregator_type = 'last'
        message_function = 'identity'
        memory_updater_type = 'gru'

        self.memory_update_at_start = True #can be customized
        raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                self.time_encoder.dimension
        message_dimension = 400 #can be customized

        '''
        initialize each module with the specified parameters
        '''
        self.memory = Memory(n_nodes=self.n_nodes,
                             memory_dimension=self.memory_dimension,  # 100
                             input_dimension=message_dimension,  # 100+100+32+100=332
                             message_dimension=message_dimension,  # 100+100+32+100=332 (because the message function is identity)
                             device=device)
        self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                         device=device)
        self.message_function = get_message_function(module_type=message_function,
                                                     raw_message_dimension=raw_message_dimension,
                                                     message_dimension=message_dimension)
        self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                 memory=self.memory,
                                                 message_dimension=message_dimension,
                                                 memory_dimension=self.memory_dimension,
                                                 device=device)
        ###################
        
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
    
    def create_time_embedds(self):
        #h:representation for head, t:representation for tail
        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # amplitude embeddings for the entities
        self.m_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        if h_or_t == "head":
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days   + self.d_phi_h(entities))
        else:
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days   + self.d_phi_t(entities))
            
        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1) #(513024,1)
        months = months.view(-1,1) #(513024,1)
        days = days.view(-1,1) #(513024,1)

        # get embedding for all samples in the batch
        h_embs1 = self.ent_embs_h(heads) #(513024,68)
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

    ######################
    #newly added
    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        # equation(3)
        # LastMessageAggregator.aggregate
        nodes = torch.Tensor(nodes).int().cuda()
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)  # equation(3)
        # GRUMemoryUpdater.get_updated_memory
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)
        return updated_memory, updated_last_update


    def get_raw_messages_head(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, absolute_time, rels):
        '''
        years, months, days = edge_times
        years = years[::501].unsqueeze(-1)
        months = months[::501].unsqueeze(-1)
        days = days[::501].unsqueeze(-1)
        '''

        absolute_time = absolute_time[::501].unsqueeze(-1)

        edge_features = self.rel_embs_f(rels)

        source_memory = self.memory.get_memory(source_nodes)
        destination_memory = self.memory.get_memory(destination_nodes)

        #source_time_embedding = self.get_time_embedd(source_nodes, years, months, days, "head")

        #year_last_update, months_last_update, days_last_update = self.memory.last_update[source_nodes]
        time_last_update = self.memory.last_update[source_nodes]

        time_delta = absolute_time.squeeze(-1) - time_last_update
        time_delta_encoding = self.time_encoder(time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    time_delta_encoding],
                                   dim=1)  # equation(1)
        messages = defaultdict(list)
        unique_sources = torch.unique(source_nodes)

        source_nodes = source_nodes.cpu().numpy()
        for i in range(len(source_nodes)):
            # append all related messages within a batch involving the specific node
            messages[source_nodes[i]].append((source_message[i], absolute_time[i]))

        return unique_sources, messages


    def get_raw_messages_tail(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, absolute_time, rels):
        '''
        years, months, days = edge_times
        years = years[::501].unsqueeze(-1)
        months = months[::501].unsqueeze(-1)
        days = days[::501].unsqueeze(-1)
        '''

        absolute_time = absolute_time[::501].unsqueeze(-1)

        edge_features = self.rel_embs_i(rels)

        source_memory = self.memory.get_memory(source_nodes)
        destination_memory = self.memory.get_memory(destination_nodes)

        #source_time_embedding = self.get_time_embedd(source_nodes, years, months, days, "tail")

        # year_last_update, months_last_update, days_last_update = self.memory.last_update[source_nodes]
        time_last_update = self.memory.last_update[source_nodes]
        time_delta = absolute_time.squeeze(-1) - time_last_update
        time_delta_encoding = self.time_encoder(time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    time_delta_encoding],
                                   dim=1)  # equation(1)
        messages = defaultdict(list)
        unique_sources = torch.unique(source_nodes)

        source_nodes = source_nodes.cpu().numpy()
        for i in range(len(source_nodes)):
            # append all related messages within a batch involving the specific node
            messages[source_nodes[i]].append((source_message[i], absolute_time[i]))

        return unique_sources, messages

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        # the message for each node should be ordered before being fed to the function
        # Here the self.memory.message get changed!!!!!!
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)
    #####################
    
    def forward(self, heads, rels, tails, absolute_time, years, months, days):
        ##############
        #newly added
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
        h_embs1_mem = memory[heads]
        t_embs1_mem = memory[tails]
        h_embs2_mem = memory[tails]
        t_embs2_mem = memory[heads]
        ##############

        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(heads, rels, tails, years, months, days)

        ##############
        #newly added
        h_embs1 = h_embs1 + h_embs1_mem
        t_embs1 = t_embs1 + t_embs1_mem
        h_embs2 = h_embs2 + h_embs2_mem
        t_embs2 = t_embs2 + t_embs2_mem
        ##############

        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)

        ###############
        #newly added
        edge_times = (years, months, days)

        #only update the memory of positive samples (研究一下TGN和Diachronic两者之间的区别)
        #positive sample:0, 501, 1002, 1503....
        positives = torch.cat([heads[::501], tails[::501]])

        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
            "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)


        unique_sources, source_id_to_messages = self.get_raw_messages_head(heads[::501],
                                                                      h_embs1[::501],
                                                                      tails[::501],
                                                                      t_embs1[::501],
                                                                      absolute_time, rels[::501])
        unique_destinations, destination_id_to_messages = self.get_raw_messages_tail(tails[::501],
                                                                                h_embs2[::501],
                                                                                heads[::501],
                                                                                t_embs2[::501],
                                                                                absolute_time, rels[::501])

        # 12.04.2021 (13.04.2012看一看是否需要要对数据预处理进行改进)
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
        ###############
        return scores

        
