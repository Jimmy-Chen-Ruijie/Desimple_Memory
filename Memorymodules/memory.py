import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    nodes = nodes.cpu().numpy()
    for node in nodes:
      #first clear messages of nodes contained in the batch, then store new raw messages of those nodes
      self.messages[node].extend(node_id_to_messages[node]) #extend: concatenate 2 lists, append: add the whole list

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      #list of tuple for each node: (message embedding, timestamp)
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_memory(self):
    #detach the message and memory from the computation graph
    self.memory.detach_()

    # Detach all stored messages (from the computation graph), the key-value pair stays unchanged
    # loop over all node
    for k, v in self.messages.items():
      new_node_messages = []
      # loop over all messages involving the node
      for message in v:
        #message embedding for the given node and the timestamp for each message
        new_node_messages.append((message[0].detach(), message[1]))
      #get the detached messages of each node
      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    nodes = nodes.cpu().numpy()
    for node in nodes:
      self.messages[node] = []
