# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import math
import copy
import time
import numpy as np
from random import shuffle
from scripts import shredFacts

class Dataset:
    """Implements the specified dataloader"""
    def __init__(self, 
                 ds_name):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        # self.ds_path = "<path-to-dataset>" + ds_name.lower() + "/"
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {} #for dataset ICEWS14 len(self.ent2id)=7128 (there are 7128 entities in total)
        self.rel2id = {} #for dataset ICEWS14 len(self.rel2id)=230 (there are 230 relations in total)
        self.data = {"train": self.readFile(self.ds_path + "train.txt"),
                     "valid": self.readFile(self.ds_path + "valid.txt"),
                     "test":  self.readFile(self.ds_path + "test.txt")}
        
        self.start_batch = 0
        self.all_facts_as_tuples = None
        
        self.convertTimes()

        #90730 events in form of tuple
        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])
        
        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])
        
    def readFile(self, 
                 filename):

        with open(filename, "r", encoding='UTF8') as f:
            data = f.readlines()
        
        facts = []
        for line in data:
            elements = line.strip().split("\t") #strip helps remove \n
            
            head_id =  self.getEntID(elements[0])
            rel_id  =  self.getRelID(elements[1])
            tail_id =  self.getEntID(elements[2])
            timestamp_absolute = elements[3]
            timestamp_ymd = elements[4]
            
            facts.append([head_id, rel_id, tail_id, timestamp_absolute, timestamp_ymd])
            
        return facts
    
    
    def convertTimes(self):      
        """
        This function spits the timestamp in the day,date and time.
        """  
        for split in ["train", "valid", "test"]:
            #iterate over all data contained in the dataset "self.data[split]"
            for i, fact in enumerate(self.data[split]):
                fact_date_ymd = fact[-1]
                fact_date_absolute = fact[-2]
                self.data[split][i] = self.data[split][i][:-2] #先把date给删去
                date_absolute = float(fact_date_absolute)
                date_ymd = list(map(float, fact_date_ymd.split("-")))
                self.data[split][i].append(date_absolute) #等效于extend,把浮点型的date添加到data中
                self.data[split][i] += date_ymd
                
                
    
    def numEnt(self):
    
        return len(self.ent2id)

    def numRel(self):
    
        return len(self.rel2id)

    
    def getEntID(self,
                 ent_name):

        if ent_name in self.ent2id:
            return self.ent2id[ent_name] #map from entity name to entity index
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]
    
    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name] 
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    
    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            #if the index surpass the #samples in the dataset
            ret_facts = self.data["train"][self.start_batch : ]
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch : self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts
    

    def addNegFacts(self, bp_facts, neg_ratio):
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio
            
            facts[s1:e1,0] = (facts[s1:e1,0] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            facts[s2:e2,2] = (facts[s2:e2,2] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            
        return facts
    
    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0]) #randomly generated negative samples
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])
        
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0 #0,501,1002,1503...
            rand_nums2[i * pos_neg_group_size] = 0
        #every 501 samples, there will be a positive sample
        #facts1: only heads are perturbated
        #facts2: only tails are perturbated
        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.numEnt()
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.numEnt()
        return np.concatenate((facts1, facts2), axis=0)
    
    def nextBatch(self, batch_size, neg_ratio=1):
        bp_facts = self.nextPosBatch(batch_size) #(512,6)
        #the final batch size: positive batchsize * (1+negative ratio)*(relation+reverse relation)
        batch = shredFacts(self.addNegFacts2(bp_facts, neg_ratio)) #513024: 512*(500+1)*2
        return batch
    
    
    def wasLastBatch(self):
        return (self.start_batch == 0)
            
