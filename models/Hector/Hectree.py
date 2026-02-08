import argparse
from collections import defaultdict
from copy import copy
import json
import os
import queue


MIN_LABELS = 2





class HecTree : 
    def __init__(self, root_label, children_dict,all_tokens):
        """
        root label : the id of the root label
        children_dict: a dict of the form id : list of childrens' id
        """
        self.root = root_label
        self._labels_per_level = {}
        self._label_to_level = {}
        self._mask_template = []
        self._label_to_mask = {}
        self._mask_to_label = {}
        self.children_dict = children_dict
        self._all_tokens : list = all_tokens
        self._max_level_tree = 0
        self._max_level_forest = 0
        self._max_width_tree = 0
        self._meta_mask = []
        self._precompute()

        



    def _precompute(self):
        explored_nodes = set()
        to_explore = set()

        to_explore.add((self.root,0))
        explored_nodes.add(self.root)
        mlevel = 0
        while len(to_explore)>0 :
            node, level = to_explore.pop()
            
            
            dlevel : list= self._labels_per_level.get(level,[])
            dlevel.append(node)
            self._labels_per_level[level] = dlevel

            self._label_to_level[node] = level

            if node not in self.children_dict: continue

            for cnode in self.children_dict[node]:
                if cnode not in explored_nodes:
                    to_explore.add((cnode,level+1))
                    mlevel = max(mlevel,level+1)
                    explored_nodes.add(cnode)

        self._max_level_tree = mlevel+1
        self._max_level_forest = self._max_level_tree
        
        labels = sorted(list(explored_nodes))
        for i in range(len(labels)):
            index_label = self._all_tokens.index(labels[i])
            self._label_to_mask[labels[i]] = index_label
            self._mask_to_label[index_label] = labels[i]
        self._mask_template = [ 0 for _ in self._all_tokens]

        for lvl in range(self._max_level_tree) :
            if lvl in self._labels_per_level :
                self._max_width_tree = max(self._max_width_tree,len(self._labels_per_level[lvl]))


    def get_max_level(self):
        return self._max_level_tree
    
    
    def set_max_level(self, max_level):
        self._max_level_forest = max_level
        self._compute_meta_mask()

    def get_width(self):
        return self._max_width_tree
        


    # def labels_to_paths(self,list_of_labels : list) :
    #     """
    #     the list of labels must have been completed. Just the label ids are enough
    #     """

    #     assert list_of_labels[0] == self.root

    #     pre_paths = [[self.root]]
    #     label_per_level = {}

    #     for label in list_of_labels :
    #         level  = self._label_to_level[label]
    #         dlevel : list= label_per_level.get(level,[])
    #         dlevel.append(label)
    #         label_per_level[level] = dlevel

    #     for level in label_per_level:
    #         if level == 0 : continue
    #         post_paths = []
    #         for path in pre_paths :
    #             was_used= False
    #             for node in label_per_level[level] :
    #                 current_leaf = path[-1]
    #                 if node in self.children_dict[current_leaf]:
    #                     was_used=True
    #                     post_paths.append(path+[node])
    #             if not was_used :
    #                 post_paths.append(path)
    #         pre_paths = post_paths 
        
    #     return post_paths
    
    def labels_to_paths(self,list_of_labels : list) :
        """
        the list of labels must have been completed. Just the label ids are enough
        """

        assert self.root in list_of_labels

        pre_paths = [[self.root]]
        label_per_level = {}

        output = []

        for label in list_of_labels :
            label = int(label)
            level  = self._label_to_level[label]
            dlevel : list= label_per_level.get(level,[])
            dlevel.append(label)
            label_per_level[level] = dlevel

        max_level = max(label_per_level.keys())

        for level in range((max_level)+1):
            if level == 0 : continue
            if level not in label_per_level : continue
            post_paths = []
            
            mask = self.get_level_mask(level)
            for path in pre_paths :
                
                children = []
                for node in label_per_level[level] :
                    current_leaf = path[-1]
                    if current_leaf in self.children_dict :
                        if node in self.children_dict[current_leaf]:
                            post_paths.append(path+[node])
                            children.append(node)
                if len(children)>0 :
                
                    output.append( (path, children, mask))
                    
            pre_paths = post_paths 

        #assert len(output)>0
        
        return output


    def get_children_mask(self, label):
        mask = list(self._mask_template)
        for childen in self.children_dict[label]:
            mask[self._label_to_mask[childen]]= 1
        return mask
    
    def get_level_mask(self,level):
        mask = list(self._mask_template)
        for childen in self._labels_per_level[level]:
            mask[self._label_to_mask[childen]]= 1
        return mask
    
    def _compute_meta_mask(self):
        full_mask = []
        for l in range(self._max_level_tree):
            mask = list(self._mask_template)
            for childen in self._labels_per_level[l]:
                mask[self._label_to_mask[childen]]= 1
            full_mask.append(mask)
        for l in range(self._max_level_tree, self._max_level_forest):
            mask = list(self._mask_template)
            full_mask.append(mask)

        self._meta_mask = full_mask
    
    def get_meta_level_mask(self):
        return self._meta_mask
    


        

    
        

