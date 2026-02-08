from copy import deepcopy

import numpy as np

from torch.nn.functional import pad
#from torch.utils.data import DataLoader, Dataset
#from torchtext.vocab import build_vocab_from_iterator

from models.Tamlec.tamlec import Tamlec
from models.Tamlec.Tamlec_tree import TAMLEC_Tree as Tree


PROBA_SUM = "SUM_PROBA"
PROBA_COMBINE = "COMBINE_PROBA"
PROBA_MAX = "MAX_PROBA"

def exp_sum(a, b):
    """
    sum of 2 proba with their log proba while circumventing precision issues
    """
    min_logproba, max_logproba = np.sort([a, b])
    small_logproba_normalized = min_logproba - max_logproba
    log_proba_combined = max_logproba + np.log(np.exp(small_logproba_normalized) + 1)
    return log_proba_combined

def exp_combine(a,b,threshold = -7): # combining two proba using the appro
    """
    combining two proba, supposing that the events are independant. Using a threshold to avoid accuracy problems, boil down to sum when proba are too small
    """
    if max(a,b)<= threshold :
        return exp_sum(a,b)
    pa = np.exp(a)
    pb = np.exp(b)

    combined_proba = pa + pb - pa*pb
    
    log_proba_combined =  np.log(combined_proba)
    return log_proba_combined



def exp_max(a,b):
    return max(a,b)


def exp_norm(array):
    min_logproba = np.max(array)
    log_norm_array = array - min_logproba
    norm_array = np.exp(log_norm_array)
    norm_array = norm_array / np.sum(norm_array)
    log_proba = min_logproba + np.log(norm_array)
    return log_proba



class CustomXMLHolder():
    def __init__(self, text_batch, task_id, beam_parameter, tamlec ,proba_operator = PROBA_COMBINE ):
        """
        :param dataset: CustomXMLDataset
        :param beam:
        :param tgt_vocab_size:
        :param config:
        """

        self.task_id = task_id
        self.tamlec : Tamlec = tamlec
        self.beam_param = beam_parameter

        self.tree  : Tree = self.tamlec.tamlec_forest[task_id]

        self.current_index=0
        self.ind = np.arange(len(text_batch))
        self.text_ref = text_batch
        self.ind_to_test = deepcopy(self.ind)

        self.paths =  [ [self.tree.root] for _ in self.ind  ]
        self.kinder = None
        
        self.level = 0  # start at level one

        self.proba = {ind: -1e9 * np.ones_like(self.tree._mask_template) for ind in self.ind}  # will store logproba for each label
        

        self.prior = [0. for _ in self.paths]  # will store log prior for beam search
        #self.scores = []

        if proba_operator == PROBA_SUM:
            self.proba_operator = exp_sum
        elif proba_operator==PROBA_MAX :
            self.proba_operator = exp_max
        elif proba_operator == PROBA_COMBINE :
            self.proba_operator = exp_combine
        else :
            raise NameError(f"Invalid proba_operator parameter : {proba_operator}. should be one of: {PROBA_SUM}, {PROBA_MAX}, {PROBA_COMBINE}")


        self._prev_index = None
        self._new_ind = []
        self._new_paths = []
        self._new_prior = []
        self._selected_paths = []


    def _get_data(self):

        txt = [ self.text_ref[i] for i in self.ind_to_test]
        path = self.paths

        lvl_mask = [self.tree.get_level_mask(self.level+1) for _ in self.ind_to_test]
        

        return txt, path, lvl_mask, self.task_id

    def _feed_prediction_partial(self, predictions):
        for ii in range(len(predictions)):

            i = self.current_index
            

            cind = self.ind_to_test[i]
            cpath = self.paths[i]
            cprior = self.prior[i]

            pred = predictions[ii]
            predict_order = np.argsort(pred)[::-1]

            if self._prev_index is None:
                self._prev_index = cind
            elif cind != self._prev_index:
                for npath, nprior in sorted(self._selected_paths, key=lambda x: -x[1])[:self.beam_param]:
                    predicted_property = npath[-1]

                    # Combine the two proba while taking into account float precision
                    oprior = self.proba[self._prev_index][predicted_property]
                    log_proba_combined = self.proba_operator(a=oprior, b=nprior)
                    self.proba[self._prev_index][predicted_property] = log_proba_combined

                    

                    if predicted_property not in self.tree.children_dict : # it is a leaf, let's stop there
                        continue
                    else:
                        
                        self._new_ind.append(self._prev_index)
                        self._new_paths.append(npath)
                        self._new_prior.append(nprior)
                        
                self._prev_index = cind
                self._selected_paths = []

            for predicted_property_idx in predict_order[:self.beam_param]:
                predicted_property = predicted_property_idx
                score = pred[predicted_property_idx]
                npath = cpath + [predicted_property]
                if score >0 : nprior = cprior + np.log(min(score,0.99))  # 0.99 to ensure that children are always slightly less likely than their parents and avoid strange behavior
                else : nprior = cprior + -1e9
                self._selected_paths.append((npath, nprior))

            self.current_index+=1

        


    def _digest_predictions(self):
        
        #for the last index !!!
        for npath, nprior in sorted(self._selected_paths, key=lambda x: -x[1])[:self.beam_param]:
                    predicted_property = npath[-1]

                    # Combine the two proba while taking into account float precision
                    oprior = self.proba[self._prev_index][predicted_property]
                    log_proba_combined = self.proba_operator(a=oprior, b=nprior)
                    self.proba[self._prev_index][predicted_property] = log_proba_combined

                    if predicted_property not in self.tree.children_dict : # it is a leaf, let's stop there
                        continue
                    else:
                        
                        self._new_ind.append(self._prev_index)
                        self._new_paths.append(npath)
                        self._new_prior.append(nprior)

        
        #saving To-do for next level
        self.ind_to_test = self._new_ind
        self.paths = self._new_paths
        self.prior = self._new_prior

        #resetting temporaty storage

        self._prev_index = None
        self._new_ind = []
        self._new_paths = []
        self._new_prior = []
        self._selected_paths = []
        self.current_index = 0

        self.level += 1

    def _finalize_prediction(self):
        for index_document in self.ind:
            self.proba[index_document][self.tree.root] =  -1e9 # hide root
        for index_label in  range(len(self.tree._mask_template)):
            if index_label not in self.tree._label_to_level :
                self.proba[index_document][index_label] = -1e9
            else : 
                self.proba[index_document][index_label] = max(self.proba[index_document][index_label], -1e8)


        ## need to go back to original labels

        
    def _get_final_prediction(self):

        self._finalize_prediction()

        predictions = []
        scores = []
        indices = sorted(self.proba.keys())

        for ind in indices:
            pred = self.proba[ind]
            predict_order = np.argsort(pred)[::-1]
            predicted_properties = predict_order#[:self.tamlec._max_padding_tgt]
            predicted_scores = pred[predict_order]#[:self.tamlec._max_padding_tgt]

            #predicted_labels = [ self.tamlec._tgt_vocab.get_itos()[i]  for i in predicted_properties]
            predicted_labels = predicted_properties

            predictions.append( predicted_labels.tolist())
            scores.append(np.exp(predicted_scores))

        return predictions, scores



    def run_predictions(self, batch_size = 15):

        while self.level +1 < self.tree._max_level_tree :
            txt, path, lvl_mask, id_task = self._get_data()

            if len(txt)==0 : # all selected paths reached dead ends, no need to continue
                break

            #predictions = []

            for i in range(0, len(txt),batch_size):
                ctxt = txt[i:i+batch_size]
                cpath = path[i:i+batch_size]
                clvl_mask = lvl_mask[i:i+batch_size]
                out = self.tamlec.test_batch(documents_tokens=ctxt, paths=cpath,lvl_mask=clvl_mask,id_task=id_task).detach().cpu().numpy()

                #predictions.append(out)
                self._feed_prediction_partial(out)
            
            #predictions =np.concatenate(predictions,axis=0) #(torch.cat(predictions, dim=0)).tolist()

            self._digest_predictions()

        result = self._get_final_prediction()
        return result
    





class CustomXMLHolderGlobal():
    def __init__(self, text_batch, task_id_list, beam_parameter, tamlec ,proba_operator = PROBA_COMBINE ):
        """
        :task_id_list must be a list of list, where for each document of text_batch there is a list of the relevant tasks.
        """

        self.task_id_list = task_id_list
        self.tamlec : Tamlec = tamlec
        self.beam_param = beam_parameter

        self.tamlec_forest  : list[Tree] = self.tamlec.tamlec_forest

        self.n_tasks = len(self.tamlec_forest)

        self.ind = np.arange(len(text_batch))
        self.text_ref = text_batch

        self.current_index=0
        self.ind_to_test = { i : [] for i in range(self.n_tasks)}
        self.paths  = { i : [] for i in range(self.n_tasks)}
        self.prior = { i : [] for i in range(self.n_tasks)}

        for i in self.ind :
            for j in self.task_id_list[i] :
                self.ind_to_test[j].append(i)
                self.paths[j].append( [self.tamlec_forest[j].root])
                self.prior[j].append(0.)

       
        self.level = 0  # start at level one

        self.proba = {ind: -1e9 * np.ones_like(self.tamlec_forest[0]._mask_template) for ind in self.ind}  # will store logproba for each label
        
        #self.scores = []

        if proba_operator == PROBA_SUM:
            self.proba_operator = exp_sum
        elif proba_operator==PROBA_MAX :
            self.proba_operator = exp_max
        elif proba_operator == PROBA_COMBINE :
            self.proba_operator = exp_combine
        else :
            raise NameError(f"Invalid proba_operator parameter : {proba_operator}. should be one of: {PROBA_SUM}, {PROBA_MAX}, {PROBA_COMBINE}")



        self._prev_index = None
        self._new_ind = []
        self._new_paths = []
        self._new_prior = []
        self._selected_paths = []


    def _get_data(self,task_index):

        txt = [ self.text_ref[i] for i in self.ind_to_test[task_index]]
        path = self.paths[task_index]
        lvl_mask = [self.tamlec_forest[task_index].get_children_mask(label=a[-1]) for a in path]



        return txt, path, lvl_mask

    def _feed_prediction_partial(self, predictions, id_task):
        for ii in range(len(predictions)):

            i = self.current_index
            

            cind = self.ind_to_test[id_task][i]
            cpath = self.paths[id_task][i]
            cprior = self.prior[id_task][i]

            pred = predictions[ii]
            predict_order = np.argsort(pred)[::-1]

            if self._prev_index is None:
                self._prev_index = cind
            elif cind != self._prev_index:
                for npath, nprior in sorted(self._selected_paths, key=lambda x: -x[1])[:self.beam_param]:
                    predicted_property = npath[-1]

                    # Combine the two proba while taking into account float precision
                    oprior = self.proba[self._prev_index][predicted_property]
                    log_proba_combined = self.proba_operator(a=oprior, b=nprior)
                    self.proba[self._prev_index][predicted_property] = log_proba_combined

                    

                    if predicted_property not in self.tamlec_forest[id_task].children_dict : # it is a leaf, let's stop there
                        continue
                    else:
                        
                        self._new_ind.append(self._prev_index)
                        self._new_paths.append(npath)
                        self._new_prior.append(nprior)
                        
                self._prev_index = cind
                self._selected_paths = []

            for predicted_property_idx in predict_order[:self.beam_param]:
                predicted_property = predicted_property_idx
                score = pred[predicted_property_idx]
                npath = cpath + [predicted_property]
                if score >0 : nprior = cprior + np.log(min(score,0.99))  # 0.99 to ensure that children are always slightly less likely than their parents and avoid strange behavior
                else : nprior = cprior + -1e9
                self._selected_paths.append((npath, nprior))
            
            self.current_index+=1

        


    def _partial_digest_predictions(self, id_task):
        
        #for the last index of the level !!!
        for npath, nprior in sorted(self._selected_paths, key=lambda x: -x[1])[:self.beam_param]:
                    predicted_property = npath[-1]

                    # Combine the two proba while taking into account float precision
                    oprior = self.proba[self._prev_index][predicted_property]
                    log_proba_combined = self.proba_operator(a=oprior, b=nprior)
                    self.proba[self._prev_index][predicted_property] = log_proba_combined

                    if predicted_property not in self.tamlec_forest[id_task].children_dict : # it is a leaf, let's stop there
                        continue
                    else:
                        
                        self._new_ind.append(self._prev_index)
                        self._new_paths.append(npath)
                        self._new_prior.append(nprior)

        
        #saving To-do for next level
        self.ind_to_test[id_task] = self._new_ind
        self.paths[id_task] = self._new_paths
        self.prior[id_task] = self._new_prior

        #resetting temporaty storage

        self._prev_index = None
        self._new_ind = []
        self._new_paths = []
        self._new_prior = []
        self._selected_paths = []
        self.current_index = 0



    def _digest_predictions(self):
        
        self.level += 1

    def _finalize_prediction(self):
        for index_document in self.ind:
            for task in range(self.n_tasks):
                #self.proba[index_document][self.tamlec_forest[task].root] = -.00000000000001# puts the root back into prediction to avoid removing labels from other tasks
                self.proba[index_document][self.tamlec_forest[task].root] = -1e9 # hide roots and task roots
        for index_label in  range(len(self.tamlec_forest[0]._mask_template)):
            self.proba[index_document][index_label] = max(self.proba[index_document][index_label], -1e8)


        ## need to go back to original labels

        
    def _get_final_prediction(self):

        self._finalize_prediction()

        predictions = []
        scores = []
        indices = sorted(self.proba.keys())

        for ind in indices:
            pred = self.proba[ind]
            predict_order = np.argsort(pred)[::-1]
            predicted_properties = predict_order#[:self.tor._max_padding_tgt]
            predicted_scores = pred[predict_order]#[:self.tor._max_padding_tgt]

            #predicted_labels = [ self.tor._tgt_vocab.get_itos()[i]  for i in predicted_properties]
            predicted_labels = predicted_properties

            predictions.append( predicted_labels.tolist())
            scores.append(np.exp(predicted_scores))

        return predictions, scores



    def run_predictions(self, batch_size = 15):

        

        max_depths = [ t._max_level_tree for t in self.tamlec_forest]
        max_depth = max(max_depths)

        while self.level +1 < max_depth :

            for id_task in range(self.n_tasks):

                if self.level+1 >= max_depths[id_task] : continue
            

                txt, path, lvl_mask = self._get_data(task_index=id_task)

                if len(txt)==0 : # all selected paths reached dead ends, no need to continue
                    continue

                #predictions = []

                for i in range(0, len(txt),batch_size):
                    ctxt = txt[i:i+batch_size]
                    cpath = path[i:i+batch_size]
                    clvl_mask = lvl_mask[i:i+batch_size]
                    out = self.tamlec.test_batch(documents_tokens=ctxt, paths=cpath,lvl_mask=clvl_mask,id_task=id_task).detach().cpu().numpy()

                    #predictions.append(out)
                    self._feed_prediction_partial(out,id_task=id_task)
                
                #predictions =np.concatenate(predictions,axis=0) #(torch.cat(predictions, dim=0)).tolist()
                self._partial_digest_predictions(id_task=id_task)

            self._digest_predictions()

        result = self._get_final_prediction()
        return result




class CustomXMLHolderCompletion():
    def __init__(self, text_batch,  known_labels_id_list,  tamlec, proba_operator):
        """
        :known_labels_id_list must be a list of dict, where for each document of text_batch there is a dict : task -> relevant labels.
        [  { 0: [label1, label2], 2: [label3] },  # for document 0
              { 1: [label4] },  ]                   # for document 1 etc.
        """

        self.known_labels_id_list = known_labels_id_list
        self.tamlec : Tamlec = tamlec
    
        self.tamlec_forest  : list[Tree] = self.tamlec.tamlec_forest

        self.n_tasks = len(self.tamlec_forest)

        self.ind = np.arange(len(text_batch))

        self.text_ref = text_batch

        self.tasks_to_do = 0
        self.index_to_do = 0
        
        self.predictions =  [ ] # task, prediction, ground truth 

        self.ind_to_test_list = [[] for _ in range(self.n_tasks)] # list task-> index
        self.labels_list = [[] for _ in range(self.n_tasks)] # list task -> list -> labels

        

        for document_index, _ in enumerate(text_batch):
            for task in known_labels_id_list[document_index] :
                     
                labels =  known_labels_id_list[document_index][task]
                self.ind_to_test_list[task].append(document_index)
                self.labels_list[task].append(labels)



              
   

    def _get_data(self,task_index):

        txts = []
        labels = []

        for i in range(len(self.ind_to_test_list[task_index])):
            ind = self.ind_to_test_list[task_index][i]
            txts.append( self.text_ref[ind])
            labels.append( self.labels_list[task_index][i])

        return txts, labels



        


    def run_predictions(self, batch_size = 15):

        

        


        for id_task in range(self.n_tasks):

            
            txt, labels = self._get_data(task_index=id_task)

            if len(txt)==0 : # no documents relevant for this task
                continue

            predictions = []
            groundtruth = []

            for i in range(0, len(txt),batch_size):
                ctxt = txt[i:i+batch_size]
                clabel = labels[i:i+batch_size]
                out,kinder = self.tamlec.completion_batch(documents_tokens=ctxt, labels_tokens=clabel,task_id=id_task)
                out = out.detach().cpu().numpy()
                kinder = kinder.detach().cpu().numpy()

                predictions.extend(out.tolist())
                groundtruth.extend(kinder.tolist())

             
        return predictions, groundtruth



if __name__ == "__main__":



    print(exp_sum(0,0))