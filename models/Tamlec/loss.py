from operator import truediv
import torch
import torch.nn as nn
#from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np



class MLabelSmoothing(nn.Module):
    "Loss function with label smoothing."
    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        super(MLabelSmoothing, self).__init__()
        self.criterion = nn.BCELoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None
        

    def forward(self, x, target, level_mask, coef_width = None):
        """
        :param x: model predictions (batch_size*(max_padding_tgt - 1) x |label_vocab|)
        :param target: true labels (batch_size*(max_padding_tgt - 1) x |real_labels|)
        :param level_mask : (batch_size*|real_labels|)
        :return: loss value
        """
        assert x.size(1) == self.vocab_size

        true_dist = self.generate_true_dist(x, target, level_mask)
        self.true_dist = true_dist
        if coef_width is not None :
            max_x = 1-self.smoothing/coef_width
            x = torch.minimum(x, max_x*torch.ones_like(x))
            
        

        return self.criterion(x, true_dist.clone().detach())

    def generate_true_dist(self, x, target, level_mask):
        "Generate target distribution with label smoothing."
        true_dist = x.data.clone()

        proba_smooth_ = self.smoothing / (
                    self.vocab_size - 2)
        # along 1 dim:
        true_dist.fill_(proba_smooth_)  # distribute _smoothing_ value throughout the vocabulary (minus true label and padding label)
        true_dist.scatter_(-1, target, self.confidence)  # set _confidence_ prob spred to the correct labels
        true_dist[:, self.padding_idx] = proba_smooth_  # set zero prob to padding label
        if level_mask is not None:
            #lvl_mask = level_mask.repeat(true_dist.shape[1], 1).float()
            true_dist = true_dist * level_mask.float()

        return true_dist




class SimpleLossCompute:
    "Generate predictions and compute loss."
    def __init__(self, criterion, widths = None, weights = None):
        """
        :param criterion: loss function
        """
        self.criterion = criterion
        self.widths = widths
        self.weights = weights

    def __call__(self, x, y, norm, level_mask, task_id = None):
        """
        :param x: decoder output (batch_size x (max_padding_tgt - 1) x d_model)
        :param y: target (batch_size x (max_padding_tgt - 1))
        :param norm: total number of relevant labels in the batch (single integer)
        :param level_mask : (batch_size*|real_labels|)
        :return: total_loss (for the report), avg_loss (for backward step)
        """
        x = x.contiguous().view(-1, x.size(-1))
        level_mask = level_mask.contiguous().view(-1, level_mask.size(-1))
        y = y 

        coef_width = None
        if (self.widths is not None) :
            if task_id is not None : coef_width = self.widths[task_id]

        sloss = self.criterion(x, y,level_mask, coef_width) / norm
        if self.weights is not None :  coef = self.weights[task_id]
        else : coef = 1
            

        return sloss.data * norm, sloss*coef


def quick_precision_at_1(x, y, level_mask):
    """
    Compute precision@1
    :param x: model predictions (batch_size x |label_vocab|)
    :param y: true labels (batch_size x |real_labels|)
    """
    with torch.no_grad():
        x_mask= x*level_mask
        _, ranks_sorted = torch.sort(x, dim=1, descending=True)
        best_preds = ranks_sorted[:,0:1]
        matches = 0
        for i in range(len(y)):
            matches += (best_preds[i] in y[i])
        return  matches

        

class CustomPrecisionLoss:
    """
    Compute precision@k for k from 1 to nprec
    """
    def __init__(self, index_pady,   nprec):
        #self.sigmoid = torch.nn.Sigmoid()
        self.nprec = nprec
        self.index_pady = index_pady
        #self.mlb = MultiLabelBinarizer()

    def __call__(self, x, y, x_mask):
        
        y = y.detach()
        z = torch.zeros_like(x,dtype=torch.int64).detach()
        for i in range(len(y[0])):
            z = z + nn.functional.one_hot(y[:,i],x.shape[1])

        z[:,self.index_pady] = 0
        
        
        x = x.detach()


        scores_sorted, ranks_sorted = torch.sort(x, dim=1, descending=True)

        labels_sorted = torch.gather(z, 1, ranks_sorted)
        tot_mass = x*z
        tot_mass = float(torch.mean(torch.sum(tot_mass, 1),0))

        precision = {}
        for k in self.nprec:

            mask = (torch.sum(z,dim=1) >=k)
            if torch.sum(mask)<=1 : prec = -1.0 
            else : prec = float(torch.mean(torch.sum(labels_sorted[mask, :k], 1) / k))
            precision[k] = prec

        return precision, tot_mass
