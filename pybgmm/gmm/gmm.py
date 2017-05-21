
import logging

import numpy as np
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score

import matplotlib.pyplot as plt

from ..utils import utils
from ..infopy.infopy import information_variation
from ..infopy.infopy import normalized_mutual_information
from ..infopy.infopy import mutual_information


from ..utils.plot_utils import plot_ellipse, plot_mixture_model


logger = logging.getLogger(__name__)



#-----------------------------------------------------------------------------#
#                                  GMM BASE CLASS                             #
#-----------------------------------------------------------------------------#

class GMM(object):
    """
    Base class for FGMM and IGMM
    """

    def __init__(self):
        pass

    def label_switch(self, idx, nplist):
        """
        sort the nplist by idx
        :param idx: idx indicates the new order
        :param nplist: sort nplist by idx
        :return: sorted nplist
        """
        return np.array(nplist)[idx]

    def setup_record_dict(self):
        """
        set up a record dictionary
        :return: record dictionary
        """
        record_dict = {}

        record_dict["sample_time"] = []
        record_dict["log_marg"] = []
        record_dict["components"] = []
        record_dict["nmi"] = []
        record_dict["mi"] = []
        record_dict["nk"] = []
        record_dict["loss"] = []
        record_dict['bic'] = []
        record_dict["vi"] = []
        record_dict["alpha"] = []

        return record_dict

    def update_record_dict(self, record_dict, i_iter, true_assignments, start_time):
        """
        Update record dictionary
        :param record_dict: record dictionary
        :param i_iter: sample iteration
        :param true_assignments: true assignment for clustering
        :param start_time: last sampling time
        :return: record dictionary
        """

        ## save
        record_dict["sample_time"].append(time.time() - start_time)

        ## save the log-marginal metric
        record_dict["log_marg"].append(self.log_marg())

        ## save how many components
        record_dict["components"].append(self.components.K)

        ## save normalized mutual information score
        nmi = normalized_mutual_information(true_assignments, self.components.assignments)
        record_dict["nmi"].append(nmi)

        ## save mutual information score
        mi = mutual_information(true_assignments, self.components.assignments)
        record_dict["mi"].append(mi)

        ## save number of samples in each cluster
        record_dict["nk"].append(str(self.components.counts[:self.components.K]))

        ## save squared sum of square loss
        loss = utils.cluster_loss_inertia(self.components.X, self.components.assignments)
        record_dict["loss"].append(loss)

        ## save bayesian information criterion (BIC)
        bic = utils.cluster_loss_inertia(self.components.X, self.components.assignments)
        record_dict["bic"].append(bic)

        ## save variation of information, here I use base 2, you can set to base e
        vi = information_variation(true_assignments, self.components.assignments, base=2)
        record_dict["vi"].append(vi)

        ## save alpha especially for finite gaussian mixture model
        record_dict["alpha"].append(self.alpha)

        ## logging every 20 steps
        if i_iter % 20 == 0:
            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            info += "."
            logger.info(info)

        return record_dict




