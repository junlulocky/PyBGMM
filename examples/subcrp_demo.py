import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score

sys.path.append("..")

from pybgmm.prior.niw import NIW
from pybgmm.igmm import SubCRPMM
from pybgmm.igmm import CRPMM
from utils.visual_2d_clustering import plot_clustering
import os

# from utils.gendata import gendata
from pybgmm.utils.gendata import gendata_1d
from utils.plot_univariate_posterior_dist import traceplot
import collections
from sklearn.metrics import mutual_info_score
from pybgmm.infopy.infopy import information_variation
from pybgmm.prior.betabern import BetaBern
from sklearn.preprocessing import scale



def main():
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)

    # Data parameters
    # D = 1           # dimensions


    # Model parameters
    alpha = 1.0
    num_saved = 3   ## save how many components
    K = 10          ## initial number of components
    n_iter = 600    ## number of iterations for the sampling, change it to larger value
    n_burnin = 500  ## number of ignored iterations for using CRP rather than SubCRP in the beginning
    n_thining = 5   ## thining by 'n_thining' draws


    ## choose a dataset
    _sim = ['hand0', 'hand1', 'hand2']
    _sim = _sim[1]

    ## use what method to update the mask vector for variable selection
    mask_update = ['gibbs', 'metropolis']
    mask_update = mask_update[1]


    if _sim == 'hand1':
        _sim = 'hand1'
        D1 = 2  # dimensions
        D2 = 10
        N = 100  # number of points to generate
        K0 = 4  # the true number of components


        mu_scale = 4.0
        covar_scale = 0.7
        _v = np.random.randint(0, K0, N)
        mu = np.random.randn(D1, K0) * mu_scale
        print "true mu: {}".format(mu)
        X = mu[:, _v] + np.random.randn(D1, N) * covar_scale
        X = X.T
        plot_clustering(X, X, _v, "true")
        plt.savefig

        ## simple conclusion
        ## in our case, we use mask_burunin = 500,
        ## the normalized mutual information (NMI) has an abrupt change in iteration of 500,
        ## NMI for iteraiton 500 is 0.46
        ## NMI for itearation 520 is 0.71

    if _sim == 'hand2':
        D1 = 20  # dimensions
        D2 = 30
        N = 100  # number of points to generate
        K0 = 4  # the true number of components

        N1 = 25
        N2 = 25
        N3 = 25
        N4 = 25
        cov1 = 1.5
        cov2 = 0.1
        cov3 = 0.5
        cov4 = 2
        X1 = 5 + np.random.randn(N1, D1) * cov1
        X2 = 2 + np.random.randn(N2, D1) * cov2
        X3 = -3 + np.random.randn(N3, D1) * cov3
        X4 = -6 + np.random.randn(N4, D1) * cov4
        X = np.concatenate((X1, X2, X3, X4), axis=0)
        _v = np.array([0]*N1 + [1]*N2 + [2]*N3 + [3]*N4)

        idx = np.random.permutation(xrange(X.shape[0]))
        X = X[idx,:]
        _v = _v[idx]



    ## data summary & scale
    print "std before: {}".format(X.std(axis=0))
    print "mean before: {}".format(X.mean(axis=0))
    X = scale(X, axis=0)
    print "std after: {}".format(X.std(axis=0))
    print "mean after: {}".format(X.mean(axis=0))

    print collections.Counter(_v)


    ## add noise data
    isAddNoiseData = True
    if isAddNoiseData:
        X_extra = np.random.randn(N, D2)
        D1 = D1+D2
        X = np.concatenate([X, X_extra], axis=1)
    print X.shape




    # plt.hist(X)
    # plt.show()

    nmi = normalized_mutual_info_score(_v, _v)
    mi = mutual_info_score(_v, _v)
    vi = information_variation(_v, _v)
    print "oracle nmi: {}".format(nmi)
    print "oracle mi: {}".format(mi)
    print "oracle vi: {}".format(vi)




    # Intialize prior
    m_0 = X.mean(axis=0)
    # k_0 = covar_scale**2/mu_scale**2
    k_0 = 1./20
    v_0 = D1 + 2
    # S_0 = covar_scale**2*v_0*np.eye(D)
    # S_0 = 1./ covar_scale  * np.eye(D)
    S_0 = 1. * np.eye(D1)
    prior = NIW(m_0, k_0, v_0, S_0)

    # seed = 1111
    # np.random.seed(seed)
    # random.seed(seed)


    ## method = 0, non reduction, which means to use CRP mixture model
    ## method = 1, reduction, which means to use SubCRP mixture model to select variable
    method = 1
    if method == 0:
        _save_path = os.path.dirname(__file__) + \
                         '/tmp_res/result_common_n{}_ki-{}_alpha-{}_sim-{}_noise-{}' \
                             .format(N, K, alpha, _sim, isAddNoiseData)
    elif method == 1:
        _save_path = os.path.dirname(__file__) + \
                     '/tmp_res/result_reduction_n{}_ki-{}_alpha-{}_sim-{}_noise-{}_D2-{}_maskupdate-{}' \
                         .format(N, K, alpha, _sim, isAddNoiseData, D2, mask_update)


    try:
        os.stat(_save_path)
    except:
        os.mkdir(_save_path)

    plot_clustering(X[:,:2], X[:,:2], _v, "true")
    plt.savefig(_save_path + '/traceplot_all.png')
    plt.savefig(_save_path + '/traceplot_all.pdf')

    logging.basicConfig(level=logging.INFO, filename=_save_path + '/log.log')
    logging.getLogger().addHandler(logging.StreamHandler())


    # Setup CRPMM of SubCSCRPMM
    if method == 0:
        igmm = CRPMM(X, prior, alpha, save_path=_save_path, assignments="rand", K=K)
    elif method == 1:
        ## D_prior = D * a/(a+b)
        bern_prior = BetaBern(a=0.4, b=1.6)
        igmm = SubCRPMM(X, prior, alpha, save_path=_save_path, assignments="rand", K=K,
                        bern_prior=bern_prior, p_bern=0.9)
    # igmm = IGMM(X, prior, alpha, assignments="one-by-one", K=K)


    # Perform Gibbs sampling
    record, distribution_dict, subcrp_record_dict = igmm.collapsed_gibbs_sampler(n_iter, _v, num_saved=num_saved, mask_update=mask_update)

    logging.info('acceptance rate: {}'.format(igmm.acc_rate))


    ## get posterior summary info
    posterior_summary(record, distribution_dict, _save_path, n_burnin, n_thining, K0, subcrp_record_dict)




def posterior_summary(record, distribution_dict, _save_path, n_burnin, n_thining, K0, subcrp_record_dict):

    loss = np.array(record['loss'])
    np.save(_save_path + '/loss', loss)
    loss = loss[n_burnin::n_thining]
    print loss.shape
    len = loss.shape[0]
    logging.info('mean loss after burn-in: {}'.format(np.mean(loss)))
    sem = loss.std() * 1. / np.sqrt(len)
    logging.info("loss sem: {}".format(sem))

    bic = np.array(record['bic'])
    np.save(_save_path + '/bic', bic)
    bic = bic[n_burnin::n_thining]
    logging.info('mean bic after burn-in: {}'.format(np.mean(bic)))
    sem = bic.std() * 1. / np.sqrt(len)
    logging.info("bic sem: {}".format(sem))

    mi = np.array(record['mi'])
    np.save(_save_path + '/mi', mi)
    mi = mi[n_burnin::n_thining]
    logging.info('mean mi after burn-in: {}'.format(np.mean(mi)))
    sem = mi.std() * 1. / np.sqrt(len)
    logging.info("mi sem: {}".format(sem))

    nmi = np.array(record['nmi'])
    np.save(_save_path + '/nmi', nmi)
    nmi = nmi[n_burnin::n_thining]
    logging.info('mean nmi after burn-in: {}'.format(np.mean(nmi)))
    sem = nmi.std() * 1. / np.sqrt(len)
    logging.info("nmi sem: {}".format(sem))

    vi = np.array(record['vi'])
    np.save(_save_path + '/vi', vi)
    vi = vi[n_burnin::n_thining]
    logging.info('mean vi after burn-in: {}'.format(np.mean(vi)))
    sem = vi.std() * 1. / np.sqrt(len)
    logging.info("vi sem: {}".format(sem))

    components = np.array(record['components'])
    np.save(_save_path + '/components', components)
    logging.info('components summary before burnin: {}'.format(collections.Counter(components)))
    components = components[n_burnin::n_thining]
    logging.info('mean components after burn-in: {}'.format(np.mean(components)))
    logging.info('components summary after burnin: {}'.format(collections.Counter(components)))
    sem = components.std() * 1. / np.sqrt(len)
    logging.info("components sem: {}".format(sem))

    included_variable = np.array(subcrp_record_dict['included_variable'])
    np.save(_save_path + '/included_variable', included_variable)
    included_variable = included_variable[n_burnin::n_thining]
    logging.info('mean included_variable after burn-in: {}'.format(np.mean(included_variable)))
    sem = included_variable.std() * 1. / np.sqrt(len)
    logging.info("included_variable sem: {}".format(sem))

    # plot posterior distribution for mu and sigma
    #plot_univarite_posterior_dist(_save_path, distribution_dict, burnin=n_burnin, thining=n_thining)

    # print distribution_dict['variance'].shape
    # print distribution_dict['mean'].shape
    # print distribution_dict['weights'].shape
    np.save(_save_path + '/mean', distribution_dict['mean'])
    np.save(_save_path + '/weights', distribution_dict['weights'])
    np.save(_save_path + '/variance', distribution_dict['variance'])

    ## compute extra weights

    weights = distribution_dict['weights']
    weights = weights[:, n_burnin::n_thining]
    weights = np.mean(weights, axis=1).flatten()
    weights = np.sort(weights)[::-1]
    logging.info('weights: {}'.format(weights))
    useful_weights = np.sum(weights[0:K0])
    logging.info('extra weights: {}'.format(1-useful_weights))



    traceplot(distribution_dict)
    plt.savefig(_save_path + '/traceplot_all.png')
    plt.savefig(_save_path + '/traceplot_all.pdf')

    traceplot(distribution_dict, burnin=n_burnin, thining=n_thining)
    plt.savefig(_save_path + '/traceplot.png')
    plt.savefig(_save_path + '/traceplot.pdf')


if __name__ == "__main__":
    main()
