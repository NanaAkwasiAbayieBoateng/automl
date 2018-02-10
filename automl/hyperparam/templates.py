"""Hyperopt templates for different models"""

# forked from hyperopt/hyperopt-sklearn

from functools import partial

import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope

import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

# Optional dependencies
try:
    import xgboost
except ImportError:
    xgboost = None


def default_name_func(name):
    return name

##############################
##==== Global variables ====##
##############################
_svm_default_cache_size = 512


###############################################
##==== Various hyperparameter generators ====##
###############################################
def hp_bool(name):
    return hp.choice(name, [False, True])


def _svm_gamma(name, n_features=1):
    '''Generator of default gamma values for SVMs.
    This setting is based on the following rationales:
    1.  The gamma hyperparameter is basically an amplifier for the
        original dot product or l2 norm.
    2.  The original dot product or l2 norm shall be normalized by
        the number of features first.
    '''
    # -- making these non-conditional variables
    #    probably helps the GP algorithm generalize
    # assert n_features >= 1
    return hp.loguniform(name,
                         np.log(1. / n_features * 1e-3),
                         np.log(1. / n_features * 1e3))


def _svm_degree(name):
    return hp.quniform(name, 1.5, 6.5, 1)


def _svm_max_iter(name):
    return hp.qloguniform(name, np.log(1e7), np.log(1e9), 1)


def _svm_C(name):
    return hp.loguniform(name, np.log(1e-5), np.log(1e5))


def _svm_tol(name):
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _svm_int_scaling(name):
    return hp.loguniform(name, np.log(1e-1), np.log(1e1))


def _svm_epsilon(name):
    return hp.loguniform(name, np.log(1e-3), np.log(1e3))


def _svm_loss_penalty_dual(name):
    """
    The combination of penalty='l1' and loss='hinge' is not supported
    penalty='l2' and loss='hinge' is only supported when dual='true'
    penalty='l1' is only supported when dual='false'.
    """
    return hp.choice(
        name, [('hinge', 'l2', True), ('squared_hinge', 'l2', True),
               ('squared_hinge', 'l1', False), ('squared_hinge', 'l2', False)])


def _knn_metric_p(name, sparse_data=False, metric=None, p=None):
    if sparse_data:
        return ('euclidean', 2)
    elif metric == 'euclidean':
        return (metric, 2)
    elif metric == 'manhattan':
        return (metric, 1)
    elif metric == 'chebyshev':
        return (metric, 0)
    elif metric == 'minkowski':
        assert p is not None
        return (metric, p)
    elif metric is None:
        return hp.pchoice(name, [
            (0.55, ('euclidean', 2)),
            (0.15, ('manhattan', 1)),
            (0.15, ('chebyshev', 0)),
            (0.15, ('minkowski', _knn_p(name + '.p'))),
        ])
    else:
        return (metric, p)  # undefined, simply return user input.


def _knn_p(name):
    return hp.quniform(name, 2.5, 5.5, 1)


def _knn_neighbors(name):
    return scope.int(hp.qloguniform(name, np.log(0.5), np.log(50.5), 1))


def _knn_weights(name):
    return hp.choice(name, ['uniform', 'distance'])


def _trees_n_estimators(name):
    return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))


def _trees_criterion(name):
    return hp.choice(name, ['gini', 'entropy'])


def _trees_max_features(name):
    return hp.pchoice(
        name,
        [
            (0.2, 'sqrt'),  # most common choice.
            (0.1, 'log2'),  # less common choice.
            (0.1, None),  # all features, less common choice.
            (0.6, hp.uniform(name + '.frac', 0., 1.))
        ])


def _trees_max_depth(name):
    return hp.pchoice(
        name,
        [
            (0.7, None),  # most common choice.
            # Try some shallow trees.
            (0.1, 2),
            (0.1, 3),
            (0.1, 4),
        ])


def _trees_min_samples_split(name):
    return 2


def _trees_min_samples_leaf(name):
    return hp.choice(
        name,
        [
            1,  # most common choice.
            scope.int(
                hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
        ])


def _trees_bootstrap(name):
    return hp.choice(name, [True, False])


def _boosting_n_estimators(name):
    return scope.int(hp.qloguniform(name, np.log(10.5), np.log(1000.5), 1))


def _ada_boost_learning_rate(name):
    return hp.lognormal(name, np.log(0.01), np.log(10.0))


def _ada_boost_loss(name):
    return hp.choice(name, ['linear', 'square', 'exponential'])


def _ada_boost_algo(name):
    return hp.choice(name, ['SAMME', 'SAMME.R'])


def _grad_boosting_reg_loss_alpha(name):
    return hp.choice(name, [('ls', 0.9), ('lad', 0.9),
                            ('huber', hp.uniform(name + '.alpha', 0.85, 0.95)),
                            ('quantile', 0.5)])


def _grad_boosting_clf_loss(name):
    return hp.choice(name, ['deviance', 'exponential'])


def _grad_boosting_learning_rate(name):
    return hp.lognormal(name, np.log(0.01), np.log(10.0))


def _grad_boosting_subsample(name):
    return hp.pchoice(
        name,
        [
            (0.2, 1.0),  # default choice.
            (0.8, hp.uniform(name + '.sgb', 0.5, 1.0)
             )  # stochastic grad boosting.
        ])


def _sgd_penalty(name):
    return hp.pchoice(name, [(0.40, 'l2'), (0.35, 'l1'), (0.25, 'elasticnet')])


def _sgd_alpha(name):
    return hp.loguniform(name, np.log(1e-6), np.log(1e-1))


def _sgd_l1_ratio(name):
    return hp.uniform(name, 0, 1)


def _sgd_epsilon(name):
    return hp.loguniform(name, np.log(1e-7), np.log(1))


def _sgdc_learning_rate(name):
    return hp.pchoice(name, [(0.50, 'optimal'), (0.25, 'invscaling'),
                             (0.25, 'constant')])


def _sgdr_learning_rate(name):
    return hp.pchoice(name, [(0.50, 'invscaling'), (0.25, 'optimal'),
                             (0.25, 'constant')])


def _sgd_eta0(name):
    return hp.loguniform(name, np.log(1e-5), np.log(1e-1))


def _sgd_power_t(name):
    return hp.uniform(name, 0, 1)


def _random_state(name, random_state):
    if random_state is None:
        return hp.randint(name, 5)
    else:
        return random_state


def _class_weight(name):
    return hp.choice(name, [None, 'balanced'])


##############################################
##==== SVM hyperparameters search space ====##
##############################################
def _svm_hp_space(kernel,
                  n_features=1,
                  C=None,
                  gamma=None,
                  coef0=None,
                  degree=None,
                  shrinking=None,
                  tol=None,
                  max_iter=None,
                  verbose=False,
                  cache_size=_svm_default_cache_size):
    '''Generate SVM hyperparamters search space
    '''
    if kernel in ['linear', 'rbf', 'sigmoid']:
        degree_ = 1
    else:
        degree_ = (_svm_degree('degree')
                   if degree is None else degree)
    if kernel in ['linear']:
        gamma_ = 'auto'
    else:
        gamma_ = (_svm_gamma('gamma', n_features=1)
                  if gamma is None else gamma)
        gamma_ /= n_features  # make gamma independent of n_features.
    if kernel in ['linear', 'rbf']:
        coef0_ = 0.0
    elif coef0 is None:
        if kernel == 'poly':
            coef0_ = hp.pchoice(
                'coef0',
                [(0.3, 0),
                 (0.7, gamma_ * hp.uniform('coef0val', 0., 10.))])
        elif kernel == 'sigmoid':
            coef0_ = hp.pchoice(
                'coef0',
                [(0.3, 0),
                 (0.7, gamma_ * hp.uniform('coef0val', -10., 10.))])
        else:
            pass
    else:
        coef0_ = coef0

    hp_space = dict(
        kernel=kernel,
        C=_svm_C('C') if C is None else C,
        gamma=gamma_,
        coef0=coef0_,
        degree=degree_,
        shrinking=(hp_bool('shrinking')
                   if shrinking is None else shrinking),
        tol=_svm_tol('tol') if tol is None else tol,
        max_iter=(_svm_max_iter('maxiter')
                  if max_iter is None else max_iter),
        verbose=verbose,
        cache_size=cache_size)
    return hp_space


def _svc_hp_space(random_state=None, probability=False):
    '''Generate SVC specific hyperparamters
    '''
    hp_space = dict(
        random_state=_random_state('rstate', random_state),
        probability=probability)
    return hp_space


def _svr_hp_space(epsilon=None):
    '''Generate SVR specific hyperparamters
    '''
    hp_space = {}
    hp_space['epsilon'] = (_svm_epsilon('epsilon')
                           if epsilon is None else epsilon)
    return hp_space


#########################################
##==== SVM classifier constructors ====##
#########################################
def svc_kernel_hp_space(kernel,
                        random_state=None,
                        probability=False,
                        **kwargs):
    """
    Return a hyperparamter template that will construct
    a sklearn.svm.SVC model with a user specified kernel.
    Supported kernels: linear, rbf, poly and sigmoid
    """

    hp_space = _svm_hp_space(kernel=kernel, **kwargs)
    hp_space.update(_svc_hp_space(random_state, probability))
    return hp_space



########################################
##==== SVM regressor constructors ====##
########################################
def svr_kernel_hp_space(kernel, epsilon=None, **kwargs):
    """
    Return a hyperparamter template that will construct
    a sklearn.svm.SVR model with a user specified kernel.
    Supported kernels: linear, rbf, poly and sigmoid
    """

    hp_space = _svm_hp_space(kernel=kernel, **kwargs)
    hp_space.update(_svr_hp_space(epsilon))
    return hp_space


##############################################
##==== KNN hyperparameters search space ====##
##############################################
def knn_hp_space(sparse_data=False,
                 n_neighbors=None,
                 weights=None,
                 algorithm='auto',
                 leaf_size=30,
                 metric=None,
                 p=None,
                 metric_params=None,
                 n_jobs=1):
    '''Generate KNN hyperparameters search space
    '''
    metric_p = _knn_metric_p('metric_p', sparse_data, metric, p)
    hp_space = dict(
        n_neighbors=(_knn_neighbors('neighbors')
                     if n_neighbors is None else n_neighbors),
        weights=(_knn_weights('weights')
                 if weights is None else weights),
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric_p[0] if metric is None else metric,
        p=metric_p[1] if p is None else p,
        metric_params=metric_params,
        n_jobs=n_jobs)
    return hp_space


####################################################################
##==== Random forest/extra trees hyperparameters search space ====##
####################################################################
def trees_hp_space(n_estimators=None,
                   max_features=None,
                   max_depth=None,
                   min_samples_split=None,
                   min_samples_leaf=None,
                   bootstrap=None,
                   oob_score=False,
                   n_jobs=1,
                   random_state=None,
                   verbose=False):
    '''Generate trees ensemble hyperparameters search space
    '''
    hp_space = dict(
        n_estimators=(_trees_n_estimators('n_estimators')
                      if n_estimators is None else n_estimators),
        max_features=(_trees_max_features('max_features')
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth('max_depth')
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(
            'min_samples_split') if min_samples_split is None else
                           min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(
            'min_samples_leaf')
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap('bootstrap')
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state('rstate', random_state),
        verbose=verbose,
    )
    return hp_space


#############################################################
##==== Random forest classifier/regressor constructors ====##
#############################################################
def random_forest_hp_space(criterion='gini', **kwargs):
    """"Return a hyperparameter template for RandomForest model.

    Parameters
    ----------
    criterion: str
        'gini' or 'entropy' and 'mse' for classification
    """

    hp_space = trees_hp_space(**kwargs)
    hp_space['criterion'] = criterion
    return hp_space


###################################################
##==== AdaBoost hyperparameters search space ====##
###################################################
def ada_boost_hp_space(base_estimator=None,
                       n_estimators=None,
                       learning_rate=None,
                       random_state=None):
    '''Generate AdaBoost hyperparameters search space
    '''
    hp_space = dict(
        base_estimator=base_estimator,
        n_estimators=(_boosting_n_estimators('n_estimators')
                      if n_estimators is None else n_estimators),
        learning_rate=(_ada_boost_learning_rate('learning_rate')
                       if learning_rate is None else learning_rate),
        random_state=_random_state('rstate', random_state))
    return hp_space


###########################################################
##==== GradientBoosting hyperparameters search space ====##
###########################################################
def grad_boosting_hp_space(learning_rate=None,
                           n_estimators=None,
                           subsample=None,
                           min_samples_split=None,
                           min_samples_leaf=None,
                           max_depth=None,
                           init=None,
                           random_state=None,
                           max_features=None,
                           verbose=0,
                           max_leaf_nodes=None,
                           warm_start=False,
                           presort='auto'):
    '''Generate GradientBoosting hyperparameters search space
    '''
    hp_space = dict(
        learning_rate=(_grad_boosting_learning_rate('learning_rate')
                        if learning_rate is None else learning_rate),
        n_estimators=(_boosting_n_estimators('n_estimators')
                        if n_estimators is None else n_estimators),
        subsample=(_grad_boosting_subsample('subsample')
                        if subsample is None else subsample),
        min_samples_split=(_trees_min_samples_split('min_samples_split')
                        if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf('min_samples_leaf')
                        if min_samples_leaf is None else min_samples_leaf),
        max_depth=(_trees_max_depth('max_depth')
                        if max_depth is None else max_depth),
        init=init,
        random_state=_random_state('rstate', random_state),
        max_features=(_trees_max_features('max_features')
                      if max_features is None else max_features),
        warm_start=warm_start,
        presort=presort)
    return hp_space


###################################################
##==== XGBoost hyperparameters search space ====##
###################################################


def _xgboost_max_depth(name):
    return scope.int(hp.uniform(name, 1, 11))


def _xgboost_learning_rate(name):
    return hp.loguniform(name, np.log(0.0001), np.log(0.5)) - 0.0001


def _xgboost_n_estimators(name):
    return scope.int(hp.quniform(name, 100, 6000, 200))


def _xgboost_gamma(name):
    return hp.loguniform(name, np.log(0.0001), np.log(5)) - 0.0001


def _xgboost_min_child_weight(name):
    return scope.int(hp.loguniform(name, np.log(1), np.log(100)))


def _xgboost_subsample(name):
    return hp.uniform(name, 0.5, 1)


def _xgboost_colsample_bytree(name):
    return hp.uniform(name, 0.5, 1)


def _xgboost_colsample_bylevel(name):
    return hp.uniform(name, 0.5, 1)


def _xgboost_reg_alpha(name):
    return hp.loguniform(name, np.log(0.0001), np.log(1)) - 0.0001


def _xgboost_reg_lambda(name):
    return hp.loguniform(name, np.log(1), np.log(4))


def xgboost_hp_space(max_depth=None,
                     learning_rate=None,
                     n_estimators=None,
                     gamma=None,
                     min_child_weight=None,
                     max_delta_step=0,
                     subsample=None,
                     colsample_bytree=None,
                     colsample_bylevel=None,
                     reg_alpha=None,
                     reg_lambda=None,
                     scale_pos_weight=1,
                     base_score=0.5,
                     random_state=None):
    '''Generate XGBoost hyperparameters search space
    '''
    hp_space = dict(
        max_depth=(_xgboost_max_depth('max_depth')
                   if max_depth is None else max_depth),
        learning_rate=(_xgboost_learning_rate('learning_rate')
                       if learning_rate is None else learning_rate),
        n_estimators=(_xgboost_n_estimators('n_estimators')
                      if n_estimators is None else n_estimators),
        gamma=(_xgboost_gamma('gamma') if gamma is None else gamma),
        min_child_weight=(_xgboost_min_child_weight(
            'min_child_weight')
                          if min_child_weight is None else min_child_weight),
        max_delta_step=max_delta_step,
        subsample=(_xgboost_subsample('subsample')
                   if subsample is None else subsample),
        colsample_bytree=(_xgboost_colsample_bytree(
            'colsample_bytree')
                          if colsample_bytree is None else colsample_bytree),
        colsample_bylevel=(_xgboost_colsample_bylevel(
            'colsample_bylevel') if colsample_bylevel is None else
                           colsample_bylevel),
        reg_alpha=(_xgboost_reg_alpha('reg_alpha')
                   if reg_alpha is None else reg_alpha),
        reg_lambda=(_xgboost_reg_lambda('reg_lambda')
                    if reg_lambda is None else reg_lambda),
        scale_pos_weight=scale_pos_weight,
        base_score=base_score,
        seed=_random_state('rstate', random_state))
    return hp_space


#################################################
##==== Naive Bayes classifiers constructor ====##
#################################################
def multinomial_nb_hp_space(class_prior=None):
    hp_space = dict(
        alpha=hp.quniform('alpha', 0, 1, 0.001),
        fit_prior=hp_bool('fit_prior'),
        class_prior=class_prior)
    return hp_space


###########################################
##==== Passive-aggressive classifier ====##
###########################################
def passive_aggressive_hp_space(loss=None,
                                C=None,
                                fit_intercept=False,
                                n_iter=None,
                                n_jobs=1,
                                random_state=None,
                                verbose=False):
    hp_space = dict(
        loss=hp.choice('loss', ['hinge', 'squared_hinge'])
        if loss is None else loss,
        C=hp.lognormal('learning_rate', np.log(0.01), np.log(10))
        if C is None else C,
        fit_intercept=fit_intercept,
        n_iter=scope.int(
            hp.qloguniform('n_iter', np.log(1), np.log(1000), q=1))
        if n_iter is None else n_iter,
        n_jobs=n_jobs,
        random_state=_random_state('rstate', random_state),
        verbose=verbose)

    return hp_space


###############################################
##==== Discriminant analysis classifiers ====##
###############################################
def linear_discriminant_analysis_hp_space(solver=None,
                                          shrinkage=None,
                                          priors=None,
                                          n_components=None,
                                          store_covariance=False,
                                          tol=0.00001):

    solver_shrinkage = hp.choice('solver_shrinkage_dual',
                                 [('svd', None), ('lsqr', None),
                                  ('lsqr', 'auto'), ('eigen', None),
                                  ('eigen', 'auto')])

    rval = dict(
        solver=solver_shrinkage[0] if solver is None else solver,
        shrinkage=solver_shrinkage[1] if shrinkage is None else shrinkage,
        priors=priors,
        n_components=4 * scope.int(
            hp.qloguniform(
                'n_components', low=np.log(0.51), high=np.log(30.5), q=1.0))
        if n_components is None else n_components,
        store_covariance=store_covariance,
        tol=tol)
    return rval


def quadratic_discriminant_analysis_hp_space(reg_param=None, priors=None):

    rval = dict(
        reg_param=hp.uniform('reg_param', 0.0, 1.0)
        if reg_param is None else 0.0,
        priors=priors)
    return rval


###############################################
##==== Various preprocessor constructors ====##
###############################################
def pca_hp_space(n_components=None, whiten=None, copy=True):
    rval = dict(
        # -- qloguniform is missing a "scale" parameter so we
        #    lower the "high" parameter and multiply by 4 out front
        n_components=4 * scope.int(
            hp.qloguniform(
                'n_components', low=np.log(0.51), high=np.log(30.5), q=1.0))
        if n_components is None else n_components,
        # n_components=(hp.uniform(name + '.n_components', 0, 1)
        #               if n_components is None else n_components),
        whiten=hp_bool('whiten') if whiten is None else whiten,
        copy=copy,
    )
    return rval


def standard_scaler(with_mean=None, with_std=None):
    rval = dict(
        with_mean=hp_bool('with_mean') if with_mean is None else with_mean,
        with_std=hp_bool('with_std') if with_std is None else with_std,
    )
    return rval


def ts_lagselector_hp_space(lower_lags=1, upper_lags=1):
    rval = dict(lag_size=scope.int(
        hp.quniform('lags', lower_lags - .5, upper_lags + .5, 1)))
    return rval


def bernoulli_rbm_hp_space(n_components=None,
                           learning_rate=None,
                           batch_size=None,
                           n_iter=None,
                           verbose=False,
                           random_state=None):

    rval = dict(
        n_components=scope.int(
            hp.qloguniform(
                'n_components', low=np.log(0.51), high=np.log(999.5), q=1.0))
        if n_components is None else n_components,
        learning_rate=hp.lognormal(
            'learning_rate',
            np.log(0.01),
            np.log(10),
        ) if learning_rate is None else learning_rate,
        batch_size=scope.int(
            hp.qloguniform(
                '.batch_size',
                np.log(1),
                np.log(100),
                q=1,
            )) if batch_size is None else batch_size,
        n_iter=scope.int(
            hp.qloguniform(
                'n_iter',
                np.log(1),
                np.log(1000),  # -- max sweeps over the *whole* train set
                q=1,
            )) if n_iter is None else n_iter,
        verbose=verbose,
        random_state=_random_state('rstate', random_state),
    )
    return rval


def colkmeans_hp_space(n_clusters=None,
                       init=None,
                       n_init=None,
                       max_iter=None,
                       tol=None,
                       precompute_distances=True,
                       verbose=0,
                       random_state=None,
                       copy_x=True,
                       n_jobs=1):
    rval = dict(
        n_clusters=scope.int(
            hp.qloguniform(
                'n_clusters', low=np.log(1.51), high=np.log(19.5), q=1.0))
        if n_clusters is None else n_clusters,
        init=hp.choice(
            'init',
            ['k-means++', 'random'],
        ) if init is None else init,
        n_init=hp.choice(
            'n_init',
            [1, 2, 10, 20],
        ) if n_init is None else n_init,
        max_iter=scope.int(
            hp.qlognormal(
                'max_iter',
                np.log(300),
                np.log(10),
                q=1,
            )) if max_iter is None else max_iter,
        tol=hp.lognormal(
            'tol',
            np.log(0.0001),
            np.log(10),
        ) if tol is None else tol,
        precompute_distances=precompute_distances,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
        n_jobs=n_jobs,
    )
    return rval

def lgbm_hp_space(**kwargs):
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 700, 1)),
        'num_leaves': scope.int(hp.quniform ('num_leaves', 10, 200, 1)),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'max_bin': scope.int(hp.quniform('max_bin', 64, 512, 1)),
        'bagging_freq': scope.int(hp.quniform('bagging_freq', 1, 5, 1)),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10),
        **kwargs
       }

    return space
# -- flake8 eofk
