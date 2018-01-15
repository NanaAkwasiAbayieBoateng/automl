"""Pipeline steps for feature selection"""
import logging
from sklearn.feature_selection import RFE
from automl.pipeline import PipelineData

import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


class FeatureSelector:
    """Class for feature selection step in pipeline. This selector
    provides selection strategy based on a single estimator. It means
    that FeatureSelector must be used after ChooseBest(1)

    Parametrs
    ---------
    max_features: int
        Desired numbers of features
    """
    def __init__(self, max_features):
        self._log = logging.getLogger(self.__class__.__name__)
        self.max_features = max_features

    def __call__(self, pipeline_data, context):
        """
        Parametrs
        ---------
        pipeline_data : PipelineData
            Data passed between PipelineStep in pipeline

        context : PiplineContext
            Global context of pipeline

        Returns
        -------
        PipelineData
            PipelineData.dataset contains changed dataset
        """
        if len(pipeline_data.return_val) > 1:
            raise ValueError(("Recurcive Feature Selector must be used with"
                              " ChooseBest(1)"))

        data_len = pipeline_data.dataset.data.shape[1]
        mask = np.array([True for _ in range(0, data_len)])
        model = pipeline_data.return_val[0].model
        # TODO this really should not be used with more than one model
        # Use ChooseBest(1)
        # For seceral models use VotingFeatureSelector
        # TODO: CV scorer in hyperopt does not fit models ???       


        if hasattr(model, "coef_"):
            f_score = [abs(coef) for coef in model.coef_]
        elif hasattr(model, "feature_importances_",):
            f_score = [abs(feature_importances) for feature_importances in model.feature_importances_]
        else:
            f_score = None

        if f_score is not None:
            if pipeline_data.dataset.data.shape[1] > self.max_features:
                threshold = sorted(f_score)[-self.max_features]
                mask = np.array([score >= threshold for score in f_score])
                self._log.info((f"Removing {sum(mask)} features for model"
                                f" {model.__class__.__name__}"))
        else:
            self._log.warn((f"Model {model.__class__.__name__} is not"
                            " supported by FeatureSelector"))

        pipeline_data.dataset.data = pipeline_data.dataset.data.compress(
            mask,
            axis=1
        )
        #pipeline_data.dataset.meta = [feature for feature, informative in zip(pipeline_data.dataset.meta, mask) if informative]
        meta = []
        for feature, informative in zip(pipeline_data.dataset.meta, mask):
            if informative:
                meta.append(feature)
        pipeline_data.dataset.meta = meta
        return PipelineData(pipeline_data.dataset, pipeline_data.return_val)


class RecursiveFeatureSelector:
    """Class for feature selection step in pipeline. This selector
    provides recurcive strategy based on a single estimator. It means
    that FeatureSelector must be used after ChooseBest(1)

    Parameters
    ----------
    n_features_to_select : int or None (default=None)
        The number of features to select. If None, half of the features are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then step corresponds to
        the (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then step corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    """
    def __init__(self, n_features_to_select=None, step=1, verbose=0):
        self._log = logging.getLogger(self.__class__.__name__)
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose

    def __call__(self, pipeline_data, context):
        """
        Parametrs
        ---------
        pipeline_data : PipelineData
            Data passed between PipelineStep in pipeline

        context : PiplineContext
            Global context of pipeline

        Returns
        -------
        PipelineData
            PipelineData.dataset contains changed dataset
        """
        if len(pipeline_data.return_val) > 1:
            raise ValueError(("Recurcive Feature Selector must be used with"
                              "ChooseBest(1)"))

        model = pipeline_data.return_val[0].model

        if hasattr(model, "coef_") or hasattr(model, "feature_importances_",):
            selector = RFE(
                model,
                self.n_features_to_select,
                self.step,
                self.verbose
            )

            assert(list(selector.get_params()['estimator'].get_params().values()) == list(model.get_params().values()))
            pipeline_data.dataset.data = selector.fit_transform(
                pipeline_data.dataset.data,
                pipeline_data.dataset.target
            )

            informative_features = zip(pipeline_data.dataset.meta,
                                       selector.get_support())
            meta = []
            for feature, informative in informative_features:
                if informative:
                    meta.append(feature)
            pipeline_data.dataset.meta = meta
        else:
            self._log.warn((f"Estimator {model.__class__.__name__} must have"
                            " coef_ or feature_importances_ attribute"))

        return PipelineData(pipeline_data.dataset, pipeline_data.return_val)


class VotingFeatureSelector:
    """Class for feature selection step in pipeline. This selector
    provides selection strategy based on several estimator. It means
    that FeatureSelector can be used with ChooseBest(n), n>1. All
    model that have feature_importances_ or coeff_ attribute have
    contribution to selection

    Parametrs
    ---------
    feature_to_select : int
        Desired numbers of features

    reverse_score : boolean
        If True, it means that bigger value of score is better,
        if False, it means that smaller value of score is worse
    """

    def __init__(self, feature_to_select, reverse_score=False):
        self._log = logging.getLogger(self.__class__.__name__)
        self.feature_to_select = feature_to_select
        self.reverse_score = reverse_score

    def __call__(self, pipeline_data, context):
        """
        Parametrs
        ---------
        pipeline_data : PipelineData
            Data passed between PipelineStep in pipeline

        context : PiplineContext
            Global context of pipeline

        Returns
        -------
        PipelineData
            PipelineData.dataset contains changed dataset
        """
        vote = []
        model_scores = []

        for value in pipeline_data.return_val:
            model = value.model

            if hasattr(model, "coef_"):
                f_score = np.array([abs(coef) for coef in model.coef_])
            elif hasattr(model, "feature_importances_",):
                f_score = np.array([abs(feature_importances)
                                    for feature_importances
                                    in model.feature_importances_])
            else:
                f_score = None

            if f_score is not None:
                vote.append(f_score)
                model_scores.append(value.score)

        model_scores = softmax(np.array(model_scores))

        if not vote:
            raise ValueError(("VotingFeatureSelector is needed at least one"
                              "model with feature_importances_ or coef_"
                              "attribute"))
        else:
            weights = np.zeros(vote[0].shape[0])

        if self.reverse_score:
            for f_score, model_score in zip(vote, model_scores):
                weights = weights + softmax(np.array(f_score)) * \
                        (1 - model_score)
        else:
            for f_score, model_score in zip(vote, model_scores):
                weights = weights + softmax(np.array(f_score)) * model_score

        threshold = sorted(weights)[-self.feature_to_select]
        mask = [score >= threshold for score in weights]

        if sum(mask):
            pipeline_data.dataset.data = pipeline_data.dataset.data.compress(
                mask,
                axis=1
            )
            #pipeline_data.dataset.meta = [feature for feature, informative in zip(pipeline_data.dataset.meta, mask) if informative]
            meta = []
            for feature, informative in zip(pipeline_data.dataset.meta, mask):
                if informative:
                    meta.append(feature)
            pipeline_data.dataset.meta = meta

        return PipelineData(pipeline_data.dataset, pipeline_data.return_val)


class CorrelatedFeatureSelector:
    def __init__(self, max_correlation):
        self._log = logging.getLogger(self.__class__.__name__)
        self.max_correlation = max_correlation

    def __call__(self, pipeline_data, context):
        data = pipeline_data.dataset.data
        meta = pipeline_data.dataset.meta
        orig_feature_num = data.shape[1]
        
        
        correlation_matrix = np.corrcoef(data.T)
        matrix_mask = np.absolute(correlation_matrix) > self.max_correlation
        pairs_of_indices = np.array(np.nonzero(matrix_mask)).T
        candidates = []
        for pair in pairs_of_indices:
            if pair[0]!=pair[1]:
                candidates.append(tuple(sorted(pair)))  #collect all pairs of feature indices which feature correlation bigger than max correlation parameter without pairs received by correlation with itself 
        candidates = list(set(candidates)) #delete pairs received by permutation of indices

        indices_to_delete = set([pair[0] for pair in candidates])
        mask = [i not in indices_to_delete for i in range(0, orig_feature_num)]#list of booleans. True if feature is not deleted by generator
        
        pipeline_data.dataset.data = data.compress(mask, axis=1)
        
        new_meta = []
        informative_features = zip(meta, mask) 
        for feature, informative in informative_features:
            if informative:
                new_meta.append(feature)
        new_meta = [feature for feature, informative in zip(meta, mask)
                    if informative]
        
        pipeline_data.dataset.meta = new_meta
        
        return PipelineData(pipeline_data.dataset, pipeline_data.return_val)