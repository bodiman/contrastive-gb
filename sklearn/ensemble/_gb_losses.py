"""Losses and corresponding default initial estimators for gradient boosting
decision trees.
"""

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from scipy.special import expit, logsumexp

from ..tree._tree import TREE_LEAF
from ..utils.stats import _weighted_percentile
from ..dummy import DummyClassifier
from ..dummy import DummyRegressor

import math
import random


class LossFunction(metaclass=ABCMeta):
    """Abstract base class for various loss functions.

    Parameters
    ----------
    n_classes : int
        Number of classes.

    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    """

    is_multi_class = False

    def __init__(self, n_classes):
        self.K = n_classes

    @abstractmethod
    def init_estimator(self):
        """Default ``init`` estimator for loss function."""

    @abstractmethod
    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """

    @abstractmethod
    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """

    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate=0.1,
        k=0,
    ):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n_samples,)
            The weight of each sample.
        sample_mask : ndarray of shape (n_samples,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.

        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(
                tree,
                masked_terminal_regions,
                leaf,
                X,
                y,
                residual,
                raw_predictions[:, k],
                sample_weight,
            )

        # update predictions (both in-bag and out-of-bag)
        raw_predictions[:, k] += learning_rate * tree.value[:, 0, 0].take(
            terminal_regions, axis=0
        )

    @abstractmethod
    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        """Template method for updating terminal regions (i.e., leaves)."""

    @abstractmethod
    def get_init_raw_predictions(self, X, estimator):
        """Return the initial raw predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data array.
        estimator : object
            The estimator to use to compute the predictions.

        Returns
        -------
        raw_predictions : ndarray of shape (n_samples, K)
            The initial raw predictions. K is equal to 1 for binary
            classification and regression, and equal to the number of classes
            for multiclass classification. ``raw_predictions`` is casted
            into float64.
        """
        pass


class RegressionLossFunction(LossFunction, metaclass=ABCMeta):
    """Base class for regression loss functions."""

    def __init__(self):
        super().__init__(n_classes=1)

    def check_init_estimator(self, estimator):
        """Make sure estimator has the required fit and predict methods.

        Parameters
        ----------
        estimator : object
            The init estimator to check.
        """
        if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
            raise ValueError(
                "The init parameter must be a valid estimator and "
                "support both fit and predict."
            )

    def get_init_raw_predictions(self, X, estimator):
        assert estimator != "contrastive", "predict cannot be called with a contrastive loss function"

        predictions = estimator.predict(X)
        return predictions.reshape(-1, 1).astype(np.float64)


class LeastSquaresError(RegressionLossFunction):
    """Loss function for least squares (LS) estimation.
    Terminal regions do not need to be updated for least squares.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    def init_estimator(self):
        return DummyRegressor(strategy="mean")

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the least squares loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        if sample_weight is None:
            return np.mean((y - raw_predictions.ravel()) ** 2)
        else:
            return (
                1
                / sample_weight.sum()
                * np.sum(sample_weight * ((y - raw_predictions.ravel()) ** 2))
            )

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute half of the negative gradient.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples,)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        return y - raw_predictions.ravel()

    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate=0.1,
        k=0,
    ):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n,)
            The weight of each sample.
        sample_mask : ndarray of shape (n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """
        # update predictions
        raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        pass


class LeastAbsoluteError(RegressionLossFunction):
    """Loss function for least absolute deviation (LAD) regression.

    Parameters
    ----------
    n_classes : int
        Number of classes
    """

    def init_estimator(self):
        return DummyRegressor(strategy="quantile", quantile=0.5)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the least absolute error.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves).

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        if sample_weight is None:
            return np.abs(y - raw_predictions.ravel()).mean()
        else:
            return (
                1
                / sample_weight.sum()
                * np.sum(sample_weight * np.abs(y - raw_predictions.ravel()))
            )

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        1.0 if y - raw_predictions > 0.0 else -1.0

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        raw_predictions = raw_predictions.ravel()
        return 2 * (y - raw_predictions > 0) - 1

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        """LAD updates terminal regions to median estimates."""
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        diff = y.take(terminal_region, axis=0) - raw_predictions.take(
            terminal_region, axis=0
        )
        tree.value[leaf, 0, 0] = _weighted_percentile(
            diff, sample_weight, percentile=50
        )


class HuberLossFunction(RegressionLossFunction):
    """Huber loss function for robust regression.

    M-Regression proposed in Friedman 2001.

    Parameters
    ----------
    alpha : float, default=0.9
        Percentile at which to extract score.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.gamma = None

    def init_estimator(self):
        return DummyRegressor(strategy="quantile", quantile=0.5)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Huber loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        gamma = self.gamma
        if gamma is None:
            if sample_weight is None:
                gamma = np.percentile(np.abs(diff), self.alpha * 100)
            else:
                gamma = _weighted_percentile(
                    np.abs(diff), sample_weight, self.alpha * 100
                )

        gamma_mask = np.abs(diff) <= gamma
        if sample_weight is None:
            sq_loss = np.sum(0.5 * diff[gamma_mask] ** 2)
            lin_loss = np.sum(gamma * (np.abs(diff[~gamma_mask]) - gamma / 2))
            loss = (sq_loss + lin_loss) / y.shape[0]
        else:
            sq_loss = np.sum(0.5 * sample_weight[gamma_mask] * diff[gamma_mask] ** 2)
            lin_loss = np.sum(
                gamma
                * sample_weight[~gamma_mask]
                * (np.abs(diff[~gamma_mask]) - gamma / 2)
            )
            loss = (sq_loss + lin_loss) / sample_weight.sum()
        return loss

    def negative_gradient(self, y, raw_predictions, sample_weight=None, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        if sample_weight is None:
            gamma = np.percentile(np.abs(diff), self.alpha * 100)
        else:
            gamma = _weighted_percentile(np.abs(diff), sample_weight, self.alpha * 100)
        gamma_mask = np.abs(diff) <= gamma
        residual = np.zeros((y.shape[0],), dtype=np.float64)
        residual[gamma_mask] = diff[gamma_mask]
        residual[~gamma_mask] = gamma * np.sign(diff[~gamma_mask])
        self.gamma = gamma
        return residual

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        gamma = self.gamma
        diff = y.take(terminal_region, axis=0) - raw_predictions.take(
            terminal_region, axis=0
        )
        median = _weighted_percentile(diff, sample_weight, percentile=50)
        diff_minus_median = diff - median
        tree.value[leaf, 0] = median + np.mean(
            np.sign(diff_minus_median) * np.minimum(np.abs(diff_minus_median), gamma)
        )


class QuantileLossFunction(RegressionLossFunction):
    """Loss function for quantile regression.

    Quantile regression allows to estimate the percentiles
    of the conditional distribution of the target.

    Parameters
    ----------
    alpha : float, default=0.9
        The percentile.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        self.percentile = alpha * 100

    def init_estimator(self):
        return DummyRegressor(strategy="quantile", quantile=self.alpha)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Quantile loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        diff = y - raw_predictions
        alpha = self.alpha

        mask = y > raw_predictions
        if sample_weight is None:
            loss = (
                alpha * diff[mask].sum() - (1 - alpha) * diff[~mask].sum()
            ) / y.shape[0]
        else:
            loss = (
                alpha * np.sum(sample_weight[mask] * diff[mask])
                - (1 - alpha) * np.sum(sample_weight[~mask] * diff[~mask])
            ) / sample_weight.sum()
        return loss

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the negative gradient.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        alpha = self.alpha
        raw_predictions = raw_predictions.ravel()
        mask = y > raw_predictions
        return (alpha * mask) - ((1 - alpha) * ~mask)

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        terminal_region = np.where(terminal_regions == leaf)[0]
        diff = y.take(terminal_region, axis=0) - raw_predictions.take(
            terminal_region, axis=0
        )
        sample_weight = sample_weight.take(terminal_region, axis=0)

        val = _weighted_percentile(diff, sample_weight, self.percentile)
        tree.value[leaf, 0] = val


class ClassificationLossFunction(LossFunction, metaclass=ABCMeta):
    """Base class for classification loss functions."""

    @abstractmethod
    def _raw_prediction_to_proba(self, raw_predictions):
        """Template method to convert raw predictions into probabilities.

        Parameters
        ----------
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        Returns
        -------
        probas : ndarray of shape (n_samples, K)
            The predicted probabilities.
        """

    @abstractmethod
    def _raw_prediction_to_decision(self, raw_predictions):
        """Template method to convert raw predictions to decisions.

        Parameters
        ----------
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        Returns
        -------
        encoded_predictions : ndarray of shape (n_samples, K)
            The predicted encoded labels.
        """

    def check_init_estimator(self, estimator):
        """Make sure estimator has fit and predict_proba methods.

        Parameters
        ----------
        estimator : object
            The init estimator to check.
        """
        if not (hasattr(estimator, "fit") and hasattr(estimator, "predict_proba")):
            raise ValueError(
                "The init parameter must be a valid estimator "
                "and support both fit and predict_proba."
            )


class BinomialDeviance(ClassificationLossFunction):
    """Binomial deviance loss function for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError(
                "{0:s} requires 2 classes; got {1:d} class(es)".format(
                    self.__class__.__name__, n_classes
                )
            )
        # we only need to fit one tree for binary clf.
        super().__init__(n_classes=1)

    def init_estimator(self):
        # return the most common class, taking into account the samples
        # weights
        return DummyClassifier(strategy="prior")

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the deviance (= 2 * negative log-likelihood).

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        # logaddexp(0, v) == log(1.0 + exp(v))
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return -2 * np.mean(
                (y * raw_predictions) - np.logaddexp(0, raw_predictions)
            )
        else:
            return (
                -2
                / sample_weight.sum()
                * np.sum(
                    sample_weight
                    * ((y * raw_predictions) - np.logaddexp(0, raw_predictions))
                )
            )

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute half of the negative gradient.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        return y - expit(raw_predictions.ravel())

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        """Make a single Newton-Raphson step.

        our node estimate is given by:

            sum(w * (y - prob)) / sum(w * prob * (1 - prob))

        we take advantage that: y - prob = residual
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        denominator = np.sum(sample_weight * (y - residual) * (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(raw_predictions.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # log(x / (1 - x)) is the inverse of the sigmoid (expit) function
        raw_predictions = np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)


class MultinomialDeviance(ClassificationLossFunction):
    """Multinomial deviance loss function for multi-class classification.

    For multi-class classification we need to fit ``n_classes`` trees at
    each stage.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """

    is_multi_class = True

    def __init__(self, n_classes):
        if n_classes < 3:
            raise ValueError(
                "{0:s} requires more than 2 classes.".format(self.__class__.__name__)
            )
        super().__init__(n_classes)

    def init_estimator(self):
        return DummyClassifier(strategy="prior")

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the Multinomial deviance.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        # create one-hot label encoding
        Y = np.zeros((y.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            Y[:, k] = y == k

        return np.average(
            -1 * (Y * raw_predictions).sum(axis=1) + logsumexp(raw_predictions, axis=1),
            weights=sample_weight,
        )

    def negative_gradient(self, y, raw_predictions, k=0, **kwargs):
        """Compute negative gradient for the ``k``-th class.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.

        k : int, default=0
            The index of the class.
        """
        return y - np.nan_to_num(
            np.exp(raw_predictions[:, k] - logsumexp(raw_predictions, axis=1))
        )

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        """Make a single Newton-Raphson step."""
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        numerator = np.sum(sample_weight * residual)
        numerator *= (self.K - 1) / self.K

        denominator = np.sum(sample_weight * (y - residual) * (1 - y + residual))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        return np.nan_to_num(
            np.exp(
                raw_predictions - (logsumexp(raw_predictions, axis=1)[:, np.newaxis])
            )
        )

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions


class ExponentialLoss(ClassificationLossFunction):
    """Exponential loss function for binary classification.

    Same loss as AdaBoost.

    Parameters
    ----------
    n_classes : int
        Number of classes.

    References
    ----------
    Greg Ridgeway, Generalized Boosted Models: A guide to the gbm package, 2007
    """

    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError(
                "{0:s} requires 2 classes; got {1:d} class(es)".format(
                    self.__class__.__name__, n_classes
                )
            )
        # we only need to fit one tree for binary clf.
        super().__init__(n_classes=1)

    def init_estimator(self):
        return DummyClassifier(strategy="prior")

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the exponential loss

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        raw_predictions = raw_predictions.ravel()
        if sample_weight is None:
            return np.mean(np.exp(-(2.0 * y - 1.0) * raw_predictions))
        else:
            return (
                1.0
                / sample_weight.sum()
                * np.sum(sample_weight * np.exp(-(2 * y - 1) * raw_predictions))
            )

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the residual (= negative gradient).

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        y_ = 2.0 * y - 1.0
        return y_ * np.exp(-y_ * raw_predictions.ravel())

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        terminal_region = np.where(terminal_regions == leaf)[0]
        raw_predictions = raw_predictions.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        y_ = 2.0 * y - 1.0

        numerator = np.sum(y_ * sample_weight * np.exp(-y_ * raw_predictions))
        denominator = np.sum(sample_weight * np.exp(-y_ * raw_predictions))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(2.0 * raw_predictions.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):
        return (raw_predictions.ravel() >= 0).astype(int)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        proba_pos_class = probas[:, 1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        # according to The Elements of Statistical Learning sec. 10.5, the
        # minimizer of the exponential loss is .5 * log odds ratio. So this is
        # the equivalent to .5 * binomial_deviance.get_init_raw_predictions()
        raw_predictions = 0.5 * np.log(proba_pos_class / (1 - proba_pos_class))
        return raw_predictions.reshape(-1, 1).astype(np.float64)

















class ContrastiveLossFunction(RegressionLossFunction):
    def __init__(self, latent_dim=12, margin_proportion=5, batch_size=12):
        super().__init__()
        self.latent_dim = latent_dim
        self.margin = margin_proportion * latent_dim**0.5
        self.batch_size = batch_size

    def init_estimator(self):
        return "contrastive"

    def single_point_loss(self, vec1, vec2, shared):
        if shared:
                return np.linalg.norm(np.array(vec1 - vec2), ord=2)
        
        return max(0, self.margin - np.linalg.norm(np.array(vec1 - vec2), ord=2))
    
    def single_point_grad(self, vec1, vec2, shared):
        if shared:
            try:
                return (vec2 - vec1) / np.linalg.norm(np.array(vec1 - vec2), ord=2)
            except:
                print(vec2, vec1, vec1-vec2)
                raise "error found"
        
        if (vec1 == vec2).all():
            return np.random.randn(*vec1.shape)
        
        if self.margin > np.linalg.norm(np.array(vec1 - vec2), ord=2):
            return (vec1 - vec2) / np.linalg.norm(np.array(vec1 - vec2), ord=2)

        return np.zeros_like(vec1)

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the contrastive loss

        Parameters
        ----------
        y : ndarray of shape (n_samples, K) K is the number of leaf nodes in the random forrest classifier
            True labels.

        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : ndarray of shape (n_samples,), default=None
            Sample weights.
        """
        total_loss = 0
        classes = []

        for class_n in range(int(np.max(y) + 1)):
            idx = np.where(y==class_n)
            classes.append(raw_predictions[idx])

        for idx1, class1 in enumerate(classes):
            for idx2, class2 in enumerate(classes):
                if idx2 > idx1:
                    continue
                for idx3, vec1 in enumerate(class1):
                    for idx4, vec2 in enumerate(class2):
                        if idx2 == idx1:
                            continue
                        total_loss += self.single_point_loss(vec1, vec2, idx1 == idx2)
    
        return total_loss
    
    def negative_gradient(self, y, raw_predictions, **kargs):
        running_gradient = []

        #batches is a list of tuples

        p = np.random.permutation(len(y))

        #randomize order of y and raw_predictions

        batches = []
        classes = []
        batch_proportions = []

        for class_n in range(int(np.max(y)+1)):
            idx = np.where(y==class_n)
            classes.append(raw_predictions[idx])
            batch_proportions.append(math.ceil(len(raw_predictions[idx])/len(raw_predictions)*self.batch_size))

        print("Batch proportions:", batch_proportions)
        print("classes")
        print(classes)

        #while you can still take sufficient samples from each class
        batching = True
        while batching:
            #sample raw_predictions without replacement for each class from classes
            cvs = []
            rps = []
            for idx, class_n in enumerate(classes):
                if len(class_n) < batch_proportions[idx]:
                    batching = False
                    break

                rp = class_n[:batch_proportions[idx]]
                print("classidxshape", classes[idx].shape)
                classes[idx] = classes[idx][batch_proportions[idx]:]
                #sample y without replacement for each class from classidxs
                cv = [idx for _ in range(self.batch_size)]
                #pack the values in a tuple and append them to batches
                rps = []
                for i in rp:
                    # print(rps, i, type(rps), type(i))
                    rps += list(i)
                cvs += cv
            
            if batching:
                batches.append((np.array(cvs), np.array(rps)))

        print("Batch length:", len(batches))
        for batch in batches:
            print("batch0 shape")
            print(batch[0].shape, "*", len(batches))
            print(batch)
            running_gradient.append(self.negative_gradient_batch(*batch))

        for i in range(len(raw_predictions) - len(batches)):
            running_gradient.append(np.zeros(self.latent_dim))

        return np.array(running_gradient)

    def negative_gradient_batch(self, y, raw_predictions, **kargs):
        # print("y")
        # print(y.shape)
        # print(y)
        # print("raw_predictions")
        # print(raw_predictions.shape)
        # print(raw_predictions)

        classes = []
        classidxs = []
        gradients = []

        for class_n in range(int(np.max(y)+1)):
            idx = np.where(y==class_n)
            classes.append(raw_predictions[idx])
            classidxs.append(list(idx))

        #compute the gradient for each vector in each class

        # print("raw predictions")
        # print(raw_predictions)
        # print("classes")
        # print(classes)

        for idx1, class1 in enumerate(classes):
            sub_gradient_list = []
            for vec1 in class1:
                sub_gradient = np.zeros_like(vec1)
                for idx2, class2 in enumerate(classes):
                    for vec2 in class2:
                        if (vec1 == vec2).all() and idx1 == idx2:
                            continue
                        sub_gradient = np.add(sub_gradient, self.single_point_grad(vec1, vec2, idx1 == idx2))
                
                sub_gradient_list.append(sub_gradient.tolist())
            gradients.append(sub_gradient_list)

        gradients = sum(gradients, [])
        gradients = np.array(gradients)
        classidxs = np.concatenate(sum(classidxs, []))

        return np.array([gradients[i] for i in classidxs])

    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate=0.1,
        k=0,
    ):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n,)
            The weight of each sample.
        sample_mask : ndarray of shape (n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.
        """
        # update predictions
        #check back on this
        raw_predictions += learning_rate * tree.predict(X).reshape(X.shape[0], self.latent_dim)
    
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y, residual, raw_predictions, sample_weight):
        pass


















LOSS_FUNCTIONS = {
    "squared_error": LeastSquaresError,
    "absolute_error": LeastAbsoluteError,
    "huber": HuberLossFunction,
    "quantile": QuantileLossFunction,
    # TODO(1.3): Remove deviance
    "deviance": None,  # for both, multinomial and binomial
    "log_loss": None,  # for both, multinomial and binomial
    "exponential": ExponentialLoss,
    "contrastive": ContrastiveLossFunction,
}
