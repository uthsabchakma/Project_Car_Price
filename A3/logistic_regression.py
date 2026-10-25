"""Multinomial Logistic Regression from scratch (based on 02 - Multinomial Logistic Regression.ipynb).

Additions for A3:
  * accuracy, precision, recall, f1 (per class)
  * macro_* and weighted_* averages, and a classification_report() like sklearn's
  * optional Ridge (L2) penalty: J = -sum(y log h) + lambda * sum(theta_j^2)
"""
import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, k, n, method='batch', lr=0.01, max_iter=5000, batch_size=64,
                 penalty=None, lam=0.01, seed=42):
        """
        k        : number of classes      n : number of features (without the bias)
        method   : 'batch', 'minibatch' or 'sto'
        lr       : learning rate          max_iter : number of gradient steps
        penalty  : None or 'ridge'        lam      : strength of the ridge penalty
        """
        assert method in ('batch', 'minibatch', 'sto')
        assert penalty in (None, 'ridge')
        self.k, self.n = k, n
        self.method, self.lr, self.max_iter, self.batch_size = method, lr, max_iter, batch_size
        self.penalty, self.lam, self.seed = penalty, lam, seed

    # ---------- training ----------
    @staticmethod
    def _add_bias(X):
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)  # subtract the row max for numerical stability
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def h_theta(self, X, W):
        return self.softmax(X @ W)

    def _one_hot(self, y):
        Y = np.zeros((len(y), self.k))
        Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1
        return Y

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = -np.sum(Y * np.log(h + 1e-12)) / m
        grad = X.T @ (h - Y) / m
        if self.penalty == 'ridge':
            # the bias row (row 0) is not penalised
            loss += self.lam * np.sum(self.W[1:] ** 2)
            reg = 2 * self.lam * self.W
            reg[0] = 0
            grad += reg
        return loss, grad

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X, Y = self._add_bias(X), self._one_hot(y)
        self.W = np.zeros((X.shape[1], self.k))
        self.losses = []
        for i in range(self.max_iter):
            if self.method == 'batch':
                Xb, Yb = X, Y
            else:
                size = self.batch_size if self.method == 'minibatch' else 1
                idx = rng.choice(X.shape[0], size=size, replace=False)
                Xb, Yb = X[idx], Y[idx]
            loss, grad = self.gradient(Xb, Yb)
            self.W -= self.lr * grad
            if i % 100 == 0:
                self.losses.append(loss)
        return self

    def predict_proba(self, X):
        return self.h_theta(self._add_bias(X), self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ---------- metrics ----------
    # TP_c: predicted c and really c ; FP_c: predicted c but not c ; FN_c: really c but predicted other
    def _counts(self, y_true, y_pred, c):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        return tp, fp, fn

    def accuracy(self, y_true, y_pred):
        return np.mean(np.asarray(y_true) == np.asarray(y_pred))

    def precision(self, y_true, y_pred, c):
        tp, fp, _ = self._counts(y_true, y_pred, c)
        return tp / (tp + fp) if tp + fp > 0 else 0.0  # no predictions of c -> 0 (as sklearn)

    def recall(self, y_true, y_pred, c):
        tp, _, fn = self._counts(y_true, y_pred, c)
        return tp / (tp + fn) if tp + fn > 0 else 0.0

    def f1(self, y_true, y_pred, c):
        p, r = self.precision(y_true, y_pred, c), self.recall(y_true, y_pred, c)
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def _per_class(self, metric, y_true, y_pred):
        return np.array([metric(y_true, y_pred, c) for c in range(self.k)])

    # macro: plain average over classes (every class counts the same)
    def macro_precision(self, y_true, y_pred):
        return self._per_class(self.precision, y_true, y_pred).mean()

    def macro_recall(self, y_true, y_pred):
        return self._per_class(self.recall, y_true, y_pred).mean()

    def macro_f1(self, y_true, y_pred):
        return self._per_class(self.f1, y_true, y_pred).mean()

    # weighted: average weighted by each class's share of the true labels (bigger classes count more).
    # The weights already sum to 1, so we do not divide by k again (this matches sklearn).
    def _weights(self, y_true):
        return np.array([np.mean(np.asarray(y_true) == c) for c in range(self.k)])

    def weighted_precision(self, y_true, y_pred):
        return np.sum(self._weights(y_true) * self._per_class(self.precision, y_true, y_pred))

    def weighted_recall(self, y_true, y_pred):
        return np.sum(self._weights(y_true) * self._per_class(self.recall, y_true, y_pred))

    def weighted_f1(self, y_true, y_pred):
        return np.sum(self._weights(y_true) * self._per_class(self.f1, y_true, y_pred))

    def classification_report(self, y_true, y_pred):
        """Same layout as sklearn.metrics.classification_report, as a DataFrame."""
        support = np.array([np.sum(np.asarray(y_true) == c) for c in range(self.k)])
        rows = {str(c): [self.precision(y_true, y_pred, c), self.recall(y_true, y_pred, c),
                         self.f1(y_true, y_pred, c), support[c]] for c in range(self.k)}
        n = support.sum()
        rows['accuracy'] = [np.nan, np.nan, self.accuracy(y_true, y_pred), n]
        rows['macro avg'] = [self.macro_precision(y_true, y_pred), self.macro_recall(y_true, y_pred),
                             self.macro_f1(y_true, y_pred), n]
        rows['weighted avg'] = [self.weighted_precision(y_true, y_pred), self.weighted_recall(y_true, y_pred),
                                self.weighted_f1(y_true, y_pred), n]
        return pd.DataFrame(rows, index=['precision', 'recall', 'f1-score', 'support']).T
