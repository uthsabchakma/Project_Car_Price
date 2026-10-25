"""Linear Regression from scratch (based on the class notebook 03 - Regularization.ipynb).

Additions for A2:
  * r2()                    - coefficient of determination
  * init='zeros'|'xavier'   - weight initialisation
  * momentum                - gradient descent with momentum (0 = off)
  * plot_feature_importance - bar chart of the coefficients
  * PolynomialRegression    - degree-2 feature expansion before fitting
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ---------------- penalties ----------------
class NoPenalty:
    """Plain linear regression: no penalty."""
    def __call__(self, theta):
        return 0.0

    def derivation(self, theta):
        return np.zeros_like(theta)


class LassoPenalty:
    """L1 penalty: l * sum(|theta|). Pushes small coefficients to exactly zero."""
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)


class RidgePenalty:
    """L2 penalty: l * sum(theta^2). Shrinks all coefficients smoothly."""
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


# ---------------- model ----------------
class LinearRegression:
    def __init__(self, regularization=None, lr=0.001, method='batch', init='zeros',
                 momentum=0.0, num_epochs=1000, batch_size=64, patience=20, tol=1e-6, seed=42):
        """
        regularization : penalty object (NoPenalty / LassoPenalty / RidgePenalty), None = no penalty
        lr             : learning rate (step size)
        method         : 'batch' (all rows per step), 'mini' (batch_size rows), 'sto' (1 row)
        init           : 'zeros' or 'xavier'
        momentum       : 0 turns momentum off; a value in (0, 1) turns it on
        num_epochs     : maximum passes over the training data
        patience, tol  : early stopping - stop when the loss has not improved by tol for patience epochs
        """
        assert method in ('batch', 'mini', 'sto')
        assert init in ('zeros', 'xavier')
        assert 0 <= momentum < 1
        self.regularization = regularization or NoPenalty()
        self.lr, self.method, self.init = lr, method, init
        self.momentum, self.num_epochs, self.batch_size = momentum, num_epochs, batch_size
        self.patience, self.tol, self.seed = patience, tol, seed

    # ---- metrics ----
    @staticmethod
    def mse(ytrue, ypred):
        return np.mean((ytrue - ypred) ** 2)

    @staticmethod
    def r2(ytrue, ypred):
        """R^2 = 1 - SS_res / SS_tot : share of the variance of y explained by the model."""
        ss_res = np.sum((ytrue - ypred) ** 2)
        ss_tot = np.sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - ss_res / ss_tot

    # ---- helpers ----
    def _transform(self, X):
        """Feature transform before the bias column. Overridden by PolynomialRegression."""
        return X

    def _add_bias(self, X):
        X = self._transform(np.asarray(X, dtype=float))
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _init_theta(self, n_features, rng):
        if self.init == 'zeros':
            return np.zeros(n_features)
        # Xavier: uniform in [-1/sqrt(m), 1/sqrt(m)], m = number of inputs
        m = n_features
        lower, upper = -1.0 / np.sqrt(m), 1.0 / np.sqrt(m)
        return lower + rng.random(n_features) * (upper - lower)

    def _gradient(self, X, y):
        error = X @ self.theta - y
        grad = X.T @ error / X.shape[0]
        reg = self.regularization.derivation(self.theta)
        reg[0] = 0  # never penalise the bias term
        return grad + reg

    def _step(self, grad):
        """One parameter update, with optional momentum.

        The brief's pseudocode writes `theta - step + momentum * prev_step`; with prev_step being a
        positive step that would *undo* part of the previous move. The standard (heavy-ball) form
        keeps moving in the previous direction: theta = theta - (step + momentum * prev_step).
        """
        step = self.lr * grad + self.momentum * self.prev_step
        self.theta = self.theta - step
        self.prev_step = step

    # ---- training ----
    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        X, y = self._add_bias(X), np.asarray(y, dtype=float)
        has_val = X_val is not None
        if has_val:
            X_val, y_val = self._add_bias(X_val), np.asarray(y_val, dtype=float)

        self.theta = self._init_theta(X.shape[1], rng)
        self.prev_step = np.zeros_like(self.theta)
        self.history, self.diverged = [], False
        best, wait = np.inf, 0
        bs = {'batch': X.shape[0], 'mini': self.batch_size, 'sto': 1}[self.method]

        for epoch in range(self.num_epochs):
            # shuffle every epoch so mini-batches / single samples are in a new random order
            idx = rng.permutation(X.shape[0]) if self.method != 'batch' else np.arange(X.shape[0])
            for start in range(0, X.shape[0], bs):
                b = idx[start:start + bs]
                self._step(self._gradient(X[b], y[b]))

            # monitor the validation loss (or training loss when there is no validation set)
            Xm, ym = (X_val, y_val) if has_val else (X, y)
            with np.errstate(over='ignore', invalid='ignore'):
                loss = self.mse(ym, Xm @ self.theta)
            self.history.append(loss)
            if not np.isfinite(loss) or loss > 1e10:  # learning rate too large -> exploded
                self.diverged = True
                break
            if loss < best - self.tol:
                best, wait = loss, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return self

    def predict(self, X):
        return self._add_bias(X) @ self.theta

    # ---- interpretation ----
    def _coef(self):
        return self.theta[1:]  # drop the bias

    def plot_feature_importance(self, feature_names, top=15, ax=None):
        """Bar chart of |coefficient| per feature. Only meaningful because inputs are standardised."""
        coef = self._coef()
        order = np.argsort(np.abs(coef))[-top:]
        ax = ax or plt.subplots(figsize=(8, 0.4 * len(order) + 1))[1]
        colors = ['tab:green' if c > 0 else 'tab:red' for c in coef[order]]
        ax.barh(np.array(feature_names)[order], np.abs(coef[order]), color=colors)
        ax.set_xlabel('|coefficient|  (green = raises price, red = lowers price)')
        ax.set_title('Feature importance')
        return ax


class NormalRegression(LinearRegression):
    def __init__(self, **kw):
        super().__init__(regularization=NoPenalty(), **kw)


class LassoRegression(LinearRegression):
    def __init__(self, l=0.01, **kw):
        super().__init__(regularization=LassoPenalty(l), **kw)


class RidgeRegression(LinearRegression):
    def __init__(self, l=0.01, **kw):
        super().__init__(regularization=RidgePenalty(l), **kw)


class PolynomialRegression(LinearRegression):
    """Adds degree-2 terms (x_i^2 and x_i*x_j) so a linear model can fit curves.

    poly_idx : column positions to expand (we pass the numeric columns). Expanding the one-hot
               dummies too would create ~800 mostly-empty columns (e.g. brand_BMW * brand_Audi = 0).
    The new terms are standardised: squaring an already-large value (a car with 2M km is ~30 std
    devs out, squared ~900) makes gradient descent explode otherwise.
    """
    def __init__(self, degree=2, poly_idx=None, **kw):
        super().__init__(regularization=NoPenalty(), **kw)
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.poly_scaler = StandardScaler()
        self.poly_idx = poly_idx
        self._poly_fitted = False

    def _transform(self, X):
        idx = self.poly_idx if self.poly_idx is not None else list(range(X.shape[1]))
        self._n_in = len(idx)
        if not self._poly_fitted:  # fitted once, on the training data passed to the first fit()
            self.poly.fit(X[:, idx])
            self.poly_scaler.fit(self.poly.transform(X[:, idx])[:, self._n_in:])
            self._poly_fitted = True
        # the first n_in poly columns are the original features (already in X) -> keep only new terms
        extra = self.poly.transform(X[:, idx])[:, self._n_in:]
        return np.hstack([X, self.poly_scaler.transform(extra)])

    def plot_feature_importance(self, feature_names, top=15, ax=None):
        idx = self.poly_idx if self.poly_idx is not None else list(range(len(feature_names)))
        extra = self.poly.get_feature_names_out(np.array(feature_names)[idx])[self._n_in:]
        return super().plot_feature_importance(list(feature_names) + list(extra), top, ax)
