import scipy
import numpy as np
from sklearn import metrics



def coef_se(X, y, y_pred):
    """Calculate standard error for beta coefficients.

    Returns
    -------
    numpy.ndarray
        An array of standard errors for the beta coefficients.
    """
    n = X.shape[0]
    X1 = np.hstack((np.ones((n, 1)), np.matrix(X)))
    se_matrix = scipy.linalg.sqrtm(
        metrics.mean_squared_error(y, y_pred) *
        np.linalg.inv(X1.T * X1)
    )
    return np.diagonal(se_matrix)






def coef_tval(X, y, coef_, intercept_, y_pred):
    """Calculate t-statistic for beta coefficients.

    Returns
    -------
    numpy.ndarray
        An array of t-statistic values.
    """
    a = np.array(intercept_ / coef_se(X, y, y_pred)[0])
    b = np.array(coef_.squeeze() / coef_se(X, y, y_pred)[1:])
    return np.append(a, b)




def coef_pval(X, y, coef_, intercept_, y_pred):
    """Calculate p-values for beta coefficients.

    Returns
    -------
    numpy.ndarray
        An array of p-values.
    """
    n = X.shape[0]
    t = coef_tval(X, y, coef_, intercept_, y_pred)
    p = 2 * (1 - scipy.stats.t.cdf(abs(t), n - 1))
    return p