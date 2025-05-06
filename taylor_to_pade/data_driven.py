import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, LinearConstraint

def generate_features(X, order_num, order_denom, bias = True):
    """
    Generate polynomial features for the numerator and denominator of a Pade approximation.

    Parameters
    ----------
    X : array-like, shape (n_features, n_samples)
        Input data.
    order_num : int
        Order of the numerator polynomial.
    order_denom : int
        Order of the denominator polynomial.
    bias : bool, optional
        If True, include a bias term (intercept) in the polynomial features. Default is True.
    Returns
    -------
    XX_p : array-like, shape (n_samples, n_features)
        Polynomial features for the numerator.
    XX_q : array-like, shape (n_samples, n_features)
        Polynomial features for the denominator.
    """
    polyFeatures_numerator = PolynomialFeatures(order_num, include_bias = False)
    XX_p  = polyFeatures_numerator.fit_transform(X.T)
    polyFeatures_denom = PolynomialFeatures(order_denom, include_bias = bias)
    XX_q  = polyFeatures_denom.fit_transform(X.T) 
    return XX_p, XX_q # these have shape (n_samples, n_features). No transpose needed


def polynomial_approximant(X, y, order, bias = True):
    """
    Fit a polynomial approximant to the data using least squares.
    Parameters
    ----------
    X : array-like, shape (n_features, n_samples)
        Input data.
    y : array-like, shape (n_samples,)
        Target values.
    order : int
        Order of the polynomial.
    bias : bool, optional
        If True, include a bias term (intercept) in the polynomial features. Default is True.
    Returns
    -------
    b : array-like, shape (n_features,)
        Coefficients of the polynomial approximant.
    """
    polyFeatures = PolynomialFeatures(order, include_bias = False)
    XX_p  = polyFeatures.fit_transform(X.T)
    b = np.linalg.pinv(XX_p)@y
    return b

def linear_error_function_scalar(y, coeffs, XX_p, XX_q):
    """
    Compute the linearized error function for a scalar output.
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Target values.
    coeffs : array-like, shape (num_unknowns_numerator*n_features + num_unknowns_denominator,)
        Coefficients of the numerator and denominator polynomials arranged into a vector.
        coeffs[:num_unknowns_denominator] are the coefficients of the denominator polynomial,
        coeffs[num_unknowns_denominator:] are the coefficients of the numerator polynomial. 
    XX_p : array-like, shape (n_samples, num_unknowns_numerator)
        Polynomial features for the numerator.
    XX_q : array-like, shape (n_samples, num_unknowns_denominator)
        Polynomial features for the denominator.
    Returns
    -------
    error : float
        evaluates |y*(b*x^n) - a*x^n|
    """
    num_unknowns_numerator = XX_p.shape[1]
    num_unknowns_denominator = XX_q.shape[1]
    b = coeffs[:num_unknowns_denominator]
    A = coeffs[num_unknowns_denominator:].reshape((num_unknowns_numerator, y.shape[1]))
    error = XX_p @ A - (XX_q@b).reshape(-1,1)*y
    return np.linalg.norm(error.ravel())**2


def nonlinear_error_function_scalar(y, coeffs, XX_p, XX_q):
    """
    Compute the full, nonlinear error function for a scalar output.

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Target values.
    coeffs : array-like, shape (num_unknowns_numerator*n_features + num_unknowns_denominator,)
        Coefficients of the numerator and denominator polynomials arranged into a vector.
        coeffs[:num_unknowns_denominator] are the coefficients of the denominator polynomial,
        coeffs[num_unknowns_denominator:] are the coefficients of the numerator polynomial. 
    XX_p : array-like, shape (n_samples, num_unknowns_numerator)
        Polynomial features for the numerator.
    XX_q : array-like, shape (n_samples, num_unknowns_denominator)
        Polynomial features for the denominator.
    Returns
    -------
    error : float
        evaluates |y- ((a*x^n) / (b*x^n))|
    """
    num_unknowns_numerator = XX_p.shape[1]
    num_unknowns_denominator = XX_q.shape[1]
    b = coeffs[:num_unknowns_denominator]
    A = coeffs[num_unknowns_denominator:].reshape((num_unknowns_numerator, y.shape[1]))
    error = (XX_p @ A) / (XX_q@b).reshape(-1,1) -y
    return np.linalg.norm(error.ravel())**2

def unpack_coeffs(coeffs, num_unknowns_numerator, num_unknowns_denominator, n_features):
    """
    Unpack the coefficients into numerator and denominator polynomials.
    Parameters
    ----------
    coeffs : array-like, shape (num_unknowns_numerator*n_features + num_unknowns_denominator,)
        Coefficients of the numerator and denominator polynomials arranged into a vector.
        coeffs[:num_unknowns_denominator] are the coefficients of the denominator polynomial,
        coeffs[num_unknowns_denominator:] are the coefficients of the numerator polynomial. 
    num_unknowns_numerator : int
        Number of unknowns in the numerator polynomial.
    num_unknowns_denominator : int
        Number of unknowns in the denominator polynomial.
    n_features : int
        Number of features in the input data.
    Returns
    -------
    A : array-like, shape (num_unknowns_numerator, n_features)
        Coefficients of the numerator polynomial.
    b : array-like, shape (num_unknowns_denominator,)
        Coefficients of the denominator polynomial.
    """
    b = coeffs[:num_unknowns_denominator]
    A = coeffs[num_unknowns_denominator:].reshape((num_unknowns_numerator, n_features))
    return A, b


def rational_approximant(X, y, order_num, order_denom, loss_type = 'linear',
                                           init_coeffs = None, constrained= True, delta = 0.5):
    """
    Computes a rational approximant to the data using least squares minimization. The loss function is either nonlinear (y - a*x^n / b*x^n)
      or linear (b*x^n y - a*x^n). 
    Constraints can be applied to the denominator polynomial to ensure that it is positive. 
    The optimization problem is solved using the SLSQP method from scipy.optimize.
    The coefficients of the numerator and denominator polynomials are returned as a flattened vector.

    Parameters
    ----------
    X : array-like, shape (n_features, n_samples)
        Input data.
    y : array-like, shape (n_samples,)
        Target values.
    order_num : int
        Order of the numerator polynomial.
    order_denom : int
        Order of the denominator polynomial.
    loss_type : str, optional
        Type of loss function to use. If 'nonlinear', we minimize (y - a*x^n / b*x^n).
         If 'linear', we minimize (b*x^n y - a*x^n). 
         Default is 'linear'.
    init_coeffs : array-like, shape (num_unknowns_numerator*n_features + num_unknowns_denominator,)
        Initial guess for the coefficients.
    constrained : bool, optional
        If True, apply linear constraints to the optimization problem, making sure that the denominator is positive. Default is True. 
    delta : float, optional
        The denominator is constrained to be larger than delta. Default is 0.5.
    Returns
    -------
    coeffs : array-like, shape (num_unknowns_numerator*n_features + num_unknowns_denominator, )
        Coefficients of the numerator and denominator polynomials arranged into a vector.
        coeffs[:num_unknowns_denominator] are the coefficients of the denominator polynomial,
        coeffs[num_unknowns_denominator:] are the coefficients of the numerator polynomial. 
    """
    XX_p, XX_q = generate_features(X, order_num, order_denom)
    num_unknowns_numerator = XX_p.shape[1]
    num_unknowns_denominator = XX_q.shape[1]
    num_features = X.shape[0]
    if init_coeffs is None:
        a_poly = polynomial_approximant(X, y.T, order_num)
        b0 = np.zeros(num_unknowns_denominator)
        init_coeffs = np.concatenate((b0, a_poly.ravel()))
    if loss_type == 'nonlinear':
        to_minimize = lambda x : nonlinear_error_function_scalar(y.T, x, XX_p, XX_q)
    if loss_type == 'linear':
        to_minimize = lambda x : linear_error_function_scalar(y.T, x, XX_p, XX_q)
    if constrained:
        num_constr_points = XX_p.shape[0]
        num_unknowns_denominator = XX_q.shape[1]
        constr_mtx = np.hstack((XX_q, np.zeros((num_constr_points, num_features*num_unknowns_numerator)) ))
        lin_cons = LinearConstraint(constr_mtx, delta, np.inf)
        result = minimize(to_minimize, init_coeffs, constraints=lin_cons, method='SLSQP', 
                          options = {'maxiter': 1000})
    else:
        result = minimize(to_minimize, init_coeffs,  method='SLSQP',
                          options = {'maxiter': 1000})
    return result.x




def evaluate_rational_model(Xa, a, b, order_num, order_denom):
    """
    Evaluate the rational model at the given input data.
    Parameters
    ----------
    Xa : array-like, shape (n_features, n_samples)
        Input data.
    a : array-like, shape (num_unknowns_numerator, n_features)
        Coefficients of the numerator polynomial.
    b : array-like, shape (num_unknowns_denominator,)
        Coefficients of the denominator polynomial.
    order_num : int
        Order of the numerator polynomial.
    order_denom : int
        Order of the denominator polynomial.
    Returns
    -------
    y : array-like, shape (n_samples,)
        Evaluated rational model at the input data.
    """
    XX_p, XX_q = generate_features(Xa, order_num, order_denom, bias=True)
    return (XX_p@a) / (XX_q@b).reshape(-1,1)
