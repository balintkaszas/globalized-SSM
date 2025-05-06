import sympy as sy
from itertools import product
import numpy as np
import scipy.interpolate as scip
from taylor_to_pade.utils import generate_powers_uptoorder_square, generate_polynomial_from_indexed
import time
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.matrices import zeros, Matrix, MatrixBase
import scipy as sc

class Coefficient:
    """ A class to represent a coefficient in a power series. Contains two representations.
    Indexed is a sympy Indexed object, e.g. n_{ijk}. Coefficient tensor is a numpy array containing the coefficients
    """
    def __init__(self, name, dimensions):
        """ Initialize the coefficient with a name and the number of dimensions. 
        Parameters:
        -------
            name (str): name of the coefficient, e.g. 'n'
            dimensions (int): number of dimensions of the coefficients, i.e., the number of indices.        
        """
        # needed to keep track of yet undetermined coefficients
        self.indices = [sy.symbols('i_%s' %ii, cls=sy.Idx) for ii in range(dimensions)] # i_1, i_2, ...
        self.Indexed = sy.Indexed(name, tuple(self.indices)) # n_{ijk}
        self.coefficient_tensor = None # the actual tensor of coefficients, empty at initialization
    def set_coefficient_tensor(self, tensor):
        self.coefficient_tensor = tensor
    

def pade_lsq(an, m, n=None):
    """Return the Pade approximant of order [m/n] of a univariate power series, given by its Taylor coefficients an.
    This is the same implementation found in scipy.interpolate. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pade.html for more information.
    The solution of the linear system of equations is done using the least squares method instead of linalg.solve, as in scipy. 

    Parameters:
        an (array): Taylor coefficients
        m (int): Order of the numerator
        n (int, optional): Order of the denominator. Default setting is m + n = len(an) -1

    Returns:
        p,q: The Pade approximant p/q of the power series an, where p and q are polynomials.
    """
 
    an = np.asarray(an)
    if n is None:
        n = len(an) - 1 - m
        if n < 0:
            raise ValueError("Order of q <m> must be smaller than len(an)-1.")
    if n < 0:
        raise ValueError("Order of p <n> must be greater than 0.")
    N = m + n
    if N > len(an)-1:
        raise ValueError("Order of q+p <m+n> must be smaller than len(an).")
    an = an[:N+1]
    Akj = np.eye(N+1, n+1, dtype=an.dtype)
    Bkj = np.zeros((N+1, m), dtype=an.dtype)
    for row in range(1, m+1):
        Bkj[row,:row] = -(an[:row])[::-1]
    for row in range(m+1, N+1):
        Bkj[row,:] = -(an[row-m:row])[::-1]
    C = np.hstack((Akj, Bkj))
    pq, c1, c2, c3 = np.linalg.lstsq(C, an, rcond=-1)
    p = pq[:n+1]
    q = np.r_[1.0, pq[n+1:]]
    return np.poly1d(p[::-1]), np.poly1d(q[::-1])


def robust_pade(an, m, n = None):
    """Returns the robust Pade approximant, as introduced by P. Gonnet, S. Guttel, L. N. Trefethen,  Robust Pad´e Approximation via SVD. SIAM Review, 2013. 
    The order of the approximant is [m/n]. The implementation is a translation of the Chebfun implementation of the robust Pade approximant. See https://github.com/chebfun/chebfun/blob/master/padeapprox.m for more information.

    Parameters:
        an (array): Taylor coefficients
        m (int): Order of the numerator
        n (int, optional): Order of the denominator. Default setting is m + n = len(an) -1
        tol (float, optional): Relative tolerance. Defaults to 1e-30.

    Returns:
        p,q: The Pade approximant p/q of the power series an, where p and q are polynomials.
    """

    # Based on the chebfun implementation of the robust Pade approximant
    if n is None:
        n = len(an) - 1 - m
        if n < 0:
            raise ValueError("Order of q <m> must be smaller than len(an)-1.")
    if n <= 0:
        raise ValueError("Order of p <n> must be greater than 0.")
    N = m + n
    if N > len(an)-1:
        raise ValueError("Order of q+p <m+n> must be smaller than len(an).")
    an = an[:N+1]


    
    # Ensure that an has enough length
    c = np.pad(an[:N+1], (0, max(0, m + n + 1 - len(an))), 'constant')
    c = c[:m + n + 1]
    row = np.concatenate([[c[0]], np.zeros(n)])
    col = c
    Z = sc.linalg.toeplitz(col[:m+n+1], row[:n+1])
    C = Z[m+1:m+n+1, :]
    if n > 0:
        U, S, V = np.linalg.svd(C, full_matrices=True)
        # Null vector gives b
        b = V[-1,:]
        # Reweighted QR for better zero preservation
        D = np.diag(np.abs(b) + np.sqrt(np.finfo(float).eps))
        Q, R = np.linalg.qr((C @ D).T)

        # Compensate for reweighting
        b = D @ Q[:, -1]
        b /= np.linalg.norm(b)
        # Compute a
        a = Z[:m+1, :n+1] @ b

    return np.poly1d(a[::-1]), np.poly1d(b[::-1])




def convert_to_pade_1d(taylor_coeffs, order_numerator, order_denominator, use_robust = False):
    """
    Wrapper function to convert a univariate Taylor series to an order [m/n] Pade approximant, using either the least squares method or the robust method.

    Parameters
    ----------
    taylor_coeffs : array
        Coefficients of the Taylor series
    order_numerator : int
        Numerator order (m)
    order_denominator : int
        Denominator order (n)
    use_robust : bool, optional
        Use robust implementation, by default False

    Returns
    -------
    polynomial, polynomial
        p,q: The Pade approximant p/q of the power series an, where p and q are polynomials.
    """
    if use_robust:
        numerator, denominator = robust_pade(taylor_coeffs, m = order_numerator, n = order_denominator)    
    else:
        numerator, denominator = pade_lsq(taylor_coeffs, m = order_numerator, n = order_denominator)
    return numerator, denominator

def generate_equations_LHS_alpha_beta_1(indexed_coeff, r, s, taylor_coeff):
    """Generate the left-hand side of the equations matching the r, r-s order of the Taylor series. The equations are generated in a way that is compatible with sympy's Eq function.
    Parameters
    ----------
    indexed_coeff : sympy.Indexed
        The indexed coefficient to be used in the equations.
    r : int
        index1. Assuming this is smaller than order_denom.
    s : int
        index2. Assuming this is smaller than order_num.
    taylor_coeff : sympy.Indexed
        The indexed coefficient of the Taylor series.
    Returns
    -------
    lhs : sympy.Add
        The left-hand side of the equations.
    """
    lhs = 0
    
    for k in range(0, r + 1):
        for n1 in range(0, k + 1):
            lhs += indexed_coeff.Indexed.subs([(indexed_coeff.indices[0], n1),
                                               (indexed_coeff.indices[1], k - n1)
                                                ]) * taylor_coeff[s - n1, r - k - s + n1]
    return lhs



def generate_equations_LHS_alpha_beta_1_robust(indexed_coeff, order_denom, r, s, taylor_coeff):
    """Generate the left-hand side of the equations matching the r, r-s order of the Taylor series for the robust Pade approximants. The equations are generated in a way that is compatible with sympy's Eq function.
    Parameters
    ----------
    indexed_coeff : sympy.Indexed
        The indexed coefficient to be used in the equations.
    r : int
        index1. Assuming this is smaller than order_denom.
    s : int
        index2. Assuming this is smaller than order_num.
    taylor_coeff : sympy.Indexed
        The indexed coefficient of the Taylor series.
    Returns
    -------
    lhs : sympy.Add
        The left-hand side of the equations.
    """
    lhs = 0
    for k in range(0, r + 1):
        for n1 in range(0, k + 1):
            if k <= order_denom:
                lhs += indexed_coeff.Indexed.subs([(indexed_coeff.indices[0], n1),
                                               (indexed_coeff.indices[1], k - n1)
                                                ]) * taylor_coeff[s - n1, r - k - s + n1]
    return lhs

def generate_equations_LHS_alpha_beta_2(indexed_coeff, order_denom, r, s, taylor_coeff):
    """Generate the left-hand side of the equations matching the r, r-s order of the Taylor series. The equations are generated in a way that is compatible with sympy's Eq function.
    Parameters
    ----------
    indexed_coeff : sympy.Indexed
        The indexed coefficient to be used in the equations.
    r : int
        index1. Assuming this is larger than order_denom.
    s : int
        index2. Assuming this is smaller than order_num.
    taylor_coeff : sympy.Indexed
        The indexed coefficient of the Taylor series.
    Returns
    -------
    lhs : sympy.Add
        The left-hand side of the equations.
    """
    lhs = 0
    for k in range(0, order_denom +1 ):
        for n1 in range(0, k + 1):
            lhs += indexed_coeff.Indexed.subs([(indexed_coeff.indices[0], n1),
                                               (indexed_coeff.indices[1], k - n1)
                                                ]) * taylor_coeff[s - n1, r - k - s + n1]
    return lhs

def generate_equations_robust_denominator(order_numerator, order_denominator, denominator_coeff, taylor_coeffs):
    """Generate the equations to compute the denominator of the Pade approximant. The equations are generated in a way that is compatible with sympy's Eq function.
    Parameters
    ----------
    order_numerator : int
        The order of the numerator polynomial.
    order_denominator : int
        The order of the denominator polynomial.
    denominator_coeff : sympy.Indexed
        The indexed coefficient of the denominator polynomial.
    taylor_coeffs : array-like
        The indexed coefficient of the Taylor series.
    Returns
    -------
    eq : list
        A list of sympy equations representing the equations to determine the denominator of the Pade approximant.
    """
    eq = []
    for r in range(order_numerator+1, order_numerator  + order_denominator + 1):
        for s in range(r + 1):
            lhs = generate_equations_LHS_alpha_beta_2(denominator_coeff, order_denominator, r, s, taylor_coeffs)

            if lhs != 0:  # exclude the trivial equation 0=0
                eq.append(sy.Eq(lhs, 0))

    return eq

def generate_symbols_robust(order_numerator, order_denominator, denominator_coeff):
    """Generate the symbols to be used in the equations for the denominator of the Pade approximant.
    Parameters
    ----------
    order_numerator : int
        The order of the numerator polynomial.
    order_denominator : int
        The order of the denominator polynomial.
    denominator_coeff : sympy.Indexed
        The indexed coefficient of the denominator polynomial.
    Returns
    -------
    symbs : list
        A list of multi-index sympy symbols representing the coefficients of the denominator polynomial.
    vectorized_symbols : array
        A list of single-index sympy symbols representing the coefficients of the denominator polynomial.
    """
    orders_for_numerator = generate_powers_uptoorder_bivariate(order_numerator)
    orders_for_denominator = generate_powers_uptoorder_bivariate(order_denominator)
    symbs = []
    for pp in orders_for_denominator:
        symbs.append(denominator_coeff.Indexed.subs([(denominator_coeff.indices[0], pp[0]),
                                                  (denominator_coeff.indices[1], pp[1])]))
    num_symbols = len(symbs)

    vectorized_symbols = sy.symbols('a0:%s' %num_symbols)
    return symbs, vectorized_symbols

def substitute_numerator_coeffs(order_numerator, order_denominator, numerator_coeff, numerator_polynomial, denominator_coeff, taylor_coeffs):
    """Substitute the coefficients of the numerator to generate a single sympy expression.
    Parameters
    ----------
    order_numerator : int
        The order of the numerator polynomial.
    order_denominator : int
        The order of the denominator polynomial.
    numerator_coeff : sympy.Indexed
        The indexed coefficient of the numerator polynomial.
    numerator_polynomial : sympy.Add
        The polynomial representing the numerator.
    denominator_coeff : sympy.Indexed
        The indexed coefficient of the denominator polynomial.
    taylor_coeffs : array-like
        The indexed coefficient of the Taylor series.
    Returns
    -------
    numerator_polynomial : sympy.Add
        The polynomial representing the numerator after substitution.
    """
    for r in range(0, order_numerator+1):
        for s in range(0, r+1):
            #print("eq1:", s, r-s)
            rhs = numerator_coeff.Indexed.subs([(numerator_coeff.indices[0], s), (numerator_coeff.indices[1], r - s)])
            lhs = generate_equations_LHS_alpha_beta_1_robust(denominator_coeff,order_denominator, r, s, taylor_coeffs)
            numerator_polynomial = numerator_polynomial.subs(rhs, lhs)
    return numerator_polynomial


def generate_equations(order_numerator, order_denominator, numerator_coeff, denominator_coeff, taylor_coeffs):
    """Generate the equations to compute the Pade approximant. Both the homogeneous and non-homogeneous equations are generated.
    Parameters
    ----------
    order_numerator : int
        The order of the numerator polynomial.
    order_denominator : int
        The order of the denominator polynomial.
    numerator_coeff : sympy.Indexed
        The indexed coefficient of the numerator polynomial.
    denominator_coeff : sympy.Indexed
        The indexed coefficient of the denominator polynomial.
    taylor_coeffs : array-like
        The indexed coefficient of the Taylor series.
    Returns
    -------
    eq : list
        A list of sympy equations representing the equations to determine the denominator and numerator of the.
    """
    eq1 = []
    for r in range(0, order_numerator+1):
        for s in range(0, r+1):
            rhs = numerator_coeff.Indexed.subs([(numerator_coeff.indices[0], s), (numerator_coeff.indices[1], r - s)])
            lhs = generate_equations_LHS_alpha_beta_1(denominator_coeff, r, s, taylor_coeffs)
            ee = sy.Eq(lhs, rhs)
            if ee is not True: # continuing to add equations that are not trivial
                eq1.append(ee)
    eq2 = []
    for r in range(order_numerator+1, order_denominator  + order_denominator + 1):
        for s in range(r + 1):
            lhs = generate_equations_LHS_alpha_beta_2(denominator_coeff, order_denominator, r, s, taylor_coeffs)
            if lhs != 0:  # exclude the trivial equation 0=0
                eq2.append(sy.Eq(lhs, 0))

    return eq1 + eq2


def generate_symbols(order_numerator, order_denominator, numerator_coeff, denominator_coeff):
    orders_for_numerator = generate_powers_uptoorder_bivariate(order_numerator)
    symbs = [numerator_coeff.Indexed.subs([(numerator_coeff.indices[0], pp[0]),
                                    (numerator_coeff.indices[1], pp[1])]) for pp in orders_for_numerator]
    orders_for_denominator = generate_powers_uptoorder_bivariate(order_denominator)
    for pp in orders_for_denominator:
        if pp[0] != 0 or pp[1] != 0: # exclude d_00
            symbs.append(denominator_coeff.Indexed.subs([(denominator_coeff.indices[0], pp[0]),
                                                  (denominator_coeff.indices[1], pp[1])]))
    num_symbols = len(symbs)

    vectorized_symbols = sy.symbols('a0:%s' %num_symbols)
    return symbs, vectorized_symbols

def swap_symbols_in_equations(equations, symbols, vectorized_symbols):
    # Create a dictionary for symbol substitution
    substitution_dict = {symbol: vectorized_symbol for symbol, vectorized_symbol in zip(symbols, vectorized_symbols)}

    # Perform symbol substitution for each equation
    eqq = [eq.subs(substitution_dict) for eq in equations]

    return eqq


def prepare_linear_system(equations, vectorized_symbols):
    eq, c = _linear_eq_to_dict(equations, vectorized_symbols) # this bit is copied from sympy without the type check
    # this allows for complex input as well
    # was AB = sy.linear_eq_to_matrix(equations, vectorized_symbols), which does not allow for complex input

    n, m = shape = len(eq), len(vectorized_symbols)
    ix = dict(zip(vectorized_symbols, range(m)))
    A = zeros(*shape)
    for row, d in enumerate(eq):
        for k in d:
            col = ix[k]
            A[row, col] = d[k]
    b = Matrix(n, 1, [-i for i in c])
    
    
    A = np.array(A, dtype = complex)
    b = np.array(b, dtype = complex)
    return A, b




def generate_powers_uptoorder_bivariate(M):
    pows = []
    for m in range(M+1):
        for s in range(m+1):
            pows.append([s, m-s])
    return np.array(pows)




def convert_to_pade_2d(variables, taylor_coeffs, order_numerator, order_denominator, numerator_coeff, denominator_coeff ):
    """Convert a 2D Taylor series to a Pade approximant of order [m/n]. We generate the set of linear equations with the helper functions above and use the least squares method to solve the linear system. The solution is then used to generate the numerator and denominator polynomials.
    The denominator is fixed to 1, i.e. d_00 = 1. The numerator is then computed by substituting the coefficients of the denominator into the numerator polynomial.
    The numerator and denominator polynomials are returned as sympy expressions.
    The function is a wrapper around the generate_equations, generate_symbols, prepare_linear_system and generate_polynomial_from_indexed functions.
    Parameters
    ----------
    variables : list
        A list of sympy symbols representing the variables in the Taylor series.
    taylor_coeffs : array-like
        The indexed coefficient of the Taylor series.
    order_numerator : int
        The order of the numerator polynomial.
    order_denominator : int
        The order of the denominator polynomial.
    numerator_coeff : sympy.Indexed
        The indexed coefficient of the numerator polynomial.
    denominator_coeff : sympy.Indexed
        The indexed coefficient of the denominator polynomial.
    Returns
    -------
    numerator_function : sympy.Add
        The polynomial representing the numerator of the Pade approximant.
    denominator_function : sympy.Add
        The polynomial representing the denominator of the Pade approximant.
    """
    equations = generate_equations(order_numerator, order_denominator, numerator_coeff, denominator_coeff, taylor_coeffs)
    symbols, vectorized_symbols = generate_symbols(order_numerator, order_denominator, numerator_coeff, denominator_coeff)
    equations = [e.subs(denominator_coeff.Indexed.subs([(denominator_coeff.indices[0], 0),
                                                         (denominator_coeff.indices[1],0)]),
                                                           1) for e in equations] # fix d_00 = 1
    coeffmatrix, rhsvector = prepare_linear_system(equations, symbols)

    two = time.time()
    

    solution, aaa, _, _ = np.linalg.lstsq(coeffmatrix, rhsvector, rcond = None)

    numerator_function = generate_polynomial_from_indexed(variables,numerator_coeff, order_numerator, include_bias=True)
    denominator_function = generate_polynomial_from_indexed(variables, denominator_coeff, order_denominator, include_bias=False) # set d_00 = 1
    numerator_function = numerator_function.subs(
        [(old, new) for old,new in zip(symbols, 
                                       solution.ravel())]
    )
    denominator_function = denominator_function.subs(
        [(old, new) for old,new in zip(symbols, 
                                       solution.ravel())]
    )
    
    return numerator_function, denominator_function



def convert_to_pade_2d_robust(variables, taylor_coeffs, order_numerator, order_denominator, numerator_coeff, denominator_coeff ):
    """Convert a 2D Taylor series to a Pade approximant of order [m/n]. We generate the set of linear equations with the helper functions above and use the least squares method to solve the linear system. The solution is then used to generate the numerator and denominator polynomials.
    The denominator is fixed to 1, i.e. d_00 = 1.
    The numerator and denominator polynomials are returned as sympy expressions.
    The function is a wrapper around the generate_equations, generate_symbols, prepare_linear_system and generate_polynomial_from_indexed functions.

    This function uses the robust Pade approximant method, i.e. it uses an SVD to compute the coefficients of the denominator first. 
    Parameters
    ----------
    variables : list
        A list of sympy symbols representing the variables in the Taylor series.
    taylor_coeffs : array-like
        The indexed coefficient of the Taylor series.
    order_numerator : int
        The order of the numerator polynomial.
    order_denominator : int
        The order of the denominator polynomial.
    numerator_coeff : sympy.Indexed
        The indexed coefficient of the numerator polynomial.
    denominator_coeff : sympy.Indexed
        The indexed coefficient of the denominator polynomial.
    Returns
    -------
    numerator_function : sympy.Add
        The polynomial representing the numerator of the Pade approximant.
    denominator_function : sympy.Add
        The polynomial representing the denominator of the Pade approximant.
    """
    start = time.time()
    equations_denom = generate_equations_robust_denominator(order_numerator, order_denominator, denominator_coeff, taylor_coeffs)
    one = time.time()
    symbols_denom, vectorized_symbols_denom = generate_symbols_robust(order_numerator, order_denominator, denominator_coeff)
    one = time.time()
    coeffmatrix, rhsvector = prepare_linear_system(equations_denom, symbols_denom) # rhsvector is simply zero here. 

    U, S, V = np.linalg.svd(coeffmatrix, full_matrices=True)
    # Null vector gives b
    b = V[-1,:]
    D = np.diag(np.abs(b) + np.sqrt(np.finfo(float).eps))
    Q, R = np.linalg.qr((coeffmatrix @ D).T)
    two = time.time()
    b = D @ Q[:, -1]
    b /= np.linalg.norm(b)
    denominator_function = generate_polynomial_from_indexed(variables, denominator_coeff, order_denominator, include_bias=True) # set d_00 = 1
    denominator_function = denominator_function.subs(
        [(old, new) for old,new in zip(symbols_denom, 
                                       b.ravel())]
    )

    numerator_function = generate_polynomial_from_indexed(variables,numerator_coeff, order_numerator, include_bias=True)
    numerator_function = substitute_numerator_coeffs(order_numerator, order_denominator, numerator_coeff, numerator_function, denominator_coeff, taylor_coeffs)

    numerator_function = numerator_function.subs(
        [(old, new) for old,new in zip(symbols_denom, 
                                       b.ravel())]
    )
    return numerator_function, denominator_function