import sympy as sy
from itertools import product
import numpy as np
from taylor_to_pade import approximant, approximant, matlab_integration


def generate_powers_uptoorder_square(dimension, M):
    """
    Generates the powers in the multivariate monomial terms up to order M, i.e., x_1^i * x_2^j * x_3^k for i,j,k = 0,1,...,M. 

    Parameters
    ----------
    dimension : int
        Number of variables
    M : int
        Order of the polynomial

    Returns
    -------
    array
        array of powers [i,j,k]
    """
    pows = []
    range_list = [range(M + 1) for _ in range(dimension)]
    for power in product(*range_list):
        pows.append([*power])
    return np.array(pows)

def generate_polynomial_from_tensor(variables, powers, coefficients):
    """Given a list of variables, powers, and coefficients, generate a polynomial.
    Parameters
    ----------
        variables (list): list of sympy variables
        powers (array): list of powers of each variable: contains [i,j,k] such that C_{ijk} * x^i * y^j * z^k
        coefficients (array):  tensor of coefficients: C_{ijk}: [i,j,k,...]
    Returns
    -------
        polynomial (sympy expression): polynomial expression
    """
    polynomial = 0
    for p in powers:
        # p is a list containing the powers of each variable.
        # Converting to tuple tuple(p) = (i, j, k): C_{ijk} = coefficients[i,j,k]  
        if len(p)> 1:
            index_tuple = tuple(p)
        else:
            index_tuple = p[0]
        polynomial += coefficients[index_tuple] * sy.prod([s**p_ for s, p_ in zip(variables, p)])
    return polynomial


def generate_polynomial_from_indexed_square(variables, indexed, order, include_bias = True):
    val = 0 
    for ii in range(order + 1):
        for jj in range(order + 1):
            val += indexed.Indexed.subs([(indexed.indices[0], ii), (indexed.indices[1], jj)]) * variables[0] ** ii *variables[1] ** jj
    if not include_bias:
        val = 1 + val - indexed.Indexed.subs([(indexed.indices[0],0), (indexed.indices[1], 0)])
    return val




def generate_polynomial_from_indexed(variables, indexed, order, include_bias = True):
    val = 0 
    for ii in range(order + 1):
        for jj in range(order + 1):
            if ii + jj <= order:
                val += indexed.Indexed.subs([(indexed.indices[0], ii), (indexed.indices[1], jj)]) * variables[0] ** ii *variables[1] ** jj
    if not include_bias:
        val = 1 + val - indexed.Indexed.subs([(indexed.indices[0],0), (indexed.indices[1], 0)])
    return val


def taylor_approximation_2d(f, x0, y0, order=2):
    """
    Analytical computation of a 2D Taylor expansion around (x0, y0).
    f(x,y) = f(x_0, y_0) + Df(x_0,y_0)*(x-x_0, y-y_0) + 1/2 D^2f(x_0,y_0)*(x-x_0, y-y_0)^2 + ...

    Parameters
    ----------
    f : sympy expression
        Function to approximante
    x0 : float
        x0 coordinate of the expansion point
    y0 : float
        y0 coordinate of the expansion point
    order : int, optional
        maximal order to compute, by default 2

    Returns
    -------
    tuple: sympy expression, array
        Sympy polynomial expression and corresponding coefficient tensor 
    """
    x, y = sy.symbols('x y')
    taylor_approx = f.subs({x: x0, y: y0})
    
    for i in range(0, order + 1):
        for j in range(i + 1):
            df = sy.diff(f, x, j, y, i-j).subs({x: x0, y: y0})
            term = df / (sy.factorial(j) * sy.factorial(i-j)) * (x - x0)**j * (y - y0)**(i-j)
            taylor_approx += term
    coeffs = taylor_approx.as_coefficients_dict()
    N = int(np.sqrt(len(coeffs)))
    taylor_coeffs = sy.zeros(N,N)
    for ii in range(0, N):
        for jj in range(0, N):
            taylor_coeffs[ii,jj] = taylor_approx.as_expr().coeff(x,ii).coeff(y, jj)
    return taylor_approx, taylor_coeffs


# from ssmlearnpy
def convert_to_polar(variables, equations):
    # should have the same number of equations as variables
    n_variables = len(variables)
    n_equations = len(equations)
    ii = sy.sqrt(-1)
    n_polar_variables = int(n_variables / 2)
    radial_variables = [sy.symbols('r_%d' %i) for i in range(n_polar_variables)]
    angle_variables = [sy.symbols('\\varphi_%d' %i) for i in range(n_polar_variables)]

    substituted_equations = []
    
    for e in equations:
        substitution_rules_variable = [(variables[i], radial_variables[i]*sy.exp(ii*angle_variables[i]) ) for i in range(n_polar_variables)]
        substitution_rules_conjugate = [(variables[i+n_polar_variables], radial_variables[i]*sy.exp(-ii*angle_variables[i]) ) for i in range(n_polar_variables)]
        f = e.subs(substitution_rules_variable)
        f = f.subs(substitution_rules_conjugate)
        substituted_equations.append(f)
    # \dot{z} = \dot{r} exp(i \varphi) + i r \dot{\varphi} exp(i \varphi)
    # \dot{\bar{z}} = \dot{r} exp(-i \varphi) - i r \dot{\varphi} exp(-i \varphi)
    # \dot{r} = (\dot{z} exp(-i \varphi) + \dot{\bar{z}} exp(i \varphi)) / 2
    # \dot{\varphi} = (i \dot{z} r exp(-i \varphi) - i \dot{\bar{z}} r exp(i \varphi)) / (2 r^2)
    r_equations = []
    phi_equations = []
    for i in range(n_polar_variables):
        z_dot = substituted_equations[i]
        zbar_dot = substituted_equations[i + n_polar_variables]
        phi_var = angle_variables[i]
        r_var = radial_variables[i]
        r_equations.append(z_dot * sy.exp(-ii * phi_var)/2  +  zbar_dot * sy.exp(ii * phi_var)/2) # dot z * exp(-i \varphi)/2 + dot \bar{z} * exp(i \varphi)/2
        phi_eq_temps = z_dot * sy.exp(-ii * phi_var)/(2*ii) - zbar_dot * sy.exp(ii * phi_var)/(2*ii) # dot z * exp(-i \varphi)/(2i) - dot \bar{z} * exp(i \varphi)/(2i)
        phi_equations.append(phi_eq_temps / (r_var))
    r_equations = [sy.simplify(r) for r in r_equations]
    phi_equations = [sy.simplify(p) for p in phi_equations]

    return radial_variables, angle_variables, r_equations, phi_equations




def backbone_curve_and_damping_curve(r_variables, phidot_eq, rdot_eq):
    # TODO: olny for single DOF for now
    # returns tuples, first element is a callable, second element is the symbolic expression
    backbone_callable = sy.lambdify(r_variables[0], phidot_eq[0])
    damping_callable = sy.lambdify(r_variables[0], -rdot_eq[0]/r_variables[0])
    return (backbone_callable, phidot_eq[0]), (damping_callable, -rdot_eq[0]/r_variables[0])

def get_coeff(expr):
    return expr.as_terms()[0][0][1][0][0] + 1j*expr.as_terms()[0][0][1][0][1]

def discard_small_coeffs(expr, tolerance = 1e-15):
    terms = sy.Add.make_args(expr)
    newexpr = 0
    for t in terms:
        if np.abs(get_coeff(t)) > tolerance:
            newexpr += t
    return newexpr


# def estimate_poles_and_residues(p, q, tol = 1e-14):
#     """
#     Estimate the poles and residues of a rational function p/q.
#     Parameters
#     ----------
#     p : sympy expression
        
#     poles = q.roots()

#     t = max(tol, 1e-7)
#     residues = t * (p(poles + t) / q(poles + t) - p(poles - t) / q(poles - t)) / 2
#     return poles, residues


def generate_list_of_taylor_approximants(polynomial_expressions, base, max_order = 30):
    """
    Generate a list of Taylor approximants from a list of polynomial expressions.
    Parameters
    ----------
    polynomial_expressions : list of sympy expressions
        List of polynomial expressions to generate Taylor approximants from.
    base : list
        List of (complex) sympy variables.
    max_order : int
        Maximum order of the Taylor approximants.
    Returns
    -------
    list_of_taylor : list of TaylorSeries objects
        List of Taylor approximants.
    """
    list_of_taylor = []
    for p in polynomial_expressions:
        coeff = matlab_integration.extract_coefficients(p[0], base, max_order)
        list_of_taylor.append(approximant.TaylorSeries(coeff, max_order, base = base))
    return list_of_taylor


def generate_list_of_pade_approximants(list_polynomial, base,
                                        order_num=3,
                                          order_denom=3,
                                            use_robust = False):
    """
    Generate a list of Pade approximants from a list of polynomials.
    Parameters
    ----------
    list_polynomial : list of TaylorSeries objects
        List of polynomials to generate Pade approximants from.
    order_num : int
        Order of the numerator polynomial.
    order_denom : int
        Order of the denominator polynomial.
    base : list
        List of (complex) sympy variables.
    use_robust : bool
        Whether to use the robust Pade approximant. Default is False.
    Returns
    -------
    list_pade : list
        List of Pade approximants.
    """
    list_pade = []
    for p in list_polynomial:
        pad = approximant.PadeApproximant(order_numerator=order_num,
                                                         order_denominator=order_denom,
                                                         base = base)
        pad.initialize_from_taylor(p.coefficients, use_robust = use_robust)
        list_pade.append(pad)
    return list_pade




def compute_polar_reduced_dyn_Taylor(reduced_dynamics_expressions, 
                                     base, max_order = 30, tolerance = 1e-10):
    """
    Given a list of reduced dynamics expressions, return the reduced dynamics in polar form to a given order.
    Parameters
    ----------
    reduced_dynamics_expressions : list of sympy expressions
        List of reduced dynamics expressions to generate Taylor approximants from.
    base : list
        List of (complex) sympy variables.
    max_order : int
        Maximum order of the Taylor approximants.
    tolerance : float
        Tolerance for small coefficients. Default is 1e-10. if None, no coefficients are discarded.
    Returns
    -------
    red_dynamics : TaylorSeries object 
        reduced dynamics up to a given order.
    frequency_Taylor : TaylorSeries object
        Taylor series of the frequency, i.e., the theta_dot dynamics
    damping_Taylor : TaylorSeries object
        Taylor series of the rdot dynamics
    damping_curve_taylor : callable
        lambda function for the damping curve -rdot/r
    radial_variables : list
        List of radial variables.
    angle_variables : list
        List of angular variables.
    """
    red_dynamics = []
    for r in reduced_dynamics_expressions:
        coeff = matlab_integration.extract_coefficients(r[0], base, max_order)
        red_dynamics.append(approximant.TaylorSeries(coeff, max_order, base = base))
    if tolerance is not None:
        for r in red_dynamics:
            r.polynomial = discard_small_coeffs(r.polynomial, tolerance = tolerance)
    radial_variables, angle_variables, r_equations, phi_equations = convert_to_polar(base, [r.polynomial for r in red_dynamics])
    
    coeff = matlab_integration.extract_coefficients_1d(phi_equations[0], radial_variables, max_order)
    frequency_Taylor = approximant.TaylorSeries(coeff, max_order, base = radial_variables)
    coeff = matlab_integration.extract_coefficients_1d(r_equations[0], radial_variables, max_order)
    damping_Taylor = approximant.TaylorSeries(coeff, max_order, base = radial_variables)
    damping_curve_taylor_ = sy.lambdify(radial_variables[0], -r_equations[0]/radial_variables[0], 'numpy')
    damping_curve_taylor = lambda x : np.real(damping_curve_taylor_(x))
    return red_dynamics, frequency_Taylor, damping_Taylor, damping_curve_taylor, radial_variables, angle_variables

def compute_polar_reduced_dyn_pade(frequency,
                                    damping,
                                    radial_variables,
                                      order_num = 3,
                                        order_denom = 3,
                                          use_robust = False):
    """
    Given a list of reduced dynamics expressions, return the reduced dynamics in polar form to a given order.
    Parameters
    ----------
    frequency : TaylorSeries object
        Frequency expression to generate Pade approximants from.
    damping : TaylorSeries object
        Damping expression to generate Pade approximants from.
    radial_variables : list
        List of radial variables.
    order_num : int
        Order of the numerator polynomial.
    order_denom : int
        Order of the denominator polynomial.
    use_robust : bool
        Whether to use the robust Pade approximant. Default is False.
    Returns
    -------
    frequency_Pade : PadeApproximant object
        Pade approximant of the frequency.
    damping_Pade : PadeApproximant object
        Pade approximant of the damping.
    """
    frequency_Pade = approximant.PadeApproximant(order_denominator=order_num, order_numerator=order_denom, base = radial_variables)
    damping_Pade = approximant.PadeApproximant(order_denominator=order_num, order_numerator=order_denom, base = radial_variables)
    damping_Pade.initialize_from_taylor(damping.coefficients, use_robust = use_robust)
    frequency_Pade.initialize_from_taylor(frequency.coefficients, use_robust = use_robust)
    return frequency_Pade, damping_Pade



def get_resp_at_r(r, idx, parametrization):
    """
    Given a list of TaylorSeries or Pade approximants, return the maximal response of a given observable (idx) at a given radius r. The periodic orbit is assumed to be r*e^{iphi}. 
    Parameters
    ----------
    r : float
        Radius at which to evaluate the response.
    idx : int
        Index of the Pade approximant to evaluate.
    parametrization : list
        List of TaylorSeries or Pade approximants.
    Returns
    -------
    response : float
        Maximal response at the given radius.
    """
    phi_sample = np.linspace(0, 2*np.pi, 100)
    z1 = r * np.exp(1j*phi_sample)
    zz = np.vstack((z1, np.conjugate(z1))).reshape(2,-1)
    response = np.real(parametrization[idx].evaluate(zz.T))
    return np.max(np.abs(response))

