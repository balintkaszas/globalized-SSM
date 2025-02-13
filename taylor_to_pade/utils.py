import sympy as sy
from itertools import product
import numpy as np


def generate_tensor(symbols, N):
    # Create an empty list to store each term
    tensor = []
    
    # Generate all combinations of powers including zero up to N
    range_list = [range(N + 1) for _ in symbols]
    
    # Iterate over the Cartesian product of [0, 1, ..., N] for each symbol
    powers = []
    for power in product(*range_list):
        # Only consider combinations where the sum of powers is <= N
        term = sy.prod([s**p for s, p in zip(symbols, power)])
        tensor.append(term)
        powers.append(power)
    return tensor, powers

def generate_powers_uptoorder_square(dimension, M):
    pows = []
    range_list = [range(M + 1) for _ in range(dimension)]
    for power in product(*range_list):
        pows.append([*power])
    return np.array(pows)

def generate_polynomial_from_tensor(variables, powers, coefficients):
    """Given a list of variables, powers, and coefficients, generate a polynomial.
    Args:
        variables (list): list of sympy variables
        powers (array): list of powers of each variable: contains [i,j,k] such that C_{ijk} * x^i * y^j * z^k
        coefficients (array):  tensor of coefficients: C_{ijk}: [i,j,k,...]
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
    # analytical computation of a 2D Taylor expansion around x0, y0
    x, y = sy.symbols('x y')
    taylor_approx = 0#f.subs({x: x0, y: y0})
    
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


def estimate_poles_and_residues(p, q, tol = 1e-14):
    poles = q.roots()

    t = max(tol, 1e-7)
    residues = t * (p(poles + t) / q(poles + t) - p(poles - t) / q(poles - t)) / 2
    return poles, residues
