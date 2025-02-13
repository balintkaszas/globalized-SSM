import sympy as sy
from taylor_to_pade.utils import generate_powers_uptoorder_square, generate_polynomial_from_tensor
from taylor_to_pade.converter import convert_to_pade_1d, convert_to_pade_2d, Coefficient, convert_to_pade_2d_robust

class TaylorSeries:

    def __init__(self, coefficients,
                order,
                base = [sy.Symbol('x')]):
        """ Initialize the Taylor series with the given coefficients. """
        self.base = base # list of variables 
        self.coefficients = coefficients  # tensor of coefficients: [i,j,k,...]
        self.dimension = len(self.base) # number of variables 
        self.order = order
        self.powers = generate_powers_uptoorder_square(self.dimension, self.order)
        self.polynomial = generate_polynomial_from_tensor(self.base, self.powers, self.coefficients)
        self.numpy_function = sy.lambdify(self.base, self.polynomial, 'numpy')
    def evaluate(self, x):
        """ Evaluate the Taylor series at a specific value of x """
        # self.numpy_function() expects self.dimension separate arguments 
        return self.numpy_function(*[x[:,i] for i in range(self.dimension)]) # need to unpack along dimensions
    

class PadeApproximant:
    def __init__(self, 
                 coefficients_numerator = None, 
                 coefficients_denominator = None, 
                 order_numerator = 2,
                 order_denominator = 2,
                 base=[sy.Symbol('x')]):
        """ Initialize the Pade approximant with the given coefficients. """
        self.base = base
        self.coefficients_numerator = coefficients_numerator  # list of coefficients
        self.coefficients_denominator = coefficients_denominator  # list of coefficients
        self.dimension = len(self.base)
        self.numerator = None
        self.denominator = None
        self.ratio = None
        self.numpy_function = None
        self.order_numerator = order_numerator
        self.order_denominator = order_denominator

        if coefficients_numerator is not None:
            self.powers_denominator = generate_powers_uptoorder_square(self.dimension, self.order_denominator)
            self.powers_numerator = generate_powers_uptoorder_square(self.dimension, self.order_numerator)
            self.numerator = generate_polynomial_from_tensor(self.base, self.powers_numerator, self.coefficients_numerator)
            self.denominator = generate_polynomial_from_tensor(self.base, self.powers_denominator, self.coefficients_denominator)
            self.ratio = self.numerator / self.denominator
            self.numpy_function = sy.lambdify(self.base, self.ratio, 'numpy')
        return 
    
    def initialize_from_taylor(self, taylor_coeffs, use_robust = False):
        """ Initialize the Pade approximant from a Taylor series. """
        if self.dimension == 1:
            self.numerator, self.denominator = convert_to_pade_1d(taylor_coeffs, self.order_numerator, self.order_denominator, use_robust  = use_robust)
            self.numpy_function = lambda x: self.numerator(x) / self.denominator(x)
        elif self.dimension == 2:
            N = Coefficient('n', 2)
            D = Coefficient('d', 2)
            if use_robust: 
                self.numerator, self.denominator = convert_to_pade_2d_robust(self.base, taylor_coeffs, self.order_numerator, self.order_denominator, N, D)
            else: 
                self.numerator, self.denominator = convert_to_pade_2d(self.base, taylor_coeffs, self.order_numerator, self.order_denominator, N, D)
            self.ratio = self.numerator / self.denominator
            self.numpy_function = sy.lambdify(self.base, self.ratio, 'numpy')
        else:
            raise NotImplementedError("Pade approximants are only implemented for 1D Taylor series.") 

        return 
    
    def evaluate(self, x):
        """ Evaluate the Taylor series at a specific value of x. """
        if self.numpy_function is None:
            return None
        return self.numpy_function(*[x[:,i] for i in range(self.dimension)]) # need to unpack along dimensions
