import sympy as sy
from taylor_to_pade.utils import generate_powers_uptoorder_square, generate_polynomial_from_tensor
from taylor_to_pade.converter import convert_to_pade_1d, convert_to_pade_2d, Coefficient, convert_to_pade_2d_robust

class TaylorSeries:
    """Represents a multivatiate Taylor series."""
    def __init__(self, coefficients,
                order,
                base = [sy.Symbol('x')]):
        """ Initialize the Taylor series with the given coefficients. 
        
        Parameters:
        -------
            coefficients (array): tensor of coefficients: C_{ijk}: [i,j,k,...]
            order (int): order of the Taylor series
            base (list): list of variables. Defaults to a single variable, labelled 'x'.
        """
        self.base = base # list of variables 
        self.coefficients = coefficients  # tensor of coefficients: [i,j,k,...]
        self.dimension = len(self.base) # number of variables 
        self.order = order
        self.powers = generate_powers_uptoorder_square(self.dimension, self.order)
        self.polynomial = generate_polynomial_from_tensor(self.base, self.powers, self.coefficients)
        self.numpy_function = sy.lambdify(self.base, self.polynomial, 'numpy')

    def evaluate(self, x):
        """ Evaluate the Taylor series at a specific value of x. 
        Parameters:
        -------

            x (array): input value(s)

        Returns:
        -------

            array: output value(s)
        """
        # self.numpy_function() expects self.dimension separate arguments 
        return self.numpy_function(*[x[:,i] for i in range(self.dimension)]) # need to unpack along dimensions
    

class PadeApproximant:
    """Represents a multivariate Pade approximant. """
    def __init__(self, 
                 coefficients_numerator = None, 
                 coefficients_denominator = None, 
                 order_numerator = 2,
                 order_denominator = 2,
                 base=[sy.Symbol('x')]):
        """ Initialize the Pade approximant with the given coefficients. The Pade approximant is defined as the ratio of two polynomials. 
        Coefficients don't need to be given at initialization, but can be set later using the initialize_from_taylor method.
        Parameters:
        -------
            coefficients_numerator (array): tensor of coefficients for the numerator: n_{ijk}: [i,j,k,...]
            coefficients_denominator (array): tensor of coefficients for the denominator: d_{ijk}: [i,j,k,...]
            order_numerator (int): order of the numerator polynomial
            order_denominator (int): order of the denominator polynomial
            base (list): list of variables. Defaults to a single variable, labelled 'x'.

        """
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
    
    def initialize_from_taylor(self, taylor_coeffs, use_robust = False):
        """Compute the coefficients of the Pade approximant from a given Taylor series. Only implemented for 1D and 2D Taylor series. The denominator and numerator are stored as separate polynomials and the ratio is computed as the quotient of the two. In addition, the lambdified function is stored in the numpy_function attribute. 
        
        Parameters:
        -------
            taylor_coeffs (array): tensor of coefficients of the Taylor series: C_{ijk}: [i,j,k,...]
            use_robust (bool): whether to use the robust Pade approximant. Defaults to False.
            """
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
        """ Evaluate the Pade approximant at a specific value of x. 
        
        Parameters:
        -------
            x (array): input value(s)
        Returns:
        -------
            array: output value(s)
        """
        if self.numpy_function is None:
            return None
        return self.numpy_function(*[x[:,i] for i in range(self.dimension)]) # need to unpack along dimensions
