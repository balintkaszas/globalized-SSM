from taylor_to_pade import converter
from taylor_to_pade.approximant import PadeApproximant
from taylor_to_pade.utils import taylor_approximation_2d
import sympy as sy
import numpy as np

def test_convert_to_pade_2d():
    nn = converter.Coefficient('n', 2)
    dd = converter.Coefficient('d', 2)
    nnn = sy.Indexed('nnn', nn.indices[0],nn.indices[1])
    ss,vss = converter.generate_symbols(3, 3, nn, dd)
    
    x, y = sy.symbols('x y')
    f = sy.sin(x)*sy.cos(y)
    pade = PadeApproximant(order_denominator = 3, order_numerator = 3, base = [x,y])
    x0 = 0
    y0 = 0
    # Compute the Taylor approximation
    taylor_approx, coeffs = taylor_approximation_2d(f, x0, y0, 25)
    pade.initialize_from_taylor(coeffs)

    true_ = sy.lambdify([x,y], taylor_approx, 'numpy')
    xx = np.linspace(-0.01, 0.01, 30)
    yy = np.linspace(-0.01, 0.01, 30)
    XX, YY = np.meshgrid(xx, yy)
    test_points = np.vstack((XX.ravel(), YY.ravel())).T
    true_values = true_(test_points[:,0], test_points[:,1])
    TT= pade.evaluate(test_points)
    assert np.allclose(
        TT,
          true_values,rtol = 1e-12)



if __name__ == '__main__':
    test_convert_to_pade_2d()