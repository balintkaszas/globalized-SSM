from taylor_to_pade.utils import generate_tensor, generate_powers_uptoorder_square, generate_polynomial_from_tensor, taylor_approximation_2d
import sympy as sy
import numpy as np


def test_generate_powers_uptoorder_square():
    pows = generate_powers_uptoorder_square(2, 2)
    assert np.allclose(pows, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])

def test_generate_polynomial_from_tensor1d():
    x = sy.symbols('x')
    variables = [x]
    powers = generate_powers_uptoorder_square(1, 6)
    np.random.seed(0)
    coeffs = np.random.rand(7)
    true = 0
    for i in range(7):
            true += coeffs[i] * x**i

    polynomial = generate_polynomial_from_tensor(variables, powers, coeffs)
    assert polynomial == true


def test_generate_polynomial_from_tensor2d():
    x, y = sy.symbols('x,y')
    variables = [x,y]
    powers = generate_powers_uptoorder_square(2, 6)
    np.random.seed(0)
    coeffs = np.random.rand(7,7)
    true = 0
    for i in range(7):
        for j in range(7):
            true += coeffs[i,j] * x**i * y**j

    polynomial = generate_polynomial_from_tensor(variables, powers, coeffs)
    assert polynomial == true


def test_taylor_approximation_2d():
    x, y = sy.symbols('x y')
    f = sy.sin(x)*sy.cos(y)
    x0 = 0
    y0 = 0
    order = 3

    # Compute the Taylor approximation
    taylor_approx, coeffs = taylor_approximation_2d(f, x0, y0, 15)
    true = float(f.subs([(x,  0.1), (y, 0.1)]))
    assert np.allclose(float(taylor_approx.subs([(x,  0.1), (y, 0.1)])), true)

# if __name__ == "__main__":
#     test_generate_polynomial_from_tensor()
#     print("All tests passed!")