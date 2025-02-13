import taylor_to_pade
from taylor_to_pade.approximant import TaylorSeries, PadeApproximant
from taylor_to_pade.utils import taylor_approximation_2d
import sympy as sy
import numpy as np

def test_taylor1d():
    x = sy.Symbol('x')
    f = sy.sin(x)
    taylor = TaylorSeries([0,1,0,-1/6,0,1/120, 0], 6, base=[x])
    true = sy.series(f, x, 0, 6).removeO()
    true_ = sy.lambdify(x, true, 'numpy')
    test_ = np.linspace(0, 0.05, 20)
    #print(np.sin(test_))

    #print(taylor.evaluate(test_))
    assert np.allclose(
        taylor.evaluate(test_.reshape(-1,1)),
          true_(test_),rtol = 1e-12)
    

def test_taylor2d():
    x, y = sy.symbols('x y')
    f = sy.sin(x)*sy.cos(y)
    x0 = 0
    y0 = 0
    # Compute the Taylor approximation
    taylor_approx, coeffs = taylor_approximation_2d(f, x0, y0, 15)
    coeffs = np.array(coeffs, dtype = float)
    taylor = TaylorSeries(coeffs, 5, base=[x,y])
    true_ = sy.lambdify([x,y], f, 'numpy')
    xx = np.linspace(-0.1, 0.1, 30)
    yy = np.linspace(-0.1, 0.1, 30)
    XX, YY = np.meshgrid(xx, yy)
    test_points = np.vstack((XX.ravel(), YY.ravel())).T
    true_values = true_(test_points[:,0], test_points[:,1])
    TT= taylor.evaluate(test_points)
    assert np.allclose(
        TT,
          true_values,rtol = 1e-12)

def test_pade1d():
    x = sy.Symbol('x')
    f = sy.sin(x)
    taylor = TaylorSeries([0,1,0,-1/6,0,1/120, 0], 6, base=[x])
    pade = PadeApproximant()
    pade.initialize_from_taylor([0,1,0,-1/6,0,1/120,0])
    true = sy.series(f, x, 0, 6).removeO()
    true_ = sy.lambdify(x, true, 'numpy')
    test_ = np.linspace(0, 0.05, 20)
    assert np.allclose(
        pade.evaluate(test_.reshape(-1,1)),
          true_(test_),rtol = 1e-12)