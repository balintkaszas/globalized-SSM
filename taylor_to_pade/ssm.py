from taylor_to_pade.utils import convert_to_polar
import numpy as np
import sympy as sy  


class SymbolicFunction:
    def __init__(self, variables, polynomial):
        self.variables = variables
        self.polynomial = polynomial
        self.numpy_function = sy.lambdify(self.variables, self.polynomial, 'numpy')

    def evaluate(self, values):
        return self.numpy_function(*values)

class SSM:
    def __init__(self, reduced_coordinates, dimensions, parametrization, chart, reduced_dynamics):
        self.reduced_coordinates = reduced_coordinates
        self.output_dimensions = dimensions
        self.dim_reduced = len(reduced_coordinates)
        self.parametrization = parametrization
        self.chart = chart
        self.normal_form = None # going from NF to reduced coords
        self.inverse_normal_form = None # going from reduced coords to NF
        self.reduced_dynamics = reduced_dynamics
        self.reduced_dynamics_polar = None
        self.reduced_radial = None
        self.reduced_angular = None
        self.backbone = None
        self.damping_curve = None
        
    def compute_reduced_dynamics_polar(self):
        """ Compute the reduced dynamics in polar coordinates. """
        if self.reduced_dynamics_polar is None:
            list_of_equations = [p.polynomial for p in self.reduced_dynamics]
            radial_variables, angle_variables, r_equations, phi_equations = convert_to_polar(self.reduced_coordinates, list_of_equations)
            self.reduced_dynamics_polar = [r_equations, phi_equations]
            self.reduced_radial = radial_variables
            self.reduced_angular = angle_variables
            self.backbone = SymbolicFunction(radial_variables[0], phi_equations[0] )
            self.damping_curve = SymbolicFunction(radial_variables[0], -r_equations[0] / radial_variables[0])
        return
def calibrate_from_amplitude(amplitude, linear_chart, outdof):
    """ Calibrate the forced response curve """
    ind = np.argmax(np.abs(linear_chart[outdof,:])) # e.g., np.abs(x[outdof,:])
    normalized_mat = linear_chart[:,ind] / linear_chart[outdof,ind]
    return amplitude * normalized_mat

def calibrateFRC(SSM, calib_point, calib_freq):
    """ Calibrate the forced response curve """
    calib_in_reduced_coords = SSM.chart(calib_point)
    calib_in_normalform = [p.evaluate(calib_in_reduced_coords.reshape(1,-1)) for p in SSM.inverse_normal_form]
    #print(calib_in_normalform)
    rho_cal = np.abs(calib_in_normalform[0])
    #print(SSM.backbone.evaluate(rho_cal))
    calibration_amplitude = np.sqrt(rho_cal**2*(SSM.backbone.evaluate(rho_cal)-calib_freq)**2 
                                    + rho_cal**2*(SSM.damping_curve.evaluate(rho_cal))**2)
    return calibration_amplitude

def generate_response(rho, SSM, outdof):
    phi_sample = np.linspace(-np.pi, np.pi, 151).reshape(-1,1) # need the reshape, otherwise z1 is has incorrect shape
    z1 = rho * np.exp(1j*phi_sample)
    zz = np.vstack((z1, np.conjugate(z1))).T
    reduced_coord = np.array([p.evaluate(zz) for p in SSM.normal_form])
    #print(SSM.parametrization(reduced_coord))
    resp_real = SSM.parametrization(reduced_coord)[outdof,:]
    return np.abs(resp_real)


def compute_FRC(SSM, calibration):
    OmegaFm = lambda x: SSM.backbone.evaluate(x) - (1./x)*np.sqrt(calibration**2-(x*SSM.damping_curve.evaluate(x))**2)
    OmegaFp = lambda x: SSM.backbone.evaluate(x) + (1./x)*np.sqrt(calibration**2-(x*SSM.damping_curve.evaluate(x))**2)
    return OmegaFm, OmegaFp