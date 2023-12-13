from yambopy import *
import copy
from math import sqrt, cos, sin, tan

class LAT():
    def __new__(cls, parameters = {}):
        self = super().__new__(cls)
        
        # Check and set required lattice parameters.
        for p in self.required_parameters:
            try: parameters[p]
            except KeyError:
                print(f"{self.name} lattice needs parameter: {p}")
                raise

        self.a = parameters.get('a', None)
        self.b = parameters.get('b', None)
        self.c = parameters.get('c', None)
        self.alpha = parameters.get('alpha', None)
        self.beta = parameters.get('beta', None)
        self.gamma = parameters.get('gamma', None)

        for param, value in parameters.items():
            if any(param == p for p in ['a', 'b', 'c']):
                if value <= 0:
                    raise ValueError(f"{param} parameter must be positive.")

        return self

    def info(self):
        print(f"Lattice type: {self.name}")
        print(f"Variation: {self.variation}")
        print(f"Required parameters: {self.required_parameters}")

        parameters = {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        }
        for key, value in parameters.items():
            if value != None:
                print(f"{key} = {value}")

        print(f"Primitive lattice:\n{self.lattice}")
        print(f"Symmetry points: {self.symmetry_points}")
        print(f"Default path: {self.default_path}")

class CUB(LAT):

    name = 'Cubic'
    
    required_parameters = ['a']
    
    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters) 

        self.variation = 'CUB'
        
        self.lattice = self.a * np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        self.symmetry_points = {
            'Gamma': [0, 0, 0],
            'M': [1/2, 1/2, 0],
            'R': [1/2, 1/2, 1/2],
            'X': [0, 1/2, 0]
        }
        
        self.default_path = [['Gamma', 'X', 'M', 'Gamma', 'R', 'X'], ['M', 'R']]

        return self

class FCC(LAT):
    
    name = 'Face-centered cubic'
    
    required_parameters = ['a']
    
    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        self.variation = 'FCC'
        
        self.lattice = (self.a / 2) * np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        self.symmetry_points = {
            'Gamma': [0, 0, 0],
            'K': [3/8, 3/8, 3/4],
            'L': [1/2, 1/2, 1/2],
            'U': [5/8, 1/4, 5/8],
            'W': [1/2, 1/4, 3/4],
            'X': [1/2, 0, 1/2]
        }
        
        self.default_path = [['Gamma', 'X', 'W', 'K', 'Gamma', 'L', 'U', 'W', 'L', 'K'], ['U', 'X']]
    
        return self

class BCC(LAT):
    
    name = 'Body-centered cubic'
    
    required_parameters = ['a']
    
    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        self.variation = 'BCC'
        
        self.lattice = (self.a / 2.0) * np.array([
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ])
        
        self.symmetry_points = {
            'Gamma': [0, 0, 0],
            'H': [1/2, -1/2, 1/2],
            'P': [1/4, 1/4, 1/4],
            'N': [0, 0, 1/2]
        }
        
        self.default_path = [['Gamma', 'H', 'N', 'Gamma', 'P', 'H'], ['P', 'N']]

        return self

class TET(LAT):
    
    name = 'Tetragonal'
    
    required_parameters = ['a', 'c']
    
    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        if self.a != self.c:

            self.variation = 'TET'

            self.lattice = np.array([
                [self.a, 0, 0],
                [0, self.a, 0],
                [0, 0, self.c]
            ])
        
            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'A': [1/2, 1/2, 1/2],
                'M': [1/2, 1/2, 0],
                'R': [0, 1/2, 1/2],
                'X': [0, 1/2, 0],
                'Z': [0, 0, 1/2]
            }
        
            self.default_path = [['Gamma', 'X', 'M', 'Gamma', 'Z', 'R', 'A', 'Z'], ['X', 'R'], ['M', 'A']]
        
            return self

        else:
            print('TET -> CUB')
            return CUB({'a': self.a})


class BCT(LAT):
    
    name = 'Body-centered tetragonal'
    
    required_parameters = ['a', 'c']
    
    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        self.lattice = np.array([
            [-self.a / 2, self.a / 2, self.c / 2],
            [self.a / 2, -self.a / 2, self.c / 2],
            [self.a / 2, self.a / 2, -self.c / 2]
        ])

        if self.c < self.a:

            self.variation = 'BCT1'
            
            eta = (1 + self.c**2 / self.a**2) / 4

            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'M': [-1/2, 1/2, 1/2],
                'N': [0, 1/2, 0],
                'P': [1/4, 1/4, 1/4],
                'X': [0, 0, 1/2],
                'Z': [eta, eta, -eta],
                'Z1': [-eta, 1 - eta, eta]
            }

            self.default_path = [['Gamma', 'X', 'M', 'Gamma', 'Z', 'P', 'N', 'Z1', 'M'], ['X', 'P']]
            
            return self

        elif self.c > self.a:

            self.variation = 'BCT2'

            eta = (1 + self.a**2 / self.c**2) / 4
            zeta = self.a**2 / (2 * self.c**2)

            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'N': [0, 1/2, 0],
                'P': [1/4, 1/4, 1/4],
                'Sigma': [-eta, eta, eta],
                'Sigma1': [eta, 1 - eta, -eta],
                'X': [0, 0, 1/2],
                'Y': [-zeta, zeta, 1/2],
                'Y1': [1/2, 1/2, -zeta],
                'Z': [1/2, 1/2, -1/2]
            }

            self.default_path = [['Gamma', 'X', 'Y', 'Sigma', 'Gamma', 'Z', 'Sigma1', 'N', 'P', 'Y1', 'Z'], ['X', 'P']]

            return self

        else:
            print('BCT -> BCC')
            return BCC({'a': self.a})

class ORC(LAT):
    
    name = 'Orthorhombic'
    
    required_parameters = ['a', 'b', 'c']
    
    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        if self.a != self.b and self.b != self.c and self.c != self.a:

            if not (self.a < self.b and self.b < self.c):
                raise ValueError("Order of parameters: a < b < c")
        
            self.variation = 'ORC'
 
            self.lattice = np.array([
                [self.a, 0, 0],
                [0, self.b, 0],
                [0, 0, self.c]
            ])

            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'R': [1/2, 1/2, 1/2],
                'S': [1/2, 1/2, 0],
                'T': [0, 1/2, 1/2],
                'U': [1/2, 0, 1/2],
                'X': [1/2, 0, 0],
                'Y': [0, 1/2, 0],
                'Z': [0, 0, 1/2]
            }

            self.default_path = [['Gamma', 'X', 'S', 'Y', 'Gamma', 'Z', 'U', 'R', 'T', 'Z'], ['Y', 'T'], ['U', 'X'], ['S', 'R']]

            return self

        elif self.a == self.b and self.b != self.c:
            print('ORC -> TET')
            return TET({'a': self.a, 'c': self.c})
        elif self.a == self.c and self.c != self.b:
            print('ORC -> TET')
            return TET({'a': self.a, 'c': self.b})
        elif self.b == self.c and self.c != self.a:
            print('ORC -> TET')
            return TET({'a': self.b, 'c': self.a})
        else: # self.a == self.b and self.b == self.c
            print('ORC -> CUB')
            return CUB({'a': self.a})

class ORCF(LAT):
    
    name = 'Face-centered orthorhombic'
    
    required_parameters = ['a', 'b', 'c']

    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        if self.a != self.b and self.b != self.c and self.c != self.a:

            if not (self.a < self.b and self.b < self.c):
                raise ValueError("Order of parameters: a < b < c")

            self.lattice = np.array([
                [0, self.b / 2, self.c / 2],
                [self.a / 2, 0, self.c / 2],
                [self.a / 2, self.b / 2, 0]
            ])

            expression = 1 / self.a**2 - 1 / self.b**2 - 1 / self.c**2

            if expression > 0:

                self.variation = 'ORCF1'
                self.default_path = [['Gamma', 'Y', 'T', 'Z', 'Gamma', 'X', 'A1', 'Y'], ['T', 'X1'], ['X', 'A', 'Z'], ['L', 'Gamma']]

            elif expression < 0:

                self.variation = 'ORCF2'

                eta = (1 + self.a**2 / self.b**2 - self.a**2 / self.c**2) / 4
                phi = (1 + self.c**2 / self.b**2 - self.c**2 / self.a**2) / 4
                delta = (1 + self.b**2 / self.a**2 - self.b**2 / self.c**2) / 4.
                
                self.symmetry_points = {
                    'Gamma': [0, 0, 0],
                    'C': [1/2, 1/2 - eta, 1 - eta],
                    'C1': [1/2, 1/2 + eta, eta],
                    'D': [1/2 - delta, 1/2, 1 - delta],
                    'D1': [1/2 + delta, 1/2, delta],
                    'L': [1/2, 1/2, 1/2],
                    'H': [1 - phi, 1/2 - phi, 1/2],
                    'H1': [phi, 1/2 + phi, 1/2],
                    'X': [0, 1/2, 1/2],
                    'Y': [1/2, 0, 1/2],
                    'Z': [1/2, 1/2, 0]
                }

                self.default_path = [['Gamma', 'Y', 'C', 'D', 'X', 'Gamma', 'Z', 'D1', 'H', 'C'], ['C1', 'Z'], ['X', 'H1'], ['H', 'Y'], ['L', 'Gamma']]
            else: # expression == 0

                self.variation = 'ORCF3'
                self.default_path = [['Gamma', 'Y', 'T', 'Z', 'Gamma', 'X', 'A1', 'Y'], ['X', 'A', 'Z'], ['L', 'Gamma']]

            if self.variation == 'ORCF1' or self.variation == 'ORCF3':

                zeta = (1 + self.a**2 / self.b**2 - self.a**2 / self.c**2) / 4
                eta = (1 + self.a**2 / self.b**2 + self.a**2 / self.c**2) / 4
                
                self.symmetry_points = {
                    'Gamma': [0, 0, 0],
                    'A': [1/2, 1/2 + zeta, zeta],
                    'A1': [1/2, 1/2 - zeta, 1 - zeta],
                    'L': [1/2, 1/2, 1/2],
                    'T': [1, 1/2, 1/2],
                    'X': [0, eta, eta],
                    'X1': [1, 1 - eta, 1 - eta],
                    'Y': [1/2, 0, 1/2],
                    'Z': [1/2, 1/2, 0]
                }

            return self

        elif self.a == self.b and self.b != self.c:
            print('ORCF -> BCT')
            return BCT({'a': self.a, 'c': self.c})
        elif self.a == self.c and self.c != self.b:
            print('ORCF -> BCT')
            return BCT({'a': self.a, 'c': self.b})
        elif self.b == self.c and self.c != self.a:
            print('ORCF -> BCT')
            return BCT({'a': self.b, 'c': self.a})
        else: # self.a == self.b and self.b == self.c
            print('ORCF -> FCC')
            return FCC({'a': self.a})
 
class ORCI(LAT):
    
    name = 'Body-centered orthorhombic'
    
    required_parameters = ['a', 'b', 'c']

    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        if self.a != self.b and self.b != self.c and self.c != self.a:

            if not (self.a < self.b and self.b < self.c):
                raise ValueError("Order of parameters: a < b < c")

            self.lattice = np.array([
                [-self.a / 2, self.b / 2, self.c / 2],
                [self.a / 2, -self.b / 2, self.c / 2],
                [self.a / 2, self.b / 2, -self.c / 2]
            ])

            self.variation = 'ORCI'

            zeta = (1 + self.a**2 / self.c**2) / 4
            eta = (1 + self.b**2 / self.c**2) / 4
            delta = (self.b**2 - self.a**2) / (4 * self.c**2)
            mu = (self.a**2 + self.b**2) / (4 * self.c**2)

            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'L': [-mu, mu, 1/2 - delta],
                'L1': [mu, -mu, 1/2 + delta],
                'L2': [1/2 - delta, 1/2 + delta, -mu],
                'R': [0, 1/2, 0],
                'S': [1/2, 0, 0],
                'T': [0, 0, 1/2],
                'W': [1/4, 1/4, 1/4],
                'X': [-zeta, zeta, zeta],
                'X1': [zeta, 1 - zeta, -zeta],
                'Y': [eta, -eta, eta],
                'Y1': [1 - eta, eta, -eta],
                'Z': [1/2, 1/2, -1/2]
            }

            self.default_path = [['Gamma', 'X', 'L', 'T', 'W', 'R', 'X1', 'Z', 'Gamma','Y', 'S', 'W'], ['L1', 'Y'], ['Y1', 'Z']]

            return self

        elif self.a == self.b and self.b != self.c:
            print('ORCI -> BCT')
            return BCT({'a': self.a, 'c': self.c})
        elif self.a == self.c and self.c != self.b:
            print('ORCI -> BCT')
            return BCT({'a': self.a, 'c': self.b})
        elif self.b == self.c and self.c != self.a:
            print('ORCI -> BCT')
            return BCT({'a': self.b, 'c': self.a})
        else: # self.a == self.b and self.b == self.c
            print('ORCI -> BCC')
            return BCC({'a': self.a})

class ORCC(LAT):
    
    name = 'Base-centered orthorhombic'
    
    required_parameters = ['a', 'b', 'c']

    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        if self.a != self.b:

            if self.a > self.b:
                raise ValueError("Order of parameters: a < b")

            self.lattice = np.array([
                [self.a / 2, -self.b / 2, 0],
                [self.a / 2, self.b / 2, 0],
                [0, 0, self.c]
            ])

            self.variation = 'ORCC'

            zeta = (1 + self.a**2 / self.b**2) / 4

            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'A': [zeta, zeta, 1/2],
                'A1': [-zeta, 1 - zeta, 1/2],
                'R': [0, 1/2, 1/2],
                'S': [0, 1/2, 0],
                'T': [-1/2, 1/2, 1/2],
                'X': [zeta, zeta, 0],
                'X1': [-zeta, 1 - zeta, 0],
                'Y': [-1/2, 1/2, 0],
                'Z': [0, 0, 1/2]
            }

            self.default_path = [['Gamma', 'X', 'S', 'R', 'A', 'Z', 'Gamma', 'Y', 'X1', 'A1', 'T', 'Y'], ['Z', 'T']]

            return self

        elif self.a == self.b and self.b == sqrt(2) * self.c:
            print('ORCC -> CUB')
            return CUB({'a': self.a})
        else:
            print('ORCC -> TET')
            return TET({'a': self.a, 'c': self.c})

class HEX(LAT):

    name = 'Hexagonal'

    required_parameters = ['a', 'c']

    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)

        self.lattice = np.array([
            [self.a / 2, -self.a * sqrt(3), 0],
            [self.a / 2, self.a * sqrt(3), 0],
            [0, 0, self.c]
        ])

        self.variation = 'HEX'

        self.symmetry_points = {
            'Gamma': [0, 0, 0],
            'A': [0, 0, 1/2],
            'H': [1/3, 1/3, 1/2],
            'K': [1/3, 1/3, 0],
            'L': [1/2, 0, 1/2],
            'M': [1/2, 0, 0]
        }

        self.default_path = [['Gamma', 'M', 'K', 'Gamma', 'A', 'L', 'H', 'A'], ['L', 'M'], ['K', 'H']]

        return self

class RHL(LAT):

    name = 'Rhombohedral'

    required_parameters = ['a', 'alpha']

    def __new__(cls, parameters = {}):
        self = super().__new__(cls, parameters)
        
        alpha = np.radians(self.alpha)

        self.lattice = np.array([
            [self.a * cos(alpha / 2), -self.a * sin(alpha / 2), 0],
            [self.a * cos(alpha / 2), self.a * sin(alpha / 2), 0],
            [self.a * cos(alpha) / cos(alpha / 2), 0, self.a * sqrt(1 - cos(alpha)**2 / cos(alpha / 2)**2)]
        ])

        if self.alpha == 60:
            print('RHL -> FCC')
            return FCC({'a': self.a})
        elif cos(alpha) == -1/3:
            print('RHL -> BCC')
            return BCC({'a': self.a})
        elif self.alpha < 90:

            self.variation = 'RHL1'

            eta = (1 + 4 * cos(alpha)) / (2 + 4 * cos(alpha))
            nu = 3/4 - eta / 2

            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'B': [eta, 1/2, 1 - eta],
                'B1': [1/2, 1 - eta, eta - 1],
                'F': [1/2, 1/2, 0],
                'L': [1/2, 0, 0],
                'L1': [0, 0, -1/2],
                'P': [eta, nu, nu],
                'P1': [1 - nu, 1 - nu, 1 - eta],
                'P2': [nu, nu, eta - 1],
                'Q': [1 - nu, nu, 0],
                'X': [nu, 0, -nu],
                'Z': [1/2, 1/2, 1/2]
            }

            self.default_path = [['Gamma', 'L', 'B1'], ['B', 'Z', 'Gamma', 'X'], ['Q', 'F', 'P1', 'Z'], ['L', 'P']]

            return self

        elif self.alpha > 90:

            self.variation = 'RHL2'

            eta = 1 / (2 * tan(alpha / 2)**2)
            nu = 3/4 - eta / 2

            self.symmetry_points = {
                'Gamma': [0, 0, 0],
                'F': [1/2, -1/2, 0],
                'L': [1/2, 0, 0],
                'P': [1 - nu, -nu, 1 - nu],
                'P1': [nu, nu - 1, nu - 1],
                'Q': [eta, eta, eta],
                'Q1': [1 - eta, -eta, -eta],
                'Z': [1/2, -1/2, 1/2]
            }

            self.default_path = [['Gamma', 'P', 'Z', 'Q', 'Gamma', 'F', 'P1', 'Q1', 'L', 'Z']]

            return self

        else: # self.alpha == 90:
            print('RHL -> CUB')
            return CUB({'a': self.a})
 
class BrillouinZone():
    """
    Constructs interpolated paths in any of the existing Brillouin zones.
    """
    def __init__(self, code = '', parameters = {}):
        """
        Initializes the Brillouin zone of a selected lattice type,
        with given required parameters.
        """
        # Select lattice type

        if code == 'CUB':
            self.bz = CUB(parameters)
        elif code == 'FCC':
            self.bz = FCC(parameters)
        elif code == 'BCC':
            self.bz = BCC(parameters)
        elif code == 'TET':
            self.bz = TET(parameters)
        elif code == 'BCT':
            self.bz = BCT(parameters)
        elif code == 'ORC':
            self.bz = ORC(parameters)
        elif code == 'ORCF':
            self.bz = ORCF(parameters)
        elif code == 'ORCI':
            self.bz = ORCI(parameters)
        elif code == 'ORCC':
            self.bz = ORCC(parameters)
        elif code == 'HEX':
            self.bz = HEX(parameters)
        elif code == 'RHL':
            self.bz = RHL(parameters)
        else:
            raise ValueError(f"Wrong lattice type: {code}")

        # Obtain reciprocal lattice vectors
        
        self.rlattice = rec_lat(self.bz.lattice)

        # Obtain symmetry points in cartesian coordinates

        symmetry_points_car_values = red_car(list(self.bz.symmetry_points.values()), self.rlattice)
        self.symmetry_points_car = dict(zip(self.bz.symmetry_points.keys(), symmetry_points_car_values))

        # Initialize path with default BZ path

        self.set_path()

    def set_path(self, path = None, ipoints = 100, in_coord_type = 'red'):
        """
        Sets the path, given list of subpaths and number of interpolating points. Each subpath represents a list
        of k-points, given by labels 'X' or coordinates and label [[Kx, Ky, Kz], 'X'], expressed in cartesian or
        reduced coordinates.
        """
        # Check input coordinates type

        if in_coord_type != 'car' and in_coord_type != 'red':
            raise ValueError(f"Unrecognized input coordinates type: {in_coord_type}")

        self.in_coord_type = in_coord_type

        # Check path structure and labels

        if path == None:
            path = self.bz.default_path

        self.check_path_structure(path)
        self.check_path_labels(path)
        
        self.path = path
        self.path_car = self.get_path_cartesian()

        # Check and parse interpolating points
        
        self.check_interpolating_points_structure(ipoints, path)

        if isinstance(ipoints, int):
            self.nipoints = ipoints
            self.interpolating_points = self.get_interpolating_points()
        else:
            self.interpolating_points = ipoints
            self.nipoints = self.get_number_interpolating_points()

        # Reset interpolated path in cartesian coordinates

        self.interpolated_path_car = None

        # Set legacy path

        self.set_legacy_path()

    def get_interpolated_path(self, out_coord_type = 'car', qe_out = False):
        """
        Interpolates path and expresses it in cartesian or reduced coordinates. Optionally, output in QE format.
        """
        # Check output coordinates type

        if out_coord_type != 'car' and out_coord_type != 'red':
            raise ValueError(f"Unrecognized output coordinates type: {out_coord_type}")

        # Interpolate path: only if not yet done

        if not isinstance(self.interpolated_path_car, np.ndarray):
            ipath = []

            for n in range(len(self.path_car)):
                for i in range(len(self.path_car[n]) - 1):
                    ipath.append(self.path_car[n][i])
                
                    delta21 = self.path_car[n][i + 1] - self.path_car[n][i]
                    
                    nipoints = self.interpolating_points[n][i]
                    
                    for j in range(nipoints):
                        ipath.append(self.path_car[n][i] + delta21 * (j + 1) / (nipoints + 1))

                ipath.append(self.path_car[n][-1])

            ipath = np.array(ipath)
            self.interpolated_path_car = ipath
        else:
            ipath = self.interpolated_path_car

        # Express it in reduced coordinates?
        
        if out_coord_type == 'red':
            ipath = car_red(ipath, self.rlattice)

        # QE output format: [ [Kx, Ky, Kz, 1], ... ]?
        
        if qe_out:
            qepath = []
            for i in range(len(ipath)):
                qepath.append(np.concatenate((ipath[i], [int(1)])))
            ipath = np.array(qepath)

        return ipath

    def get_interpolating_points(self):
        """
        Given path in cartesian coordinates, spread given number of k-points proportionally to inter-point lengths
        """
        # Obtain distances between consecutive points on path

        distances = []

        for subpath in self.path_car:
            kpoints = np.array(subpath)
            subpath_distances = []
            for i in range(1, len(kpoints)):
                subpath_distances.append(np.linalg.norm(kpoints[i - 1] - kpoints[i]))
            distances.append(np.array(subpath_distances))

        # Obtain total path length

        path_length = 0
        for subpath_distances in distances:
            path_length += np.sum(subpath_distances)

        # Obtain number of interpolated k-points on each subpath

        interpolating_points = [[int(self.nipoints * distance / path_length) for distance in subdistances] for subdistances in distances]

        return interpolating_points

    def get_number_interpolating_points(self):
        """
        Obtain number of interpolating k-points as given by interpolating points
        """
        return sum([i for subipoints in self.interpolating_points for i in subipoints])

    def get_path_cartesian(self):
        """
        Express path in cartesian coordinates.
        """
        # Transform labels and k-points to cartesian coordinates

        path_car = []
        
        for subpath in self.path:
            subpath_car = []
            for element in subpath:
                if isinstance(element, str):
                    subpath_car.append(self.symmetry_points_car[element])
                elif isinstance(element, list):
                    if self.in_coord_type == 'car':
                        subpath_car.append(element[0])
                    elif self.in_coord_type == 'red':
                        subpath_car.append(red_car([element[0]], self.rlattice)[0])
            path_car.append(subpath_car)

        return path_car

    def get_path_reduced(self):
        """
        Express path in reduced coordinates.
        """
        # Transform labels and k-points to reduced coordinates

        path_red = []
        
        for subpath in self.path:
            subpath_red = []
            for element in subpath:
                if isinstance(element, str):
                    subpath_red.append(self.bz.symmetry_points[element])
                elif isinstance(element, list):
                    if self.in_coord_type == 'car':
                        subpath_red.append(car_red([element[0]], self.rlattice)[0])
                    elif self.in_coord_type == 'red':
                        subpath_red.append(element[0])
            path_red.append(subpath_red)

        return path_red

    def get_path_labels(self):
        """
        Obtain list of labels of k-points on path 
        """
        labels = []

        for subpath in self.path:
            for element in subpath:
                if isinstance(element, str):
                    labels.append(element)
                elif isinstance(element, list):
                    labels.append(element[1])

        return labels

    def get_intervals(self):
        """
        Obtain intervals
        """
        intervals = []
        
        for subpath in self.interpolating_points:
            for ipoints in subpath:
                intervals.append(ipoints + 1)
            intervals.append(1)
        intervals.pop()

        return intervals

    def set_legacy_path(self):
        """"
        Set path in reduced coordinates, labels and intervals
        """
        # Obtain path in reduced coordinates

        path_red = self.get_path_reduced()
        
        # Flatten path

        path_red = [kpoint for subpath in path_red for kpoint in subpath]

        # Legacy data structures

        self.kpoints = np.array(path_red)
        self.klabels = self.get_path_labels()
        self.intervals = self.get_intervals()

    def get_legacy_path(self):
        """
        Construct and return a legacy Path object
        """
        # Construct list of k-points and labels

        klist = [list(k) for k in zip(self.kpoints.tolist(), self.klabels)]

        return Path(klist, self.intervals)

    def as_dict(self):
        """
        LEGACY: Obtain path as dictionary
        """
        d = {
            'kpoints': self.kpoints.tolist(),
            'klabels': self.klabels,
            'intervals': self.intervals
        }
        return d

    def distances(self):
        """
        LEGACY: Obtain list of distances between first and consecutive k-points on path.
        """
        distance = 0
        distances = []
        kpoint1 = self.kpoints[0]

        for kpoint2 in self.kpoints:
            distance += np.linalg.norm(kpoint1 - kpoint2)
            distances.append(distance)
            kpoint1 = kpoint2
    
        return distances

    def set_xticks(self, ax):
        """
        LEGACY: Set x-axis ticks and labels
        """
        ax.set_xticks(self.distances())
        ax.set_xticklabels(self.klabels)

    def __iter__(self):
        """
        LEGACY: Iterator???
        """
        return iter(zip(self.kpoints, self.klabels, self.distances()))

    def get_klist(self):
        """
        LEGACY: Output in the format of QE == [ [Kx, Ky, Kz, 1], ... ]
        """
        return self.get_interpolated_path('red', True)

    def get_indexes(self):
        """
        LEGACY: Obtain index of each k-point on path
        """
        indexes = []
        index = 0

        for n, interval in enumerate(self.intervals):
            indexes.append([index, self.klabels[n]])
            index += interval
        indexes.append([index, self.klabels[-1]])

        return indexes

    def check_path_structure(self, path):
        """
        Check if given path is a valid list of subpaths (lists) and if enough points (>1), etc.
        """
        if not isinstance(path, list):
            raise ValueError("Path must be list of subpaths.")

        if all(isinstance(subpath, list) for subpath in path):
            for subpath in path:
                for element in subpath:
                    if not isinstance(element, list) and not isinstance(element, str):
                        raise ValueError(f"K-Point must be specified by label 'X' or by coordinates with label [[Kx, Ky, Kz], 'X']: {element}")
                    if isinstance(element, list):
                        if len(element) != 2:
                            raise ValueError(f"Unrecognized k-point specification: {element}")
                        if not isinstance(element[0], list) or not isinstance(element[1], str):
                            raise ValueError(f"Unrecognized k-point specification: {element}")
                        if len(element[0]) != 3: 
                            raise ValueError(f"K-Point must have 3 coordinates: {element[0]}")
                        if not all(isinstance(coord, (float, int)) for coord in element[0]):
                            raise ValueError(f"K-Point coordinate must be of type float or int: {element[0]}")
                if len(subpath) < 2:
                    raise ValueError(f"Not enough elements in subpath: {subpath}")
        else:
            raise ValueError("Path must be list of subpaths.")

    def check_path_labels(self, path):
        """
        Check if given path contains valid high symmetry points labels of selected Brillouin zone.
        """
        for subpath in path:
            for element in subpath:
                if isinstance(element, str):
                    try: self.bz.symmetry_points[element]
                    except KeyError as err:
                        print(err.args[0], 'is not a high symmetry point of', self.bz.name, 'lattice.')
                        raise

    def check_interpolating_points_structure(self, ipoints, path):
        """
        Check if given interpolating points are valid and consistent with given path
        """
        # Check structure

        if not isinstance(ipoints, list) and not isinstance(ipoints, int):
            raise ValueError("Interpolating k-points must be positive integer number, zero or list of lists.")

        if isinstance(ipoints, int):
            if ipoints < 0:
                raise ValueError("Number of interpolating k-points must be positive integer or zero.")
        elif all(isinstance(subipoints, list) for subipoints in ipoints):
            for subipoints in ipoints:
                if not all(isinstance(i, int) and i >= 0 for i in subipoints):
                    raise ValueError(f"Interpolating k-point number must be an positive integer or zero: {subipoints}")
            # Check consistency between ipoints and path
            nipoints1 = [len(subpath) - 1 for subpath in path]
            nipoints2 = [len(subipoints) for subipoints in ipoints]
            if nipoints1 != nipoints2:
                raise ValueError("Path and interpolating k-points mismatch.")
        else:
            raise ValueError("Interpolating k-points must be positive integer number, zero or list of lists.")
