from math import sqrt, cos, sin, radians
from ase.cell import Cell
import numpy as np



class BrillouinZone():
    """
    Constructs interpolated paths in any of the existing Brillouin zones.
    Follows QE's classification criterion (ibrav)
    See: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm226
    """

    required_parameters = {
        1: ['a'],
        2: ['a'],
        3: ['a'],
        -3: ['a'],
        4: ['a', 'c'],
        5: ['a', 'gamma'],
        -5: ['a', 'gamma'],
        6: ['a', 'c'],
        7: ['a', 'c'],
        8: ['a', 'b', 'c'],
        9: ['a', 'b', 'c'],
        -9: ['a', 'b', 'c'],
        91: ['a', 'b', 'c'],
        10: ['a', 'b', 'c'],
        11: ['a', 'b', 'b'],
        12: ['a', 'b', 'c', 'gamma'],
        -12: ['a', 'b', 'c', 'beta'],
        13: ['a', 'b', 'c', 'gamma'],
        -13: ['a', 'b', 'c', 'beta'],
        14: ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    }

    def __init__(self, ibrav, parameters = {}):
        """
        Initializes the Brillouin zone of a selected lattice type,
        with given required parameters.
        """

        # Check if valid Bravais-lattice index

        if not ibrav in self.required_parameters:
            raise ValueError(f"ibrav: {ibrav} not supported.")

        # Check required lattice parameters

        for p in self.required_parameters[ibrav]:
            try: parameters[p]
            except KeyError:
                print(f"ibrav: {ibrav} lattice needs parameter: {p}")
                raise

        # Check if a, b, and c parameters are positive nonzero values

        for param, value in parameters.items():
            if any(param == p for p in ['a', 'b', 'c']):
                if value <= 0:
                    raise ValueError(f"{param} parameter must be positive.")

        # Set parameters

        a = parameters.get('a', None)
        b = parameters.get('b', None)
        c = parameters.get('c', None)
        alpha = parameters.get('alpha', None)
        beta = parameters.get('beta', None)
        gamma = parameters.get('gamma', None)

        # Set lattice vectors

        match ibrav:
            # CUB
            case 1:
                v1 = [a, 0, 0]
                v2 = [0, a, 0]
                v3 = [0, 0, a]
            # FCC
            case 2:
                v1 = [-a / 2, 0, a / 2]
                v2 = [0, a / 2, a / 2]
                v3 = [-a / 2, a / 2 , 0]
            # BCC
            case 3:
                v1 = [a / 2, a / 2, a / 2]
                v2 = [-a / 2, a / 2, a / 2]
                v3 = [-a / 2, -a / 2, a / 2]
            case -3:
                v1 = [-a / 2, a / 2, a / 2]
                v2 = [a / 2, -a / 2, a / 2]
                v3 = [a / 2, a / 2, -a / 2]
            # HEX
            case 4:
                v1 = [a, 0, 0]
                v2 = [-a / 2, a * sqrt(3) / 2, 0]
                v3 = [0, 0, c]
            # RHL
            case 5:
                c = cos(radians(gamma))
                tx = sqrt((1 - c) / 2)
                ty = sqrt((1 - c) / 6)
                tz = sqrt((1 + 2 * c) / 3)
                v1 = [a * tx, -a * ty, a * tz]
                v2 = [0, a * 2 * ty, a * tz]
                v3 = [-a * tx, -a * ty, a * tz]
            case -5:
                a /= sqrt(3)
                c = cos(radians(gamma))
                ty = sqrt((1 - c) / 6)
                tz = sqrt((1 + 2 * c) / 3)
                u = tz - 2 * sqrt(2) * ty
                v = tz + sqrt(2) * ty
                v1 = [a * u, a * v, a * v]
                v2 = [a * v, a * u, a * v]
                v3 = [a * v, a * v, a * u]
            # TET
            case 6:
                v1 = [a, 0, 0]
                v2 = [0, a, 0]
                v3 = [0, 0, c]
            # BCT
            case 7:
                v1 = [a / 2, -a / 2, c / 2]
                v2 = [a / 2, a / 2, c / 2]
                v3 = [-a / 2, -a / 2, c / 2]
            # ORC
            case 8:
                v1 = [a, 0, 0]
                v2 = [0, b, 0]
                v3 = [0, 0, c]
            # ORCC
            case 9:
                v1 = [a / 2, b / 2, 0]
                v2 = [-a / 2, b / 2, 0]
                v3 = [0, 0, c]
            case -9:
                v1 = [a / 2, -b / 2, 0]
                v2 = [a / 2, b / 2, 0]
                v3 = [0, 0, c]
            case 91:
                v1 = [a, 0, 0]
                v2 = [0, b / 2, -c / 2]
                v3 = [0, b / 2, c / 2]
            # ORCF
            case 10:
                v1 = [a / 2, 0, c / 2]
                v2 = [a / 2, b / 2, 0]
                v3 = [0, b / 2, c / 2]
            # ORCI
            case 11:
                v1 = [a / 2, b / 2, c / 2]
                v2 = [-a / 2, b / 2, c / 2]
                v3 = [-a / 2, -b / 2, c / 2]
            # MCL
            case 12:
                v1 = [a, 0, 0]
                v2 = [b * cos(radians(gamma)), b * sin(radians(gamma)), 0]
                v3 = [0, 0, c]
            case -12:
                v1 = [a, 0, 0]
                v2 = [0, b, 0]
                v3 = [c * cos(radians(beta)), 0, c * sin(radians(beta))]
            # MCLC
            case 13:
                v1 = [a / 2, 0, -c / 2]
                v2 = [b * cos(radians(gamma)), b * sin(radians(gamma)), 0]
                v3 = [a / 2, 0, c / 2]
            case -13:
                v1 = [a / 2, b / 2, 0]
                v2 = [-a / 2, b / 2, 0]
                v3 = [c * cos(radians(beta)), 0, c * sin(radians(beta))]
            # TRI
            case 14:
                v1 = [a, 0, 0]
                v2 = [b * cos(radians(gamma)), b * sin(radians(gamma)), 0]
                v3 = [c * cos(radians(beta)),
                      c * (cos(radians(alpha)) - cos(radians(beta)) * cos(radians(gamma))) / sin(radians(gamma)),
                      c * sqrt(1 + 2 * cos(radians(alpha)) * cos(radians(beta)) * cos(radians(gamma)) - cos(radians(alpha)) ** 2 - cos(radians(beta)) ** 2 - cos(radians(gamma)) ** 2) / sin(radians(gamma))]

        # Set Cell, Bravais and reciprocal lattices

        self.cell = Cell([v1, v2, v3])
        self.blat = self.cell.get_bravais_lattice()
        self.rcell = self.cell.reciprocal()

        # Initialize path with default path

        self.set_path()



    def info(self):
        """
        Prints description of the Bravais and reciprocal lattices
        """

        print(self.blat.description())
        print(f"Reciprocal lattice:\n{self.rcell[:]}")



    def set_path(self, path = None, npoints = None, density = None, special_points = None):
        """
        Sets the path as ASE's BandPath.
        Augment high symmetry points with special_points
        If no arguments given: default path.
        """

        total_special_points = None

        if isinstance(special_points, dict):
            total_special_points = special_points | self.blat.get_special_points()

        self.bandpath = self.cell.bandpath(path = path, npoints = npoints, density = density, special_points = total_special_points)



    def path_cartesian(self):
        """
        Return path in cartesian coordinates.
        """

        return self.bandpath.cartesian_kpts()



    def path_reduced(self):
        """
        Return path in reduced coordinates.
        """

        return self.bandpath.kpts;



    def path_qe(self):
        """
        Returns path in QE format: [ [Kx, Ky, Kz, 1], ... ]
        """

        # Pad kpts ndarray of shape (nktps, 3) with a single value 1 as last element of 2nd dimension
        # So new shape is (nkpts, 4)

        return np.pad(self.bandpath.kpts, [(0, 0), (0, 1)], 'constant', constant_values = 1)



    def special_points(self):
        return self.blat.get_special_points()
