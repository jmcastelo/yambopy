from math import sqrt, cos, sin, radians, pi
from ase.cell import Cell
from ase.dft.kpoints import parse_path_string
import numpy as np



class BrillouinZone():
    """
    Constructs interpolated paths in any of the existing Brillouin zones.
    Follows QE's classification criterion (ibrav).
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

    def __init__(self, ibrav, parameters = {}, path = None, extra_points = None):
        """
        Initializes the Brillouin zone of a selected lattice type, with given required parameters and path.
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

        # Set lattice vectors as QE does

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

        # Set Cell and Bravais lattice

        self.cell = Cell([v1, v2, v3])
        self.blat = self.cell.get_bravais_lattice()

        # Save arguments as dictionary

        self.arguments = {
            'ibrav': ibrav,
            'parameters' : parameters,
            'path': path,
            'extra_points': extra_points
        }

        # Construct path in the BZ consisting only of the special points, i.e. no interpolation made

        #self.special_points = self.blat.get_special_points()
        self.special_points = self.transformed_special_points()

        if isinstance(extra_points, dict):
            self.special_points = extra_points | self.special_points

        if path == None:
            path = self.blat.special_path

        self.bandpath = self.cell.bandpath(path = path, npoints = 0, special_points = self.special_points)
        self.blat_bandpath = self.blat.bandpath(path = path, npoints = 0)



    def as_dict(self):
        """
        Return as dictionary all arguments needed to construct object of this class
        """

        return self.arguments



    @classmethod
    def from_dict(cls, args):
        """
        Construct a new object of this class, given a dictionary with all arguments needed
        """

        return cls(ibrav = args['ibrav'], parameters = args['parameters'], path = args['path'], extra_points = args['extra_points'])



    def info(self):
        """
        Prints description of the Bravais lattice
        """

        print(self.blat.description())

        print(self.blat.cellpar())
        print(np.round(self.blat.tocell()[:] / self.arguments['parameters']['a'], 6))
        print(np.round(self.blat.tocell().reciprocal()[:] * self.arguments['parameters']['a'], 6))
        print(self.blat_bandpath.kpts)
        print(np.round(self.blat_bandpath.cartesian_kpts() * self.arguments['parameters']['a'], 6))
        print('### Cell ###')
        print(self.cell.cellpar())
        print(np.round(self.cell[:] / self.arguments['parameters']['a'], 6))
        print(np.round(self.cell.reciprocal()[:] * self.arguments['parameters']['a'], 6))
        print(np.round(self.bandpath.kpts, 6))
        print(np.round(self.bandpath.cartesian_kpts() * self.arguments['parameters']['a'], 6))



    def transformed_special_points(self):
        Asc = self.blat.tocell()[:]
        Aqe = self.cell[:]
        Asc_inv = np.linalg.inv(Asc)
        S = np.matmul(Aqe, Asc_inv)

        #Bsc = self.blat.tocell().reciprocal()[:]
        #Bqe = self.cell.reciprocal()[:]
        #Bqe_inv = np.linalg.inv(Bqe)
        #S = np.transpose(np.matmul(Bsc, Bqe_inv))

        special_points = {}
        for label, coords in self.blat.get_special_points().items():
            special_points[label] = S.dot(coords)

        print(special_points)

        return special_points



    def kpoints(self, coords = 'red'):
        """
        Returns ndarray of the k-points of the path (no interpolation made).

        Input:
            coords: 'red' for reduced (default), or 'car' for Cartesian coordinates.

        Output:
            ndarray containing the k-points of the path in the selected coordinates.
        """

        if coords == 'red':
            return self.bandpath.kpts
        elif coords == 'car':
            return self.bandpath.cartesian_kpts()
        else:
            raise ValueError(f"coords: {coords} not supported.")



    def kpoints_interpolated(self, coords = 'red', npoints = None, density = None, qe = False):
        """
        Returns ndarray of interpolated k-points along the path.

        Input:
            coords: 'red' for reduced (default), or 'car' for Cartesian coordinates.
            npoints (int): Total number of k-points to interpolate. At least one point is added for each special point in the path.
            density (float): Density of k-points along the path in Angstron**-1.
            qe: True for output in Quantum ESPRESSO's format [ [Kx, Ky, Kz, 1], ... ]

        Output:
            ndarray containing the interpolated k-points along the path in the selected coordinates and format.
        """

        interpolated_bandpath = self.bandpath.interpolate(npoints = npoints, density = density)

        # To get path with Quantum ESPRESSO's format:
        #   Pad kpts ndarray of shape (nktps, 3) with a single value 1 as last element of 2nd dimension, so new shape is (nkpts, 4)

        if coords == 'red':
            if qe:
                return np.pad(interpolated_bandpath.kpts, [(0, 0), (0, 1)], 'constant', constant_values = 1)
            else:
                return interpolated_bandpath.kpts
        elif coords == 'car':
            if qe:
                return np.pad(interpolated_bandpath.cartesian_kpts(), [(0, 0), (0, 1)], 'constant', constant_values = 1)
            else:
                return interpolated_bandpath.cartesian_kpts()
        else:
            raise ValueError(f"coords: {coords} not supported.")


    def linear_kpoint_axis(self):
        """
        Get an x-axis to be used when plotting a band structure.

        The first of the returned lists is a list of cumulative distances between k-points.
        The second is list of x-coordinates of the special points (can be used as xticks).
        The third is a list of the special points as strings (can be used as xticklabels).
        """
        return self.bandpath.get_linear_kpoint_axis()



    def kpoints_piecewise(self, coords = 'red'):
        sections = parse_path_string(self.bandpath.path)

        if coords == 'red':
            return [[self.special_points[label] for label in section] for section in sections]
        elif coords == 'car':
            reciprocal_cell = self.cell.reciprocal()
            return [[reciprocal_cell.cartesian_positions(self.special_points[label]) for label in section] for section in sections]
        else:
            raise ValueError(f"coords: {coords} not supported.")
