import numpy as np
from math import sqrt, cos, tan, radians, isclose
from itertools import product
import re

from yambopy.kpoints import expand_kpoints
from yambopy.lattice import red_car, isbetween



def ibrav_required_parameters():
    required_parameters = {
        1: ['a'],
        2: ['a'],
        3: ['a'],
        -3: ['a'],
        4: ['a', 'c'],
        5: ['a', 'alpha'],
        # -5: ['a', 'gamma'],
        6: ['a', 'c'],
        7: ['a', 'c'],
        8: ['a', 'b', 'c'],
        9: ['a', 'b', 'c'],
        -9: ['a', 'b', 'c'],
        # 91: ['a', 'b', 'c'],
        10: ['a', 'b', 'c'],
        11: ['a', 'b', 'c'],
        # 12: ['a', 'b', 'c', 'gamma'],
        # -12: ['a', 'b', 'c', 'beta'],
        # 13: ['a', 'b', 'c', 'gamma'],
        # -13: ['a', 'b', 'c', 'beta'],
        # 14: ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    }
    return required_parameters



def get_lattice_data(ibrav:int, parameters=None):
    # Each Bravais lattice (ibrav) requires setting specific parameters

    required_parameters = ibrav_required_parameters()

    # Check if valid Bravais-lattice index

    if not ibrav in required_parameters:
        raise ValueError(f"ibrav: {ibrav} not supported.")

    # Check if parameters is dictionary

    if not isinstance(parameters, dict):
        raise TypeError(f"{parameters} is not a dictionary")

    # Check required lattice parameters

    for p in required_parameters[ibrav]:
        try:
            parameters[p]
        except KeyError:
            print(f"ibrav: {ibrav} lattice needs parameter: {p}")
            raise

    # Check if a, b, and c parameters are positive nonzero values

    for param, value in parameters.items():
        if any(param == p for p in ['a', 'b', 'c']):
            if value <= 0:
                raise ValueError(f"{param} parameter must be positive.")

    # Set parameters

    a = parameters.get('a', 1)
    b = parameters.get('b', 1)
    c = parameters.get('c', 1)
    alpha = parameters.get('alpha', np.pi / 2)
    beta = parameters.get('beta', np.pi / 2)
    gamma = parameters.get('gamma', np.pi / 2)

    # Set lattice vectors and high symmetry points

    match ibrav:
        # CUB
        case 1:
            v1 = [a, 0, 0]
            v2 = [0, a, 0]
            v3 = [0, 0, a]

            cell = np.array([v1, v2, v3])

            variant = 'CUB'

            high_symmetry_points = {
                'G': [0, 0, 0],
                'M': [1 / 2, 1 / 2, 0],
                'R': [1 / 2, 1 / 2, 1 / 2],
                'X': [0, 1 / 2, 0]
            }

            default_path = 'GXMGRX,MR'

        # FCC
        case 2:
            v1 = [-a / 2, 0, a / 2]
            v2 = [0, a / 2, a / 2]
            v3 = [-a / 2, a / 2, 0]

            cell = np.array([v1, v2, v3])

            variant = 'FCC'

            high_symmetry_points = {
                'G': [0, 0, 0],
                'K': [-3 / 8, 3 / 8, 0],
                'L': [0, 1 / 2, 0],
                'U': [0, 5 / 8, 3 / 8],
                'W': [-1 / 4, 1 / 2, 1 / 4],
                'X': [0, 1 / 2, 1 / 2]
            }

            default_path = 'GXWKGLUWLK,UX'

        # BCC
        case 3:
            v1 = [a / 2, a / 2, a / 2]
            v2 = [-a / 2, a / 2, a / 2]
            v3 = [-a / 2, -a / 2, a / 2]

            cell = np.array([v1, v2, v3])

            variant = 'BCC'

            high_symmetry_points = {
                'G': [0, 0, 0],
                'H': [1 / 2, 1 / 2, -1 / 2],
                'P': [3 / 4, 1 / 4, -1 / 4],
                'N': [1 / 2, 0, -1 / 2]
            }

            default_path = 'GHNGPH,PN'

        # BCC
        case -3:
            v1 = [-a / 2, a / 2, a / 2]
            v2 = [a / 2, -a / 2, a / 2]
            v3 = [a / 2, a / 2, -a / 2]

            cell = np.array([v1, v2, v3])

            variant = 'BCC'

            high_symmetry_points = {
                'G': [0, 0, 0],
                'H': [1 / 2, -1 / 2, 1 / 2],
                'P': [1 / 4, 1 / 4, 1 / 4],
                'N': [0, 0, 1 / 2]
            }

            default_path = 'GHNGPH,PN'

        # HEX
        case 4:
            v1 = [a, 0, 0]
            v2 = [-a / 2, a * sqrt(3) / 2, 0]
            v3 = [0, 0, c]

            cell = np.array([v1, v2, v3])

            variant = 'HEX'

            high_symmetry_points = {
                'G': [0, 0, 0],
                'A': [0, 0, 1 / 2],
                'H': [2 / 3, -1 / 3, 1 / 2],
                'K': [2 / 3, -1 / 3, 0],
                'L': [1 / 2, 0, 1 / 2],
                'M': [1 / 2, 0, 0]
            }

            default_path = 'GMKGALHA,LM,KH'

        # RHL
        case 5:
            cs = cos(radians(alpha))

            tx = sqrt((1 - cs) / 2)
            ty = sqrt((1 - cs) / 6)
            tz = sqrt((1 + 2 * cs) / 3)

            v1 = [a * tx, -a * ty, a * tz]
            v2 = [0, a * 2 * ty, a * tz]
            v3 = [-a * tx, -a * ty, a * tz]

            cell = np.array([v1, v2, v3])

            if 0 < alpha < 90:
                variant = 'RHL1'

                eta = (1 + 4 * cos(radians(alpha))) / (2 + 4 * cos(radians(alpha)))
                nu = 3 / 4 - eta / 2

                high_symmetry_points = {
                    'G': [0, 0, 0],
                    'B': [eta, 1 / 2, 1 - eta],
                    'B1': [1 / 2, 1 - eta, eta - 1],
                    'F': [1 / 2, 1 / 2, 0],
                    'L': [1 / 2, 0, 0],
                    'L1': [0, 0, -1 / 2],
                    'P': [eta, nu, nu],
                    'P1': [1 - nu, 1 - nu, 1 - eta],
                    'P2': [nu, nu, eta - 1],
                    'Q': [1 - nu, nu, 0],
                    'X': [nu, 0, -nu],
                    'Z': [1 / 2, 1 / 2, 1 / 2]
                }

                default_path = 'GLB1,BZGX,QFP1Z,LP'

            elif 90 < alpha < 120:
                variant = 'RHL2'

                eta = 1 / (2 * tan(radians(alpha / 2)) ** 2)
                nu = 3 / 4 - eta / 2

                high_symmetry_points = {
                    'G': [0, 0, 0],
                    'F': [0, 1 / 2, -1 / 2],
                    'L': [0, 1 / 2, 0],
                    'P': [1 - nu, 1 - nu, -nu],
                    'P1': [nu - 1, nu, nu - 1],
                    'Q': [eta, eta, eta],
                    'Q1': [-eta, 1 - eta, -eta],
                    'Z': [1 / 2, 1 / 2, -1 / 2]
                }

                default_path = 'GPZQGFP1Q1LZ'

            else:
                raise ValueError('Invalid alpha value')

        # case -5:
        #     a /= sqrt(3)
        #     c = cos(radians(gamma))
        #     ty = sqrt((1 - c) / 6)
        #     tz = sqrt((1 + 2 * c) / 3)
        #     u = tz - 2 * sqrt(2) * ty
        #     v = tz + sqrt(2) * ty
        #     v1 = [a * u, a * v, a * v]
        #     v2 = [a * v, a * u, a * v]
        #     v3 = [a * v, a * v, a * u]

        # TET
        case 6:
            v1 = [a, 0, 0]
            v2 = [0, a, 0]
            v3 = [0, 0, c]

            cell = np.array([v1, v2, v3])

            variant = 'TET'

            high_symmetry_points = {
                'G': [0, 0, 0],
                'A': [1 / 2, 1 / 2, 1 / 2],
                'M': [1 / 2, 1 / 2, 0],
                'R': [0, 1 / 2, 1 / 2],
                'X': [0, 1 / 2, 0],
                'Z': [0, 0, 1 / 2]
            }

            default_path = 'GXMGZRAZ,XR,MA'

        # BCT
        case 7:
            v1 = [a / 2, -a / 2, c / 2]
            v2 = [a / 2, a / 2, c / 2]
            v3 = [-a / 2, -a / 2, c / 2]

            cell = np.array([v1, v2, v3])

            if isclose(a, c, rel_tol=1e-5):
                raise ValueError('Select either c<a or c>a')

            elif c < a:
                variant = 'BCT1'

                eta = (1 + c ** 2 / a ** 2) / 4

                high_symmetry_points = {
                    'G': [0, 0, 0],
                    'M': [1 / 2, 1 / 2, -1 / 2],
                    'N': [1 / 2, 1 / 2, 0],
                    'P': [1 / 4, 3 / 4, -1 / 4],
                    'X': [0, 1 / 2, -1 / 2],
                    'Z': [eta, eta, eta],
                    'Z1': [1 - eta, 1 - eta, -eta]
                }

                default_path = 'GXMGZPNZ1M,XP'

            elif c > a:
                variant = 'BCT2'

                eta = (1 + a ** 2 / c ** 2) / 4
                zeta = a ** 2 / (2 * c ** 2)

                high_symmetry_points = {
                    'G': [0, 0, 0],
                    'N': [1 / 2, 1 / 2, 0],
                    'P': [1 / 4, 3 / 4, -1 / 4],
                    'S': [eta, eta, -eta],
                    'S1': [1 - eta, 1 - eta, eta],
                    'X': [0, 1 / 2, -1 / 2],
                    'Y': [zeta, 1 / 2, -1 / 2],
                    'Y1': [1 / 2, 1 - zeta, zeta],
                    'Z': [1 / 2, 1 / 2, 1 / 2]
                }

                default_path = 'GXYSGZS1NPY1Z,XP'

        # ORC
        case 8:
            v1 = [a, 0, 0]
            v2 = [0, b, 0]
            v3 = [0, 0, c]

            cell = np.array([v1, v2, v3])

            variant = 'ORC'

            high_symmetry_points = {
                'G': [0, 0, 0],
                'R': [1 / 2, 1 / 2, 1 / 2],
                'S': [1 / 2, 1 / 2, 0],
                'T': [0, 1 / 2, 1 / 2],
                'U': [1 / 2, 0, 1 / 2],
                'X': [1 / 2, 0, 0],
                'Y': [0, 1 / 2, 0],
                'Z': [0, 0, 1 / 2]
            }

            default_path = 'GXSYGZURTZ,YT,UX,SR'

        # ORCC
        case 9:
            v1 = [a / 2, b / 2, 0]
            v2 = [-a / 2, b / 2, 0]
            v3 = [0, 0, c]

            cell = np.array([v1, v2, v3])

            variant = 'ORCC'

            zeta = (1 + a ** 2 / b ** 2) / 4

            high_symmetry_points = {
                'G': [0, 0, 0],
                'A': [zeta, -zeta, 1 / 2],
                'A1': [1 - zeta, zeta, 1 / 2],
                'R': [1 / 2, 0, 1 / 2],
                'S': [1 / 2, 0, 0],
                'T': [1 / 2, 1 / 2, 1 / 2],
                'X': [zeta, -zeta, 0],
                'X1': [1 - zeta, zeta, 0],
                'Y': [1 / 2, 1 / 2, 0],
                'Z': [0, 0, 1 / 2]
            }

            default_path = 'GXSRAZGYX1A1TY,ZT'

        # ORCC
        case -9:
            v1 = [a / 2, -b / 2, 0]
            v2 = [a / 2, b / 2, 0]
            v3 = [0, 0, c]

            cell = np.array([v1, v2, v3])

            variant = 'ORCC'

            zeta = (1 + a ** 2 / b ** 2) / 4

            high_symmetry_points = {
                'G': [0, 0, 0],
                'A': [zeta, zeta, 1 / 2],
                'A1': [-zeta, 1 - zeta, 1 / 2],
                'R': [0, 1 / 2, 1 / 2],
                'S': [0, 1 / 2, 0],
                'T': [-1 / 2, 1 / 2, 1 / 2],
                'X': [zeta, zeta, 0],
                'X1': [-zeta, 1 - zeta, 0],
                'Y': [-1 / 2, 1 / 2, 0],
                'Z': [0, 0, 1 / 2]
            }

            default_path = 'GXSRAZGYX1A1TY,ZT'

        # case 91:
        #     v1 = [a, 0, 0]
        #     v2 = [0, b / 2, -c / 2]
        #     v3 = [0, b / 2, c / 2]

        # ORCF
        case 10:
            v1 = [a / 2, 0, c / 2]
            v2 = [a / 2, b / 2, 0]
            v3 = [0, b / 2, c / 2]

            cell = np.array([v1, v2, v3])

            ia = 1 / a ** 2
            ibc = 1 / b ** 2 + 1 / c ** 2

            if isclose(ia, ibc, rel_tol=1e-5):
                variant = 'ORCF3'
            elif ia < ibc:
                variant = 'ORCF2'
            else:
                variant = 'ORCF1'

            if variant in ['ORCF1', 'ORCF3']:
                zeta = (1 + a ** 2 / b ** 2 - a ** 2 / c ** 2) / 4
                eta = (1 + a ** 2 / b ** 2 + a ** 2 / c ** 2) / 4

                high_symmetry_points = {
                    'G': [0, 0, 0],
                    'A': [1 / 2 + zeta, zeta, 1 / 2],
                    'A1': [1 / 2 - zeta, 1 - zeta, 1 / 2],
                    'L': [1 / 2, 1 / 2, 1 / 2],
                    'T': [1 / 2, 1 / 2, 1],
                    'X': [eta, eta, 0],
                    'X1': [1 - eta, 1 - eta, 1],
                    'Y': [0, 1 / 2, 1 / 2],
                    'Z': [1 / 2, 0, 1 / 2]
                }

                if variant == 'ORCF1':
                    default_path = 'GYTZGXA1Y,TX1,XAZ,LG'
                else:
                    default_path = 'GYTZGXA1Y,XAZ,LG'

            else:
                eta = (1 + a ** 2 / b ** 2 - a ** 2 / c ** 2) / 4
                delta = (1 + b ** 2 / a ** 2 - b ** 2 / c ** 2) / 4
                phi = (1 + c ** 2 / b ** 2 - c ** 2 / a ** 2) / 4

                high_symmetry_points = {
                    'G': [0, 0, 0],
                    'C': [1 / 2 - eta, 1 - eta, 1 / 2],
                    'C1': [1 / 2 + eta, eta, 1 / 2],
                    'D': [1 / 2, 1 - delta, 1 / 2 - delta],
                    'D1': [1 / 2, delta, 1 / 2 + delta],
                    'L': [1 / 2, 1 / 2, 1 / 2],
                    'H': [1 / 2 - phi, 1 / 2, 1 - phi],
                    'H1': [1 / 2 + phi, 1 / 2, phi],
                    'X': [1 / 2, 1 / 2, 0],
                    'Y': [0, 1 / 2, 1 / 2],
                    'Z': [1 / 2, 0, 1 / 2]
                }

                default_path = 'GYCDXGZD1HC,C1Z,XH1,HY,LG'

        # ORCI
        case 11:
            v1 = [a / 2, b / 2, c / 2]
            v2 = [-a / 2, b / 2, c / 2]
            v3 = [-a / 2, -b / 2, c / 2]

            cell = np.array([v1, v2, v3])

            variant = 'ORCI'

            mu = (a ** 2 + b ** 2) / (4 * c ** 2)
            delta = (b ** 2 - a ** 2) / (4 * c ** 2)
            zeta = (1 + a ** 2 / c ** 2) / 4
            eta = (1 + b ** 2 / c ** 2) / 4

            high_symmetry_points = {
                'G': [0, 0, 0],
                'L': [1 / 2 - delta, -mu, delta - 1 / 2],
                'L1': [1 / 2 + delta, mu, -1 / 2 - delta],
                'L2': [1 - mu, 1 / 2 - delta, mu],
                'R': [1 / 2, 0, 0],
                'S': [1 / 2, 1 / 2, 0],
                'T': [1 / 2, 0, -1 / 2],
                'W': [3 / 4, 1 / 4, -1 / 4],
                'X': [zeta, -zeta, -zeta],
                'X1': [1 - zeta, zeta, zeta],
                'Y': [eta, eta, -eta],
                'Y1': [1 - eta, 1 - eta, eta],
                'Z': [1 / 2, 1 / 2, 1 / 2]
            }

            default_path = 'GXLTWRX1ZGYSW,L1Y,Y1Z'

        # MCL
        # case 12:
        #     v1 = [a, 0, 0]
        #     v2 = [b * cos(radians(gamma)), b * sin(radians(gamma)), 0]
        #     v3 = [0, 0, c]
        # case -12:
        #     v1 = [a, 0, 0]
        #     v2 = [0, b, 0]
        #     v3 = [c * cos(radians(beta)), 0, c * sin(radians(beta))]
        # MCLC
        # case 13:
        #     v1 = [a / 2, 0, -c / 2]
        #     v2 = [b * cos(radians(gamma)), b * sin(radians(gamma)), 0]
        #     v3 = [a / 2, 0, c / 2]
        # case -13:
        #     v1 = [a / 2, b / 2, 0]
        #     v2 = [-a / 2, b / 2, 0]
        #     v3 = [c * cos(radians(beta)), 0, c * sin(radians(beta))]
        # TRI
        # case 14:
        #     v1 = [a, 0, 0]
        #     v2 = [b * cos(radians(gamma)), b * sin(radians(gamma)), 0]
        #     v3 = [c * cos(radians(beta)),
        #           c * (cos(radians(alpha)) - cos(radians(beta)) * cos(radians(gamma))) / sin(radians(gamma)),
        #           c * sqrt(1 + 2 * cos(radians(alpha)) * cos(radians(beta)) * cos(radians(gamma)) - cos(radians(alpha)) ** 2 - cos(radians(beta)) ** 2 - cos(radians(gamma)) ** 2) / sin(radians(gamma))]

        case _:
            raise ValueError(f"ibrav: {ibrav} not supported.")

    return cell, variant, high_symmetry_points, default_path



class BrillouinZone:
    """
    Constructs interpolated paths in any of the existing Brillouin zones.

    Follows Quantum ESPRESSO's classification criterion (ibrav [1]).

    In general, ASE framework (following Setyawan and Curtarolo's [2] standard) and QE may employ different direct/reciprocal basis vector conventions.
    On each convention, high-symmetry k-points (a.k.a. special k-points) may be characterized by different coordinates.
    Special k-points are obtained from ASE framework and transformed into QE representation criterion. (Ref. to derivation of transformation)
    Note: not all Bravais lattice indices (ibrav) are supported.

    References:
        [1]: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm226
        [2]: https://doi.org/10.1016/j.commatsci.2010.05.010
    """

    def __init__(self, ibrav: int, parameters=None, path_string=None, extra_points=None, npoints: int = None, density: float = None):
        """
        Initializes the Brillouin zone of a selected lattice type, with given required parameters.
        Path can be determined via special k-points labels, corresponding to high symmetry points of the BZ and/or user-given k-point labels.
        Interpolation of k-points along the path can be performed, either setting the number of k-points or their density.

        Input:
            * ibrav: (int) QE's Bravais lattice type index (1 to 12).
            * parameters: (dict) Dictionary with lattice parameters. (e.g. {'a': 0.123, 'c': 0.321})
            * path: (string) Piecewise special k-point path in the Brillouin zone. Defaults to standard path. (e.g. 'GMKGALHA,LM,KH')
            * extra_points: (dict) Dictionary defining extra special k-points to be used in the path. May overwrite pre-existing special k-points. (e.g. {'M': [0, 0.5, 0], 'K': [1/3, 1/3, 0.0]}
            * npoints: (int) Number of k-points along the path, incompatible with 'density' option.
            * density: (float) Density of k-points (units: 1/Angstrom), incompatible with 'npoints' option.
        """

        # Check and get lattice data

        self.cell, self.variant, self.high_symmetry_points, self.default_path = get_lattice_data(ibrav, parameters)

        # Reciprocal cell

        self.rcell = np.linalg.inv(np.transpose(self.cell))

        # If extra_points given, add them to special k-points dictionary, possibly overwriting existing ones

        if isinstance(extra_points, dict):
            for label, coords in extra_points.items():
                self.high_symmetry_points[label] = np.array(coords)

        # Set default path if none given

        if path_string is None:
            path_string = self.default_path

        # Obtain path sections

        self.path_sections = []
        for section in path_string.split(','):
            labels = [label for label in re.split(r'([A-Z][a-z0-9]*)', section) if label]
            if len(labels) > 0:
                self.path_sections.append(labels)

        all_labels_recognized = np.all([label in self.high_symmetry_points.keys() for section in self.path_sections for label in section])
        if not all_labels_recognized:
            raise ValueError('Given path contains unrecognized labels')

        # Reconstruct path string

        section_strings = []
        for section in self.path_sections:
            section_strings.append(''.join(section))
        self.path_string = ','.join(section_strings)

        self.interpolate(npoints, density)

        # Save arguments as dictionary

        self.arguments = {
            'ibrav': ibrav,
            'parameters' : parameters,
            'path': path_string,
            'extra_points': extra_points,
            'npoints': npoints,
            'density': density
        }


    def as_dict(self):
        """
        Return as dictionary all arguments needed to construct object of this class.
        """

        return self.arguments



    @classmethod
    def from_dict(cls, args):
        """
        Construct a new object of this class, given a dictionary with all arguments needed.
        """

        return cls(ibrav=args['ibrav'], parameters=args['parameters'], path_string=args['path_string'], extra_points=args['extra_points'], npoints=args['npoints'], density=args['density'])



    def interpolate_bak(self, npoints: int = None, density: float = None):
        if npoints is not None and density is not None:
            raise ValueError('You may define npoints or density, but not both.')

        points = self.special_kpoints('red', True)
        dists = points[1:] - points[:-1]
        lengths = self.special_kpoints_distances(True)[1:]
        length = sum(lengths)

        if npoints is None:
            if density is None:
                density = 5
            # npoints = int(round(length * 2 * np.pi * density))
            npoints = int(round(length * density))

        self.kpts_red = []
        x0 = 0
        x = []

        for kpt, dist, l in zip(points[:-1], dists, lengths):
            diff = length - x0
            if abs(diff) < 1e-6:
                n = 0
            else:
                n = max(2, int(round(l * (npoints - len(x)) / diff)))

            for t in np.linspace(0, 1, n)[:-1]:
                self.kpts_red.append(kpt + t * dist)
                x.append(x0 + t * l)

            x0 += l

        if len(points) > 0:
            self.kpts_red.append(points[-1])

        if len(self.kpts_red) == 0:
            self.kpts_red = np.empty((0, 3))

        self.kpts_car = [np.matmul(kpt, self.rcell) for kpt in self.kpts_red]

    def interpolate(self, npoints: int = None, density: float = None):
        if npoints is not None and density is not None:
            raise ValueError('You may define npoints or density, but not both.')

        length = sum(self.special_kpoints_distances(True))

        if npoints is None:
            if density is None:
                density = 5
            # npoints = int(round(length * 2 * np.pi * density))
            npoints = int(round(length * density))

        points = self.special_kpoints('red', False)
        lengths = self.special_kpoints_distances(True)

        self.kpts_red = []
        self.kpts_car = []
        x = []
        x0 = 0
        i = 1

        for spoints in points:
            kpoints = np.array(spoints)
            dists = kpoints[1:] - kpoints[:-1]

            kpts_red = []

            for kpt, dist, l in zip(kpoints[:-1], dists, lengths[i:i+len(kpoints)-1]):
                diff = length - x0
                if abs(diff) < 1e-6:
                    n = 0
                else:
                    n = max(2, int(round(l * (npoints - len(x)) / diff)))

                for t in np.linspace(0, 1, n)[:-1]:
                    kpts_red.append(kpt + t * dist)
                    x.append(x0 + t * l)

                x0 += l

            i += len(kpoints)-1

            if len(kpoints) > 0:
                kpts_red.append(kpoints[-1])
                # x.append(x0)

            if len(kpts_red) == 0:
                kpts_red = np.empty((0, 3))

            self.kpts_car.append([np.matmul(kpt, self.rcell) for kpt in kpts_red])
            self.kpts_red.append(kpts_red)



    def kpoints(self, coords='red', qe=False):
        """
        Obtain coordinates of the k-points of the path.

        Input:
            * coords: 'red' for reduced (default), or 'car' for Cartesian coordinates.
            * qe: True for output in Quantum ESPRESSO's format [ [Kx, Ky, Kz, 1], ... ]

        Output:
            * (ndarray) Coordinates of the k-points of the path.
        """

        # To get path with Quantum ESPRESSO's format:
        # Pad kpts ndarray of shape (nktps, 3) with a single value 1 as last element of 2nd dimension, so new shape is (nkpts, 4)

        if coords == 'red':
            kpts_red = np.concatenate(self.kpts_red)
            if qe:
                return np.pad(kpts_red, [(0, 0), (0, 1)], 'constant', constant_values=1)
            else:
                return np.array(kpts_red)
        elif coords == 'car':
            kpts_car = np.concatenate(self.kpts_car)
            if qe:
                return np.pad(kpts_car, [(0, 0), (0, 1)], 'constant', constant_values=1)
            else:
                return np.array(kpts_car)
        else:
            raise ValueError(f"coords: {coords} not supported.")



    def special_kpoints(self, coords='red', merge_sections=False):
        """
        Obtain coordinates of special k-points defining the path.

        Input:
            * coords: 'red' for reduced or 'car' for Cartesian coordinates.
            * merge_sections: Join path segments in simple list, or keep each segment as a sub-list.

        Output:
            * Special k-point coordinates (as list of ndarray, or as list of sub-lists of ndarray for each segment).
        """

        if coords == 'red':
            sub_lists = [[np.array(self.high_symmetry_points[label]) for label in section] for section in self.path_sections]
            if merge_sections:
                return np.concatenate(sub_lists)
            else:
                return sub_lists
        elif coords == 'car':
            sub_lists = [[np.matmul(self.high_symmetry_points[label], self.rcell) for label in section] for section in self.path_sections]
            if merge_sections:
                return np.concatenate(sub_lists)
            else:
                return sub_lists
        else:
            raise ValueError(f"coords: {coords} not supported.")



    def kpoints_distances(self):
        """
        Obtain consecutive distances between k-points on the path.
        Suitable to plot band-structures.

        Output:
            * (ndarray) Distances of k-points on the path.
        """

        # spoints_piecewise = self.special_kpoints('car')
        # kpoints = self.kpoints('car')

        # spoints_distance = 0
        # kpoints_distances = []

        # nk = 0

        # for spoints in spoints_piecewise:
        #     for ns in range(len(spoints) - 1):
        #         while nk < len(kpoints):
        #             if np.allclose(kpoints[nk], spoints[ns + 1]):
        #                 spoints_distance += np.linalg.norm(spoints[ns + 1] - spoints[ns])
        #                 kpoints_distances.append(spoints_distance)
        #                 nk += 1
        #                 break
        #             if isbetween(spoints[ns], spoints[ns + 1], kpoints[nk]):
        #                 kpoints_distances.append(spoints_distance + np.linalg.norm(kpoints[nk] - spoints[ns]))
        #                 nk += 1

        # return np.array(kpoints_distances)

        kpt_dists = [0]
        dist = 0
        for kpts in self.kpts_car:
            for nk in range(len(kpts) - 1):
                dist += np.linalg.norm(kpts[nk + 1] - kpts[nk])
                kpt_dists.append(dist)

        return np.array(kpt_dists)


    def special_kpoints_distances(self, merge_sections=False):
        """
        Obtain cumulative distances between consecutive special k-points along the path.

        Input:
            * merge_sections: Whether to keep consecutive segments boundary distances (coincident) or not.

        Output:
            * (ndarray) Distances of special k-points on the path.
        """

        spoints_piecewise = self.special_kpoints('car')

        distances = []
        distance = 0

        if merge_sections:
            distances.append(distance)

        for section in spoints_piecewise:
            if not merge_sections:
                distances.append(distance)
            for nk in range(len(section) - 1):
                distance += np.linalg.norm(section[nk + 1] - section[nk])
                distances.append(distance)

        return np.array(distances)



    def path_labels_list(self, merge_sections=False):
        """
        Obtain simple list of path's special point labels.
        Suitable to plot band-structure x-axis labels.

        Input:
            * merge_sections: Join path segment labels with a hyphen.

        Output:
            * (list) Path's special k-point labels.
        """

        labels_list = [label for labels in self.path_sections for label in labels]

        if merge_sections:
            sections_lengths = [len(labels) for labels in self.path_sections]
            sections_lengths.pop()

            if len(sections_lengths) > 0:
                boundaries = np.cumsum(sections_lengths)
                for k in boundaries:
                    labels_list[k - 1] = labels_list[k - 1] + " - " + labels_list[k]
                for k in sorted(boundaries, reverse=True):
                    labels_list.pop(k)

        return labels_list



    def get_collinear_kpoints(self, kpoints_car, sym_car=None, debug=False):
        """
        Given a set of k-points, obtain those which are collinear with the path.

        Input:
            * kpoints_car: Cartesian coordinates of k-points (irreducible BZ or full BZ)
            * sym_car: Symmetry operations to expand k-points of IBZ, or None if k-points given in full BZ.

        Output:
            * (ndarray) Collinear k-point cartesian coordinates.
            * (ndarray) Collinear k-point indices.
            * (ndarray) Collinear k-point distances along the path.
        """

        rlat = self.rcell

        if sym_car is None:
            kpoints_indices = list(range(len(kpoints_car)))
        else:
            _, kpoints_indices, _, kpoints_car = expand_kpoints(kpoints_car, sym_car, rlat)

        spoints_car = self.special_kpoints('car')
        spoints_distances = self.special_kpoints_distances()

        collinear_kpoints = []
        collinear_indices = []
        collinear_distances = []

        kdist = 0

        for section in spoints_car:
            for k in range(len(section) - 1):

                collinear_kpoints_data = {}

                start_kpoint = section[k]
                end_kpoint = section[k + 1]

                for x, y, z in product(list(range(-1, 2)), list(range(-1, 2)), list(range(-1, 2))):

                    shift = red_car([np.array([x, y, z])], rlat)[0]

                    for index, kpoint in zip(kpoints_indices, kpoints_car):

                        kpoint_shift = kpoint + shift

                        if isbetween(start_kpoint, end_kpoint, kpoint_shift):
                            key = tuple([np.round(kpt, 4) for kpt in kpoint_shift])

                            distance = np.linalg.norm(start_kpoint - kpoint_shift)
                            distance_within_path = spoints_distances[kdist] + distance

                            value = [index, distance, kpoint_shift, distance_within_path]

                            collinear_kpoints_data[key] = value

                kdist += 1

                collinear_kpoints_data = sorted(list(collinear_kpoints_data.values()), key = lambda i: i[1])

                for index, distance, kpoint_shift, distance_within_path in collinear_kpoints_data:
                    collinear_indices.append(index)
                    collinear_kpoints.append(kpoint_shift)
                    collinear_distances.append(distance_within_path)

                    if debug:
                        print(("%12.8lf " * 3)%tuple(kpoint_shift), index, distance, distance_within_path)

        return np.array(collinear_kpoints), np.array(collinear_indices), np.array(collinear_distances)