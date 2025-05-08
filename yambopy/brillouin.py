import numpy as np
from math import sqrt, cos, sin, radians
from ase.cell import Cell
from ase.dft.kpoints import parse_path_string
from itertools import product

from yambopy.kpoints import expand_kpoints
from yambopy.lattice import red_car, isbetween



class BrillouinZone():
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

    # Each Bravais lattice (ibrav) requires setting specific parameters

    required_parameters = {
        1: ['a'],
        2: ['a'],
        3: ['a'],
        #-3: ['a'],
        4: ['a', 'c'],
        5: ['a', 'gamma'],
        #-5: ['a', 'gamma'],
        6: ['a', 'c'],
        7: ['a', 'c'],
        8: ['a', 'b', 'c'],
        9: ['a', 'b', 'c'],
        #-9: ['a', 'b', 'c'],
        #91: ['a', 'b', 'c'],
        10: ['a', 'b', 'c'],
        11: ['a', 'b', 'c'],
        12: ['a', 'b', 'c', 'gamma'],
        #-12: ['a', 'b', 'c', 'beta'],
        #13: ['a', 'b', 'c', 'gamma'],
        #-13: ['a', 'b', 'c', 'beta'],
        #14: ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    }

    # For each ibrav and variant, we have sampled three special k-point reduced coordinates using QE.
    # Used to obtain transformation matrices (ASE/SC -> QE).

    selected_points_qe = {
        1: {
            'CUB': {
                'M': [1/2, 1/2, 0],
                'R': [1/2, 1/2, 1/2],
                'X': [0, 1/2, 0]
            }
        },
        2: {
            'FCC': {
                'K': [-3/8, 3/8, 0],
                'L': [0, 1/2, 0],
                'U': [0, 5/8, 3/8]
            }
        },
        3: {
            'BCC': {
                'H': [1/2, 1/2, -1/2],
                'P': [3/4, 1/4, -1/4],
                'N': [1/2, 0, -1/2]
            }
        },
        4: {
            'HEX': {
                'K': [2/3, -1/3, 0],
                'L': [1/2, 0, 1/2],
                'M': [1/2, 0, 0]
            }
        },
        5: {
            'RHL1': {
                'F': [1/2, 1/2, 0],
                'L': [1/2, 0, 0],
                'L1': [0, 0, -1/2]
            },
            'RHL2': {
                'F': [0, 1/2, -1/2],
                'L': [0, 1/2, 0],
                'Z': [1/2, 1/2, -1/2]
            }
        },
        6: {
            'TET': {
                'M': [1/2, 1/2, 0],
                'X': [0, 1/2, 0],
                'Z': [0, 0, 1/2]
            }
        },
        7: {
            'BCT1': {
                'M': [1/2, 1/2, -1/2],
                'N': [1/2, 1/2, 0],
                'P': [1/4, 3/4, -1/4]
            },
            'BCT2': {
                'N': [1/2, 1/2, 0],
                'P': [1/4, 3/4, -1/4],
                'X': [0, 1/2, -1/2]
            }
        },
        8: {
            'ORC': {
                'X': [1/2, 0, 0],
                'Y': [0, 1/2, 0],
                'Z': [0, 0, 1/2]
            }
        },
        9: {
            'ORCC': {
                'R': [1/2, 0, 1/2],
                'S': [1/2, 0, 0],
                'T': [1/2, 1/2, 1/2]
            }
        },
        10: {
            'ORCF1': {
                'L': [1/2, 1/2, 1/2],
                'Y': [0, 1/2, 1/2],
                'Z': [1/2, 0, 1/2]
            },
            'ORCF2': {
                'L': [1/2, 1/2, 1/2],
                'Y': [0, 1/2, 1/2],
                'Z': [1/2, 0, 1/2]
            },
            'ORCF3': {
                'L': [1/2, 1/2, 1/2],
                'Y': [0, 1/2, 1/2],
                'Z': [1/2, 0, 1/2]
            }
        },
        11: {
            'ORCI': {
                'R': [1/2, 0, 0],
                'S': [1/2, 1/2, 0],
                'T': [1/2, 0, -1/2]
            }
        },
        12: {
            'MCL': {
                'X': [1/2, 0, 0],
                'Y': [0, 1/2, 0],
                'Z': [0, 0, 1/2]
            }
        }
        # -12: {
        #     'MCL': {
        #         'X': [1/2, 0, 0],
        #         'A': [1/2, 1/2, 0],
        #         'Y': [0, 1/2, 0],
        #         'Z': [0, 0, 1/2]
        #     }
        # }
    }

    def __init__(self, ibrav, parameters=None, path=None, extra_points=None, npoints=None, density=None):
        """
        Initializes the Brillouin zone of a selected lattice type, with given required parameters.
        Path can be determined via special k-points labels, corresponding to high symmetry points of the BZ and/or user-given k-point labels.
        Interpolation of k-points along the path can be performed, either setting the number of k-points or their density.

        Input:
            * ibrav: (int) QE's Bravais lattice type index (1 to 12).
            * parameters: (dict) Dictionary with lattice parameters. (e.g. {'a': 0.123, 'c': 0.321})
            * path: (string) Piecewise special k-point path in the Brillouin zone. Defaults to standard path. (e.g. 'GMKGALHA,LM,KH')
            * extra_points: (dict) Dictionary defining extra special k-points to be used in the path. May overwrite pre-existing special k-points. (e.g. {'M': [0, 0.5, 0], 'K': [1/3, 1/3, 0.0]}
            * npoints: (int) Number of k-points on the path. If none given and density not specified, only the special k-points defining the path are considered. Not compatible with density argument.
            * density: (float) Density of k-points per 1/Angstrom. If none given and npoints not specified, only the special k-points defining the path are considered. Not compatible with npoints argument.
        """

        # Check if valid Bravais-lattice index

        if not ibrav in self.required_parameters:
            raise ValueError(f"ibrav: {ibrav} not supported.")

        # Check if parameters is dictionary

        if not isinstance(parameters, dict):
            raise TypeError(f"{parameters} is not a dictionary")

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

        # Save arguments as dictionary

        self.arguments = {
            'ibrav': ibrav,
            'parameters' : parameters,
            'path': path,
            'extra_points': extra_points,
            'npoints': npoints,
            'density': density
        }

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
            # case -3:
            #     v1 = [-a / 2, a / 2, a / 2]
            #     v2 = [a / 2, -a / 2, a / 2]
            #     v3 = [a / 2, a / 2, -a / 2]
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
            # case -9:
            #     v1 = [a / 2, -b / 2, 0]
            #     v2 = [a / 2, b / 2, 0]
            #     v3 = [0, 0, c]
            # case 91:
            #     v1 = [a, 0, 0]
            #     v2 = [0, b / 2, -c / 2]
            #     v3 = [0, b / 2, c / 2]
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

        # Set Cell object and Bravais lattice (ASE resources):
        # Cell object: represents a Cell given direct QE's lattice vectors, corresponding to specified ibrav
        # 'get_bravais_lattice' method: identifies the Bravais lattice type of the Cell object and returns BravaisLattice object
        # BravaisLattice object: contains high-symmetry points and default path, following SC criterion for basis vectors

        self.cell = Cell([v1, v2, v3])
        self.blat = self.cell.get_bravais_lattice()

        # Transform BravaisLattice object's high-symmetry points from ASE/SC to QE representation
        # We store ASE/SC special points to check results later

        self.special_points = self.transformed_special_points()
        self.blat_special_points = self.blat.get_special_points()

        # If extra_points given, add them to special k-points dictionary, possibly overwriting existing ones

        if isinstance(extra_points, dict):
            for label, coords in extra_points.items():
                self.special_points[label] = np.array(coords)
                # Transform to ASE/SC representation
                self.blat_special_points[label] = np.matmul(np.array(coords), np.linalg.inv(self.P))

        # Set default path if none given

        if path is None:
            path = self.blat.special_path

        # Set no interpolation of k-points along the path if npoints and density not given

        if npoints is None and density is None:
            npoints = 0

        # Set BandPath objects (ASE resources):
        # 'bandpath' method: builds a BandPath object for a cell
        # BandPath objects: represent Brillouin zone paths, either following QE (actual path) or ASE/SC criteria (for checking purposes), depending on special_points given

        self.bandpath = self.cell.bandpath(path=path, special_points=self.special_points, npoints=npoints, density=density)
        self.blat_bandpath = self.blat.bandpath(path=path, special_points=self.blat_special_points, npoints=npoints, density=density)



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

        return cls(ibrav=args['ibrav'], parameters=args['parameters'], path=args['path'], extra_points=args['extra_points'], npoints=args['npoints'], density=args['density'])



    def info(self, debug=False):
        """
        Prints description of the direct and reciprocal lattices, and tests of correctness of transformation.
        """

        print(f"Lattice name: {self.blat.name} ({self.blat.longname})")
        print(f"Variant name: {self.blat.variant}")
        print('Parameters:')
        for name, value in zip(['a', 'b', 'c', 'alpha', 'beta', 'gamma'], self.cell.cellpar()):
            print(f"\t{name}: {np.round(value, 3)}")
        print(f"Special k-point names: {self.blat.special_point_names}")
        print(f"Default path: {self.blat.special_path}")

        print('Special k-point reduced coordinates:')
        for label, coords in self.special_points.items():
            print(f"\t{label}:\t{coords[0]:.4f}\t{coords[1]:.4f}\t{coords[2]:.4f}")

        if debug:
            a = self.arguments['parameters']['a']

            sc_cell = self.blat.tocell()[:] / a
            qe2sc_cell = np.linalg.inv(np.matmul(self.U, np.matmul(np.linalg.inv(self.cell[:]), np.transpose(self.P)))) / a

            sc_reciprocal = self.blat.tocell().reciprocal()[:] * a
            qe2sc_reciprocal = np.matmul(self.P, np.matmul(self.cell.reciprocal()[:], np.transpose(self.U))) * a

            sc_car_kpoints = self.blat_bandpath.cartesian_kpts() * a
            qe2sc_car_kpoints = np.einsum('ik,kj->ij', self.bandpath.cartesian_kpts(), np.linalg.inv(self.U)) * a

            sc_red_kpoints = self.blat_bandpath.kpts
            qe2sc_red_kpoints = np.einsum('ik,kj->ij', self.bandpath.kpts, np.linalg.inv(self.P))

            print('\n### Transformation matrices: ASE/SC -> QE ###')

            print('Reciprocal basis transformation matrix P:')
            print(np.round(self.P, 6))
            print(f"det(P) = {np.round(np.linalg.det(self.P), 6)}")

            print('Cartesian basis transformation matrix U:')
            print(np.round(self.U, 6))
            print(f"det(U) = {np.round(np.linalg.det(self.U), 6)}")

            print('Cartesian basis vector angles (deg):')
            print(np.round(np.degrees(np.arccos(np.round(self.U, 4))), 3))

            print('\n### ASE/SC Cell ###')

            print(self.blat.description())

            print('Parameters:')
            print(self.blat.cellpar())

            print('Direct:')
            print(np.round(sc_cell, 6))

            if np.allclose(sc_cell, qe2sc_cell):
                print('Direct QE -> SC: Match')
            else:
                print('Direct QE -> SC: Mismatch!')
                print(np.round(qe2sc_cell, 6))

            print('Reciprocal:')
            print(np.round(sc_reciprocal, 6))

            if np.allclose(sc_reciprocal, qe2sc_reciprocal):
                print('Reciprocal QE -> SC: Match')
            else:
                print('Reciprocal QE -> SC: Mismatch!')
                print(np.round(qe2sc_reciprocal, 6))

            print('Cartesian:')
            print(f"{len(sc_car_kpoints)} k-points")
            print(np.round(sc_car_kpoints, 6))

            if len(sc_car_kpoints) == len(qe2sc_car_kpoints) and np.allclose(sc_car_kpoints, qe2sc_car_kpoints):
                print('Cartesian QE -> SC: Match')
            else:
                print('Cartesian QE -> SC: Mismatch!')
                print(f"{len(qe2sc_car_kpoints)} k-points")
                print(np.round(qe2sc_car_kpoints, 6))

            print('Fractional:')
            print(f"{len(sc_red_kpoints)} k-points")
            print(sc_red_kpoints)

            if len(sc_red_kpoints) == len(qe2sc_red_kpoints) and np.allclose(sc_red_kpoints, qe2sc_red_kpoints):
                print('Fractional QE -> SC: Match')
            else:
                print('Fractional QE -> SC: Mismatch!')
                print(f"{len(qe2sc_red_kpoints)} k-points")
                print(np.round(qe2sc_red_kpoints, 6))

            print('\n### QE Cell ###')

            print('Parameters:')
            print(self.cell.cellpar())

            print('Direct:')
            print(np.round(self.cell[:] / a, 6))

            print('Reciprocal:')
            print(np.round(self.cell.reciprocal()[:] * a, 6))

            print('Cartesian:')
            print(np.round(self.bandpath.cartesian_kpts() * a, 6))

            print('Fractional:')
            print(np.round(self.bandpath.kpts, 6))



    def change_of_basis_matrices(self):
        """
        Construct change of basis matrices: ASE/SC -> QE

        Matrix P: connects special k-point reduced coordinates on both representations.
        Matrix U: connects reciprocal lattices bases on both representations.

        Output:
            * (ndarray) Matrices P and U.
        """

        Bqe = self.cell.reciprocal()[:]
        Bsc = self.blat.tocell().reciprocal()[:]

        ibrav = self.arguments['ibrav']
        variant = self.blat.variant

        if ibrav in self.selected_points_qe and variant in self.selected_points_qe[ibrav]:
            fcoords_qe = self.selected_points_qe[ibrav][variant]
            fcoords_sc = self.blat.get_special_points()

            labels = list(self.selected_points_qe[ibrav][variant].keys())

            Fqe = np.array([fcoords_qe[labels[0]], fcoords_qe[labels[1]], fcoords_qe[labels[2]]])
            Fsc = np.array([fcoords_sc[labels[0]], fcoords_sc[labels[1]], fcoords_sc[labels[2]]])
            Fsc_inv = np.linalg.inv(Fsc)

            P = np.matmul(Fsc_inv, Fqe)
            U = np.matmul(np.transpose(Bsc), np.linalg.inv(np.transpose(np.matmul(P, Bqe))))
        else:
            P = np.eye(3)
            U = np.eye(3)

        return P, U



    def transformed_special_points(self):
        """
        Transform Brillouin zone special k-points from ASE/SC to QE representation.

        Output:
            * (dict) Transformed special k-points.
        """

        self.P, self.U = self.change_of_basis_matrices()

        special_points = {}
        for label, coords in self.blat.get_special_points().items():
            special_points[label] = np.matmul(coords, self.P)

        return special_points



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
        #   Pad kpts ndarray of shape (nktps, 3) with a single value 1 as last element of 2nd dimension, so new shape is (nkpts, 4)

        if coords == 'red':
            if qe:
                return np.pad(self.bandpath.kpts, [(0, 0), (0, 1)], 'constant', constant_values = 1)
            else:
                return self.bandpath.kpts
        elif coords == 'car':
            if qe:
                return np.pad(self.bandpath.cartesian_kpts(), [(0, 0), (0, 1)], 'constant', constant_values = 1)
            else:
                return self.bandpath.cartesian_kpts()
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

        sections = parse_path_string(self.bandpath.path)

        if coords == 'red':
            if merge_sections:
                return np.array([self.special_points[label] for section in sections for label in section])
            else:
                return [[self.special_points[label] for label in section] for section in sections]
        elif coords == 'car':
            reciprocal_cell = self.cell.reciprocal()
            if merge_sections:
                return np.array([reciprocal_cell.cartesian_positions(self.special_points[label]) for section in sections for label in section])
            else:
                return [[reciprocal_cell.cartesian_positions(self.special_points[label]) for label in section] for section in sections]
        else:
            raise ValueError(f"coords: {coords} not supported.")



    def kpoints_distances(self):
        """
        Obtain consecutive distances between k-points on the path.
        Suitable to plot band-structures.

        Output:
            * (ndarray) Distances of special k-points on the path.
        """

        spoints_piecewise = self.special_kpoints('car')
        kpoints = self.kpoints('car')

        spoints_distance = 0
        kpoints_distances = []

        nk = 0

        for spoints in spoints_piecewise:
            for ns in range(len(spoints) - 1):
                while nk < len(kpoints):
                    if np.allclose(kpoints[nk], spoints[ns + 1]):
                        spoints_distance += np.linalg.norm(spoints[ns + 1] - spoints[ns])
                        kpoints_distances.append(spoints_distance)
                        nk += 1
                        break
                    if isbetween(spoints[ns], spoints[ns + 1], kpoints[nk]):
                        kpoints_distances.append(spoints_distance + np.linalg.norm(kpoints[nk] - spoints[ns]))
                        nk += 1

        return np.array(kpoints_distances)



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

        labels_sections = parse_path_string(self.bandpath.path)

        labels_list = [label for labels in labels_sections for label in labels]

        if merge_sections:
            sections_lengths = [len(labels) for labels in labels_sections]
            sections_lengths.pop()

            if len(sections_lengths) > 0:
                boundaries = np.cumsum(sections_lengths)
                for k in boundaries:
                    labels_list[k - 1] = labels_list[k - 1] + " - " + labels_list[k]
                for k in sorted(boundaries, reverse=True):
                    labels_list.pop(k)

        return labels_list



    def path_labels(self):
        """
        Obtain list of path segment labels, where each segment is stored in a sub-list.

        Output:
            * (list of sub-lists) Path's segment labels.
        """

        return parse_path_string(self.bandpath.path)



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

        rlat = self.cell.reciprocal()[:]

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