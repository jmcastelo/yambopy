from yambopy import *

def interpolate(kpoint1, kpoint2, nkpoints):
    delta21 = kpoint2 - kpoint1

    interpolated_points = []

    for i in range(nkpoints):
        interpolated_points.append(kpoint1 + delta21 * (i + 1) / (nkpoints + 1))

    return interpolated_points

def check_list_level(lst, level = 0):
    if level > 1:
        raise ValueError('Path must be list of subpaths specified by k-points labels.')
    for item in lst:
        if isinstance(item, list):
            check_list_level(item, level + 1)

class CUB():
    def __init__(self):
        self.name = 'Cubic'
        self.required_params = ['a']
 
        self.symmetry_points = {
            'Gamma': [0.0, 0.0, 0.0],
            'M': [1.0 / 2.0, 1.0 / 2.0, 0.0],
            'R': [1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0],
            'X': [0.0, 1.0 / 2.0, 0.0]
        }

        self.default_path = [['Gamma', 'X', 'M', 'Gamma', 'R', 'X'], ['M', 'R']]
   
    def set_parameters(self, parameters):
        a = parameters['a']

        self.lattice = a * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

class FCC():
    def __init__(self):
        self.name = 'Face-centered cubic'
        self.required_params = ['a']
 
        self.symmetry_points = {
            'Gamma': [0.0, 0.0, 0.0],
            'K': [3.0 / 8.0, 3.0 / 8.0, 3.0 / 4.0],
            'L': [1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0],
            'U': [5.0 / 8.0, 1.0 / 4.0, 5.0 / 8.0],
            'W': [1.0 / 2.0, 1.0 / 4.0, 3.0 / 4.0],
            'X': [1.0 / 2.0, 0.0, 1.0 / 2.0]
        }

        self.default_path = [['Gamma', 'X', 'W', 'K', 'Gamma', 'L', 'U', 'W', 'L', 'K'], ['U', 'X']]
   
    def set_parameters(self, parameters):
        a = parameters['a']

        self.lattice = (a / 2.0) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        
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
            self.bz = CUB()
        elif code == 'FCC':
            self.bz = FCC()
        else:
            raise ValueError('Wrong lattice type: {}'.format(code))

        # Check required lattice parameters

        for p in self.bz.required_params:
            try: parameters[p]
            except KeyError:
                print(self.bz.name, 'lattice needs parameter:', p)
                raise
        
        # Set lattice parameters

        self.bz.set_parameters(parameters)

        # Obtain reciprocal lattice vectors
        
        self.rlattice = rec_lat(self.bz.lattice)

        # Obtain symmetry points in cartesian coordinates

        symmetry_points_car_values = red_car(list(self.bz.symmetry_points.values()), self.rlattice)
        self.symmetry_points_car = dict(zip(self.bz.symmetry_points.keys(), symmetry_points_car_values))

    def check_path_structure(self, path):
        """
        Check if given path is list of subpaths (lists) and if enough points (>1).
        """
        if not isinstance(path, list):
            raise ValueError('Path must be list of subpaths.')

        if all(isinstance(subpath, list) for subpath in path):
            for subpath in path:
                for kpoint in subpath:
                    if isinstance(kpoint, list):
                        if any(isinstance(coordinate, list) for coordinate in kpoint):
                            raise ValueError('Path must be list of subpaths.')
                        elif len(kpoint) != 3:
                            raise ValueError('k-point must have 3 coordinates:', kpoint)
                if len(subpath) < 2:
                    raise ValueError('Not enough elements:', element)
        else:
            raise ValueError('Path must be list of subpaths.')

    def check_path_labels(self, path):
        """
        Check if given path contains valid high symmetry points labels of selected Brillouin zone.
        """
        for subpath in path:
            for kpoint in subpath:
                if isinstance(kpoint, str):
                    try: self.bz.symmetry_points[kpoint]
                    except KeyError as err:
                        print(err.args[0], 'is not a high symmetry point of', self.bz.name, 'lattice.')
                        raise
   
    def get_path_interpolated(self, path, in_coord_type = 'red', out_coord_type = 'car', qe_out = False,  nkpoints = 100):
        """
        Constructs path given high symmetry points labels of selected Brillouin zone, with given number of
        interpolated points, expressed in cartesian or reduced coordinates. Optionally, output in QE format.
        """
        # Check nkpoints

        if nkpoints < 0:
            raise ValueError('Number of k-points must be positive integer.')

        # Check output coordinates type

        if out_coord_type != 'car' and out_coord_type != 'red':
            raise ValueError('Unrecognized output coordinates type:', out_coord_type)

        # Transform given path to cartesian coordinates
        
        path_car = self.transform_path_to_cartesian(path, in_coord_type)

        # Interpolate path

        ipath = self.interpolate_path(path_car, nkpoints)
        self.interpolated_path_car = ipath

        # Express it in reduced coordinates
        
        if out_coord_type == 'red':
            ipath = car_red(self.interpolate_path(path_car, nkpoints), self.rlattice)

        # QE output format: [ [Kx, Ky, Kz, 1], ... ]
        
        if qe_out:
            qepath = []
            for i in range(len(ipath)):
                qepath.append(np.concatenate((ipath[i], [int(1)])))
            ipath = np.array(qepath)

        return ipath

    def get_default_path_interpolated(self, out_coord_type = 'car', qe_out = False, nkpoints = 100):
        """
        Constructs default path of selected Brillouin zone with given number of interpolated points,
        expressed in cartesian or reduced coordinates. Optionally, output in QE format.
        """
        return self.get_path_interpolated(self.bz.default_path, 'red', out_coord_type, qe_out, nkpoints)

    def transform_path_to_cartesian(self, path, in_coord_type):
        """
        Given path expressed by high symmetry points labels, interpolate given number of k-points,
        returning interpolated path in cartesian coordinates.
        """
        # Check path structure and labels

        self.check_path_structure(path)
        self.check_path_labels(path)

        # Transform labels and k-points to cartesian coordinates

        path_car = []
        for subpath in path:
            subpath_car = []
            for kpoint in subpath:
                if isinstance(kpoint, str):
                    subpath_car.append(self.symmetry_points_car[kpoint])
                elif isinstance(kpoint, list):
                    if in_coord_type == 'car':
                        subpath_car.append(kpoint)
                    elif in_coord_type == 'red':
                        subpath_car.append(red_car([kpoint], self.rlattice)[0])
            path_car.append(subpath_car)

        return path_car

    def interpolate_path(self, path, nkpoints):
        """
        Given path in cartesian coordinates, interpolate given number of k-points,
        returning interpolated path as array of k-points.
        """
        nkpoints = int(nkpoints)

        #self.check_path_structure(path)

        # Obtain distances between consecutive points on path

        distances = [self.inter_distances(subpath) for subpath in path]

        # Obtain total path length

        path_length = 0
        for subpath_distances in distances:
            path_length += np.sum(subpath_distances)

        # Obtain number of interpolated k-points on each subpath

        self.subpath_nkpoints = [[int(nkpoints * distance / path_length) for distance in subdistances] for subdistances in distances]

        # Obtain interpolated path

        interpolated_path = []
        for n in range(len(path)):
            for i in range(len(path[n]) - 1):
                interpolated_path.append(path[n][i])
                interpolated_path += interpolate(path[n][i], path[n][i + 1], self.subpath_nkpoints[n][i])
            interpolated_path.append(path[n][-1])

        return np.array(interpolated_path)

    def inter_distances(self, kpoints):
        """
        Given list of k-points, obtain the distances between them
        """
        kpoints = np.array(kpoints)
        distances = []

        for nk in range(1, len(kpoints)):
            distances.append(np.linalg.norm(kpoints[nk - 1] - kpoints[nk]))

        return np.array(distances)

    def distances(self):
        """
        Obtain list of distances between first and consecutive k-points on interpolated path.
        """
        distance = 0
        distances = []
        kpoint1 = self.interpolated_path_car[0]

        for kpoint2 in self.interpolated_path_car:
            distance += np.linalg.norm(kpoint1 - kpoint2)
            distances.append(distance)
            kpoint1 = kpoint2
    
        return distances

