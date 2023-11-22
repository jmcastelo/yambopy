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

        # Initialize path with default BZ path

        self.set_path()

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
                        raise ValueError(f"K-Point must be specified by label ('X') or by coordinates with label [[Kx, Ky, Kz], 'X']: {element}")
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

    def check_intervals_structure(self, intervals, path):
        """
        Check if given intervals is valid and consistent with given path
        """
        # Check structure

        if not isinstance(intervals, list):
            raise ValueError("Intervals must be list of subintervals.")

        if all(isinstance(subinterval, list) for subinterval in intervals):
            for subinterval in intervals:
                if not all(isinstance(i, int) for i in subinterval):
                    raise ValueError(f"Interval must be an integer number: {i}")
        else:
            raise ValueError("Intervals must be list of subintervals.")

        # Check consistency

        nintervals1 = [len(subpath) - 1 for subpath in path]
        nintervals2 = [len(subinterval) for subinterval in intervals]

        if nintervals1 != nintervals2:
            raise ValueError("Path and intervals mismatch.")

    def set_path(self, **kwargs):
        """
        Sets the path, given list of subpaths. Each subpath represents a list of k-points,
        given by labels 'X' or coordinates and label [[Kx, Ky, Kz], 'X']
        """
        # Get arguments

        path = kwargs.get('path', self.bz.default_path)
        self.intervals = kwargs.get('intervals', None)
        in_coord_type = kwargs.get('in_coord_type', 'red')

        # Check input coordinates type

        if in_coord_type != 'car' and in_coord_type != 'red':
            raise ValueError(f"Unrecognized input coordinates type: {in_coord_type}")

        self.in_coord_type = in_coord_type

        # Check path structure and labels

        self.check_path_structure(path)
        self.check_path_labels(path)
        
        self.path = path

        # Check intervals structure

        if self.intervals != None:
            self.check_intervals_structure(self.intervals, self.path)

    def get_interpolated_path(self, out_coord_type = 'car', qe_out = False,  nkpoints = 100):
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

        # Transform path to cartesian coordinates
        
        path_car = self.transform_path_to_cartesian(self.path, self.in_coord_type)

        # Interpolate path

        ipath = self.interpolate_path(path_car, nkpoints)
        self.interpolated_path_car = ipath

        # Express it in reduced coordinates
        
        if out_coord_type == 'red':
            ipath = car_red(ipath, self.rlattice)

        # QE output format: [ [Kx, Ky, Kz, 1], ... ]
        
        if qe_out:
            qepath = []
            for i in range(len(ipath)):
                qepath.append(np.concatenate((ipath[i], [int(1)])))
            ipath = np.array(qepath)

        return ipath

    def transform_path_to_cartesian(self, path, in_coord_type):
        """
        Express given path in cartesian coordinates.
        """
        # Transform labels and k-points to cartesian coordinates

        path_car = []
        
        for subpath in path:
            subpath_car = []
            for element in subpath:
                if isinstance(element, str):
                    subpath_car.append(self.symmetry_points_car[element])
                elif isinstance(element, list):
                    if in_coord_type == 'car':
                        subpath_car.append(element[0])
                    elif in_coord_type == 'red':
                        subpath_car.append(red_car([element[0]], self.rlattice)[0])
            path_car.append(subpath_car)

        return path_car

    def transform_path_to_reduced(self, path, in_coord_type):
        """
        Express given path in reduced coordinates.
        """
        # Transform labels and k-points to cartesian coordinates

        path_red = []
        
        for subpath in path:
            subpath_red = []
            for element in subpath:
                if isinstance(element, str):
                    subpath_red.append(self.bz.symmetry_points[element])
                elif isinstance(element, list):
                    if in_coord_type == 'car':
                        subpath_red.append(car_red([element[0]], self.rlattice)[0])
                    elif in_coord_type == 'red':
                        subpath_red.append(element[0])
            path_red.append(subpath_red)

        return path_red

    def interpolate_path(self, path, nkpoints):
        """
        Given path in cartesian coordinates, interpolate given number of k-points,
        returning interpolated path as array of k-points.
        """
        nkpoints = int(nkpoints)

        # Obtain number of interpolating points between consecutive k-points on path
        
        if self.intervals != None:
            subpath_nkpoints = self.intervals
        else:
            subpath_nkpoints = self.get_interpolating_points(path, nkpoints)

        # Obtain interpolated path

        interpolated_path = []
        for n in range(len(path)):
            for i in range(len(path[n]) - 1):
                interpolated_path.append(path[n][i])
                interpolated_path += interpolate(path[n][i], path[n][i + 1], subpath_nkpoints[n][i])
            interpolated_path.append(path[n][-1])

        return np.array(interpolated_path)

    def get_interpolating_points(self, path, nkpoints):
        """
        Given path in cartesian coordinates, spread given number of k-points proportionally to inter-point lengths
        """
        nkpoints = int(nkpoints)

        # Obtain distances between consecutive points on path

        distances = [self.inter_distances(subpath) for subpath in path]

        # Obtain total path length

        path_length = 0
        for subpath_distances in distances:
            path_length += np.sum(subpath_distances)

        # Obtain number of interpolated k-points on each subpath

        subpath_nkpoints = [[int(nkpoints * distance / path_length) for distance in subdistances] for subdistances in distances]

        return subpath_nkpoints

    def inter_distances(self, kpoints):
        """
        Given list of k-points in cartesian coordinates, obtain the distances between them
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

    def get_path_labels(self, path):
        """
        Obtain list of labels of k-points on path 
        """
        labels = []

        for subpath in path:
            for element in subpath:
                if isinstance(element, str):
                    labels.append(element)
                elif isinstance(element, list):
                    labels.append(element[1])

        return labels

    def get_legacy_path(self, nkpoints):
        """
        Construct and return a legacy Path object
        """
        # Check nkpoints

        if nkpoints < 0:
            raise ValueError('Number of k-points must be positive integer.')

        # Obtain path labels

        labels = self.get_path_labels(self.path)

        # Obtain path in reduced coordinates

        path_red = self.transform_path_to_reduced(self.path, self.in_coord_type)
        path_red = [kpoint for subpath in path_red for kpoint in subpath]

        # Construct list of k-points and labels

        klist = [list(k) for k in zip(path_red, labels)]

        # Obtain path in cartesian coordinates

        path_car = self.transform_path_to_cartesian(self.path, self.in_coord_type)

        # Obtain intervals

        if self.intervals != None:
            interpolating_points = self.intervals
        else:
            interpolating_points = self.get_interpolating_points(path_car, nkpoints)

        intervals = []

        for subpath in interpolating_points:
            for ipoints in subpath:
                intervals.append(ipoints + 1)
            intervals.append(1)
        intervals.pop()

        return Path(klist, intervals)
