from qepy import PwXML
from yambopy import BrillouinZone

# k-points map
npoints = 50
intervals=[npoints,npoints,npoints,npoints,npoints]
extra_points = { 'H': [0.0, 0.0, 1.0], 'N': [1/2, 0.0, 1/2], 'P': [1/2, 1/2, 1/2] }
bz = BrillouinZone(ibrav=3, parameters={'a': 5.42}, path_string='GHNGPN', extra_points=extra_points, intervals=intervals)

# Class PwXML. QE database reading
xml = PwXML(prefix='pw',path='bands')

# Class PwXML. QE database reading
xml.plot_eigen(bz)
