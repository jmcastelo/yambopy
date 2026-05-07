from qepy import PwXML
from yambopy import BrillouinZone
from math import sqrt

# k-points map
npoints = 50
intervals = [int(npoints*2), int(npoints), int(sqrt(5)*npoints)]
bz = BrillouinZone(ibrav=4, parameters={'a': 4.7, 'c': 12.0}, path_string='GMKG', intervals=intervals)

# Class PwXML. QE database reading
xml = PwXML(prefix='bn', path='bands')

# Class PwXML. QE database reading
xml.plot_eigen(bz)

# Alternative option with more plot control
## Matplotlib options 
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)

## Class PwXML. QE database reading
#xml.plot_eigen_ax(ax, bz)
#plt.show()
