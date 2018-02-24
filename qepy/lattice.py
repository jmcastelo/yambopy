# Copyright (c) 2017, Henrique Miranda
# All rights reserved.
#
# This file is part of the yambopy project
#
from __future__ import division
import numpy as np

def calculate_distances(kpoints):
    """
    take a list of k-points and calculate the distances between all of them
    """
    kpoints = np.array(kpoints)
    distances = [0]
    distance = 0
    for nk in range(1,len(kpoints)):
        distance += np.linalg.norm(kpoints[nk-1]-kpoints[nk])
        distances.append(distance)   
    return distances

class Path(object):
    """ Class that defines a path in the brillouin zone
    """
    def __init__(self,klist,intervals):
        """
        Generation of a path in reciprocal space by specifying a list of k-points
        """
        self.intervals = intervals
        klabels = []
        kpoints = []

        for kline in klist:
            kpoint, klabel = kline
            kpoints.append(kpoint)
            klabels.append(klabel)
        self.kpoints = np.array(kpoints)
        self.klabels = klabels

    def get_klist(self):
        """ 
        Output in the format of quantum espresso == [ [kx, ky, kz, 1], ... ]
        """
        kpoints = self.kpoints
        intervals = self.intervals
        kout  = np.zeros([sum(intervals)+1,4])
        kout[:,3] = 1
        io = 0
        for ik,interval in enumerate(intervals):
          for ip in range(interval):
            kout[io,:3] = kpoints[ik] + float(ip)/interval*(kpoints[ik+1] - kpoints[ik])
            io = io + 1
        kout[io,:3] = kpoints[ik] + float(ip+1)/interval*(kpoints[ik+1] - kpoints[ik])

        return kout

    def get_indexes(self):
        """ get the index of each point of the path
        """

        indexes = []
        index = 0
        for n,label in enumerate(self.intervals):
            indexes.append([index,self.klabels[n]])
            index += self.intervals[n] 
        indexes.append([index,self.klabels[-1]])
        return indexes

def vec_in_list(veca,vec_list,atol=1e-6):
    """ check if a vector exists in a list of vectors
    """
    return np.array([ np.allclose(veca,vecb,rtol=atol,atol=atol) for vecb in vec_list ]).any()

def red_car(red,lat):
    """
    Convert reduced coordinates to cartesian
    """
    return np.array([coord[0]*lat[0]+coord[1]*lat[1]+coord[2]*lat[2] for coord in red])

def car_red(car,lat):
    """
    Convert cartesian coordinates to reduced
    """
    return np.array([np.linalg.solve(np.array(lat).T,coord) for coord in car])

def rec_lat(lat):
    """
    Calculate the reciprocal lattice vectors
    """
    a1,a2,a3 = np.array(lat)
    v = np.dot(a1,np.cross(a2,a3))
    b1 = np.cross(a2,a3)/v
    b2 = np.cross(a3,a1)/v
    b3 = np.cross(a1,a2)/v
    return np.array([b1,b2,b3])
