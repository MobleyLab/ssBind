#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse

class MyParser(argparse.ArgumentParser):
    """display help message for every error"""
    def error(self, message):
        self.print_help()
        sys.stderr.write('\nERROR:\n  %s\n' % message)
        sys.exit(2)

def get_args():
    parser = MyParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--in_map',   help='input .map', required=True)
    parser.add_argument('-o', '--out_map',  help='output .map', required=True)
    parser.add_argument('-x', '--coords',   help='attraction center', required=True, nargs=3, type=float)
    parser.add_argument('-g', '--gap',      help='no penalty radius', default=1.2)
    parser.add_argument('-s', '--slope',    help='penalty slope', default=1e2)
    args = parser.parse_args()
    return args

class MyMap():
    def __init__(self, mapfname):

        self.header = ''

        with open(mapfname) as f:
            count = 0
            for line in f:
                count += 1
                self.header += line
                if line.startswith('NELEMENTS'):
                    _, nx, ny, nz = line.split()
                    # number of gridpoints increases by one if even
                    nx = int(nx) + (int(nx) + 1) % 2
                    ny = int(ny) + (int(ny) + 1) % 2
                    nz = int(nz) + (int(nz) + 1) % 2
                    self.nelements = (nx, ny, nz)
                elif line.startswith('CENTER'):
                    _, cx, cy, cz = line.split()
                    cx = float(cx)
                    cy = float(cy)
                    cz = float(cz)
                    self.center = (cx, cy, cz)
                elif line.startswith('SPACING'):
                    self.spacing = float(line.split()[1])
                if count == 6:
                    break

        xmin = cx - (nx/2) * self.spacing
        ymin = cy - (ny/2) * self.spacing
        zmin = cz - (nz/2) * self.spacing

        self.x_coords = [xmin + i * self.spacing for i in range(nx)]
        self.y_coords = [ymin + i * self.spacing for i in range(ny)]
        self.z_coords = [zmin + i * self.spacing for i in range(nz)]

        self.onedee = self._read_as_1D(mapfname)
        self.threedee = self._obsolete_1D_to_3D(self.onedee)
        
    def _read_as_1D(self, mapfname):
        with open(mapfname) as f:
            lines = f.readlines()
        onedee = np.array([float(line) for line in lines[6:]])
        return onedee

    def _obsolete_1D_to_3D(self, onedeearray):
        """
            this one is obsolute but may be useful in the future
            to convert 1D to 3D numpy arrays
            onedeearray is a 1D array in the z -> y -> x order
        """
        nx, ny, nz = self.nelements
        threedee = np.reshape(onedeearray, (nz, ny, nx))
        threedee = np.swapaxes(threedee, 0, 2)
        return threedee


def modify_and_write(mymap, point, fname, slope, radius):

    # distance from point of interest in each coordinate
    dx = np.abs([coord - point[0] for coord in mymap.x_coords])
    dy = np.abs([coord - point[1] for coord in mymap.y_coords])
    dz = np.abs([coord - point[2] for coord in mymap.z_coords])
    
    nx, ny, nz = mymap.nelements
    
    o = open(fname, 'w')
    o.write(mymap.header)
    
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                dist = np.sqrt(dx[ix]**2 + dy[iy]**2 + dz[iz]**2)
                if dist > radius:
                    o.write('%.3f\n' % (dist * slope))
                else:
                    o.write('%.3f\n' % mymap.threedee[ix, iy, iz])
    o.close()
    return


#if __name__ == '__main__':

args = get_args()
mymap = MyMap(args.in_map)
modify_and_write(mymap, args.coords, args.out_map, args.slope, args.gap)
