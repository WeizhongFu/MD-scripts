import numpy as np
import numba
from numba import jit
'''
Read me:
This is a python module for analysing H-bonds of MD trajectory.
To use this module firstly you should use the ase.read() function to input the trajectory files (this read() function support so many formats such .xyz and XDATCAR and so on) into a Atoms object (which is a list of frames of the trajectory).
Then you could create a HbondsAnalyse object and input the frames to analyse one by one. For initiating, you should input the first frame and a cell (a 3*1 array) and the ids of those H,O atoms you care.
To input a new frame, use the next_frame() function.
Here I have offered two analysing function, donate_hbonds_statistics() and accept_hbonds_statistics(). They can give a statistics of two kinds of H-bonds between the two groups of O atoms you assign.
I use the 3.5-30 as a default judgement here, and you can change it if necessary.

An using example:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from ase.io import read
import Hbonds

Atoms = read('filepath', index=':')
O_list, H_list, cell = ...
my_hbonds_analyse = Hbonds.HbondsAnalyse(Atoms[0], cell, O_list, H_list) # to initiating
for atoms in Atoms:
    my_hbonds_analyse.next_frame(atoms)
    # and do some analysis here
# save the data
# over
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

!!! Please notice that here cell must be a 3*1 array but not a 3*3 array, which means this script can only deal with systems whose cell is orthorhombic. To support a 3*3 cell, you should improve the distance function yourself.
'''


@jit
def distance(pos1, pos2, cell):
    delta = np.abs(pos1 - pos2)
    delta = np.where(delta > 0.5 * cell, delta - cell, delta)
    return np.linalg.norm(delta)


@jit
def angle(a, b, c):
    return np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))


class HbondsAnalyse():
    def __init__(self,
                 atoms,
                 cell,
                 Olist,
                 Hlist,
                 cut_off_dist=3.5,
                 judge_angle=30):
        self.atoms = atoms
        self.cell = cell
        self.Hlist = Hlist
        self.Olist = Olist
        self.Oown = self.init_Oown()
        self.Hbelong = self.init_Hbelong()
        self.OHpairs_analyse_1st()
        self.cut_off_dist = cut_off_dist
        self.judge_angle = judge_angle

    def init_Oown(self):
        res = {}
        for id in self.Olist:
            res[id] = []
        return res

    def init_Hbelong(self):
        res = {}
        for id in self.Hlist:
            res[id] = [-1, 0]
        return res

    def OHpairs_analyse_1st(self):
        for hid in self.Hlist:
            Oid_tmp = -1
            OHdis_tmp = 100
            for oid in self.Olist:
                oh_dis = distance(self.atoms[hid].position,
                                  self.atoms[oid].position, self.cell)
                if oh_dis < OHdis_tmp:
                    Oid_tmp = oid
                    OHdis_tmp = oh_dis
            if Oid_tmp == -1:
                print(
                    "Here's something wrong in function OHpairs_analyse_1st!")
            else:
                self.Hbelong[hid] = [Oid_tmp, OHdis_tmp]
                self.Oown[Oid_tmp].append(hid)

    def OHpairs_analyse(self):
        for hid in self.Hlist:
            newdis = distance(self.atoms[hid].position,
                              self.atoms[self.Hbelong[hid][0]].position,
                              self.cell)
            if newdis < 1.15:
                self.Hbelong[hid][1] = newdis
            else:
                Oid_tmp = -1
                OHdis_tmp = 100
                for oid in self.Olist:
                    oh_dis = distance(self.atoms[hid].position,
                                      self.atoms[oid].position, self.cell)
                    if oh_dis < OHdis_tmp:
                        Oid_tmp = oid
                        OHdis_tmp = oh_dis
                if Oid_tmp == -1:
                    print(
                        "Here's something wrong in function OHpairs_analyse!")
                else:
                    self.Oown[self.Hbelong[hid][0]].remove(hid)
                    self.Hbelong[hid] = [Oid_tmp, OHdis_tmp]
                    self.Oown[Oid_tmp].append(hid)

    def next_frame(self, atoms):
        self.atoms = atoms
        self.OHpairs_analyse()

    def donate_hbonds_statistics(self, donate_Olist, other_Olist):
        res = []
        for Od_id in donate_Olist:
            if self.Oown[Od_id] == []:
                pass
            else:
                for Oa_id in other_Olist:
                    if abs(self.atoms[Oa_id].position[2] -
                           self.atoms[Od_id].position[2]) < self.cut_off_dist:
                        OOdis = distance(self.atoms[Od_id].position,
                                         self.atoms[Oa_id].position, self.cell)
                        if OOdis < self.cut_off_dist:
                            for H_id in self.Oown[Od_id]:
                                HOdis = distance(self.atoms[Oa_id].position,
                                                 self.atoms[H_id].position,
                                                 self.cell)
                                if angle(OOdis, self.Hbelong[H_id][1],
                                         HOdis) < self.judge_angle:
                                    res.append(HOdis)
        return res

    def accept_hbonds_statistics(self, accept_Olist, other_Olist):
        res = []
        for Oa_id in accept_Olist:
            for Od_id in other_Olist:
                if self.Oown[Od_id] == []:
                    pass
                else:
                    if abs(self.atoms[Oa_id].position[2] -
                           self.atoms[Od_id].position[2]) < self.cut_off_dist:
                        OOdis = distance(self.atoms[Od_id].position,
                                         self.atoms[Oa_id].position, self.cell)
                        if OOdis < self.cut_off_dist:
                            for H_id in self.Oown[Od_id]:
                                HOdis = distance(self.atoms[H_id].position,
                                                 self.atoms[Oa_id].position,
                                                 self.cell)
                                if angle(OOdis, self.Hbelong[H_id][1],
                                         HOdis) < self.judge_angle:
                                    res.append(HOdis)
        return res
