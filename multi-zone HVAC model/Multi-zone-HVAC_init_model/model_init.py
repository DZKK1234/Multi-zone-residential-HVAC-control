import numpy as np
import pandas as pd
from function_building import *
import math
from scipy.integrate import odeint

class five_room_model(object):

    def __init__(self, location, C_EXTwall, C_INTwall, C_ceil, C_floor, C_window, dx_EXTwall, dx_INTwall,
                 dx_ceil, dx_floor, N_EXTwall, N_INTwall, N_ceil, N_floor, V, Rou_air, Cp_air, Lamda_EXTwall,
                 h_out, h_in_wall, hr, Lamda_INTwall, N_window, Lamda_ceil, h_in_ceil, h_in_floor, h_window_12, dt, f,
                 f_tol, eb, Ta_targ_min, Ta_targ_max, Lamda_floor ,n_air, HC_mode):
        self.location = location
        self.C_EXTwall = C_EXTwall
        self.C_INTwall = C_INTwall
        self.C_ceil = C_ceil
        self.C_floor = C_floor
        self.C_window = C_window
        self.dx_EXTwall = dx_EXTwall
        self.dx_INTwall = dx_INTwall
        self.dx_ceil = dx_ceil
        self.dx_floor = dx_floor
        self.N_EXTwall = N_EXTwall
        self.N_INTwall = N_INTwall
        self.N_ceil = N_ceil
        self.N_floor = N_floor
        self.V = V
        self.Cp_air = Cp_air
        self.Rou_air = Rou_air
        self.Lamda_EXTwall = Lamda_EXTwall
        self.h_out = h_out
        self.h_in_wall = h_in_wall
        self.hr = hr
        self.N_window = N_window
        self.Lamda_ceil = Lamda_ceil
        self.h_in_ceil = h_in_ceil
        self.h_in_floor = h_in_floor
        self.h_window_12 = h_window_12
        self.dt = dt
        self.f = f
        self.f_tol = f_tol
        self.eb = eb
        self.Ta_targ_min = Ta_targ_min
        self.Ta_targ_max = Ta_targ_max
        self.Lamda_INTwall = Lamda_INTwall
        self.Lamda_floor = Lamda_floor
        self.n_air = n_air
        self.HC_mode = HC_mode
        # 考虑所有的温度计算节点(建筑构件单元的温度节点和室内空气节点)，分别为5个房间构建整个热容矩阵C
        self.C_main = np.vstack((np.hstack((self.C_EXTwall, np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                       np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                       np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                       np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                       np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                       np.zeros((len(self.C_EXTwall), len(self.C_window))))),
                       np.hstack((np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                       self.C_INTwall, np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                       np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                       np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                       np.zeros((len(self.C_INTwall), len(self.C_floor))),
                       np.zeros((len(self.C_INTwall), len(self.C_window))))),
                       np.hstack((np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                       np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                       self.C_EXTwall, np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                       np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                       np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                       np.zeros((len(self.C_EXTwall), len(self.C_window))))),
                       np.hstack((np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                       np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                       np.zeros((len(self.C_INTwall), len(self.C_EXTwall))), self.C_INTwall,
                       np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                       np.zeros((len(self.C_INTwall), len(self.C_floor))),
                       np.zeros((len(self.C_INTwall), len(self.C_window))))),
                       np.hstack((np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                       np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                       np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                       np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                       self.C_ceil, np.zeros((len(self.C_ceil), len(self.C_floor))),
                       np.zeros((len(self.C_ceil), len(self.C_window))))),
                       np.hstack((np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                       np.zeros((len(self.C_floor), len(self.C_INTwall))),
                       np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                       np.zeros((len(self.C_floor), len(self.C_INTwall))),
                       np.zeros((len(self.C_floor), len(self.C_ceil))),
                       self.C_floor, np.zeros((len(self.C_floor), len(self.C_window))))),
                       np.hstack((np.zeros((len(self.C_window), len(self.C_EXTwall))),
                       np.zeros((len(self.C_window), len(self.C_INTwall))),
                       np.zeros((len(self.C_window), len(self.C_EXTwall))),
                       np.zeros((len(self.C_window), len(self.C_INTwall))),
                       np.zeros((len(self.C_window), len(self.C_ceil))),
                       np.zeros((len(self.C_window), len(self.C_floor))), self.C_window))))
        self.C_main2 = np.vstack((np.hstack((self.C_EXTwall, np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                        np.zeros((len(self.C_EXTwall), len(self.C_window))),
                        np.zeros((len(self.C_EXTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        self.C_INTwall, np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))),
                        np.zeros((len(self.C_INTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))), self.C_INTwall,
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))),
                        np.zeros((len(self.C_INTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))), self.C_INTwall,
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))),
                        np.zeros((len(self.C_INTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        self.C_ceil, np.zeros((len(self.C_ceil), len(self.C_floor))),
                        np.zeros((len(self.C_ceil), len(self.C_window))),
                        np.zeros((len(self.C_ceil), 1)))),
                        np.hstack((np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                        np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_ceil))),
                        self.C_floor, np.zeros((len(self.C_floor), len(self.C_window))),
                        np.zeros((len(self.C_floor), 1)))),
                        np.hstack((np.zeros((len(self.C_window), len(self.C_EXTwall))),
                        np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_ceil))),
                        np.zeros((len(self.C_window), len(self.C_floor))), self.C_window,
                        np.zeros((len(self.C_window), 1)))),
                        np.hstack((np.zeros((1, len(self.C_EXTwall))),
                        np.zeros((1, len(self.C_INTwall))),
                        np.zeros((1, len(self.C_INTwall))),
                        np.zeros((1, len(self.C_INTwall))),
                        np.zeros((1, len(self.C_ceil))),
                        np.zeros((1, len(self.C_floor))),
                        np.zeros((1, len(self.C_window))),
                        np.array([[self.Cp_air * self.Rou_air * self.V[1]]])))))
        self.C_main3 = np.vstack((np.hstack((self.C_EXTwall, np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                        np.zeros((len(self.C_EXTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        self.C_INTwall, np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        self.C_INTwall, np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_INTwall))), self.C_EXTwall,
                        np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                        np.zeros((len(self.C_EXTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                        self.C_ceil, np.zeros((len(self.C_ceil), len(self.C_floor))),
                        np.zeros((len(self.C_ceil), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                        np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                        np.zeros((len(self.C_floor), len(self.C_ceil))),
                        self.C_floor, np.zeros((len(self.C_floor), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_window), len(self.C_EXTwall))),
                        np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_EXTwall))),
                        np.zeros((len(self.C_window), len(self.C_ceil))),
                        np.zeros((len(self.C_window), len(self.C_floor))),
                                   self.C_window))))
        self.C_main4 = np.vstack((np.hstack((self.C_INTwall, np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))),
                        np.zeros((len(self.C_INTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        self.C_EXTwall, np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                        np.zeros((len(self.C_EXTwall), len(self.C_window))),
                        np.zeros((len(self.C_EXTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        self.C_INTwall, np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))),
                        np.zeros((len(self.C_INTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_INTwall))), self.C_EXTwall,
                        np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                        np.zeros((len(self.C_EXTwall), len(self.C_window))),
                        np.zeros((len(self.C_EXTwall), 1)))),
                        np.hstack((np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                        self.C_ceil, np.zeros((len(self.C_ceil), len(self.C_floor))),
                        np.zeros((len(self.C_ceil), len(self.C_window))),
                        np.zeros((len(self.C_ceil), 1)))),
                        np.hstack((np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                        np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                        np.zeros((len(self.C_floor), len(self.C_ceil))),
                        self.C_floor, np.zeros((len(self.C_floor), len(self.C_window))),
                        np.zeros((len(self.C_floor), 1)))),
                        np.hstack((np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_EXTwall))),
                        np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_EXTwall))),
                        np.zeros((len(self.C_window), len(self.C_ceil))),
                        np.zeros((len(self.C_window), len(self.C_floor))), self.C_window,
                        np.zeros((len(self.C_window), 1)))),
                        np.hstack((np.zeros((1, len(self.C_INTwall))),
                        np.zeros((1, len(self.C_EXTwall))),
                        np.zeros((1, len(self.C_INTwall))),
                        np.zeros((1, len(self.C_EXTwall))),
                        np.zeros((1, len(self.C_ceil))),
                        np.zeros((1, len(self.C_floor))),
                        np.zeros((1, len(self.C_window))),
                        np.array([[self.Cp_air * self.Rou_air * self.V[3]]])))))
        self.C_main5 = np.vstack((np.hstack((self.C_INTwall, np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        self.C_EXTwall, np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                        np.zeros((len(self.C_EXTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_EXTwall))),
                        self.C_EXTwall, np.zeros((len(self.C_EXTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_EXTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_EXTwall), len(self.C_floor))),
                        np.zeros((len(self.C_EXTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_INTwall), len(self.C_INTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_EXTwall))),
                        np.zeros((len(self.C_INTwall), len(self.C_EXTwall))), self.C_INTwall,
                        np.zeros((len(self.C_INTwall), len(self.C_ceil))),
                        np.zeros((len(self.C_INTwall), len(self.C_floor))),
                        np.zeros((len(self.C_INTwall), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_EXTwall))),
                        np.zeros((len(self.C_ceil), len(self.C_INTwall))),
                        self.C_ceil, np.zeros((len(self.C_ceil), len(self.C_floor))),
                        np.zeros((len(self.C_ceil), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                        np.zeros((len(self.C_floor), len(self.C_EXTwall))),
                        np.zeros((len(self.C_floor), len(self.C_INTwall))),
                        np.zeros((len(self.C_floor), len(self.C_ceil))),
                        self.C_floor, np.zeros((len(self.C_floor), len(self.C_window))))),
                        np.hstack((np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_EXTwall))),
                        np.zeros((len(self.C_window), len(self.C_EXTwall))),
                        np.zeros((len(self.C_window), len(self.C_INTwall))),
                        np.zeros((len(self.C_window), len(self.C_ceil))),
                        np.zeros((len(self.C_window), len(self.C_floor))), self.C_window))))
    def Construct_heat_flux_flow_relationship_room1(self):
        # 从相邻房间的对流和辐射换热
        # east wall
        A_wall_1_R1 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[0][1], self.hr[0][2], self.hr[0][3], self.hr[0][4], self.hr[0][5], self.hr[0][6])
        # west wall
        A_wall_2_R1 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[0][0], self.hr[0][2], self.hr[0][3], self.hr[0][4], self.hr[0][5], self.hr[0][6],
                                   self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][5], self.hr[4][6])
        # south wall
        A_wall_3_R1 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[0][0], self.hr[0][1], self.hr[0][3], self.hr[0][4], self.hr[0][5], self.hr[0][6])
        # north wall
        A_wall_4_R1 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[0][0], self.hr[0][1], self.hr[0][2], self.hr[0][4], self.hr[0][5], self.hr[0][6],
                                   self.hr[1][0], self.hr[1][1], self.hr[1][3], self.hr[1][4], self.hr[1][5], self.hr[1][6])
        # ceiling
        A_ceil_R1 = setA_INTwall(self.N_ceil, self.Lamda_ceil, self.dx_ceil, self.h_in_ceil, self.h_in_floor, self.hr[0][0],
                                 self.hr[0][1], self.hr[0][2], self.hr[0][3], self.hr[0][5], self.hr[0][6],self.hr[0][0],
                                 self.hr[0][1], self.hr[0][2], self.hr[0][3], self.hr[0][4], self.hr[0][6])
        A_ceil_R1[0][np.sum(self.N_ceil[:], dtype=int)] = self.hr[0][4]
        # floor
        A_floor_R1 = setA_INTwall(self.N_floor, self.Lamda_floor, self.dx_floor, self.h_in_floor, self.h_in_ceil,
                                  self.hr[0][0], self.hr[0][1], self.hr[0][2], self.hr[0][3], self.hr[0][4], self.hr[0][6],
                                  self.hr[0][0], self.hr[0][1], self.hr[0][2], self.hr[0][3], self.hr[0][5], self.hr[0][6])
        A_floor_R1[0][np.sum(self.N_floor[:], dtype=int)] = self.hr[0][5]
        # window
        A_window_R1 = setA_window(self.N_window, self.h_out, self.h_in_wall, self.h_window_12, self.hr[0][0], self.hr[0][1],
                                  self.hr[0][2], self.hr[0][3], self.hr[0][4], self.hr[0][5])
        # 东外墙的辐射换热
        A12_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][0])
        A13_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][0])
        A14_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][0])
        A15_R1 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[0][0])
        A16_R1 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[0][0])
        A17_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[0][0])
        # 西内墙辐射换热
        A21_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][1])
        A23_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][1])
        A24_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][1])
        A25_R1 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[0][1])
        A26_R1 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[0][1])
        A27_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[0][1])
        # 南外墙的辐射换热
        A31_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][2])
        A32_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][2])
        A34_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][2])
        A35_R1 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[0][2])
        A36_R1 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[0][2])
        A37_R1 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[0][2])
        # 北内墙的辐射换热
        A41_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][3])
        A42_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][3])
        A43_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][3])
        A45_R1 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[0][3])
        A46_R1 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[0][3])
        A47_R1 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[0][3])
        # 天花板的辐射换热
        A51_R1 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][4])
        A52_R1 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][4])
        A53_R1 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][4])
        A54_R1 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][4])
        A56_R1 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[0][4])
        A57_R1 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), self.N_window, self.hr[0][4])
        # 窗户的辐射换热
        A61_R1 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][5])
        A62_R1 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][5])
        A63_R1 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][5])
        A64_R1 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][5])
        A65_R1 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[0][5])
        A67_R1 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), self.N_window, self.hr[0][5])
        # 室内空气与第i个建筑围护结构之间的对流传热
        A71_R1 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][6])
        A72_R1 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][6])
        A73_R1 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[0][6])
        A74_R1 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[0][6])
        A75_R1 = A_lwr_cyc(self.N_window, (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[0][6])
        A76_R1 = A_lwr_cyc(self.N_window, (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[0][6])
        # 建筑围护结构部件之间的对流传热
        # 房间1 -卧室的室内空气
        # Aconv_air_wall1_R1 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall2_R1 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall3_R1 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall4_R1 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_ceil_R1 = A_conv_air_CYCwall((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_floor, self.h_in_ceil)
        # Aconv_air_floor_R1 = A_conv_air_CYCwall((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_ceil, self.h_in_floor)
        # Acon_air_window_R1 = A_conv_air_wall(self.N_window, self.h_in_wall)
        # 矩阵A表示由温度引起的热流运动
        # 相邻温度节点的差异
        # Aconv_wall1_R1_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[0][0])
        # Aconv_wall2_R1_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[0][1])
        # Aconv_wall3_R1_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[0][2])
        # Aconv_wall4_R1_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[0][3])
        # Aconv_ceil_R1_air = A_conv_wall_air((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_ceil, self.f[0][4])
        # Aconv_floor_R1_air = A_conv_wall_air((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_floor, self.f[0][5])
        # Aconv_window_R1_air = A_conv_wall_air(self.N_window, self.h_in_wall, self.f[0][6])
        A_main = np.vstack((np.hstack((A_wall_1_R1, A21_R1, A31_R1, A41_R1, A51_R1, A61_R1, A71_R1)),
                  np.hstack((A12_R1, A_wall_2_R1, A32_R1, A42_R1, A52_R1, A62_R1, A72_R1)),
                  np.hstack((A13_R1, A23_R1, A_wall_3_R1, A43_R1, A53_R1, A63_R1, A73_R1)),
                  np.hstack((A14_R1, A24_R1, A34_R1, A_wall_4_R1, A54_R1, A64_R1, A74_R1)),
                  np.hstack((A15_R1, A25_R1, A35_R1, A45_R1, A_ceil_R1, A65_R1, A75_R1)),
                  np.hstack((A16_R1, A26_R1, A36_R1, A46_R1, A56_R1, A_floor_R1, A76_R1)),
                  np.hstack((A17_R1, A27_R1, A37_R1, A47_R1, A57_R1, A67_R1, A_window_R1))))

        return A_main

    def Construct_heat_flux_flow_relationship_room2(self):
        # 2号房间(厕所)的墙壁、天花板地板和窗户矩阵
        A_wall_1_R2 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[1][1], self.hr[1][2], self.hr[1][3], self.hr[1][4], self.hr[1][5], self.hr[1][6])
        A_wall_2_R2 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[1][0], self.hr[1][2], self.hr[1][3], self.hr[1][4], self.hr[1][5], self.hr[1][6],
                                   self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][5], self.hr[4][6]) # 邻近墙- 5号房间的东墙
        A_wall_3_R2 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[1][0], self.hr[1][1], self.hr[1][3], self.hr[1][4], self.hr[1][5], self.hr[1][6],
                                   self.hr[0][0], self.hr[0][1], self.hr[0][2], self.hr[0][4], self.hr[0][5], self.hr[0][6]) # 相邻墙- 1号房间的北墙
        A_wall_4_R2 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[1][0], self.hr[1][1], self.hr[1][2], self.hr[1][4], self.hr[1][5], self.hr[1][6],
                                   self.hr[2][0], self.hr[2][1], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6]) # 相邻的墙，3号房间的南墙
        A_ceil_R2 = setA_INTwall(self.N_ceil, self.Lamda_ceil, self.dx_ceil, self.h_in_ceil, self.h_in_floor, self.hr[1][0],
                                 self.hr[1][1], self.hr[1][2], self.hr[1][3], self.hr[1][5], self.hr[1][6], self.hr[1][0],
                                 self.hr[1][1], self.hr[1][2], self.hr[1][3], self.hr[1][4], self.hr[1][6])
        A_ceil_R2[0][np.sum(N_ceil[:], dtype=int)]= self.hr[1][4]
        A_floor_R2 = setA_INTwall(self.N_floor, self.Lamda_floor, self.dx_floor, self.h_in_floor, self.h_in_ceil, self.hr[1][0],
                                  self.hr[1][1], self.hr[1][2], self.hr[1][3], self.hr[1][5], self.hr[1][6], self.hr[1][0],
                                  self.hr[1][1], self.hr[1][2], self.hr[1][3], self.hr[1][5], self.hr[1][6])
        A_floor_R2[0][np.sum(self.N_floor[:], dtype=int)] = self.hr[1][5]
        A_window_R2 = setA_window(self.N_window, self.h_out, self.h_in_wall, self.h_window_12, self.hr[1][0], self.hr[1][1],
                                  self.hr[1][2], self.hr[1][3], self.hr[1][4], self.hr[1][5])
        # 东外墙的辐射换热
        A12_R2 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][0])
        A13_R2 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][0])
        A14_R2 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][0])
        A15_R2 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[1][0])
        A16_R2 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[1][0])
        A17_R2 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[1][0])
        # 西内墙辐射换热
        A21_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[1][1])
        A23_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][1])
        A24_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][1])
        A25_R2 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[1][1])
        A26_R2 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[1][1])
        A27_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[1][1])
        # 南外墙的辐射换热
        A31_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[1][2])
        A32_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][2])
        A34_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][2])
        A35_R2 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[1][2])
        A36_R2 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[1][2])
        A37_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[1][2])
        # 北内墙的辐射换热
        A41_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[1][3])
        A42_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][3])
        A43_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][3])
        A45_R2 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[1][3])
        A46_R2 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[1][3])
        A47_R2 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[1][3])
        # 天花板的辐射换热
        A51_R2 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[1][4])
        A52_R2 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][4])
        A53_R2 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][4])
        A54_R2 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][4])
        A56_R2 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                       self.hr[1][4])
        A57_R2 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), self.N_window, self.hr[1][4])
        # 窗户的辐射换热
        A61_R2 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[1][5])
        A62_R2 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][5])
        A63_R2 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][5])
        A64_R2 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[1][5])
        A65_R2 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                       self.hr[1][5])
        A67_R2 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), self.N_window, self.hr[1][5])
        # 室内空气与第i个建筑围护结构之间的对流传热
        A71_R2 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[1][6])
        A72_R2 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[1][6])
        A73_R2 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[1][6])
        A74_R2 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[1][6])
        A75_R2 = A_lwr_cyc(self.N_window, (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[1][6])
        A76_R2 = A_lwr_cyc(self.N_window, (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[1][6])
        # 建筑围护结构部件之间的对流传热
        # 和房间2的室内空气
        Aconv_air_wall1_R2 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_wall2_R2 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_wall3_R2 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_wall4_R2 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_ceil_R2 = A_conv_air_CYCwall((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_floor, self.h_in_ceil)
        Aconv_air_floor_R2 = A_conv_air_CYCwall((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_ceil, self.h_in_floor)
        Acon_air_window_R2 = A_conv_air_wall(self.N_window, self.h_in_wall)
        # 相邻温度节点的差异
        Aconv_wall1_R2_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[1][0])
        Aconv_wall2_R2_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[1][1])
        Aconv_wall3_R2_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[1][2])
        Aconv_wall4_R2_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[1][3])
        Aconv_ceil_R2_air = A_conv_wall_air((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_ceil, self.f[1][4])
        Aconv_floor_R2_air = A_conv_wall_air((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_floor, self.f[1][5])
        Aconv_window_R2_air = A_conv_wall_air(self.N_window, self.h_in_wall, self.f[1][6])

        A_main2 = np.vstack((np.hstack((A_wall_1_R2, A21_R2, A31_R2, A41_R2, A51_R2, A61_R2, A71_R2, Aconv_air_wall1_R2)),
                   np.hstack((A12_R2, A_wall_2_R2, A32_R2, A42_R2, A52_R2, A62_R2, A72_R2, Aconv_air_wall2_R2)),
                   np.hstack((A13_R2, A23_R2, A_wall_3_R2, A43_R2, A53_R2, A63_R2, A73_R2, Aconv_air_wall3_R2)),
                   np.hstack((A14_R2, A24_R2, A34_R2, A_wall_4_R2, A54_R2, A64_R2, A74_R2, Aconv_air_wall4_R2)),
                   np.hstack((A15_R2, A25_R2, A35_R2, A45_R2, A_ceil_R2, A65_R2, A75_R2, Aconv_air_ceil_R2)),
                   np.hstack((A16_R2, A26_R2, A36_R2, A46_R2, A56_R2, A_floor_R2, A76_R2, Aconv_air_floor_R2)),
                   np.hstack((A17_R2, A27_R2, A37_R2, A47_R2, A57_R2, A67_R2, A_window_R2, Acon_air_window_R2)),
                   np.hstack((Aconv_wall1_R2_air, Aconv_wall2_R2_air, Aconv_wall3_R2_air, Aconv_wall4_R2_air,
                   Aconv_ceil_R2_air, Aconv_floor_R2_air, Aconv_window_R2_air,
                   np.array([[-self.h_in_wall * (np.sum(self.f[1, 0 : 4])) - self.h_in_wall * self.f[1][6] - self.h_in_ceil *
                              self.f[1][4] - self.h_in_floor * self.f[1][5] - self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[2]]])))))

        return A_main2

    def Construct_heat_flux_flow_relationship_room3(self):
        # 从相邻房间的对流和辐射换热
        # east wall
        A_wall_1_R3 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[2][1], self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6])
        # west wall
        A_wall_2_R3 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[2][0], self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6],
                                   0.667 * self.hr[3][1] + 0.333 * self.hr[4][1], 0.667 * self.hr[3][2] + 0.333 * self.hr[4][2],
                                   0.667 * self.hr[3][3] + 0.333 * self.hr[4][3], 0.667 * self.hr[3][4] + 0.333 * self.hr[4][4],
                                   0.667 * self.hr[3][5] + 0.333 * self.hr[4][5], 0.667 * self.hr[3][6] + 0.333 * self.hr[4][6])
        # south wall
        A_wall_3_R3 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[2][0], self.hr[2][1], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6],
                                   self.hr[1][0], self.hr[1][1], self.hr[1][2], self.hr[1][4], self.hr[1][5], self.hr[1][6])
        # north wall
        A_wall_4_R3 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                   self.hr[2][0], self.hr[2][1], self.hr[2][2], self.hr[2][4], self.hr[2][5], self.hr[2][6])
        # ceiling
        A_ceil_R3 = setA_INTwall(self.N_ceil, self.Lamda_ceil, self.dx_ceil, self.h_in_ceil, self.h_in_floor, self.hr[2][0],
                                 self.hr[2][1], self.hr[2][2], self.hr[2][3], self.hr[2][5], self.hr[2][6],self.hr[2][0],
                                 self.hr[2][1], self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][6])
        A_ceil_R3[0][np.sum(self.N_ceil[:], dtype=int)] = self.hr[2][4]
        # floor
        A_floor_R3 = setA_INTwall(self.N_floor, self.Lamda_floor, self.dx_floor, self.h_in_floor, self.h_in_ceil,
                                  self.hr[2][0], self.hr[2][1], self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][6],
                                  self.hr[2][0], self.hr[2][1], self.hr[2][2], self.hr[2][3], self.hr[2][5], self.hr[2][6])
        A_floor_R3[0][np.sum(self.N_floor[:], dtype=int)] = self.hr[2][5]
        # window
        A_window_R3 = setA_window(self.N_window, self.h_out, self.h_in_wall, self.h_window_12, self.hr[2][0], self.hr[2][1],
                                  self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][5])
        # 东外墙的辐射换热
        A12_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][0])
        A13_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][0])
        A14_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][0])
        A15_R3 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[2][0])
        A16_R3 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[2][0])
        A17_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[2][0])
        # 西内墙辐射换热
        A21_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][1])
        A23_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][1])
        A24_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][1])
        A25_R3 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[2][1])
        A26_R3 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[2][1])
        A27_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[2][1])
        # 南外墙的辐射换热
        A31_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][2])
        A32_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][2])
        A34_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][2])
        A35_R3 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[2][2])
        A36_R3 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[2][2])
        A37_R3 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[2][2])
        # 北内墙的辐射换热
        A41_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[2][3])
        A42_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[2][3])
        A43_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[2][3])
        A45_R3 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[2][3])
        A46_R3 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[2][3])
        A47_R3 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[2][3])
        # 天花板的辐射换热
        A51_R3 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][4])
        A52_R3 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][4])
        A53_R3 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][4])
        A54_R3 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][4])
        A56_R3 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[2][4])
        A57_R3 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), self.N_window, self.hr[2][4])
        # 窗户的辐射换热
        A61_R3 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][5])
        A62_R3 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][5])
        A63_R3 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][5])
        A64_R3 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][5])
        A65_R3 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[2][5])
        A67_R3 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), self.N_window, self.hr[2][5])
        # 室内空气与第i个建筑围护结构之间的对流传热
        A71_R3 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][6])
        A72_R3 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][6])
        A73_R3 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[2][6])
        A74_R3 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[2][6])
        A75_R3 = A_lwr_cyc(self.N_window, (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[2][6])
        A76_R3 = A_lwr_cyc(self.N_window, (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[2][6])
        # 建筑围护结构部件之间的对流传热
        # 房间1 -卧室的室内空气
        # Aconv_air_wall1_R3 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall2_R3 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall3_R3 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall4_R3 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_ceil_R3 = A_conv_air_CYCwall((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_floor, self.h_in_ceil)
        # Aconv_air_floor_R3 = A_conv_air_CYCwall((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_ceil, self.h_in_floor)
        # Acon_air_window_R3 = A_conv_air_wall(self.N_window, self.h_in_wall)
        # 矩阵A表示由温度引起的热流运动
        # 相邻温度节点的差异
        # Aconv_wall1_R3_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][0])
        # Aconv_wall2_R3_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][1])
        # Aconv_wall3_R3_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][2])
        # Aconv_wall4_R3_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][3])
        # Aconv_ceil_R3_air = A_conv_wall_air((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_ceil, self.f[3][4])
        # Aconv_floor_R3_air = A_conv_wall_air((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_floor, self.f[3][5])
        # Aconv_window_R3_air = A_conv_wall_air(self.N_window, self.h_in_wall, self.f[3][6])
        A_main3 = np.vstack((np.hstack((A_wall_1_R3, A21_R3, A31_R3, A41_R3, A51_R3, A61_R3, A71_R3)),
                   np.hstack((A12_R3, A_wall_2_R3, A32_R3, A42_R3, A52_R3, A62_R3, A72_R3)),
                   np.hstack((A13_R3, A23_R3, A_wall_3_R3, A43_R3, A53_R3, A63_R3, A73_R3)),
                   np.hstack((A14_R3, A24_R3, A34_R3, A_wall_4_R3, A54_R3, A64_R3, A74_R3)),
                   np.hstack((A15_R3, A25_R3, A35_R3, A45_R3, A_ceil_R3, A65_R3, A75_R3)),
                   np.hstack((A16_R3, A26_R3, A36_R3, A46_R3, A56_R3, A_floor_R3, A76_R3)),
                   np.hstack((A17_R3, A27_R3, A37_R3, A47_R3, A57_R3, A67_R3, A_window_R3))))

        return A_main3

    def Construct_heat_flux_flow_relationship_room4(self):

        A_wall_2_R4 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[3][0], self.hr[3][2], self.hr[3][3], self.hr[3][4], self.hr[3][5], self.hr[3][6])
        A_wall_1_R4 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[3][1], self.hr[3][2], self.hr[3][3], self.hr[3][4], self.hr[3][5], self.hr[3][6],
                                   self.hr[2][0], self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6]) # 邻近墙- 3号房间的西墙
        A_wall_3_R4 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[3][0], self.hr[3][1], self.hr[3][3], self.hr[3][4], self.hr[3][5], self.hr[3][6],
                                   self.hr[4][0], self.hr[4][1], self.hr[4][2], self.hr[4][4], self.hr[4][5], self.hr[4][6]) # 相邻墙- 5号房间的北墙
        A_wall_4_R4 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[3][0], self.hr[3][1], self.hr[3][2], self.hr[3][4], self.hr[3][5], self.hr[3][6])
        A_ceil_R4 = setA_INTwall(self.N_ceil, self.Lamda_ceil, self.dx_ceil, self.h_in_ceil, self.h_in_floor, self.hr[3][0],
                                 self.hr[3][1], self.hr[3][2], self.hr[3][3], self.hr[3][5], self.hr[3][6], self.hr[3][0],
                                 self.hr[3][1], self.hr[3][2], self.hr[3][3], self.hr[3][4], self.hr[3][6])
        A_ceil_R4[0][np.sum(N_ceil[:], dtype=int)]= self.hr[3][4]
        A_floor_R4 = setA_INTwall(self.N_floor, self.Lamda_floor, self.dx_floor, self.h_in_floor, self.h_in_ceil, self.hr[3][0],
                                  self.hr[3][1], self.hr[3][2], self.hr[3][3], self.hr[3][5], self.hr[3][6], self.hr[3][0],
                                  self.hr[3][1], self.hr[3][3], self.hr[3][3], self.hr[3][5], self.hr[3][6])
        A_floor_R4[0][np.sum(self.N_floor[:], dtype=int)] = self.hr[3][5]
        A_window_R4 = setA_window(self.N_window, self.h_out, self.h_in_wall, self.h_window_12, self.hr[3][0], self.hr[3][1],
                                  self.hr[3][2], self.hr[3][3], self.hr[3][4], self.hr[3][5])
        # 东外墙的辐射换热
        A12_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][0])
        A13_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][0])
        A14_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][0])
        A15_R4 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[3][0])
        A16_R4 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[3][0])
        A17_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[3][0])
        # 西内墙辐射换热
        A21_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][1])
        A23_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][1])
        A24_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][1])
        A25_R4 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[3][1])
        A26_R4 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[3][1])
        A27_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[3][1])
        # 南外墙的辐射换热
        A31_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][2])
        A32_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][2])
        A34_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][2])
        A35_R4 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[3][2])
        A36_R4 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[3][2])
        A37_R4 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[3][2])
        # 北内墙的辐射换热
        A41_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][3])
        A42_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][3])
        A43_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][3])
        A45_R4 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[3][3])
        A46_R4 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[3][3])
        A47_R4 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[3][3])
        # 天花板的辐射换热
        A51_R4 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][4])
        A52_R4 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][4])
        A53_R4 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][4])
        A54_R4 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][4])
        A56_R4 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                       self.hr[3][4])
        A57_R4 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), self.N_window, self.hr[3][4])
        # 窗户的辐射换热
        A61_R4 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][5])
        A62_R4 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][5])
        A63_R4 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[3][5])
        A64_R4 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[3][5])
        A65_R4 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                       self.hr[3][5])
        A67_R4 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), self.N_window, self.hr[3][5])
        # 室内空气与第i个建筑围护结构之间的对流传热
        A71_R4 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[3][6])
        A72_R4 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[3][6])
        A73_R4 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[3][6])
        A74_R4 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[3][6])
        A75_R4 = A_lwr_cyc(self.N_window, (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[3][6])
        A76_R4 = A_lwr_cyc(self.N_window, (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[3][6])
        # 建筑围护结构部件之间的对流传热
        # 和房间2的室内空气
        Aconv_air_wall1_R4 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_wall2_R4 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_wall3_R4 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_wall4_R4 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        Aconv_air_ceil_R4 = A_conv_air_CYCwall((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_floor, self.h_in_ceil)
        Aconv_air_floor_R4 = A_conv_air_CYCwall((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_ceil, self.h_in_floor)
        Acon_air_window_R4 = A_conv_air_wall(self.N_window, self.h_in_wall)
        # 相邻温度节点的差异
        Aconv_wall1_R4_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][0])
        Aconv_wall2_R4_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][1])
        Aconv_wall3_R4_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][2])
        Aconv_wall4_R4_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[3][3])
        Aconv_ceil_R4_air = A_conv_wall_air((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_ceil, self.f[3][4])
        Aconv_floor_R4_air = A_conv_wall_air((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_floor, self.f[3][5])
        Aconv_window_R4_air = A_conv_wall_air(self.N_window, self.h_in_wall, self.f[3][6])

        A_main4 = np.vstack((np.hstack((A_wall_1_R4, A21_R4, A31_R4, A41_R4, A51_R4, A61_R4, A71_R4, Aconv_air_wall1_R4)),
                   np.hstack((A12_R4, A_wall_2_R4, A32_R4, A42_R4, A52_R4, A62_R4, A72_R4, Aconv_air_wall2_R4)),
                   np.hstack((A13_R4, A23_R4, A_wall_3_R4, A43_R4, A53_R4, A63_R4, A73_R4, Aconv_air_wall3_R4)),
                   np.hstack((A14_R4, A24_R4, A34_R4, A_wall_4_R4, A54_R4, A64_R4, A74_R4, Aconv_air_wall4_R4)),
                   np.hstack((A15_R4, A25_R4, A35_R4, A45_R4, A_ceil_R4, A65_R4, A75_R4, Aconv_air_ceil_R4)),
                   np.hstack((A16_R4, A26_R4, A36_R4, A46_R4, A56_R4, A_floor_R4, A76_R4, Aconv_air_floor_R4)),
                   np.hstack((A17_R4, A27_R4, A37_R4, A47_R4, A57_R4, A67_R4, A_window_R4, Acon_air_window_R4)),
                   np.hstack((Aconv_wall1_R4_air, Aconv_wall2_R4_air, Aconv_wall3_R4_air, Aconv_wall4_R4_air,
                   Aconv_ceil_R4_air, Aconv_floor_R4_air, Aconv_window_R4_air,
                   np.array([[-self.h_in_wall * (np.sum(self.f[3, 0 : 4])) - self.h_in_wall * self.f[3][6] - self.h_in_ceil *
                    self.f[3][4] - self.h_in_floor * self.f[3][5] - self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[3]]])))))

        return A_main4

    def Construct_heat_flux_flow_relationship_room5(self):
        # 从相邻房间的对流和辐射换热
        # east wall
        A_wall_2_R5 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[4][0], self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][5], self.hr[4][6])
        # west wall
        A_wall_1_R5 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][5], self.hr[4][6],
                                   0.53 * self.hr[0][0] + 0.34 * self.hr[1][0] + 0.31 * self.hr[2][0],
                                   0.53 * self.hr[0][2] + 0.34 * self.hr[1][2] + 0.13 * self.hr[2][2],
                                   0.53 * self.hr[0][3] + 0.34 * self.hr[1][3] + 0.13 * self.hr[2][3],
                                   0.53 * self.hr[0][4] + 0.34 * self.hr[1][4] + 0.13 * self.hr[2][4],
                                   0.53 * self.hr[0][5] + 0.34 * self.hr[1][5] + 0.13 * self.hr[2][5],
                                   0.53 * self.hr[0][6] + 0.34 * self.hr[1][6] + 0.13 * self.hr[2][6]) # 相邻的墙- 1、2号房间的西墙和3号房间的一部分
        # south wall
        A_wall_3_R5 = setA_wall(self.N_EXTwall, self.Lamda_EXTwall, self.dx_EXTwall, self.h_out, self.h_in_wall,
                                self.hr[4][0], self.hr[4][1], self.hr[4][3], self.hr[4][4], self.hr[4][5], self.hr[4][6])
        # north wall
        A_wall_4_R5 = setA_INTwall(self.N_INTwall, self.Lamda_INTwall, self.dx_INTwall, self.h_in_wall, self.h_in_wall,
                                   self.hr[4][0], self.hr[4][1], self.hr[4][2], self.hr[4][4], self.hr[4][5], self.hr[4][6],
                                   self.hr[3][0], self.hr[3][1], self.hr[3][3], self.hr[3][4], self.hr[3][5], self.hr[3][6]) # 邻近墙- 4号房间的南墙
        # ceiling
        A_ceil_R5 = setA_INTwall(self.N_ceil, self.Lamda_ceil, self.dx_ceil, self.h_in_ceil, self.h_in_floor,
                                 self.hr[4][0], self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][5], self.hr[4][6],
                                 self.hr[4][0], self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][6])
        A_ceil_R5[0][np.sum(self.N_ceil[:], dtype=int)] = self.hr[4][4]
        # floor
        A_floor_R5 = setA_INTwall(self.N_floor, self.Lamda_floor, self.dx_floor, self.h_in_floor, self.h_in_ceil,
                                  self.hr[4][0], self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][6],
                                  self.hr[4][0], self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][5], self.hr[4][6])
        A_floor_R5[0][np.sum(self.N_floor[:], dtype=int)] = self.hr[4][5]
        # window
        A_window_R5 = setA_window(self.N_window, self.h_out, self.h_in_wall, self.h_window_12, self.hr[4][0], self.hr[4][1],
                                  self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][5])
        # 东外墙的辐射换热
        A12_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][0])
        A13_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][0])
        A14_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][0])
        A15_R5 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[4][0])
        A16_R5 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[4][0])
        A17_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[3][0])
        # 西内墙辐射换热
        A21_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][1])
        A23_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][1])
        A24_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][1])
        A25_R5 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[4][1])
        A26_R5 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[4][1])
        A27_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[4][1])
        # 南外墙的辐射换热
        A31_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][2])
        A32_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][2])
        A34_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][2])
        A35_R5 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[4][2])
        A36_R5 = A_lwr_cyc((np.sum(self.N_EXTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[4][2])
        A37_R5 = A_lwr((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.N_window, self.hr[4][2])
        # 北内墙的辐射换热
        A41_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1),
                       self.hr[4][3])
        A42_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[4][3])
        A43_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1),
                       self.hr[4][3])
        A45_R5 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1),
                           self.hr[4][3])
        A46_R5 = A_lwr_cyc((np.sum(self.N_INTwall[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1),
                           self.hr[4][3])
        A47_R5 = A_lwr((np.sum(self.N_INTwall[:], dtype=int) + 1), self.N_window, self.hr[4][3])
        # 天花板的辐射换热
        A51_R5 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][4])
        A52_R5 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][4])
        A53_R5 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][4])
        A54_R5 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][4])
        A56_R5 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[4][4])
        A57_R5 = A_lwr((np.sum(self.N_ceil[:], dtype=int) + 1), self.N_window, self.hr[4][4])
        # 窗户的辐射换热
        A61_R5 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][5])
        A62_R5 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][5])
        A63_R5 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][5])
        A64_R5 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][5])
        A65_R5 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[4][5])
        A67_R5 = A_lwr((np.sum(self.N_floor[:], dtype=int) + 1), self.N_window, self.hr[4][5])
        # 室内空气与第i个建筑围护结构之间的对流传热
        A71_R5 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][6])
        A72_R5 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][6])
        A73_R5 = A_lwr(self.N_window, (np.sum(self.N_EXTwall[:], dtype=int) + 1), self.hr[4][6])
        A74_R5 = A_lwr(self.N_window, (np.sum(self.N_INTwall[:], dtype=int) + 1), self.hr[4][6])
        A75_R5 = A_lwr_cyc(self.N_window, (np.sum(self.N_ceil[:], dtype=int) + 1), self.hr[4][6])
        A76_R5 = A_lwr_cyc(self.N_window, (np.sum(self.N_floor[:], dtype=int) + 1), self.hr[4][6])
        # 建筑围护结构部件之间的对流传热
        # 房间1 -卧室的室内空气
        # Aconv_air_wall1_R5 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall2_R5 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall3_R5 = A_conv_air_wall((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_wall4_R5 = A_conv_air_wall((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall)
        # Aconv_air_ceil_R5 = A_conv_air_CYCwall((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_floor, self.h_in_ceil)
        # Aconv_air_floor_R5 = A_conv_air_CYCwall((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_ceil, self.h_in_floor)
        # Acon_air_window_R5 = A_conv_air_wall(self.N_window, self.h_in_wall)
        # 矩阵A表示由温度引起的热流运动
        # 相邻温度节点的差异
        # Aconv_wall1_R5_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[4][0])
        # Aconv_wall2_R5_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[4][1])
        # Aconv_wall3_R5_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[4][2])
        # Aconv_wall4_R5_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[4][3])
        # Aconv_ceil_R5_air = A_conv_wall_air((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_ceil, self.f[4][4])
        # Aconv_floor_R5_air = A_conv_wall_air((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_floor, self.f[4][5])
        # Aconv_window_R5_air = A_conv_wall_air(self.N_window, self.h_in_wall, self.f[4][6])
        A_main5 = np.vstack((np.hstack((A_wall_1_R5, A21_R5, A31_R5, A41_R5, A51_R5, A61_R5, A71_R5)),
                   np.hstack((A12_R5, A_wall_2_R5, A32_R5, A42_R5, A52_R5, A62_R5, A72_R5)),
                   np.hstack((A13_R5, A23_R5, A_wall_3_R5, A43_R5, A53_R5, A63_R5, A73_R5)),
                   np.hstack((A14_R5, A24_R5, A34_R5, A_wall_4_R5, A54_R5, A64_R5, A74_R5)),
                   np.hstack((A15_R5, A25_R5, A35_R5, A45_R5, A_ceil_R5, A65_R5, A75_R5)),
                   np.hstack((A16_R5, A26_R5, A36_R5, A46_R5, A56_R5, A_floor_R5, A76_R5)),
                   np.hstack((A17_R5, A27_R5, A37_R5, A47_R5, A57_R5, A67_R5, A_window_R5))))

        return A_main5

    def input_weather_data(self):
        if self.location == "chongqing":
            r = 29.57
            longitude = 106.5
            p_tot = 982
            a = pd.read_csv("chongqing.csv")
            chongqing_tmy = np.array(a)
            #  p_tot_sum = 968.5; % 96.4~97.3, SUMMER TOTAL ATMOSPHERIC PRESSURE OF CHONGQING
            #  p_tot_win = 989; % 98.3~99.5, WINTER TOTAL ATMOSPHERIC PRESSURE OF CHONGQING
            # CONTROL WINTER ROOM RELATIVE HUMIDITY - UPPER LIMIT(控制冬季室内相对湿度上限)
            w_e_in_set_1 = Humidity_Ratio(Ta_targ_min, 65, p_tot)
            # SETTING THE UPPER LIMIT OF RELATIVE HUMIDITY TO BE RH = 65( %) FOR AIR CONDITIONING
            # 设置空调相对湿度上限值为rh = 65(%)
            w_e_in_set_2 = Humidity_Ratio(Ta_targ_max, 65, p_tot)
            nhour = len(chongqing_tmy)
            T = nhour * 3600 / self.dt
            ta_out = np.zeros((int(T), 1))
            q_solar_out_1 = np.zeros((int(T), 1))
            q_solar_out_2 = np.zeros((int(T), 1))
            q_solar_out_3 = np.zeros((int(T), 1))
            q_solar_out_4 = np.zeros((int(T), 1))
            q_solar_out_5 = np.zeros((int(T), 1))
            Mon =  np.zeros((int(T), 1))
            Day = np.zeros((int(T), 1))
            Hour = np.zeros((int(T), 1))
            Min = np.zeros((int(T), 1))
            RH = np.zeros((int(T), 1))
            w_e_out = np.zeros((int(T), 1))
            nh = np.zeros((int(T), 1))
            for i in range(int(T - 3600/self.dt)):
                nx = math.ceil(i * self.dt / 3600)
                ta_out[i][0] = chongqing_tmy[nx][4] + (i * self.dt / 3600 + 1 - nx) * (chongqing_tmy[nx + 1][4] - chongqing_tmy[nx][4])
                q_solar_out_1[i][0] = chongqing_tmy[nx][6] + (i * self.dt / 3600 + 1 - nx) * \
                                      (chongqing_tmy[nx + 1][6] - chongqing_tmy[nx][6]) # EXTERNAL WALL SOLAR RADIATION ON EAST WALL, W / m2
                q_solar_out_2[i][0] = chongqing_tmy[nx][8] + (i * self.dt / 3600 + 1 - nx) * \
                                      (chongqing_tmy[nx + 1][8] - chongqing_tmy[nx][8]) # EXTERNAL WALL SOLAR RADIATION ON WEST WALL, W / m2
                q_solar_out_3[i][0] = chongqing_tmy[nx][7] + (i * self.dt / 3600 + 1 - nx) * \
                                      (chongqing_tmy[nx + 1][7] - chongqing_tmy[nx][7]) # EXTERNAL WALL SOLAR RADIATION ON SOUTH WALL, W / m2
                q_solar_out_4[i][0] = chongqing_tmy[nx][9] + (i * self.dt / 3600 + 1 - nx) * \
                                      (chongqing_tmy[nx + 1][9] - chongqing_tmy[nx][9])  # EXTERNAL WALL SOLAR RADIATION ON NORTH WALL, W / m2
                q_solar_out_5[i][0] = chongqing_tmy[nx][5] + (i * self.dt / 3600 + 1 - nx) * \
                                      (chongqing_tmy[nx + 1][5] - chongqing_tmy[nx][5]) # EXTERNAL SOLAR RADIATION OF ROOF(HORIZONTAL), W / m2
                RH[i][0] = chongqing_tmy[nx][10] + (i * self.dt / 3600 + 1 - nx) * \
                           (chongqing_tmy[nx + 1][10] - chongqing_tmy[nx][10]) # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
                nh[i][0] = chongqing_tmy[nx][2] + (i * self.dt / 3600 + 1 - nx) * \
                           (chongqing_tmy[nx + 1][2] - chongqing_tmy[nx][2]) # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR(用于安排乘员行为的每日小时数)
                Mon[i][0] = chongqing_tmy[nx][0] # MONTHS
                Day[i][0] = chongqing_tmy[nx][1] # DATE NUMBER
                Hour[i][0] = chongqing_tmy[0][2] + i * self.dt / 3600 # HOURLY TIME
                Min[i][0] = chongqing_tmy[nx][3]
            q_solar_out_6 = 0
            for i in range(int(T - 3600 / self.dt), int(T)):
                nx = math.ceil(i * self.dt / 3600)
                ta_out[i][0] = chongqing_tmy[nx][4]
                q_solar_out_1[i][0] = chongqing_tmy[nx][6] # EXTERNAL WALL SOLAR RADIATION OF EAST WALL, W / m2
                q_solar_out_2[i][0] = chongqing_tmy[nx][8] # EXTERNAL WALL SOLAR RADIATION OF WEST WALL, W / m2
                q_solar_out_3[i][0] = chongqing_tmy[nx][7] # EXTERNAL WALL SOLAR RADIATION OF SOUTH WALL, W / m2
                q_solar_out_4[i][0] = chongqing_tmy[nx][9] # EXTERNAL WALL SOLAR RADIATION OF NORTH WALL, W / m2
                q_solar_out_5[i][0] = chongqing_tmy[nx][5] # EXTERNAL WALL SOLAR RADIATION OF FLOOR, W / m2
                RH[i][0] = chongqing_tmy[nx][10] # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
                nh[i][0] = chongqing_tmy[nx][2] # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR
                Mon[i][0] = chongqing_tmy[nx][0] # MONTHS
                Day[i][0] = chongqing_tmy[nx][1] # DATE NUMBER IN A YEAR
                Hour[i][0] = chongqing_tmy[0][2] + i * self.dt / 3600 # HOURLY NUMBER
                Min[i][0] = chongqing_tmy[nx][3] # MINUTES

        elif self.location == "shanghai":
            r = 31.4
            longitude = 121.45
            p_tot = 1016.8
            a = pd.read_csv("shanghai.csv")
            shanghai_tmy = np.array(a)
            # CONTROL WINTER ROOM RELATIVE HUMIDITY - UPPER LIMIT(控制冬季室内相对湿度上限)
            w_e_in_set_1 = Humidity_Ratio(Ta_targ_min, 65, p_tot)
            # SETTING THE UPPER LIMIT OF RELATIVE HUMIDITY TO BE RH = 65( %) FOR AIR CONDITIONING
            # 设置空调相对湿度上限值为rh = 65(%)
            w_e_in_set_2 = Humidity_Ratio(Ta_targ_max, 65, p_tot)
            nhour = len(shanghai_tmy)
            T = nhour * 3600 / self.dt
            ta_out = np.zeros((int(T), 1))
            q_solar_out_1 = np.zeros((int(T), 1))
            q_solar_out_2 = np.zeros((int(T), 1))
            q_solar_out_3 = np.zeros((int(T), 1))
            q_solar_out_4 = np.zeros((int(T), 1))
            q_solar_out_5 = np.zeros((int(T), 1))
            Mon =  np.zeros((int(T), 1))
            Day = np.zeros((int(T), 1))
            Hour = np.zeros((int(T), 1))
            Min = np.zeros((int(T), 1))
            RH = np.zeros((int(T), 1))
            w_e_out = np.zeros((int(T), 1))
            nh = np.zeros((int(T), 1))
            for i in range(int(T - 3600/self.dt)):
                nx = math.ceil(i * self.dt / 3600)
                ta_out[i][0] = shanghai_tmy[nx][4] + (i * self.dt / 3600 + 1 - nx) * (shanghai_tmy[nx + 1][4] - shanghai_tmy[nx][4])
                q_solar_out_1[i][0] = shanghai_tmy[nx][6] + (i * self.dt / 3600 + 1 - nx) * \
                                      (shanghai_tmy[nx + 1][6] - shanghai_tmy[nx][6]) # EXTERNAL WALL SOLAR RADIATION ON EAST WALL, W / m2
                q_solar_out_2[i][0] = shanghai_tmy[nx][8] + (i * self.dt / 3600 + 1 - nx) * \
                                      (shanghai_tmy[nx + 1][8] - shanghai_tmy[nx][8]) # EXTERNAL WALL SOLAR RADIATION ON WEST WALL, W / m2
                q_solar_out_3[i][0] = shanghai_tmy[nx][7] + (i * self.dt / 3600 + 1 - nx) * \
                                      (shanghai_tmy[nx + 1][7] - shanghai_tmy[nx][7]) # EXTERNAL WALL SOLAR RADIATION ON SOUTH WALL, W / m2
                q_solar_out_4[i][0] = shanghai_tmy[nx][9] + (i * self.dt / 3600 + 1 - nx) * \
                                      (shanghai_tmy[nx + 1][9] - shanghai_tmy[nx][9])  # EXTERNAL WALL SOLAR RADIATION ON NORTH WALL, W / m2
                q_solar_out_5[i][0] = shanghai_tmy[nx][5] + (i * self.dt / 3600 + 1 - nx) * \
                                      (shanghai_tmy[nx + 1][5] - shanghai_tmy[nx][5]) # EXTERNAL SOLAR RADIATION OF ROOF(HORIZONTAL), W / m2
                RH[i][0] = shanghai_tmy[nx][10] + (i * self.dt / 3600 + 1 - nx) * \
                           (shanghai_tmy[nx + 1][10] - shanghai_tmy[nx][10]) # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
                nh[i][0] = shanghai_tmy[nx][2] + (i * self.dt / 3600 + 1 - nx) * \
                           (shanghai_tmy[nx + 1][2] - shanghai_tmy[nx][2]) # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR(用于安排乘员行为的每日小时数)
                Mon[i][0] = shanghai_tmy[nx][0] # MONTHS
                Day[i][0] = shanghai_tmy[nx][1] # DATE NUMBER
                Hour[i][0] = shanghai_tmy[0][2] + i * self.dt / 3600 # HOURLY TIME
                Min[i][0] = shanghai_tmy[nx][3]
            q_solar_out_6 = 0
            for i in range(T - 3600/self.dt, T):
                nx = math.ceil(i * self.dt / 3600)
                ta_out[i][0] = shanghai_tmy[nx][4]
                q_solar_out_1[i][0] = shanghai_tmy[nx][6] # EXTERNAL WALL SOLAR RADIATION OF EAST WALL, W / m2
                q_solar_out_2[i][0] = shanghai_tmy[nx][8] # EXTERNAL WALL SOLAR RADIATION OF WEST WALL, W / m2
                q_solar_out_3[i][0] = shanghai_tmy[nx][7] # EXTERNAL WALL SOLAR RADIATION OF SOUTH WALL, W / m2
                q_solar_out_4[i][0] = shanghai_tmy[nx][9] # EXTERNAL WALL SOLAR RADIATION OF NORTH WALL, W / m2
                q_solar_out_5[i][0] = shanghai_tmy[nx][5] # EXTERNAL WALL SOLAR RADIATION OF FLOOR, W / m2
                RH[i][0] = shanghai_tmy[nx][10] # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
                nh[i][0] = shanghai_tmy[nx][2] # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR
                Mon[i][0] = shanghai_tmy[nx][0] # MONTHS
                Day[i][0] = shanghai_tmy[nx][1] # DATE NUMBER IN A YEAR
                Hour[i][0] = shanghai_tmy[0][2] + i * self.dt / 3600 # HOURLY NUMBER
                Min[i][0] = shanghai_tmy[nx][3] # MINUTES

        elif self.location == "changsha":
            r = 28.22
            longitude = 112.92
            p_tot = 1008
            a = pd.read_csv("changsha.csv")
            changsha_tmy = np.array(a)
            # CONTROL WINTER ROOM RELATIVE HUMIDITY - UPPER LIMIT(控制冬季室内相对湿度上限)
            w_e_in_set_1 = Humidity_Ratio(Ta_targ_min, 65, p_tot)
            # SETTING THE UPPER LIMIT OF RELATIVE HUMIDITY TO BE RH = 65( %) FOR AIR CONDITIONING
            # 设置空调相对湿度上限值为rh = 65(%)
            w_e_in_set_2 = Humidity_Ratio(Ta_targ_max, 65, p_tot)
            nhour = len(changsha_tmy)
            T = int(nhour * 3600 / self.dt)
            ta_out = np.zeros((int(T), 1))
            q_solar_out_1 = np.zeros((int(T), 1))
            q_solar_out_2 = np.zeros((int(T), 1))
            q_solar_out_3 = np.zeros((int(T), 1))
            q_solar_out_4 = np.zeros((int(T), 1))
            q_solar_out_5 = np.zeros((int(T), 1))
            Mon =  np.zeros((int(T), 1))
            Day = np.zeros((int(T), 1))
            Hour = np.zeros((int(T), 1))
            Min = np.zeros((int(T), 1))
            RH = np.zeros((int(T), 1))
            w_e_out = np.zeros((int(T), 1))
            nh = np.zeros((int(T), 1))
            for i in range(int(T - 3600/self.dt)):
                nx = math.ceil(i * self.dt / 3600)
                ta_out[i][0] = changsha_tmy[nx][4] + (i * self.dt / 3600 + 1 - nx) * (changsha_tmy[nx + 1][4] - changsha_tmy[nx][4])
                q_solar_out_1[i][0] = changsha_tmy[nx][6] + (i * self.dt / 3600 + 1 - nx) * \
                                      (changsha_tmy[nx + 1][6] - changsha_tmy[nx][6]) # EXTERNAL WALL SOLAR RADIATION ON EAST WALL, W / m2
                q_solar_out_2[i][0] = changsha_tmy[nx][8] + (i * self.dt / 3600 + 1 - nx) * \
                                      (changsha_tmy[nx + 1][8] - changsha_tmy[nx][8]) # EXTERNAL WALL SOLAR RADIATION ON WEST WALL, W / m2
                q_solar_out_3[i][0] = changsha_tmy[nx][7] + (i * self.dt / 3600 + 1 - nx) * \
                                      (changsha_tmy[nx + 1][7] - changsha_tmy[nx][7]) # EXTERNAL WALL SOLAR RADIATION ON SOUTH WALL, W / m2
                q_solar_out_4[i][0] = changsha_tmy[nx][9] + (i * self.dt / 3600 + 1 - nx) * \
                                      (changsha_tmy[nx + 1][9] - changsha_tmy[nx][9])  # EXTERNAL WALL SOLAR RADIATION ON NORTH WALL, W / m2
                q_solar_out_5[i][0] = changsha_tmy[nx][5] + (i * self.dt / 3600 + 1 - nx) * \
                                      (changsha_tmy[nx + 1][5] - changsha_tmy[nx][5]) # EXTERNAL SOLAR RADIATION OF ROOF(HORIZONTAL), W / m2
                RH[i][0] = changsha_tmy[nx][10] + (i * self.dt / 3600 + 1 - nx) * \
                           (changsha_tmy[nx + 1][10] - changsha_tmy[nx][10]) # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
                nh[i][0] = changsha_tmy[nx][2] + (i * self.dt / 3600 + 1 - nx) * \
                           (changsha_tmy[nx + 1][2] - changsha_tmy[nx][2]) # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR(用于安排乘员行为的每日小时数)
                Mon[i][0] = changsha_tmy[nx][0] # MONTHS
                Day[i][0] = changsha_tmy[nx][1] # DATE NUMBER
                Hour[i][0] = changsha_tmy[0][2] + i * self.dt / 3600 # HOURLY TIME
                Min[i][0] = changsha_tmy[nx][3]
            q_solar_out_6 = 0
            for i in range(int(T - 3600/self.dt), int(T)):
                nx = math.ceil(i * self.dt / 3600)
                ta_out[i][0] = changsha_tmy[nx][4]
                q_solar_out_1[i][0] = changsha_tmy[nx][6] # EXTERNAL WALL SOLAR RADIATION OF EAST WALL, W / m2
                q_solar_out_2[i][0] = changsha_tmy[nx][8] # EXTERNAL WALL SOLAR RADIATION OF WEST WALL, W / m2
                q_solar_out_3[i][0] = changsha_tmy[nx][7] # EXTERNAL WALL SOLAR RADIATION OF SOUTH WALL, W / m2
                q_solar_out_4[i][0] = changsha_tmy[nx][9] # EXTERNAL WALL SOLAR RADIATION OF NORTH WALL, W / m2
                q_solar_out_5[i][0] = changsha_tmy[nx][5] # EXTERNAL WALL SOLAR RADIATION OF FLOOR, W / m2
                RH[i][0] = changsha_tmy[nx][10] # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
                nh[i][0] = changsha_tmy[nx][2] # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR
                Mon[i][0] = changsha_tmy[nx][0] # MONTHS
                Day[i][0] = changsha_tmy[nx][1] # DATE NUMBER IN A YEAR
                Hour[i][0] = changsha_tmy[0][2] + i * self.dt / 3600 # HOURLY NUMBER
                Min[i][0] = changsha_tmy[nx][3] # MINUTES
        mon_nd = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334] # 1-12months
        n0 = mon_nd[int(Mon[0][0] - 1)] + Day[0][0]
        # 计算初始化
        NN_main = len(self.C_main)
        NN_main2 = len(self.C_main2)
        NN_main3 = len(self.C_main3)
        NN_main4 = len(self.C_main4)
        NN_main5 = len(self.C_main5)
        t1 = 10 * np.ones((NN_main, int(T)))
        tR1_wall1_in = ta_out.copy()
        tR1_wall2_in = ta_out.copy()
        tR1_wall3_in = ta_out.copy()
        tR1_wall4_in = ta_out.copy()
        tR1_ceil_in = ta_out.copy()
        tR1_floor_in = ta_out.copy()
        tR1_window_in = ta_out.copy()
        tR1_air = ta_out.copy()
        t2 = 10 * np.ones((NN_main2, int(T)))
        tR2_wall1_in = ta_out.copy()
        tR2_wall2_in = ta_out.copy()
        tR2_wall3_in = ta_out.copy()
        tR2_wall4_in = ta_out.copy()
        tR2_ceil_in = ta_out.copy()
        tR2_floor_in = ta_out.copy()
        tR2_window_in = ta_out.copy()
        tR2_air = ta_out.copy()
        t3 = 10 * np.ones((NN_main3, int(T)))
        tR3_wall1_in = ta_out.copy()
        tR3_wall2_in = ta_out.copy()
        tR3_wall3_in = ta_out.copy()
        tR3_wall4_in = ta_out.copy()
        tR3_ceil_in = ta_out.copy()
        tR3_floor_in = ta_out.copy()
        tR3_window_in = ta_out.copy()
        tR3_air = ta_out.copy()
        t4 = 10 * np.ones((NN_main4, int(T)))
        tR4_wall1_in = ta_out.copy()
        tR4_wall2_in = ta_out.copy()
        tR4_wall3_in = ta_out.copy()
        tR4_wall4_in = ta_out.copy()
        tR4_ceil_in = ta_out.copy()
        tR4_floor_in = ta_out.copy()
        tR4_window_in = ta_out.copy()
        tR4_air = ta_out.copy()
        t5 = 10 * np.ones((NN_main5, int(T)))
        tR5_wall1_in = ta_out.copy()
        tR5_wall2_in = ta_out.copy()
        tR5_wall3_in = ta_out.copy()
        tR5_wall4_in = ta_out.copy()
        tR5_ceil_in = ta_out.copy()
        tR5_floor_in = ta_out.copy()
        tR5_window_in = ta_out.copy()
        tR5_air = ta_out.copy()
        tR1_window_out = ta_out.copy()
        tR2_window_out = ta_out.copy()
        tR3_window_out = ta_out.copy()
        tR4_window_out = ta_out.copy()
        tR5_window_out = ta_out.copy()
        Q_hload_R1 = np.zeros((int(T), 1)) # heating load of room1
        Q_hload_R3 = np.zeros((int(T), 1)) # heating load of room3
        Q_hload_R5 = np.zeros((int(T), 1)) # heating load of room5
        Q_h_dehumid_R1 = np.zeros((int(T), 1)) # dehumidification of latent heat for room 1(1室潜热除湿)
        Q_h_dehumid_R3 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 3
        Q_h_dehumid_R5 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 5
        Q_c_sens_R1 = np.zeros((int(T), 1))
        Q_c_latent_R1 = np.zeros((int(T), 1))  # sensible and latent cooling loads of room 1,respectively (1室的显冷负荷和潜冷负荷)
        Q_c_sens_R3 = np.zeros((int(T), 1))
        Q_c_latent_R3 = np.zeros((int(T), 1))  # sensible and latent cooling loads of room 3,respectively (3室的显冷负荷和潜冷负荷)
        Q_c_sens_R5 = np.zeros((int(T), 1))
        Q_c_latent_R5 = np.zeros((int(T), 1))  # sensible and latent cooling loads of room 5,respectively (5室的显冷负荷和潜冷负荷)
        w_e_in_R1 = np.zeros((int(T), 1))
        w_e_in_R1[0][0] = 6
        w_e_in_R3 = np.zeros((int(T), 1))
        w_e_in_R3[0][0] = 6
        w_e_in_R5 = np.zeros((int(T), 1))
        w_e_in_R5[0][0] = 6
        q_heatemss_p = 70 + 60  # 居住者对房间的感热和潜热散发量，70瓦特/人
        m_w_gp = 50 / 3600 * self.dt # 人的吸湿量(增湿量)，50克/(h人)
        n_p = 3  # 公寓里住了三个人
        # 设置内部热增益(人员、设备和照明)人体代谢产热率: 70瓦特 / 人 - 按
        # 一天约1500千卡代谢热量;在这里，乘员的辐射热增益比为0.50
        # 设备密度: 4.3 w / m2，辐射热增益比0.20;
        # 照明密度: 6 w / m2，辐射热增益比0.52;
        # 居住者的热增益随时间而变化
        #  房间1 - 主卧室，占用时间为2人:
        #  星期一至星期日: 0:00 - 6: 00 & 20:00 - 24: 00
        #  房间3 - 次要卧室，占用时间为1人:
        #  星期一至星期日: 0:00 - 6: 00 & 20:00 - 24: 00
        #  房间5 - 客厅，3人停留时间:
        #  周一至周五: 6:00 - 8: 00 & 18:00 - 20: 00
        #  星期六至星期日: 6:00 - 20: 00
        OCCUPIED_BED = np.zeros((int(T), 1))       # BEDROOM OCCUPATION
        LIGHT_ON_BED = np.zeros((int(T), 1))       # LIGHTS & EQUIPMENT ON
        OCCUPIED_SITT = np.zeros((int(T), 1))      # SITTING ROOM OCCUPATION
        LIGHT_ON_SITT = np.zeros((int(T), 1))
        HEATING_BEDROOM = np.zeros((int(T), 1))
        HEATING_SITTROOM = np.zeros((int(T), 1))
        COOLING_BEDROOM = np.zeros((int(T), 1))
        COOLING_SITTROOM = np.zeros((int(T), 1))
        Q_internal_rad_R1 = np.zeros((int(T), 1))
        Q_internal_cov_R1 = np.zeros((int(T), 1))
        Q_internal_rad_R3 = np.zeros((int(T), 1))
        Q_internal_cov_R3 = np.zeros((int(T), 1))
        Q_internal_rad_R5 = np.zeros((int(T), 1))
        Q_internal_cov_R5 = np.zeros((int(T), 1))
        Q_INTrad_wall1_R1 = np.zeros((1, int(T)))
        Q_INTrad_wall2_R1 = np.zeros((1, int(T)))
        Q_INTrad_wall3_R1 = np.zeros((1, int(T)))
        Q_INTrad_wall4_R1 = np.zeros((1, int(T)))
        Q_INTrad_ceil_R1 = np.zeros((1, int(T)))
        Q_INTrad_floor_R1 = np.zeros((1, int(T)))
        Q_INTrad_win_R1 = np.zeros((1, int(T)))
        Q_conv1 = np.zeros((1, int(T)))
        Q_hvac1 = np.zeros((1, int(T)))
        Q_conv2 = np.zeros((1, int(T)))
        Q_hvac2 = np.zeros((1, int(T)))
        Q_INTrad_wall1_R3 = np.zeros((1, int(T)))
        Q_INTrad_wall2_R3 = np.zeros((1, int(T)))
        Q_INTrad_wall3_R3 = np.zeros((1, int(T)))
        Q_INTrad_wall4_R3 = np.zeros((1, int(T)))
        Q_INTrad_ceil_R3 = np.zeros((1, int(T)))
        Q_INTrad_floor_R3 = np.zeros((1, int(T)))
        Q_INTrad_win_R3 = np.zeros((1, int(T)))
        Q_conv3 = np.zeros((1, int(T)))
        Q_hvac3 = np.zeros((1, int(T)))
        Q_conv4 = np.zeros((1, int(T)))
        Q_hvac4 = np.zeros((1, int(T)))
        Q_INTrad_wall1_R5 = np.zeros((1, int(T)))
        Q_INTrad_wall2_R5 = np.zeros((1, int(T)))
        Q_INTrad_wall3_R5 = np.zeros((1, int(T)))
        Q_INTrad_wall4_R5 = np.zeros((1, int(T)))
        Q_INTrad_ceil_R5 = np.zeros((1, int(T)))
        Q_INTrad_floor_R5 = np.zeros((1, int(T)))
        Q_INTrad_win_R5 = np.zeros((1, int(T)))
        Q_conv5 = np.zeros((1, int(T)))
        Q_hvac5 = np.zeros((1, int(T)))
        # Q_internal_rad_R1 = 0.03 * self.f[0][5] * 70.0 * 0.50 + 4.3 * self.f[0][5] * 0.20 + 6 * self.f[0][5] * 0.52
        # Q_internal_cov_R1 = 0.03 * self.f[0][5] * 70.0 * 0.50 + 4.3 * self.f[0][5] * 0.80 + 6 * self.f[0][5] * 0.48
        Q_internal_rad_R2 = 0   # NO INTERNAL HEAT GAIN FOR TOILET
        Q_internal_cov_R2 = 0   # NO INTERNAL HEAT GAIN FOR TOILET
        # Q_internal_rad_R3 = 0.03 * self.f[2][4] * 70.0 * 0.50 + 4.3 * self.f[2][4] * 0.20 + 6 * self.f[2][4] * 0.52
        # Q_internal_cov_R3 = 0.03 * self.f[2][4] * 70.0 * 0.50 + 4.3 * self.f[2][4] * 0.80 + 6 * self.f[2][4] * 0.48
        # Q_internal_rad_R5 = 0.03 * self.f[4][5] * 70.0 * 0.50 + 4.3 * self.f[4][5] * 0.20 + 6 * self.f[4][5] * 0.52
        # Q_internal_cov_R5 = 0.03 * self.f[4][5] * 70.0 * 0.50 + 4.3 * self.f[4][5] * 0.80 + 6 * self.f[4][5] * 0.48
        Q_solar_in_R1 = np.zeros((1, int(T)))
        Q_solar_in_R2 = np.zeros((1, int(T)))
        Q_solar_in_R3 = np.zeros((1, int(T)))
        Q_solar_in_R4 = np.zeros((1, int(T)))
        Q_solar_in_R5 = np.zeros((1, int(T)))
        Q_solar_in_wall = np.zeros((5, int(T)))
        Q_solar_in_ceil = np.zeros((5, int(T)))
        Q_solar_in_floor = np.zeros((5, int(T)))
        for i in range(1, int(T)):
            time = nh[i][0] % 24
            if (time >= 0 and time <= 6) or (time >= 20 and time < 24):      # STAYING IN BEDROOMS AT TIMEFRAME 0:00-6:00 am; 20:00-24:00 pm
                OCCUPIED_BED[i][0] = 1
            elif (time >= 6 and time <= 8) or (time >= 20 and time <= 22):    # LIGHTS & EQUIPMENT ON IN THE BEDCHAMBER
                LIGHT_ON_BED[i][0] = 1
            if (((nh[i][0] / 24) % 7) >= 1) and (((nh[i][0] / 24) % 7) < 6):   # WEEKDAYS(MONDAY TO FRIDAY)
                if (time >= 6 and time <= 8) and (time >= 18 and time <= 20):   # TIMEFRAME 6:00-20:00 pm IN SITTING ROOM AND LIGHTS ON WHEN OCCUPYING
                    OCCUPIED_SITT[i][0] = 1                                        # 时间范围:下午6:00-20:00在客厅，占用时灯亮着
                    LIGHT_ON_SITT[i][0] = 1
            else:
                if (time >= 6 and time <= 20):
                    OCCUPIED_SITT[i][0] = 1
                    LIGHT_ON_SITT[i][0] = 1

            Q_internal_rad_R1[i][0] = OCCUPIED_BED[i][0] * 2 * q_heatemss_p * 0.50 + LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[0][5] * 0.20 + 6 * self.f[0][5] * 0.52) # 2 OCCUPANTS
            Q_internal_cov_R1[i][0] = OCCUPIED_BED[i][0] * 2 * q_heatemss_p * 0.50 + LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[0][5] * 0.80 + 6 * self.f[0][5] * 0.48)
            Q_internal_rad_R3[i][0] = OCCUPIED_BED[i][0] * 1 * q_heatemss_p * 0.50 + LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[2][5] * 0.20 + 6 * self.f[2][5] * 0.52) # 1 OCCUPANT
            Q_internal_cov_R3[i][0] = OCCUPIED_BED[i][0] * 1 * q_heatemss_p * 0.50 + LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[2][5] * 0.80 + 6 * self.f[2][5] * 0.48)
            Q_internal_rad_R5[i][0] = OCCUPIED_SITT[i][0] * 3 * q_heatemss_p * 0.50 + LIGHT_ON_SITT[i][0] * \
                                      (4.3 * self.f[4][5] * 0.20 + 6 * self.f[4][5] * 0.52) # 3 OCCUPANTS
            Q_internal_cov_R5[i][0] = OCCUPIED_SITT[i][0] * 3 * q_heatemss_p * 0.50 + LIGHT_ON_SITT[i][0] * \
                                      (4.3 * self.f[4][5] * 0.80 + 6 * self.f[4][5] * 0.48)
            Q_INTrad_wall1_R1[0][i] = Q_internal_rad_R1[i][0] * self.f[0][0] / self.f_tol[0]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall2_R1[0][i] = Q_internal_rad_R1[i][0] * self.f[0][1] / self.f_tol[0]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall3_R1[0][i] = Q_internal_rad_R1[i][0] * self.f[0][2] / self.f_tol[0]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall4_R1[0][i] = Q_internal_rad_R1[i][0] * self.f[0][3] / self.f_tol[0]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_ceil_R1[0][i] = Q_internal_rad_R1[i][0] * self.f[0][4] / self.f_tol[0]   # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_floor_R1[0][i] = Q_internal_rad_R1[i][0] * self.f[0][5] / self.f_tol[0]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_win_R1[0][i] = Q_internal_rad_R1[i][0] * self.f[0][6] / self.f_tol[0]    # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_conv1[0][i] = Q_internal_cov_R1[i][0]   # 从室内空气产生的内部热量中获得的对流热量，w
            Q_hvac1[0][i] = 0  # 房间1空调供热 w

            Q_conv2[0][i] = 0 # IT'S ASSUMED NO INTERNAL HEAT GAIN FROM ROOM 2 - TOILET AND ROOM 4 - STORAGE AND KITCHEN
            Q_hvac2[0][i] = 0

            Q_INTrad_wall1_R3[0][i] = Q_internal_rad_R3[i][0] * self.f[2][0] / self.f_tol[2]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall2_R3[0][i] = Q_internal_rad_R3[i][0] * self.f[2][1] / self.f_tol[2]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall3_R3[0][i] = Q_internal_rad_R3[i][0] * self.f[2][2] / self.f_tol[2]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall4_R3[0][i] = Q_internal_rad_R3[i][0] * self.f[2][3] / self.f_tol[2]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_ceil_R3[0][i] = Q_internal_rad_R3[i][0] * self.f[2][4] / self.f_tol[2]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_floor_R3[0][i] = Q_internal_rad_R3[i][0] * self.f[2][5] / self.f_tol[2]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_win_R3[0][i] = Q_internal_rad_R3[i][0] * self.f[2][6] / self.f_tol[2]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_conv3[0][i] = Q_internal_cov_R3[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
            Q_hvac3[0][i] = 0  # 房间3空调供热 w

            Q_conv4[0][i] = 0  # IT'S ASSUMED NO INTERNAL HEAT GAIN FROM ROOM 4 - TOILET AND ROOM 4 - STORAGE AND KITCHEN
            Q_hvac4[0][i] = 0

            Q_INTrad_wall1_R5[0][i] = Q_internal_rad_R5[i][0] * self.f[4][0] / self.f_tol[4]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall2_R5[0][i] = Q_internal_rad_R5[i][0] * self.f[4][1] / self.f_tol[4]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall3_R5[0][i] = Q_internal_rad_R5[i][0] * self.f[4][2] / self.f_tol[4]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_wall4_R5[0][i] = Q_internal_rad_R5[i][0] * self.f[4][3] / self.f_tol[4]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_ceil_R5[0][i] = Q_internal_rad_R5[i][0] * self.f[4][4] / self.f_tol[4]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_floor_R5[0][i] = Q_internal_rad_R5[i][0] * self.f[4][5] / self.f_tol[4]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_INTrad_win_R5[0][i] = Q_internal_rad_R5[i][0] * self.f[4][6] / self.f_tol[4]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
            Q_conv5[0][i] = Q_internal_cov_R5[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
            Q_hvac5[0][i] = 0  # 房间5空调供热 w

            if i < 143 * (3600 / self.dt):
                n = n0
            else:
                n = n0 + int((i * (self.dt / 3600) - 143) / 24)
            ws = i * (self.dt / 3600) - 24 * int(i * (self.dt / 3600) / 24)
            angle3 = solar_angle(n, r, longitude, ws, 90, 0)
            [win_transp3, win_absorp3, win_refelect3, e_all3, tao_all3] = solar_win2(angle3)
            Q_solar_in_R1[0][i] = win_transp3 * q_solar_out_3[i][0] * self.f[0][6] # SOUTH WINDOW SOLAR
            Q_solar_in_R5[0][i] = win_transp3 * q_solar_out_3[i][0] * self.f[4][6] # SOUTH WINDOW SOLAR

            angle1 = solar_angle(n, r, longitude, ws, 90, -90)
            [win_transp1, win_absorp1, win_refelect1, e_all1, tao_all1] = solar_win2(angle1)
            Q_solar_in_R2[0][i] = win_transp1 * q_solar_out_1[i][0] * self.f[1][6] # EAST WINDOW SOLAR
            Q_solar_in_R3[0][i] = win_transp1 * q_solar_out_1[i][0] * self.f[2][6] # EAST WINDOW SOLAR

            angle4 = solar_angle(n, r, longitude, ws, 90, 180)
            [win_transp4, win_absorp4, win_refelect4, e_all4, tao_all4] = solar_win2(angle4)
            Q_solar_in_R4[0][i] = win_transp4 * q_solar_out_4[i][0] * self.f[3][6] # NORTH WINDOW SOLAR

            # 通过玻璃渗透获得内部太阳热量
            Q_solar_in_wall[0][i] = Q_solar_in_R1[0][i] * 0.1
            Q_solar_in_ceil[0][i] = Q_solar_in_R1[0][i] * 0.1
            Q_solar_in_floor[0][i] = Q_solar_in_R1[0][i] * 0.5
            Q_solar_in_wall[1][i] = Q_solar_in_R2[0][i] * 0.1
            Q_solar_in_ceil[1][i] = Q_solar_in_R2[0][i] * 0.1
            Q_solar_in_floor[1][i] = Q_solar_in_R2[0][i] * 0.5
            Q_solar_in_wall[2][i] = Q_solar_in_R3[0][i] * 0.1
            Q_solar_in_ceil[2][i] = Q_solar_in_R3[0][i] * 0.1
            Q_solar_in_floor[2][i] = Q_solar_in_R3[0][i] * 0.5
            Q_solar_in_wall[3][i] = Q_solar_in_R4[0][i] * 0.1
            Q_solar_in_ceil[3][i] = Q_solar_in_R4[0][i] * 0.1
            Q_solar_in_floor[3][i] = Q_solar_in_R4[0][i] * 0.5
            Q_solar_in_wall[3][i] = Q_solar_in_R5[0][i] * 0.1
            Q_solar_in_ceil[4][i] = Q_solar_in_R5[0][i] * 0.1
            Q_solar_in_floor[4][i] = Q_solar_in_R5[0][i] * 0.5

            w_e_out[i][0] = Humidity_Ratio(ta_out[i][0], RH[i][0], p_tot) # HUMIDITY RATIO OF ATMOSHPHERIC STATE POING

            # b向量元素-修正热负荷计算矩阵
            # 房间1 - REVISED(修改)
            B_wall1_R1 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0], Q_solar_in_wall[0][i],
                                        Q_INTrad_wall1_R1[0][i], ta_out[i][0], self.f[0][0], self.h_in_wall, tR1_air[i - 1][0])
            B_wall2_R1 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[0][i], Q_INTrad_wall2_R1[0][i],
                                           Q_solar_in_wall[4][i], Q_INTrad_wall1_R5[0][i], self.f[4][0], tR5_air[i - 1][0],
                                           tR5_wall2_in[i - 1][0], tR5_wall3_in[i - 1][0], tR5_wall4_in[i - 1][0], tR5_ceil_in[i - 1][0],
                                           tR5_floor_in[i - 1][0], tR5_window_in[i - 1][0], self.hr[4][1], self.hr[4][2], self.hr[4][3],
                                           self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[0][1], tR1_air[i - 1][0])
            B_wall3_R1 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_3[i][0], Q_solar_in_wall[0][i],
                                        Q_INTrad_wall3_R1[0][i], ta_out[i][0], self.f[0][2], self.h_in_wall, tR1_air[i - 1][0])
            B_wall4_R1 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[0][i], Q_INTrad_wall2_R1[0][i],
                                           Q_solar_in_wall[1][i], 0, self.f[1][2], tR2_air[i - 1][0], tR2_wall1_in[i - 1][0],
                                           tR2_wall2_in[i - 1][0], tR2_wall4_in[i - 1][0], tR2_ceil_in[i - 1][0],
                                           tR2_floor_in[i - 1][0], tR2_window_in[i - 1][0], self.hr[1][0],
                                           self.hr[1][1], self.hr[1][3],
                                           self.hr[1][4], self.hr[1][5], self.hr[1][6], self.f[0][3], tR1_air[i - 1][0])
            B_ceil_R1 = setB_ceiling_load(self.N_ceil, self.eb, Q_solar_in_floor[0][i], Q_INTrad_floor_R1[0][i], self.f[0][5],
                                          Q_solar_in_ceil[0][i], Q_INTrad_ceil_R1[0][i], self.f[0][4], self.h_in_ceil, self.h_in_floor,
                                          tR1_air[i - 1][0])
            B_floor_R1 = setB_ceiling_load(self.N_floor, self.eb, Q_solar_in_ceil[0][i], Q_INTrad_ceil_R1[0][i], self.f[0][4],
                                           Q_solar_in_floor[0][i], Q_INTrad_floor_R1[0][i], self.f[0][5], self.h_in_floor, self.h_in_ceil,
                                           tR1_air[i - 1][0])
            B_window_R1 = setB_window_load(self.N_window, self.h_out, e_all3[0][1], e_all3[0][3], q_solar_out_3[i][0], Q_INTrad_win_R1[0][i],
                                           ta_out[i][0], self.f[0][6], self.h_in_wall, tR1_air[i - 1][0]) # SOUTH WINDWOW
            # B_air_R1 = self.Cp_air * self.Rou_air * n_air / 3600 * V[0] * ta_out[i][0] + Q_conv1[0][i] + Q_hvac1[0][i]
            B_main = np.vstack((B_wall1_R1, B_wall2_R1, B_wall3_R1, B_wall4_R1, B_ceil_R1, B_floor_R1, B_window_R1)) # B_air_R1
            # SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            # [m1,temp_R1]=ode15s(@fun,[(i-1)*dt,i*dt],t1(:,i-1))
            # t1(:, i)=temp_R1(length(m1),:)
            a, b = function(self.Construct_heat_flux_flow_relationship_room1(), B_main, self.C_main)
            def deriv(y, t):
                return (np.dot(a, y.reshape(NN_main, 1)) + b)[:, 0]
            time1 = np.linspace((i - 1) * dt, i * dt, 1000)
            temp_R1 = odeint(deriv, t1[:, i - 1], time1)
            t1[:, i] = temp_R1[-1, :].T
            tR1_wall1_in[i][0] = t1[np.sum(self.N_EXTwall[:], dtype=int)][i]
            tR1_wall2_in[i][0] = t1[np.sum(self.N_EXTwall[:], dtype=int) + 1][i]
            tR1_wall3_in[i][0] = t1[np.sum(self.N_EXTwall[:], dtype=int) + 2][i]
            tR1_wall4_in[i][0] = t1[np.sum(self.N_EXTwall[:], dtype=int) + 3][i]
            tR1_ceil_in[i][0] = t1[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2
                                   + np.sum(self.N_ceil[:], dtype=int) + 4][i]
            tR1_floor_in[i][0] = t1[NN_main - np.sum(self.N_window, dtype=int) - 1][i]
            tR1_window_in[i][0] = t1[NN_main - 1][i]
            tR1_window_out[i][0] = t1[NN_main - 2][i]
            dQair_change_x1 = self.h_in_wall * self.f[0][0] * (tR1_wall1_in[i][0] - tR1_air[i - 1][0]) + self.h_in_wall * \
                              self.f[0][1] * (tR1_wall2_in[i][0] - tR1_air[i - 1][0]) + self.h_in_wall * self.f[0][2] * \
                              (tR1_wall3_in[i][0] - tR1_air[i - 1][0]) + self.h_in_wall * self.f[0][3] * \
                              (tR1_wall4_in[i][0] - tR1_air[i - 1][0]) + self.h_in_ceil * self.f[0][4] * \
                              (tR1_ceil_in[i][0] - tR1_air[i - 1][0]) + self.h_in_floor * self.f[0][5] * \
                              (tR1_ceil_in[i][0] - tR1_air[i - 1][0]) +self.h_in_wall * self.f[0][6] * \
                              np.float(tR1_window_in[i][0] - tR1_air[i - 1][0]) + np.float(self.Cp_air) * np.float(self.Rou_air) * np.float(self.n_air) * \
                              np.float(self.V[0]) / np.float(3600)  * np.float(ta_out[i][0] - tR1_air[i - 1][0]) + Q_conv1[0][i]
            denominator1 = (self.h_in_wall * np.sum(f[0, 0:4]) + self.h_in_ceil * self.f[0][4] + self.h_in_floor * self.f[0][5] + self.h_in_wall * self.f[0][6]) \
                           * self.dt / self.Cp_air / self.Rou_air / self.V[0] + (self.n_air * self.dt / 3600 + 1)
            dQair_change1 = dQair_change_x1 / denominator1
            # SOLVING ROOM 1 TEMPERATURE
            tR1_air[i][0] = dQair_change1 * self.dt / self.Cp_air / self.Rou_air / self.V[0] + tR1_air[i - 1][0]
            # HIMIDITY MIXING CALCULATION OF ROOM 1; 2 OCCUPANTS IN MAIN BEDCHAMBER
            w_e_in_R1[i][0] = w_e_in_R1[i - 1][0] + self.n_air * self.dt / 3600 * (w_e_out[i][0] - w_e_in_R1[i - 1][0]) \
                              + m_w_gp * 2 * OCCUPIED_BED[i][0] / self.Rou_air / self.V[0]
            # HEATING & COOLING LOADS CALCULATION ENCOMPASSING DEHUMIDIFICATION/LATENT HEAT
            if self.HC_mode == 1:    # FULL TIME SPACE HEATING AND COOLING - 24 H AVAILABLE IN THE PERIODS
                HEATING_BEDROOM[i][0] = float((Hour[i][0] >= 0 and Hour[i][0] < 1560) or (Hour[i][0] > 8160 and Hour[i][0] <= 8904))
                HEATING_SITTROOM[i][0] = float((Hour[i][0] >= 0 and Hour[i][0] < 1560) or (Hour[i][0] > 8160 and Hour[i][0] <= 8904))
                COOLING_BEDROOM[i][0] = float(Hour[i][0] >= 4104 and Hour[i][0] <= 5976)
                COOLING_SITTROOM[i][0] = float(Hour[i][0] >= 4104 and Hour[i][0] <= 5976)
            elif self.HC_mode == 2: # PART TIME SPACE HEATING AND COOLING AS PER SPECIFIED OCCUPANTS BEHAVIOUR PROFILE
                HEATING_BEDROOM[i][0] = float(OCCUPIED_BED[i][0] and ((Hour[i][0] >= 0 and Hour[i][0] < 1560) or (Hour[i][0] > 8160 and Hour[i][0] <= 8904)))
                HEATING_SITTROOM[i][0] = float(OCCUPIED_SITT[i][0] and ((Hour[i][0] >= 0 and Hour[i][0] < 1560) or (Hour[i][0] > 8160 and Hour[i][0] <= 8904)))
                COOLING_BEDROOM[i][0] = float(OCCUPIED_BED[i][0] and (Hour[i][0] >= 4104 and Hour[i][0] <= 5976))
                COOLING_SITTROOM[i][0] = float(OCCUPIED_SITT[i][0] and (Hour[i][0] >= 4104 and Hour[i][0] <= 5976))
            if HEATING_BEDROOM[i][0]:
                if w_e_in_R1[i][0]> w_e_in_set_1:
                    Q_h_dehumid_R1[i - 1][0] = 2260 * (w_e_in_R1[i][0] - w_e_in_set_1) * self.Rou_air * self.V[0] / self.dt
                    w_e_in_R1[i][0] = w_e_in_set_1
                if tR1_air[i][0] < self.Ta_targ_min:
                    Q_hload_R1[i - 1][0] = self.Cp_air * self.Rou_air * self.V[0] * (self.Ta_targ_min - tR1_air[i - 1][0]) / \
                                           self.dt  +self.h_in_wall * self.f[0][0] * (self.Ta_targ_min - tR1_wall1_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[0][1] * (self.Ta_targ_min - tR1_wall2_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[0][2] * (self.Ta_targ_min - tR1_wall3_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[0][3] * (self.Ta_targ_min - tR1_wall4_in[i - 1][0]) + \
                                           self.h_in_ceil * self.f[0][4] * (self.Ta_targ_min - tR1_ceil_in[i - 1][0]) + \
                                           self.h_in_floor * self.f[0][5] * (self.Ta_targ_min - tR1_ceil_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[0][6] * (self.Ta_targ_min - tR1_window_in[i - 1][0]) + \
                                           self.Cp_air * self.Rou_air * self.n_air * self.V[0] / 3600 * (self.Ta_targ_min - ta_out[i][0]) \
                                           - Q_conv1[0][i]
                    tR1_air[i][0] = self.Ta_targ_min
            # TARGETED ROOM TEMPERATURE 26 C FOR SPACE COOLIN
            elif COOLING_BEDROOM[i][0]:
                if w_e_in_R1[i][0] > w_e_in_set_2:
                    Q_c_latent_R1[i - 1][0] = 2260 * (w_e_in_R1[i][0] - w_e_in_set_2) * self.Rou_air * self.V[0] / self.dt
                    w_e_in_R1[i][0] = w_e_in_set_2
                if tR1_air[i][0] > self.Ta_targ_max:
                    Q_c_sens_R1[i - 1][0] = self.Cp_air * self.Rou_air * self.V[0] * (tR1_air[i - 1][0] - self.Ta_targ_max) / self.dt + dQair_change_x1
                    tR1_air[i][0] = self.Ta_targ_max
            # room2
            B_wall1_R2 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0], Q_solar_in_wall[1][i], 0, ta_out[i][0], self.f[1][0])
            B_wall2_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[1][i], 0, Q_solar_in_wall[4][i],
                                      Q_INTrad_wall1_R5[0][i], self.f[4][0], tR5_air[i - 1][0], tR5_wall2_in[i - 1][0],
                                      tR5_wall3_in[i - 1][0], tR5_wall4_in[i - 1][0], tR5_ceil_in[i - 1][0], tR5_floor_in[i - 1][0],
                                      tR5_window_in[i - 1][0], self.hr[4][1], self.hr[4][2], self.hr[4][3], self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[1][1])
            B_wall3_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[1][i], 0, Q_solar_in_wall[0][i],
                                      Q_INTrad_wall1_R1[0][i], self.f[0][3], tR1_air[i][0], tR1_wall2_in[i][0],
                                      tR1_wall3_in[i][0], tR1_wall4_in[i][0], tR1_ceil_in[i][0], tR1_floor_in[i][0],
                                      tR1_window_in[i][0], self.hr[0][0], self.hr[0][1], self.hr[0][2],
                                      self.hr[0][4], self.hr[0][5], self.hr[0][6], self.f[1][2])
            B_wall4_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[1][i], 0,
                                      Q_solar_in_wall[2][i],
                                      Q_INTrad_wall1_R3[0][i], self.f[2][2], tR3_air[i - 1][0], tR3_wall2_in[i - 1][0],
                                      tR3_wall3_in[i - 1][0], tR3_wall4_in[i - 1][0], tR3_ceil_in[i - 1][0],
                                      tR3_floor_in[i - 1][0],
                                      tR3_window_in[i - 1][0], self.hr[2][0], self.hr[2][1], self.hr[2][3],
                                      self.hr[2][4], self.hr[2][5], self.hr[2][6], self.f[1][3])
            B_ceil_R2 = setB_ceiling(self.N_ceil, self.eb, Q_solar_in_floor[1][i], 0, self.f[1][5], Q_solar_in_ceil[1][i], 0, self.f[1][4])
            B_floor_R2 = setB_ceiling(self.N_floor, self.eb, Q_solar_in_ceil[1][i], 0, self.f[1][4], Q_solar_in_floor[1][i], 0,f[1][5])
            B_window_R2 = setB_window(self.N_window, self.h_out, e_all1[0][1], e_all1[0][3], q_solar_out_1[i][0], 0, ta_out[i][0], self.f[1][6]) # EAST WINDOW
            B_air_R2 = self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[1] * ta_out[i][0] + Q_conv2[0][i] + Q_hvac2[0][i]
            B_main2 = np.vstack((B_wall1_R2, B_wall2_R2, B_wall3_R2, B_wall4_R2, B_ceil_R2, B_floor_R2, B_window_R2, B_air_R2))
            #  % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m2,temp_R2]=ode15s(@fun2,[(i-1)*dt,i*dt],t2(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t2(:,i)=temp_R2(length(m2),:)';
            a2, b2 = function(self.Construct_heat_flux_flow_relationship_room2(), B_main2, self.C_main2)
            def deriv2(y, t):
                return (np.dot(a2, y.reshape(NN_main2, 1)) + b2)[:, 0]
            temp_R2 = odeint(deriv2, t2[:, i - 1], time1)
            t2[:, i] = temp_R2[-1, :].T
            tR2_wall1_in[i][0] = t2[np.sum(self.N_EXTwall[:], dtype=int)][i]
            tR2_wall2_in[i][0] = t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            tR2_wall3_in[i][0] = t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
            tR2_wall4_in[i][0] = t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 3 + 3][i]
            tR2_ceil_in[i][0] = t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 3 + np.sum(self.N_ceil[:], dtype=int) + 4][i]
            tR2_floor_in[i][0] = t2[NN_main2 - np.sum(self.N_window, dtype=int) - 2][i]
            tR2_window_in[i][0] = t2[NN_main2 - 2][i]
            tR2_window_out[i][0] = t2[NN_main2 - 3][i]
            tR2_air[i][0] = t2[NN_main2 - 1][i]
            # room3
            B_wall1_R3 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0], Q_solar_in_wall[2][i],
                                        Q_INTrad_wall1_R3[0][i], ta_out[i][0], self.f[2][0], self.h_in_wall, tR3_air[i][0])
            B_wall2_R3 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * (0.667 * np.float(tR4_air[i - 1][0]) + 0.333 * np.float(tR5_air[i - 1][0])),
                                            eb, Q_solar_in_wall[2][i], Q_INTrad_wall1_R3[0][i],
                                            [Q_solar_in_wall[3][i] + (1 / 7.55) * Q_solar_in_wall[4][i]],(1.0 / 7.55) * Q_INTrad_wall1_R5[0][i],
                                            [0.667 * (self.hr[3][1] * tR4_wall2_in[i - 1][0] + self.hr[3][2] * tR4_wall3_in[i - 1][0] +
                                                      self.hr[3][3] * tR4_wall4_in[i - 1][0] + self.hr[3][4] * tR4_ceil_in[i - 1][0] +
                                                      self.hr[3][5] * tR4_floor_in[i - 1][0] + self.hr[3][6] * tR4_window_in[i - 1][0]) +
                                             0.333 * (self.hr[4][1] * tR5_wall2_in[i - 1][0] + self.hr[4][2] * tR5_wall3_in[i - 1][0] +
                                                      self.hr[4][3] * tR5_wall4_in[i - 1][0] + self.hr[4][4] * tR5_ceil_in[i - 1][0] +
                                                      self.hr[4][5] * tR5_floor_in[i - 1][0] + self.hr[4][6] * tR5_window_in[i - 1][0])],
                                            self.f[2][1], h_in_wall * tR3_air[i - 1][0])
            B_wall3_R3 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[2][i], Q_INTrad_wall3_R3[0][i],
                                           Q_solar_in_wall[1][i], 0, self.f[1][3], tR2_air[i][0], tR2_wall1_in[i][0],
                                           tR2_wall2_in[i][0], tR2_wall3_in[i][0], tR2_ceil_in[i][0], tR2_floor_in[i][0],
                                           tR2_window_in[i][0], self.hr[1][0], self.hr[1][1], self.hr[1][2], self.hr[1][4],
                                           self.hr[1][5], self.hr[1][6], self.f[2][2], tR3_air[i - 1][0])
            B_wall4_R3 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_4[i][0], Q_solar_in_wall[2][i],
                                        Q_INTrad_wall4_R3[0][i], ta_out[i][0], self.f[2][3], self.h_in_wall, tR3_air[i - 1][0])
            B_ceil_R3 = setB_ceiling_load(self.N_ceil, self.eb, Q_solar_in_floor[2][i], Q_INTrad_floor_R3[0][i], self.f[2][5],
                                          Q_solar_in_ceil[2][i], Q_INTrad_ceil_R3[0][i], self.f[2][4], self.h_in_ceil, self.h_in_floor,
                                          tR3_air[i - 1][0])
            B_floor_R3 = setB_ceiling_load(self.N_floor, self.eb, Q_solar_in_ceil[2][i], Q_INTrad_ceil_R3[0][i], self.f[2][4],
                                           Q_solar_in_floor[2][i], Q_INTrad_floor_R3[0][i], self.f[2][5], self.h_in_floor, self.h_in_ceil,
                                           tR3_air[i - 1][0])
            B_window_R3 = setB_window_load(self.N_window, self.h_out, e_all1[0][1], e_all1[0][3], q_solar_out_1[i][0], Q_INTrad_win_R3[0][i],
                                           ta_out[i][0], self.f[2][6], self.h_in_wall, tR3_air[i - 1][0])  # EAST WINDOW
            B_main3 = np.vstack((B_wall1_R3, B_wall2_R3, B_wall3_R3, B_wall4_R3, B_ceil_R3, B_floor_R3, B_window_R3))
            # SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m3,temp_R3]=ode15s(@fun3,[(i-1)*dt,i*dt],t3(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t3(:,i)=temp_R3(length(m3),:)';
            a3, b3 = function(self.Construct_heat_flux_flow_relationship_room3(), B_main3, self.C_main3)
            def deriv3(y, t):
                return (np.dot(a3, y.reshape(NN_main3, 1)) + b3)[:, 0]
            temp_R3 = odeint(deriv3, t3[:, i - 1], time1)
            t3[:, i] = temp_R3[-1, :].T
            tR3_wall1_in[i][0] = t3[np.sum(self.N_EXTwall[:], dtype=int)][i]
            tR3_wall2_in[i][0] = t3[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            tR3_wall3_in[i][0] = t3[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
            tR3_wall4_in[i][0] = t3[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
            tR3_ceil_in[i][0] = t3[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(self.N_ceil[:],dtype=int) + 4][i]
            tR3_floor_in[i][0] = t3[NN_main3 - np.sum(self.N_window, dtype=int) - 2][i]
            tR3_window_in[i][0] = t3[NN_main3 - 2][i]
            tR3_window_out[i][0] = t3[NN_main3 - 3][i]
            tR3_air[i][0] = t3[NN_main3 - 1][i]
            dQair_change_x3 = self.h_in_wall * self.f[2][0] * (tR3_wall1_in[i][0] - tR3_air[i - 1][0]) + self.h_in_wall * \
                              self.f[2][1] * (tR3_wall2_in[i][0] - tR3_air[i - 1][0]) + self.h_in_wall * self.f[2][2] * \
                              (tR3_wall3_in[i][0] - tR3_air[i - 1][0]) + self.h_in_wall * self.f[2][3] * (tR3_wall4_in[i][0] - tR3_air[i - 1][0]) \
                              + self.h_in_ceil * self.f[2][4] * (tR3_ceil_in[i][0] - tR3_air[i - 1][0]) + self.h_in_floor * \
                              self.f[2][5] * np.float(tR3_ceil_in[i][0] - tR3_air[i - 1][0]) + self.h_in_wall * self.f[2][6] * \
                              np.float(tR3_window_in[i][0] - tR3_air[i - 1][0]) + np.float(self.Cp_air) * np.float(self.Rou_air) * \
                              np.float(self.n_air) * np.float(self.V[2]) / 3600  * np.float(ta_out[i][0] - tR1_air[i - 1][0]) + Q_conv3[0][i]
            denominator3 = (self.h_in_wall * np.sum(self.f[2, 0:4]) + self.h_in_ceil * self.f[2][4] + self.h_in_floor *
                            self.f[2][5] + self.h_in_wall * self.f[2][6]) * self.dt / self.Cp_air / self.Rou_air / self.V[2] \
                           + (self.n_air * self.dt / 3600 + 1)
            dQair_change3 = dQair_change_x3 / denominator3
            tR3_air[i][0] = dQair_change3 * self.dt / self.Cp_air / self.Rou_air / self.V[2] + tR3_air[i - 1][0] # SOLVING ROOM 3 TEMPERATURE
            w_e_in_R3[i][0] = w_e_in_R3[i - 1][0] + self.n_air * self.dt / 3600 * (w_e_out[i][0] - w_e_in_R3[i - 1][0]) + \
                              m_w_gp * 1 * OCCUPIED_BED[i][0] / self.Rou_air / self.V[2]
            if HEATING_BEDROOM[i][0]:
                if w_e_in_R3[i][0] > w_e_in_set_1:
                    Q_h_dehumid_R3[i - 1][0] = 2260 * (w_e_in_R3[i][0] - w_e_in_set_1) * self.Rou_air * self.V[2] / self.dt
                    w_e_in_R3[i][0] = w_e_in_set_1
                if tR3_air[i][0] < self.Ta_targ_min:
                    Q_hload_R3[i - 1][0] = self.Cp_air * self.Rou_air * self.V[2] * (self.Ta_targ_min - tR3_air[i - 1][0]) / self.dt \
                                           + self.h_in_wall * self.f[2][0] * (self.Ta_targ_min - tR3_wall1_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[2][1] * (self.Ta_targ_min - tR3_wall2_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[2][2] * (self.Ta_targ_min - tR3_wall3_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[2][3] * (self.Ta_targ_min - tR3_wall4_in[i - 1][0]) + \
                                           self.h_in_ceil * self.f[2][4] * (self.Ta_targ_min - tR3_ceil_in[i - 1][0]) + \
                                           self.h_in_floor * self.f[2][5] * (self.Ta_targ_min - tR3_ceil_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[2][6] * (self.Ta_targ_min - tR3_window_in[i - 1][0]) + \
                                           self.Cp_air * self.Rou_air * self.n_air * self.V[2] / 3600 * \
                                           (self.Ta_targ_min - ta_out[i][0]) - Q_conv3[0][i]
                    tR3_air[i][0] = self.Ta_targ_min
            # room4
            B_wall1_R4 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[3][i], 0, Q_solar_in_wall[2][i],
                                      Q_INTrad_wall2_R3[0][i], self.f[2][1], tR3_air[i][0], tR3_wall2_in[i][0], tR3_wall3_in[i][0],
                                      tR3_wall4_in[i][0], tR3_ceil_in[i][0], tR3_floor_in[i][0], tR3_window_in[i][0], self.hr[2][0],
                                      self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6], self.f[3][0])
            B_wall2_R4 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_2[i][0], Q_solar_in_wall[3][i], 0, ta_out[i][0], self.f[3][1])
            B_wall3_R4 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, Q_solar_in_wall[3][i], 0, Q_solar_in_wall[4][i],
                                      Q_INTrad_wall4_R5[0][i], self.f[4][3], tR5_air[i - 1][0], tR5_wall1_in[i - 1][0],
                                      tR5_wall2_in[i - 1][0], tR5_wall3_in[i - 1][0], tR5_ceil_in[i - 1][0], tR5_floor_in[i - 1][0],
                                      tR5_window_in[i - 1][0], self.hr[4][0], self.hr[4][1], self.hr[4][2],
                                      self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[3][2])
            B_wall4_R4 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_4[i][0], Q_solar_in_wall[3][i], 0, ta_out[i][0], self.f[3][3])
            B_ceil_R4 = setB_ceiling(self.N_ceil, self.eb, Q_solar_in_floor[3][i], 0, self.f[3][5], Q_solar_in_ceil[3][i], 0, self.f[3][4])
            B_floor_R4 = setB_ceiling(self.N_floor, self.eb, Q_solar_in_ceil[3][i], 0, self.f[3][4], Q_solar_in_floor[3][i], 0, self.f[3][5])
            B_window_R4 = setB_window(self.N_window, self.h_out, e_all4[0][1], e_all4[0][3], q_solar_out_4[i][0], 0, ta_out[i][0], self.f[3][6])
            B_air_R4 = self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[3] * ta_out[i][0] + Q_conv4[0][i] + Q_hvac4[0][i]
            B_main4 = np.vstack((B_wall1_R4, B_wall2_R4, B_wall3_R4, B_wall4_R4, B_ceil_R4, B_floor_R4, B_window_R4, B_air_R4))
            # % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m4,temp_R4]=ode15s(@fun4,[(i-1)*dt,i*dt],t4(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t4(:,i)=temp_R4(length(m4),:)';
            a4, b4 = function(self.Construct_heat_flux_flow_relationship_room4(), B_main4, self.C_main4)
            def deriv4(y, t):
                return (np.dot(a4, y.reshape(NN_main4, 1)) + b4)[:, 0]
            temp_R4 = odeint(deriv4, t4[:, i - 1], time1)
            t4[:, i] = temp_R4[-1, :].T
            tR4_wall1_in[i][0] = t4[np.sum(self.N_EXTwall[:], dtype=int)][i]
            tR4_wall2_in[i][0] = t4[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            tR4_wall3_in[i][0] = t4[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
            tR4_wall4_in[i][0] = t4[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
            tR4_ceil_in[i][0] = t4[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(self.N_ceil[:], dtype=int) + 4][i]
            tR4_floor_in[i][0] = t4[NN_main4 - np.sum(self.N_window, dtype=int) - 2][i]
            tR4_window_in[i][0] = t4[NN_main4 - 2][i]
            tR4_window_out[i][0] = t4[NN_main4 - 3][i]
            tR4_air[i][0] = t4[NN_main4 - 1][i]
            # room5
            B_wall1_R5 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * (0.53 * tR1_air[i][0] + 0.34 * tR2_air[i][0] + 0.13 * tR3_air[i][0]),
                                            self.eb, Q_solar_in_wall[4][i], Q_INTrad_wall1_R5[0][i],
                                            Q_solar_in_wall[0][i] + Q_solar_in_wall[1][i] + 0.333 * Q_solar_in_wall[2][i],
                                            Q_INTrad_wall1_R1[0][i] + 0 + 0.333 * Q_INTrad_wall1_R3[0][i],
                                            0.53 * (self.hr[0][0] * tR1_wall1_in[i][0] + self.hr[0][2] * tR1_wall3_in[i][0] +
                                                     self.hr[0][3] * tR1_wall4_in[i][0] + self.hr[0][4] * tR1_ceil_in[i][0] +
                                                     self.hr[0][5] * tR1_floor_in[i][0] + self.hr[0][6] * tR1_window_in[i][0]) +
                                             0.34 * (self.hr[1][0] * tR2_wall1_in[i][0] + self.hr[1][2] * tR2_wall3_in[i][0] +
                                                     self.hr[1][3] * tR2_wall4_in[i][0] + self.hr[1][4] * tR2_ceil_in[i][0] +
                                                     self.hr[1][5] * tR2_floor_in[i][0] + self.hr[1][6] * tR2_window_in[i][0]) +
                                             0.13 * (self.hr[2][0] * tR3_wall1_in[i][0] + self.hr[2][2] * tR3_wall3_in[i][0] +
                                                     self.hr[2][3] * tR3_wall4_in[i][0] + self.hr[2][4] * tR3_ceil_in[i][0] +
                                                     self.hr[2][5] * tR3_floor_in[i][0] + self.hr[2][6] * tR3_window_in[i][0]),
                                            self.f[4][0],  self.h_in_wall * tR5_air[i - 1][0])
            B_wall2_R5 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_2[i][0], Q_solar_in_wall[4][i],
                                        Q_INTrad_wall2_R5[0][i], ta_out[i][0], self.f[4][1], self.h_in_wall, tR5_air[i - 1][0])
            B_wall3_R5 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_3[i][0], Q_solar_in_wall[4][i],
                                        Q_INTrad_wall3_R5[0][i], ta_out[i][0], self.f[4][2], self.h_in_wall, tR5_air[i - 1][0])
            B_wall4_R5 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * tR4_air[i][0], eb, Q_solar_in_wall[4][i],
                                            Q_INTrad_wall4_R5[0][i], Q_solar_in_wall[3][i], 0,
                                            [self.hr[3][0] * tR4_wall1_in[i][0] + self.hr[3][1] * tR4_wall2_in[i][0] +
                                             self.hr[3][3] * tR4_wall4_in[i][0] + self.hr[3][4] * tR4_ceil_in[i][0] +
                                             self.hr[3][5] * tR4_floor_in[i][0] + self.hr[3][6] * tR4_window_in[i][0]],
                                            self.f[4][3], self.h_in_wall * tR5_air[i - 1][0])
            B_ceil_R5 = setB_ceiling_load(self.N_ceil, self.eb, Q_solar_in_floor[4][i], Q_INTrad_floor_R5[0][i], self.f[4][5],
                                          Q_solar_in_ceil[4][i], Q_INTrad_ceil_R5[0][i], self.f[4][4], self.h_in_ceil, self.h_in_floor,
                                          tR5_air[i - 1][0])
            B_floor_R5 = setB_ceiling_load(self.N_floor, self.eb, Q_solar_in_ceil[4][i], Q_INTrad_ceil_R5[0][i], self.f[4][4],
                                           Q_solar_in_floor[4][i], Q_INTrad_floor_R5[0][i],self.f[4][5], self.h_in_floor, self.h_in_ceil,
                                           tR5_air[i - 1][0])
            B_window_R5 = setB_window_load(self.N_window, self.h_out, e_all3[0][1], e_all3[0][3], q_solar_out_3[i][0], Q_INTrad_win_R5[0][i],
                                           ta_out[i][0], self.f[4][6], self.h_in_wall, tR5_air[i - 1][0])
            B_main5 = np.vstack((B_wall1_R5, B_wall2_R5, B_wall3_R5, B_wall4_R5, B_ceil_R5, B_floor_R5, B_window_R5))
            #  % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m5,temp_R5]=ode15s(@fun5,[(i-1)*dt,i*dt],t5(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t5(:,i)=temp_R5(length(m5),:)';
            a5, b5 = function(self.Construct_heat_flux_flow_relationship_room5(), B_main5, self.C_main5)
            def deriv5(y, t):
                return (np.dot(a5, y.reshape(NN_main5, 1)) + b5)[:, 0]
            temp_R5 = odeint(deriv5, t5[:, i - 1], time1)
            t5[:, i] = temp_R5[-1, :].T
            tR5_wall1_in[i][0] = t5[np.sum(self.N_EXTwall[:], dtype=int)][i]
            tR5_wall2_in[i][0] = t5[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            tR5_wall3_in[i][0] = t5[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) + 2][i]
            tR5_wall4_in[i][0] = t5[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
            tR5_ceil_in[i][0] = t5[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(self.N_ceil[:], dtype=int) + 4][i]
            tR5_floor_in[i][0] = t5[NN_main5 - np.sum(self.N_window, dtype=int) - 2][i - 1]
            tR5_window_in[i][0] = t5[NN_main5 - 2][i]
            tR5_window_out[i][0] = t5[NN_main5 - 3][i]
            tR5_air[i][0] = t5[NN_main5 - 1][i]
            dQair_change_x5 = self.h_in_wall * self.f[4][0] * (tR5_wall1_in[i][0] - tR5_air[i - 1][0]) + self.h_in_wall * \
                              self.f[4][1] * (tR5_wall2_in[i][0] - tR5_air[i - 1][0]) + self.h_in_wall * self.f[4][2] * \
                              (tR5_wall3_in[i][0] - tR5_air[i - 1][0]) + self.h_in_wall * self.f[4][3] * \
                              (tR5_wall4_in[i][0] - tR5_air[i - 1][0]) + self.h_in_ceil * self.f[4][4] * \
                              (tR5_ceil_in[i][0] - tR5_air[i - 1][0]) + self.h_in_floor * self.f[4][5] * \
                              (tR5_ceil_in[i][0] - tR5_air[i - 1][0]) + self.h_in_wall * self.f[4][6] * \
                              (tR5_window_in[i][0] - tR5_air[i - 1][0]) + self.Cp_air * self.Rou_air * self.n_air * self.V[4] \
                              / 3600 * (ta_out[i][0] - tR5_air[i - 1][0]) + Q_conv5[0][i]
            denominator5 = (self.h_in_wall * np.sum(f[4, 0:4]) + self.h_in_ceil * self.f[4][4] + self.h_in_floor *
                            self.f[4][5] + self.h_in_wall * self.f[4][6]) * self.dt / self.Cp_air / self.Rou_air / self.V[4] + \
                           (self.n_air * self.dt / 3600 + 1)
            dQair_change5 = dQair_change_x5 / denominator5
            tR5_air[i][0] = dQair_change5 * self.dt / self.Cp_air / self.Rou_air / self.V[4] + tR5_air[i - 1][0]
            w_e_in_R5[i][0] = w_e_in_R5[i - 1][0] + self.n_air * self.dt / 3600 * (w_e_out[i][0] - w_e_in_R5[i - 1][0]) + \
                              m_w_gp * 3 * OCCUPIED_SITT[i][0] / self.Rou_air / self.V[4]
            if HEATING_SITTROOM[i][0]:
                if w_e_in_R5[i][0] > w_e_in_set_1:
                    Q_h_dehumid_R5[i - 1][0] = 2260 * (w_e_in_R5[i][0] - w_e_in_set_1) * self.Rou_air * self.V[4] / self.dt
                    w_e_in_R5[i][0] = w_e_in_set_1
                if tR5_air[i][0] < self.Ta_targ_min:
                    Q_hload_R5[i - 1][0] = self.Cp_air * self.Rou_air * self.V[4] * (self.Ta_targ_min - tR5_air[i - 1][0]) \
                                           / self.dt + self.h_in_wall * self.f[4][0] * (self.Ta_targ_min - tR5_wall1_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[4][1] * (self.Ta_targ_min - tR5_wall2_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[4][2] * (self.Ta_targ_min - tR5_wall3_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[4][3] * (self.Ta_targ_min - tR5_wall4_in[i - 1][0]) + \
                                           self.h_in_ceil * self.f[4][4] * (self.Ta_targ_min - tR5_ceil_in[i - 1][0]) + \
                                           self.h_in_floor * self.f[4][5] * (self.Ta_targ_min - tR5_ceil_in[i - 1][0]) + \
                                           self.h_in_wall * self.f[4][6] * (self.Ta_targ_min - tR5_window_in[i - 1][0]) + \
                                           self.Cp_air * self.Rou_air * self.n_air * self.V[4] / 3600 * (self.Ta_targ_min - ta_out[i][0]) - Q_conv5[0][i]
                    tR5_air[i][0] = self.Ta_targ_min
            elif COOLING_SITTROOM[i][0]:
                if w_e_in_R5[i][0] > w_e_in_set_2:
                    Q_c_latent_R5[i - 1][0] = 2260 * (w_e_in_R5[i][0] - w_e_in_set_2) * self.Rou_air * self.V[4] / self.dt
                    w_e_in_R5[i][0] = w_e_in_set_2
                if tR5_air[i][0] > self.Ta_targ_max:
                    Q_c_sens_R5[i - 1][0] = self.Cp_air * self.Rou_air * self.V[4] * (tR5_air[i - 1][0] - self.Ta_targ_max) / self.dt + dQair_change_x5
                    tR5_air[i][0] = self.Ta_targ_max

        BSC = (np.sum(f[0:4, 0]) + np.sum(f[3:5, 3])+np.sum(f[3:5, 1]) + self.f[4][2] + self.f[0][2] +
               np.sum(f[0:5, 4])+np.sum(f[0:5, 5])) / np.sum(V[:])   # BUILDING SHAPE FACTOR
        U_value_w = 1 / (0.13 + np.sum(D_EXTwall / 1000. / Lamda_EXTwall) + 0.04)   # U-VALUE OF EXTERNAL WALLS
        Q_h_SENS = Q_hload_R1 + Q_hload_R3 + Q_hload_R5 # WINTER SENSIBLE HEATING LOAD
        Q_h_dehumid = Q_h_dehumid_R1 + Q_h_dehumid_R3 + Q_h_dehumid_R5 # DEHUMIDIFICATION LOAD IN WINTER
        Q_hload_tot = Q_h_SENS + Q_h_dehumid
        Dev = 24 * 6 * 3600 / self.dt
        Q_h_SENS_sum = np.sum(Q_h_SENS[int(Dev - 1): , 0]) / 1000 * self.dt / 3600 / np.sum(f[0:5, 5]) # WINTER SENSIBLE HEATING LOAD IN kWh / m ^ 2
        q_hload = Q_h_SENS_sum * 1000 / 24 / 121
        Q_h_LATT_sum = np.sum(Q_h_dehumid[int(Dev - 1): , 0]) / 1000 * self.dt / 3600 / np.sum(f[0:5, 5])
        Q_hload_sum = Q_h_SENS_sum + Q_h_LATT_sum
        # %PRINT SUM OF HEATING LOAD PER MONTH
        Q_h_Jan = (np.sum(Q_h_SENS[int(Dev - 1):int(888 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(Dev - 1): int(888 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
        Q_h_Feb = (np.sum(Q_h_SENS[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Mar = (np.sum(Q_h_SENS[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(1560 * 3600 / self.dt) - 1: int(2304 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Apr = (np.sum(Q_h_SENS[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(2304 * 3600 / self.dt) - 1: int(3024 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_May = (np.sum(Q_h_SENS[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(3024 * 3600 / self.dt) - 1: int(3768 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Jun = (np.sum(Q_h_SENS[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(3768 * 3600 / self.dt) - 1: int(4488 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Jul = (np.sum(Q_h_SENS[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(4488 * 3600 / self.dt) - 1: int(5232 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Aug = (np.sum(Q_h_SENS[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(5232 * 3600 / self.dt) - 1: int(5976 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Sep = (np.sum(Q_h_SENS[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(5976 * 3600 / self.dt) - 1: int(6696 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Oct = (np.sum(Q_h_SENS[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(6696 * 3600 / self.dt) - 1: int(7440 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Nov = (np.sum(Q_h_SENS[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(7440 * 3600 / self.dt) - 1: int(8160 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Dec = (np.sum(Q_h_SENS[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(8160 * 3600 / self.dt) - 1: int(8904 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_MONTH = [Q_h_Jan, Q_h_Feb, Q_h_Mar, Q_h_Apr, Q_h_May, Q_h_Jun, Q_h_Jul, Q_h_Aug, Q_h_Sep, Q_h_Oct, Q_h_Nov, Q_h_Dec]
        Q_c_SENS = Q_c_sens_R1 + Q_c_sens_R3 + Q_c_sens_R5
        Q_c_LATENT = Q_c_latent_R1 + Q_c_latent_R3 + Q_c_latent_R5
        Q_cload_tot = Q_c_SENS + Q_c_LATENT
        Q_c_SENS_sum = np.sum(Q_c_SENS[:, 0]) / 1000 * self.dt / 3600 / np.sum(f[0:5, 5])
        Q_c_LATENT_sum = np.sum(Q_c_LATENT[:, 0]) / 1000 * self.dt / 3600 / sum(f[0:5, 5])
        Q_cload_sum = Q_c_SENS_sum + Q_c_LATENT_sum
        Qload_sum = Q_hload_sum + Q_cload_sum
        Q_c_Jan = (np.sum(Q_c_SENS[int(Dev - 1):int(888 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(Dev - 1): int(888 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Feb = (np.sum(Q_c_SENS[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0])) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Mar = (np.sum(Q_c_SENS[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(1560 * 3600 / self.dt) - 1: int(2304 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Apr = (np.sum(Q_c_SENS[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(2304 * 3600 / self.dt) - 1: int(3024 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_May = (np.sum(Q_c_SENS[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(3024 * 3600 / self.dt) - 1: int(3768 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Jun = (np.sum(Q_c_SENS[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(3768 * 3600 / self.dt) - 1: int(4488 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Jul = (np.sum(Q_c_SENS[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(4488 * 3600 / self.dt) - 1: int(5232 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Aug = (np.sum(Q_c_SENS[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(5232 * 3600 / self.dt) - 1: int(5976 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Sep = (np.sum(Q_c_SENS[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(5976 * 3600 / self.dt) - 1: int(6696 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Oct = (np.sum(Q_c_SENS[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(6696 * 3600 / self.dt) - 1: int(7440 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Nov = (np.sum(Q_c_SENS[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(7440 * 3600 / self.dt) - 1: int(8160 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Dec = (np.sum(Q_c_SENS[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(8160 * 3600 / self.dt) - 1: int(8904 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_MONTH = [Q_c_Jan, Q_c_Feb, Q_c_Mar, Q_c_Apr, Q_c_May, Q_c_Jun, Q_c_Jul, Q_c_Aug, Q_c_Sep, Q_c_Oct, Q_c_Nov, Q_c_Dec]
        # 通过窗户获得内部太阳热量的统计 - full time?
        Q_solar_in_win = Q_solar_in_R1 + Q_solar_in_R2 + Q_solar_in_R3 + Q_solar_in_R4 + Q_solar_in_R5
        Q_s_win_Jan = np.sum(Q_solar_in_win[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
        Q_s_win_Feb = np.sum(Q_solar_in_win[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Mar = np.sum(Q_solar_in_win[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Apr = np.sum(Q_solar_in_win[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_May = np.sum(Q_solar_in_win[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Jun = np.sum(Q_solar_in_win[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Jul = np.sum(Q_solar_in_win[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Aug = np.sum(Q_solar_in_win[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Sep = np.sum(Q_solar_in_win[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Oct = np.sum(Q_solar_in_win[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Nov = np.sum(Q_solar_in_win[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_Dec = np.sum(Q_solar_in_win[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_s_win_MONTH = [Q_s_win_Jan, Q_s_win_Feb, Q_s_win_Mar, Q_s_win_Apr, Q_s_win_May, Q_s_win_Jun, Q_s_win_Jul,
                         Q_s_win_Aug, Q_s_win_Sep, Q_s_win_Oct, Q_s_win_Nov, Q_s_win_Dec]
        # 空气渗透热损益统计——本质上应分为加热和冷却考虑部分时间目标
        Q_air_infiltra = self.Cp_air * self.Rou_air * self.n_air * (self.V[0] / 3600 * (ta_out - tR1_air) +
                                                                    self.V[2] / 3600 * (ta_out - tR3_air) + self.V[4] / 3600 * (ta_out - tR5_air))
        Q_air_inf_Jan = np.sum(Q_air_infiltra[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
        Q_air_inf_Feb = np.sum(Q_air_infiltra[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Mar = np.sum(Q_air_infiltra[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Apr = np.sum(Q_air_infiltra[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_May = np.sum(Q_air_infiltra[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Jun = np.sum(Q_air_infiltra[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Jul = np.sum(Q_air_infiltra[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Aug = np.sum(Q_air_infiltra[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Sep = np.sum(Q_air_infiltra[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Oct = np.sum(Q_air_infiltra[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Nov = np.sum(Q_air_infiltra[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_Dec = np.sum(Q_air_infiltra[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_air_inf_MONTH = [Q_air_inf_Jan, Q_air_inf_Feb, Q_air_inf_Mar, Q_air_inf_Apr, Q_air_inf_May, Q_air_inf_Jun,
                           Q_air_inf_Jul, Q_air_inf_Aug, Q_air_inf_Sep, Q_air_inf_Oct, Q_air_inf_Nov, Q_air_inf_Dec]
        # 统计从人员、设备和照明中获得的内热
        Q_int_hgain = Q_internal_rad_R1 + Q_internal_cov_R1 + Q_internal_rad_R3 + Q_internal_cov_R3 + Q_internal_rad_R5 + Q_internal_cov_R5
        Q_int_hg_Jan = np.sum(Q_int_hgain[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Feb = np.sum(Q_int_hgain[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Mar = np.sum(Q_int_hgain[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Apr = np.sum(Q_int_hgain[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_May = np.sum(Q_int_hgain[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Jun = np.sum(Q_int_hgain[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Jul = np.sum(Q_int_hgain[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Aug = np.sum(Q_int_hgain[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Sep = np.sum(Q_int_hgain[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Oct = np.sum(Q_int_hgain[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Nov = np.sum(Q_int_hgain[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_Dec = np.sum(Q_int_hgain[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_int_hg_MONTH = [Q_int_hg_Jan, Q_int_hg_Feb, Q_int_hg_Mar, Q_int_hg_Apr, Q_int_hg_May, Q_int_hg_Jun,
                          Q_int_hg_Jul, Q_int_hg_Aug, Q_int_hg_Sep, Q_int_hg_Oct, Q_int_hg_Nov, Q_int_hg_Dec]
        # 窗热传输的统计
        Q_winTRANs_h = self.h_window_12 * self.f[0][6] * (tR1_window_in - tR1_window_out) * HEATING_BEDROOM + self.h_window_12 * \
                       self.f[2][6] * (tR3_window_in - tR3_window_out) * HEATING_BEDROOM + self.h_window_12 * \
                       self.f[4][6] * (tR5_window_in - tR5_window_out) * HEATING_SITTROOM
        Q_winTRANs_c = self.h_window_12 * self.f[0][6] * (tR1_window_out - tR1_window_in) * COOLING_BEDROOM + self.h_window_12 * \
                       self.f[2][6] * (tR3_window_out - tR3_window_in) * COOLING_BEDROOM + self.h_window_12 * \
                       self.f[4][6] * (tR5_window_out - tR5_window_in) * COOLING_SITTROOM
        Q_winTranh_Jan = np.sum(Q_winTRANs_h[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Feb = np.sum(Q_winTRANs_h[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / np.sum(self.f[0: 5, 5])
        Q_winTranh_Mar = np.sum(Q_winTRANs_h[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Apr = np.sum(Q_winTRANs_h[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_May = np.sum(Q_winTRANs_h[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Jun = np.sum(Q_winTRANs_h[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Jul = np.sum(Q_winTRANs_h[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Aug = np.sum(Q_winTRANs_h[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Sep = np.sum(Q_winTRANs_h[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Oct = np.sum(Q_winTRANs_h[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Nov = np.sum(Q_winTRANs_h[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_Dec = np.sum(Q_winTRANs_h[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranh_MONTH = [Q_winTranh_Jan, Q_winTranh_Feb, Q_winTranh_Mar, Q_winTranh_Apr, Q_winTranh_May, Q_winTranh_Jun,
                            Q_winTranh_Jul, Q_winTranh_Aug, Q_winTranh_Sep, Q_winTranh_Oct, Q_winTranh_Nov, Q_winTranh_Dec]
        Q_winTranc_Jan = sum(Q_winTRANs_c[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Feb = sum(Q_winTRANs_c[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Mar = sum(Q_winTRANs_c[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Apr = sum(Q_winTRANs_c[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_May = sum(Q_winTRANs_c[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Jun = sum(Q_winTRANs_c[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Jul = sum(Q_winTRANs_c[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Aug = sum(Q_winTRANs_c[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Sep = sum(Q_winTRANs_c[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Oct = sum(Q_winTRANs_c[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Nov = sum(Q_winTRANs_c[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_Dec = sum(Q_winTRANs_c[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_winTranc_MONTH = [Q_winTranc_Jan, Q_winTranc_Feb, Q_winTranc_Mar, Q_winTranc_Apr, Q_winTranc_May, Q_winTranc_Jun,
                            Q_winTranc_Jul, Q_winTranc_Aug, Q_winTranc_Sep, Q_winTranc_Oct, Q_winTranc_Nov, Q_winTranc_Dec]
        # 外墙吸收太阳辐射的统计
        Q_ext_solarr = np.sum(f[0:3, 0]) * self.eb * q_solar_out_1 + np.sum(f[3:5, 1]) * self.eb * q_solar_out_2 + \
                       (self.f[0][2] + self.f[4][2]) * self.eb * q_solar_out_3 + np.sum(f[2:4, 3]) * self.eb * q_solar_out_4
        Q_ext_sol_Jan = sum(Q_ext_solarr[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Feb = sum(Q_ext_solarr[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Mar = sum(Q_ext_solarr[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Apr = sum(Q_ext_solarr[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_May = sum(Q_ext_solarr[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Jun = sum(Q_ext_solarr[int(3768 * 3600 / self.dt) - 1:int(4488 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Jul = sum(Q_ext_solarr[int(4488 * 3600 / self.dt) - 1:int(5232 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Aug = sum(Q_ext_solarr[int(5232 * 3600 / self.dt) - 1:int(5976 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Sep = sum(Q_ext_solarr[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Oct = sum(Q_ext_solarr[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Nov = sum(Q_ext_solarr[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_Dec = sum(Q_ext_solarr[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_ext_sol_MONTH = [Q_ext_sol_Jan, Q_ext_sol_Feb, Q_ext_sol_Mar, Q_ext_sol_Apr, Q_ext_sol_May, Q_ext_sol_Jun,
                           Q_ext_sol_Jul, Q_ext_sol_Aug, Q_ext_sol_Sep, Q_ext_sol_Oct, Q_ext_sol_Nov, Q_ext_sol_Dec]
        outcome = [Mon[int(Dev - 1):int(T), 0], Day[int(Dev - 1):int(T), 0], nh[int(Dev - 1):int(T), 0], Min[int(Dev - 1):int(T), 0], ta_out[int(Dev - 1):int(T), 0],
                   tR1_air[int(Dev - 1):int(T), 0], tR2_air[int(Dev - 1):int(T), 0], tR3_air[int(Dev - 1):int(T), 0], tR4_air[int(Dev - 1):int(T), 0],
                   tR5_air[int(Dev - 1):int(T), 0], Q_hload_R1[int(Dev - 1):int(T), 0], Q_c_sens_R1[int(Dev - 1):int(T), 0], Q_hload_R3[int(Dev - 1):int(T), 0],
                   Q_c_sens_R3[int(Dev - 1):int(T), 0], Q_hload_R5[int(Dev - 1):int(T), 0], Q_c_sens_R5[int(Dev - 1):int(T), 0], Q_hload_tot[int(Dev - 1):int(T), 0],
                   Q_cload_tot[int(Dev - 1):int(T), 0], Q_h_dehumid[int(Dev - 1):int(T), 0], Q_c_LATENT[int(Dev - 1):int(T), 0]]
        summary = [D_insulation[0][0], U_value_w, 0, Q_h_SENS_sum, q_hload, Q_h_LATT_sum, Q_hload_sum, 0, Q_c_SENS_sum,
                   Q_c_LATENT_sum, Q_cload_sum, Qload_sum]

        load = np.array([summary, Q_h_MONTH, Q_c_MONTH, Q_s_win_MONTH, Q_air_inf_MONTH, Q_int_hg_MONTH, Q_winTranc_MONTH, Q_ext_sol_MONTH]).T
        load_data = pd.DataFrame(load, columns=['summary', 'Q_h_MONTH', 'Q_c_MONTH', 'Q_s_win_MONTH', 'Q_air_inf_MONTH', 'Q_int_hg_MONTH', 'Q_winTranc_MONTH', 'Q_ext_sol_MONTH'])
        temperatures = np.hstack((Mon[int(Dev - 1):int(T), 0:1], Day[int(Dev - 1):int(T), 0:1], nh[int(Dev - 1):int(T), 0:1], Min[int(Dev - 1):int(T), 0:1], ta_out[int(Dev - 1):int(T), 0:1],
                                  tR1_air[int(Dev - 1):int(T), 0:1], tR2_air[int(Dev - 1):int(T), 0:1], tR3_air[int(Dev - 1):int(T), 0:1], tR4_air[int(Dev - 1):int(T), 0:1],
                                  tR5_air[int(Dev - 1):int(T), 0:1], Q_hload_R1[int(Dev - 1):int(T), 0:1], Q_c_sens_R1[int(Dev - 1):int(T), 0:1], Q_hload_R3[int(Dev - 1):int(T), 0:1],
                                  Q_c_sens_R3[int(Dev - 1):int(T), 0:1], Q_hload_R5[int(Dev - 1):int(T), 0:1], Q_c_sens_R5[int(Dev - 1):int(T), 0:1], Q_hload_tot[int(Dev - 1):int(T), 0:1],
                                  Q_cload_tot[int(Dev - 1):int(T), 0:1], Q_h_dehumid[int(Dev - 1):int(T), 0:1], Q_c_LATENT[int(Dev - 1):int(T), 0:1]))
        temperatures_data = pd.DataFrame(temperatures)
        load_data.to_csv("load_data.csv")
        temperatures_data.to_csv("temperatures_data.csv")


if __name__ == '__main__':

    # 三个地点
    local_place = {1: "chongqing", 2: "shanghai", 3: "changsha"}
    # heating or cooling model
    HC_model = 1
    # 墙体热质量:1-粘土砖墙；2-钢筋混凝土；3充气混凝土砌块
    wall_mass = 2
    # 绝缘选项(1-外部绝缘；2-内部绝缘)
    INS_OPT = 1
    # 窗墙比WWR(南S，北N，东E)
    WWR_S = 0.4
    WWR_N = 0.3
    WWR_E = 0.3
    # 分配5个房间的面积信息 areas of building envelope components
    f = np.zeros((5, 7))
    # building envelope components of every room(m^2)
    f_tol = np.zeros((5, 1))
    # room1(main bedroom)
    f[0][0] = 4.0 * 2.8  # east wall
    f[0][1] = 4.0 * 2.8  # west wall
    f[0][2] = 3.5 * 2.8 * (1 - WWR_S)  # south wall
    f[0][3] = 3.5 * 2.8  # north wall
    f[0][4] = 4.0 * 3.5  # ceiling
    f[0][5] = 4.0 * 3.5  # floor
    f[0][6] = 3.5 * 2.8 * WWR_S  # south window
    f_tol[0][0] = np.sum(f[0, :])
    # room2(toilet)
    f[1][0] = 2.55 * 2.8 * (1 - WWR_E)  # east wall
    f[1][1] = 2.55 * 2.8  # west wall
    f[1][2] = 3.5 * 2.8   # south wall
    f[1][3] = 3.5 * 2.8  # north wall
    f[1][4] = 3.5 * 2.55  # ceiling
    f[1][5] = 3.5 * 2.55  # floor
    f[1][6] = 2.55 * 2.8 * WWR_E  # east window
    f_tol[1][0] = np.sum(f[1, :])
    # room3(secondary bedroom)
    f[2][0] = 3.0 * 2.8 * (1 - WWR_E)  # east wall
    f[2][1] = 3.0 * 2.8  # west wall
    f[2][2] = 3.5 * 2.8   # south wall
    f[2][3] = 3.5 * 2.8  # north wall
    f[2][4] = 3.5 * 3.0  # ceiling
    f[2][5] = 3.5 * 3.0  # floor
    f[2][6] = 3 * 2.8 * WWR_E  # east window
    f_tol[2][0] = np.sum(f[2, :])
    # room4(storage room & kitichen)
    f[3][0] = 2.0 * 2.8  # east wall
    f[3][1] = 2.0 * 2.8  # west wall
    f[3][2] = 3.85 * 2.8  # south wall
    f[3][3] = 3.85 * 2.8 * (1 - WWR_N)  # north wall
    f[3][4] = 3.85 * 2.0  # ceiling
    f[3][5] = 3.85 * 2.0  # floor
    f[3][6] = 3.85 * 2.8 * WWR_N  # north window
    f_tol[3][0] = np.sum(f[3, :])
    # room5()
    f[4][0] = 7.55 * 2.8  # east wall
    f[4][1] = 7.55 * 2.8  # west wall
    f[4][2] = 3.85 * 2.8 * (1 - WWR_S)  # south wall
    f[4][3] = 3.85 * 2.8  # north wall
    f[4][4] = 3.85 * 7.55  # ceiling
    f[4][5] = 3.85 * 7.55  # floor
    f[4][6] = 3.85 * 2.8 * WWR_S  # south window
    f_tol[4][0] = np.sum(f[4, :])
    # print(f)
    # print(f_tol)
    # volume of the rooms
    V = f[:, -2] * 2.8  # shape(5, 1)
    # set Air Exchange Rate per hour of each room
    n_air = 1
    # set window heat transmittance coefficient(bouble layer glazing)双层玻璃
    k_window = 2.8
    # eternal insulation of building envelope
    D_insulation = np.zeros((5, 7))
    # REINFORCED CONCRETE(reinforce concrete)强制混凝土
    D_insulation[0][0] = 38.7
    # 迭代时间(1h=3600s)
    dt = 3600
    # set room target temperatures for heating and cooling(设定房间加热和冷却的目标温度)
    Ta_targ_min = 18  # 加热的目标温度
    Ta_targ_max = 26  # 制冷的目标温度
    # set thermoproperties of each type of thermal mass composition for external
    # walls, including thickness, density, conductivity and specific capacity
    if wall_mass == 1:
        if INS_OPT == 1:
            # external wall - thickness(mm) external to internal
            D_EXTwall = np.array([D_insulation[0][0], 15, 240, 15])
            # external wall - density(kg/m^3)
            Rou_EXTwall = np.array([19, 1800, 1800, 1100])
            # external wall - conductivity(电导率w/(m*k))
            Lamda_EXTwall = np.array([0.046, 0.93, 0.81, 0.41])
            # external wall - specific capacity(比容量j/(kg*k))
            Cp_EXTwall = np.array([2500, 840, 880, 840])
        elif INS_OPT == 2:
            # external wall - thickness(mm) internal to external
            D_EXTwall = np.array([15, 240, 15, D_insulation[0][0]])
            # external wall - density(kg/m^3)
            Rou_EXTwall = np.array([1800, 1800, 1100, 19])
            # external wall - conductivity(电导率w/(m*k))
            Lamda_EXTwall = np.array([0.93, 0.81, 0.41, 0.046])
            # external wall - specific capacity(比容量j/(kg*k))
            Cp_EXTwall = np.array([840, 880, 840, 2500])
    elif wall_mass == 2:
        if INS_OPT == 1:
            # external wall - thickness(mm) external to internal
            D_EXTwall = np.array([D_insulation[0][0], 15, 240, 15])
            # external wall - density(kg/m^3)
            Rou_EXTwall = np.array([19, 1800, 2500, 1100])
            # external wall - conductivity(电导率w/(m*k))
            Lamda_EXTwall = np.array([0.046, 0.93, 1.70, 0.41])
            # external wall - specific capacity(比容量j/(kg*k))
            Cp_EXTwall = np.array([2500, 840, 920, 840])
        elif INS_OPT == 2:
            # external wall - thickness(mm) internal to external
            D_EXTwall = np.array([15, 240, 15, D_insulation(1, 1)])
            # external wall - density(kg/m^3)
            Rou_EXTwall = np.array([1800, 2500, 1100, 19])
            # external wall - conductivity(电导率w/(m*k))
            Lamda_EXTwall = np.array([0.93, 1.70, 0.41, 0.046])
            # external wall - specific capacity(比容量j/(kg*k))
            Cp_EXTwall = np.array([840, 920, 840, 2500])
    elif wall_mass == 3:
        if INS_OPT == 1:
            # external wall - thickness(mm) external to internal
            D_EXTwall = np.array([D_insulation[0][0], 15, 240, 15])
            # external wall - density(kg/m^3)
            Rou_EXTwall = np.array([19, 1800, 680, 1100])
            # external wall - conductivity(电导率w/(m*k))
            Lamda_EXTwall = np.array([0.046, 0.93, 0.2, 0.41])
            # external wall - specific capacity(比容量j/(kg*k))
            Cp_EXTwall = np.array([2500, 840, 1050, 840])
        elif INS_OPT == 2:
            # external wall - thickness(mm) internal to external
            D_EXTwall = np.array([15, 240, 15, D_insulation[0][0]])
            # external wall - density(kg/m^3)
            Rou_EXTwall = np.array([1800, 680, 1100, 19])
            # external wall - conductivity(电导率w/(m*k))
            Lamda_EXTwall = np.array([0.93, 0.2, 0.41, 0.046])
            # external wall - specific capacity(比容量j/(kg*k))
            Cp_EXTwall = np.array([840, 1050, 840, 2500])
    # set thermoproperties for internal walls, floor, ceiling and windows
    # 设置内墙、地板、天花板和窗户的热属性
    D_INTwall = np.array([15, 120, 15])  # internal wall - thickness (mm)
    D_ceil = np.array([20, 150, 20])  # ceiling - thickness (mm)
    D_floor = np.array([20, 150, 20])  # floor -  thickness (mm)
    D_window = 3  # single glazing thickness (mm)
    N_window = 2  # 玻璃窗户的层数
    Rou_INTwall = np.array([1100, 1800, 1100])  # internal wall of materials - density(kg/m^3)
    Rou_ceil = np.array([1800, 2400, 1100])  # ceiling of materials - density(kg/m^3)
    Rou_floor = np.array([1100, 2400, 1800])  # floor of materials - density(kg/m^3)
    Rou_window = 2500  # density of window glass (kg/m^3)
    Rou_air = 1.200  # density if indoor air (kg/m^3)
    Lamda_INTwall = np.array([0.41, 0.81, 0.41])  # internal wall of layers - thermal conductivity(电导率w/(m*k))
    Lamda_ceil = np.array([0.93, 1.54, 0.41])  # ceiling of layers - thermal conductivity(电导率w/(m*k))
    Lamda_floor = np.array([0.41, 1.54, 0.93])  # floor of layers - thermal conductivity(电导率w/(m*k))
    Lamda_window = 0.76  # the thermal conductivity(电导率w/(m*k)) of window glass
    R = 0.158  # thermal resistance of window surfaces (m^2*k/w) 窗户表面的热阻
    Cp_INTwall = np.array([840, 880, 840])  # specific heat of wall - - specific capacity(比容量j/(kg*k))
    Cp_ceil = np.array([840, 840, 840])  # specific heat of ceiling - - specific capacity(比容量j/(kg*k))
    Cp_floor = np.array([840, 840, 840])  # specific heat of floor - - specific capacity(比容量j/(kg*k))
    Cp_window = 840  # specific heat of window glazing - - specific capacity(比容量j/(kg*k))
    Cp_air = 1005  # specific heat of air - - specific capacity(比容量j/(kg*k))
    # set emittance and absorptance of wall surfaces for calculating radiative heat
    # transfer processes.设置壁表面的发射率和吸收率，用于计算辐射传热过程
    # the emittance of materials
    e_brick = 0.94  # the emittance of plastering surface
    e_glass = 0.84  # the emittance of glazing
    eb = 0.7  # wall absorptance 一般设定为0.55
    # long wave radiation heat transfer coefficients
    # 长波辐射传热系数
    hr = h_lwrr(f, f_tol, e_brick, e_glass)
    #  external comprehensive heat transfer coefficient of external walls in
    # different cities according to the TMY weather data(室外综合传热系数)
    location = local_place[3]
    if location == "chongqing":
        h_out = 10.8
    elif location == "shanghai":
        h_out = 17.4
    elif location == "changsha":
        h_out = 17.4
    # set internal convective heat transfer coefficients of walls, floors, ceilings,
    # windows(设置墙壁、地板、天花板、窗户的内部对流传热系数)
    h_in_wall = 3.5  # indoor vertical wall heat transfer coefficients(室内垂直墙体传热系数) (w/(m^2 * k))
    h_in_ceil = 1.0  # indoor ceiling heat transfer coefficients(室内天花板传热系数) (w/(m^2 * k))
    h_in_floor = 4.0  # indoor floor heat transfer coefficients(室内地板传热系数) (w/(m^2 * k))
    h_window_12 = 1 / ((1 / k_window) - R)  # convective heat transfer coefficients of multi-layer according to K-SC model
    # 根据k-sc模型计算多层玻璃的对流换热系数
    # diffusivity settings for building envelope components(建筑围护结构组件的扩散系数设置)
    a_EXTwall = Lamda_EXTwall / Rou_EXTwall / Cp_EXTwall  # THERMAL DIFFUSIVITY,M^2/S
    a_INTwall = Lamda_INTwall / Rou_INTwall / Cp_INTwall  # THERMAL DIFFUSIVITY,M^2/S
    a_ceil = Lamda_ceil / Rou_ceil / Cp_ceil
    a_floor = Lamda_floor / Rou_floor / Cp_floor
    # calculate the spatial step of each component by assuming Fourier number equals 1.
    # 假设傅里叶数等于1，计算每个分量的空间步长
    dx_EXTwall = (a_EXTwall * dt) ** 0.5
    dx_INTwall = (a_INTwall * dt) ** 0.5
    dx_ceil = (a_ceil * dt) ** 0.5
    dx_floor = (a_floor * dt) ** 0.5
    # 根据估计的空间步长确定每个组件的元素数量。数字被强制为整数
    N_EXTwall = np.ceil(D_EXTwall / 1000 / dx_EXTwall)  # DIVIDED LAYERS OF EXTERNAL WALLS
    N_INTwall = np.ceil(D_INTwall / 1000 / dx_INTwall)  # DIVIDED LAYERS OF INTERNAL WALLS
    N_ceil = np.ceil(D_ceil / 1000 / dx_ceil)
    N_floor = np.ceil(D_floor / 1000 / dx_floor)
    # 当分割元素的数量被强制为整数时，重新计算每个组件的空间步长
    dx_EXTwall = D_EXTwall / 1000 / N_EXTwall  # EXACT DIVIDED THICKNESSES OF COMPONENT ELEMENTS
    dx_INTwall = D_INTwall / 1000 / N_INTwall
    dx_ceil = D_ceil / 1000 / N_ceil
    dx_floor = D_floor / 1000 / N_floor
    # 为建筑围护结构组件(墙壁、地板、天花板、窗户)设置热容量矩阵
    C_EXTwall = setC_wall(N_EXTwall, Cp_EXTwall, Rou_EXTwall, dx_EXTwall)  # construct of external wall
    C_INTwall = setC_INTwall(N_INTwall, Cp_INTwall, Rou_INTwall, dx_INTwall)  # construct of internal wall
    C_ceil = setC_INTwall(N_ceil, Cp_ceil, Rou_ceil, dx_ceil)  # construct of ceiling
    C_floor = setC_INTwall(N_floor, Cp_floor, Rou_floor, dx_floor)  # construct of floor
    C_window = setC_window(N_window, Cp_window, Rou_window, D_window / 1000)  # construct of window

    model = five_room_model(location, C_EXTwall, C_INTwall, C_ceil, C_floor, C_window, dx_EXTwall, dx_INTwall,
                 dx_ceil, dx_floor, N_EXTwall, N_INTwall, N_ceil, N_floor, V, Rou_air, float(Cp_air), Lamda_EXTwall,
                 h_out, h_in_wall, hr, Lamda_INTwall, N_window, Lamda_ceil, h_in_ceil, h_in_floor, h_window_12, int(dt), f,
                 f_tol, eb, Ta_targ_min, Ta_targ_max, Lamda_floor, n_air, HC_mode=1)

    model.input_weather_data()

