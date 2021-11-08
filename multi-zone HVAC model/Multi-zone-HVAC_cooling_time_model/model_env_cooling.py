import numpy as np
import pandas as pd
import torch
from function_building import *
import math
from scipy.integrate import odeint
from DDPG_AGENT import DDPG_agent1
from weather_data import *
from ttt import comfort_NET1
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from joblib import load, dump
import joblib
from sklearn.linear_model import LinearRegression


class five_room_model(object):

    def __init__(self, location, C_EXTwall, C_INTwall, C_ceil, C_floor, C_window, dx_EXTwall, dx_INTwall,
                 dx_ceil, dx_floor, N_EXTwall, N_INTwall, N_ceil, N_floor, V, Rou_air, Cp_air, Lamda_EXTwall, D_EXTwall,
                 h_out, h_in_wall, hr, Lamda_INTwall, N_window, Lamda_ceil, h_in_ceil, h_in_floor, h_window_12, dt, f,
                 f_tol, eb, Ta_targ_min, Ta_targ_max, Lamda_floor ,n_air, HC_mode):
        self.location = location
        self.C_EXTwall = C_EXTwall
        self.D_EXTwall = D_EXTwall
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
        self.terp_vio = 0
        self.HVAC1 = np.zeros((T, 1))
        self.HVAC3 = np.zeros((T, 1))
        self.HVAC5 = np.zeros((T, 1))
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
        self.NN_main = len(self.C_main)
        self.NN_main2 = len(self.C_main2)
        self.NN_main3 = len(self.C_main3)
        self.NN_main4 = len(self.C_main4)
        self.NN_main5 = len(self.C_main5)
        self.t1 = 10 * np.ones((self.NN_main, int(T)))
        self.tR1_wall1_in = ta_out.copy()
        self.tR1_wall2_in = ta_out.copy()
        self.tR1_wall3_in = ta_out.copy()
        self.tR1_wall4_in = ta_out.copy()
        self.tR1_ceil_in = ta_out.copy()
        self.tR1_floor_in = ta_out.copy()
        self.tR1_window_in = ta_out.copy()
        self.tR1_air = ta_out.copy()
        self.tR1_air_time = ta_out.copy()
        self.t2 = 10 * np.ones((self.NN_main2, int(T)))
        self.tR2_wall1_in = ta_out.copy()
        self.tR2_wall2_in = ta_out.copy()
        self.tR2_wall3_in = ta_out.copy()
        self.tR2_wall4_in = ta_out.copy()
        self.tR2_ceil_in = ta_out.copy()
        self.tR2_floor_in = ta_out.copy()
        self.tR2_window_in = ta_out.copy()
        self.tR2_air = ta_out.copy()
        self.t3 = 10 * np.ones((self.NN_main3, int(T)))
        self.tR3_wall1_in = ta_out.copy()
        self.tR3_wall2_in = ta_out.copy()
        self.tR3_wall3_in = ta_out.copy()
        self.tR3_wall4_in = ta_out.copy()
        self.tR3_ceil_in = ta_out.copy()
        self.tR3_floor_in = ta_out.copy()
        self.tR3_window_in = ta_out.copy()
        self.tR3_air = ta_out.copy()
        self.tR3_air_time = ta_out.copy()
        self.t4 = 10 * np.ones((self.NN_main4, int(T)))
        self.tR4_wall1_in = ta_out.copy()
        self.tR4_wall2_in = ta_out.copy()
        self.tR4_wall3_in = ta_out.copy()
        self.tR4_wall4_in = ta_out.copy()
        self.tR4_ceil_in = ta_out.copy()
        self.tR4_floor_in = ta_out.copy()
        self.tR4_window_in = ta_out.copy()
        self.tR4_air = ta_out.copy()
        self.t5 = 10 * np.ones((self.NN_main5, int(T)))
        self.tR5_wall1_in = ta_out.copy()
        self.tR5_wall2_in = ta_out.copy()
        self.tR5_wall3_in = ta_out.copy()
        self.tR5_wall4_in = ta_out.copy()
        self.tR5_ceil_in = ta_out.copy()
        self.tR5_floor_in = ta_out.copy()
        self.tR5_window_in = ta_out.copy()
        self.tR5_air = ta_out.copy()
        self.tR5_air_time = ta_out.copy()
        self.tR1_window_out = ta_out.copy()
        self.tR2_window_out = ta_out.copy()
        self.tR3_window_out = ta_out.copy()
        self.tR4_window_out = ta_out.copy()
        self.tR5_window_out = ta_out.copy()
        self.Q_hload_R1 = np.zeros((int(T), 1))  # heating load of room1
        self.Q_hload_R3 = np.zeros((int(T), 1))  # heating load of room3
        self.Q_hload_R5 = np.zeros((int(T), 1))  # heating load of room5
        self.Q_h_dehumid_R1 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 1(1室潜热除湿)
        self.Q_h_dehumid_R3 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 3
        self.Q_h_dehumid_R5 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 5
        self.Q_c_sens_R1 = np.zeros((int(T), 1))
        self.Q_c_latent_R1 = np.zeros((int(T), 1))  # sensible and latent cooling loads of room 1,respectively (1室的显冷负荷和潜冷负荷)
        self.Q_c_sens_R3 = np.zeros((int(T), 1))
        self.Q_c_latent_R3 = np.zeros((int(T), 1))  # sensible and latent cooling loads of room 3,respectively (3室的显冷负荷和潜冷负荷)
        self.Q_c_sens_R5 = np.zeros((int(T), 1))
        self.Q_c_latent_R5 = np.zeros((int(T), 1))  # sensible and latent cooling loads of room 5,respectively (5室的显冷负荷和潜冷负荷)
        self.w_e_in_R1 = np.zeros((int(T), 1))
        self.w_e_in_R1_time = np.zeros((int(T), 1))
        self.w_e_in_R1[0][0] = 6
        self.w_e_in_R1_time[0][0] = 6
        self.w_e_in_R3 = np.zeros((int(T), 1))
        self.w_e_in_R3_time = np.zeros((int(T), 1))
        self.w_e_in_R3[0][0] = 6
        self.w_e_in_R3_time[0][0] = 6
        self.w_e_in_R5 = np.zeros((int(T), 1))
        self.w_e_in_R5_time = np.zeros((int(T), 1))
        self.w_e_in_R5[0][0] = 6
        self.w_e_in_R5_time[0][0] = 6
        self.q_heatemss_p = 70 + 60  # 居住者对房间的感热和潜热散发量，70瓦特/人
        self.m_w_gp = 50 / 3600 * self.dt  # 人的吸湿量(增湿量)，50克/(h人)
        self.n_p = 3  # 公寓里住了三个人
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
        self.OCCUPIED_BED = np.zeros((int(T), 1))  # BEDROOM OCCUPATION
        self.LIGHT_ON_BED = np.zeros((int(T), 1))  # LIGHTS & EQUIPMENT ON
        self.OCCUPIED_SITT = np.zeros((int(T), 1))  # SITTING ROOM OCCUPATION
        self.LIGHT_ON_SITT = np.zeros((int(T), 1))
        self.HEATING_BEDROOM = np.zeros((int(T), 1))
        self.HEATING_SITTROOM = np.zeros((int(T), 1))
        self.COOLING_BEDROOM = np.zeros((int(T), 1))
        self.COOLING_SITTROOM = np.zeros((int(T), 1))
        self.Q_internal_rad_R1 = np.zeros((int(T), 1))
        self.Q_internal_cov_R1 = np.zeros((int(T), 1))
        self.Q_internal_rad_R3 = np.zeros((int(T), 1))
        self.Q_internal_cov_R3 = np.zeros((int(T), 1))
        self.Q_internal_rad_R5 = np.zeros((int(T), 1))
        self.Q_internal_cov_R5 = np.zeros((int(T), 1))
        self.Q_INTrad_wall1_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_wall2_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_wall3_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_wall4_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_ceil_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_floor_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_win_R1 = np.zeros((1, int(T)))
        self.Q_conv1 = np.zeros((1, int(T)))
        self.Q_hvac1 = np.zeros((1, int(T)))
        self.Q_conv2 = np.zeros((1, int(T)))
        self.Q_hvac2 = np.zeros((1, int(T)))
        self.Q_INTrad_wall1_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_wall2_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_wall3_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_wall4_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_ceil_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_floor_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_win_R3 = np.zeros((1, int(T)))
        self.Q_conv3 = np.zeros((1, int(T)))
        self.Q_hvac3 = np.zeros((1, int(T)))
        self.Q_conv4 = np.zeros((1, int(T)))
        self.Q_hvac4 = np.zeros((1, int(T)))
        self.Q_INTrad_wall1_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_wall2_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_wall3_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_wall4_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_ceil_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_floor_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_win_R5 = np.zeros((1, int(T)))
        self.Q_conv5 = np.zeros((1, int(T)))
        self.Q_hvac5 = np.zeros((1, int(T)))
        # Q_internal_rad_R1 = 0.03 * self.f[0][5] * 70.0 * 0.50 + 4.3 * self.f[0][5] * 0.20 + 6 * self.f[0][5] * 0.52
        # Q_internal_cov_R1 = 0.03 * self.f[0][5] * 70.0 * 0.50 + 4.3 * self.f[0][5] * 0.80 + 6 * self.f[0][5] * 0.48
        self.Q_internal_rad_R2 = 0  # NO INTERNAL HEAT GAIN FOR TOILET
        self.Q_internal_cov_R2 = 0  # NO INTERNAL HEAT GAIN FOR TOILET
        # Q_internal_rad_R3 = 0.03 * self.f[2][4] * 70.0 * 0.50 + 4.3 * self.f[2][4] * 0.20 + 6 * self.f[2][4] * 0.52
        # Q_internal_cov_R3 = 0.03 * self.f[2][4] * 70.0 * 0.50 + 4.3 * self.f[2][4] * 0.80 + 6 * self.f[2][4] * 0.48
        # Q_internal_rad_R5 = 0.03 * self.f[4][5] * 70.0 * 0.50 + 4.3 * self.f[4][5] * 0.20 + 6 * self.f[4][5] * 0.52
        # Q_internal_cov_R5 = 0.03 * self.f[4][5] * 70.0 * 0.50 + 4.3 * self.f[4][5] * 0.80 + 6 * self.f[4][5] * 0.48
        self.Q_solar_in_R1 = np.zeros((1, int(T)))
        self.Q_solar_in_R2 = np.zeros((1, int(T)))
        self.Q_solar_in_R3 = np.zeros((1, int(T)))
        self.Q_solar_in_R4 = np.zeros((1, int(T)))
        self.Q_solar_in_R5 = np.zeros((1, int(T)))
        self.Q_solar_in_wall = np.zeros((5, int(T)))
        self.Q_solar_in_ceil = np.zeros((5, int(T)))
        self.Q_solar_in_floor = np.zeros((5, int(T)))


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
        # Aconv_wall1_R3_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[2][0])
        # Aconv_wall2_R3_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[2][1])
        # Aconv_wall3_R3_air = A_conv_wall_air((np.sum(self.N_INTwall[:], dtype=int) + 1), self.h_in_wall, self.f[2][2])
        # Aconv_wall4_R3_air = A_conv_wall_air((np.sum(self.N_EXTwall[:], dtype=int) + 1), self.h_in_wall, self.f[2][3])
        # Aconv_ceil_R3_air = A_conv_wall_air((np.sum(self.N_ceil[:], dtype=int) + 1), self.h_in_ceil, self.f[3][2])
        # Aconv_floor_R3_air = A_conv_wall_air((np.sum(self.N_floor[:], dtype=int) + 1), self.h_in_floor, self.f[2][5])
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

    def reward(self, Q1, Q2, Q3, a, s, i, M_t):
        reward1 = 0
        reward2 = 0
        reward3 = 0
        time_occupied = nh[i][0] % 24
        if 0 <= time_occupied <= 7:
            M1_ = 0.15
            M3_ = 0.2
            M5_ = 0.35
            # reward1 += np.abs(M_t[0][0] - M1_) + 0.1 * Q1 / 1000
            # reward2 += np.abs(M_t[1][0] - M3_) + 0.1 * Q2 / 1000
            # reward3 += np.abs(M_t[2][0] - M5_) + 0.1 * Q3 / 1000
            reward1 += 10 * np.abs(M_t[0][0] - M1_) + Q1 / 1000
            reward2 += 10 * np.abs(M_t[1][0] - M3_) + Q2 / 1000
            reward3 += 10 * np.abs(M_t[2][0] - M5_) + Q3 / 1000
        if 8 <= time_occupied <= 20:
            M1_ = 0.35
            M3_ = 0.35
            M5_ = - 0.1
            # reward1 += np.abs(M_t[0][0] - M1_) + 0.1 * Q1 / 1000
            # reward2 += np.abs(M_t[1][0] - M3_) + 0.1 * Q2 / 1000
            # reward3 += np.abs(M_t[2][0] - M5_) + 0.1 * Q3 / 1000
            reward1 += 10 * np.abs(M_t[0][0] - M1_) + Q1 / 1000
            reward2 += 10 * np.abs(M_t[1][0] - M3_) + Q2 / 1000
            reward3 += 10 * np.abs(M_t[2][0] - M5_) + Q3 / 1000
        elif 21 <= time_occupied <= 24:
            M1_ = 0.1
            M3_ = 0.15
            M5_ = 0.35
            # reward1 += np.abs(M_t[0][0] - M1_) + 0.1 * Q1 / 1000
            # reward2 += np.abs(M_t[1][0] - M3_) + 0.1 * Q2 / 1000
            # reward3 += np.abs(M_t[2][0] - M5_) + 0.1 * Q3 / 1000
            reward1 += 10 * np.abs(M_t[0][0] - M1_) + Q1 / 1000
            reward2 += 10 * np.abs(M_t[1][0] - M3_) + Q2 / 1000
            reward3 += 10 * np.abs(M_t[2][0] - M5_) + Q3 / 1000
            # if np.abs(M_t[0][0] - M1_) >= 0.1:
            #     reward1 += 1
            # if np.abs(M_t[1][0] - M3_) >= 0.1:
            #     reward2 += 1
            # if np.abs(M_t[2][0] - M5_) >= 0.1:
            #     reward3 += 1
            # if 0.05 <= np.abs(M_t[0][0] - M1_) < 0.1:
            #     reward1 += 0.5
            # if 0.05 <= np.abs(M_t[1][0] - M3_) < 0.1:
            #     reward2 += 0.5
            # if 0.05 <= np.abs(M_t[2][0] - M5_) < 0.1:
            #     reward3 += 0.5

        # reward1 += np.exp(- 0.2 * (a[0] - 25) ** 2) - 0.1 * (max(24 -a[0], 0) + max(a[0] - 26, 0))
        # reward1 += np.exp(- 0.2 * (a[1] - 25) ** 2) - 0.1 * (max(24 -a[1], 0) + max(a[1] - 26, 0))
        # reward1 += np.exp(- 0.2 * (a[2] - 25) ** 2) - 0.1 * (max(24 -a[2], 0) + max(a[2] - 26, 0))

        self.terp_vio += np.abs(M_t[0][0] - M1_) + np.abs(M_t[1][0] - M3_) + np.abs(M_t[2][0] - M5_)
        reward =  - (reward1 + reward2 + reward3) / 153
        return reward

    # def comfortable_terp(self, s, a, i):
    #     model_ = comfort_NET1(2, 1)
    #     model_.load_state_dict(torch.load("tem.pth"))
    #     model_.eval()
    #     s_t = np.array([a[0], a[1], a[2]])
    #     s_h = np.array([60, 60, 60])
    #     x = torch.FloatTensor(np.vstack((s_t, s_h)).T)
    #     M_t = model_(x).detach().numpy()
    #     return M_t

    def compute_comfortable_terp(self, s, i):
        SVR_comfort = load('linear_svr.joblib')
        ss_x = StandardScaler()
        ss_y = StandardScaler()
        s_t = s
        s_h = np.array([60, 60, 60])
        x_SVR = np.vstack((s_t, s_h)).T
        x_SVR = ss_x.fit_transform(x_SVR)
        y_svr = SVR_comfort.predict(x_SVR).reshape(-1, 1)
        model_ = comfort_NET1(3, 1)
        model_.load_state_dict(torch.load("tem2.pth"))
        model_.eval()
        x = torch.FloatTensor(np.vstack((s_t, s_h, y_svr.T)).T)
        M_t = model_(x).detach().numpy()
        M_t = np.squeeze(M_t, 1)
        return M_t

    def get_action(self, a, s, i):
        action = []
        time_action = nh[i][0] % 24
        # RBC
        # if 0 <= time_action <= 7:
        #     action.append(25)
        #     action.append(25)
        #     action.append(27)
        # if 8 <= time_action <= 20:
        #     action.append(27)
        #     action.append(27)
        #     action.append(24)
        # if 21 <= time_action <= 24:
        #     action.append(24)
        #     action.append(25)
        #     action.append(27)
        for j in range(3):
            action.append(25 + 2 * a[j])
        return action

    def init_compute(self):
        # 计算初始化
        # action = 25
        for i in range(1, 144):
            # w_e_in_set_2 = Humidity_Ratio(action, 65, p_tot)
            time = nh[i][0] % 24
            if (time >= 0 and time <= 7) or (
                    time > 20 and time < 24):  # STAYING IN BEDROOMS AT TIMEFRAME 0:00-6:00 am; 20:00-24:00 pm
                self.OCCUPIED_BED[i][0] = 1
            elif (time >= 6 and time <= 8) or (time > 20 and time <= 22):  # LIGHTS & EQUIPMENT ON IN THE BEDCHAMBER
                self.LIGHT_ON_BED[i][0] = 1
            # if (((nh[i][0] / 24) % 7) >= 1) and (((nh[i][0] / 24) % 7) < 6):  # WEEKDAYS(MONDAY TO FRIDAY)
            #     if (time > 6 and time <= 8) or (
            #             time >= 18 and time <= 20):  # TIMEFRAME 6:00-20:00 pm IN SITTING ROOM AND LIGHTS ON WHEN OCCUPYING
            #         self.OCCUPIED_SITT[i][0] = 1  # 时间范围:下午6:00-20:00在客厅，占用时灯亮着
            #         self.LIGHT_ON_SITT[i][0] = 1
            # else:
            #     if (time > 6 and time <= 20):
            #         self.OCCUPIED_SITT[i][0] = 1
            #         self.LIGHT_ON_SITT[i][0] = 1
            if (time > 7 and time <= 20):
                self.OCCUPIED_SITT[i][0] = 1
                self.LIGHT_ON_SITT[i][0] = 1

            self.Q_internal_rad_R1[i][0] = self.OCCUPIED_BED[i][0] * 2 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[0][5] * 0.20 + 6 * self.f[0][5] * 0.52)  # 2 OCCUPANTS
            self.Q_internal_cov_R1[i][0] = self.OCCUPIED_BED[i][0] * 2 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[0][5] * 0.80 + 6 * self.f[0][5] * 0.48)
            self.Q_internal_rad_R3[i][0] = self.OCCUPIED_BED[i][0] * 1 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[2][5] * 0.20 + 6 * self.f[2][5] * 0.52)  # 1 OCCUPANT
            self.Q_internal_cov_R3[i][0] = self.OCCUPIED_BED[i][0] * 1 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][0] * \
                                      (4.3 * self.f[2][5] * 0.80 + 6 * self.f[2][5] * 0.48)
            self.Q_internal_rad_R5[i][0] = self.OCCUPIED_SITT[i][0] * 3 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_SITT[i][0] * \
                                      (4.3 * self.f[4][5] * 0.20 + 6 * self.f[4][5] * 0.52)  # 3 OCCUPANTS
            self.Q_internal_cov_R5[i][0] = self.OCCUPIED_SITT[i][0] * 3 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_SITT[i][0] * \
                                      (4.3 * self.f[4][5] * 0.80 + 6 * self.f[4][5] * 0.48)
            self.Q_INTrad_wall1_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][0] / self.f_tol[
                0]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall2_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][1] / self.f_tol[
                0]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall3_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][2] / self.f_tol[
                0]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall4_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][3] / self.f_tol[
                0]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_ceil_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][4] / self.f_tol[
                0]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_floor_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][5] / self.f_tol[
                0]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_win_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][6] / self.f_tol[
                0]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_conv1[0][i] = self.Q_internal_cov_R1[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
            self.Q_hvac1[0][i] = 0  # 房间1空调供热 w

            self.Q_conv2[0][i] = 0  # IT'S ASSUMED NO INTERNAL HEAT GAIN FROM ROOM 2 - TOILET AND ROOM 4 - STORAGE AND KITCHEN
            self.Q_hvac2[0][i] = 0

            self.Q_INTrad_wall1_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][0] / self.f_tol[
                2]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall2_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][1] / self.f_tol[
                2]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall3_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][2] / self.f_tol[
                2]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall4_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][3] / self.f_tol[
                2]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_ceil_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][4] / self.f_tol[
                2]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_floor_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][5] / self.f_tol[
                2]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_win_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][6] / self.f_tol[
                2]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_conv3[0][i] = self.Q_internal_cov_R3[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
            self.Q_hvac3[0][i] = 0  # 房间3空调供热 w

            self.Q_conv4[0][i] = 0  # IT'S ASSUMED NO INTERNAL HEAT GAIN FROM ROOM 4 - TOILET AND ROOM 4 - STORAGE AND KITCHEN
            self.Q_hvac4[0][i] = 0

            self.Q_INTrad_wall1_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][0] / self.f_tol[
                4]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall2_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][1] / self.f_tol[
                4]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall3_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][2] / self.f_tol[
                4]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_wall4_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][3] / self.f_tol[
                4]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_ceil_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][4] / self.f_tol[
                4]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_floor_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][5] / self.f_tol[
                4]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_INTrad_win_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][6] / self.f_tol[
                4]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
            self.Q_conv5[0][i] = self.Q_internal_cov_R5[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
            self.Q_hvac5[0][i] = 0  # 房间5空调供热 w

            if i < 144 * (3600 / self.dt):
                n = n0
            else:
                n = n0 + int((i * (self.dt / 3600) - 144) / 24)
            ws = i * (self.dt / 3600) - 24 * int(i * (self.dt / 3600) / 24)
            angle3 = solar_angle(n, r, longitude, ws, 90, 0)
            [win_transp3, win_absorp3, win_refelect3, e_all3, tao_all3] = solar_win2(angle3)
            self.Q_solar_in_R1[0][i] = win_transp3 * q_solar_out_3[i][0] * self.f[0][6]  # SOUTH WINDOW SOLAR
            self.Q_solar_in_R5[0][i] = win_transp3 * q_solar_out_3[i][0] * self.f[4][6]  # SOUTH WINDOW SOLAR

            angle1 = solar_angle(n, r, longitude, ws, 90, -90)
            [win_transp1, win_absorp1, win_refelect1, e_all1, tao_all1] = solar_win2(angle1)
            self.Q_solar_in_R2[0][i] = win_transp1 * q_solar_out_1[i][0] * self.f[1][6]  # EAST WINDOW SOLAR
            self.Q_solar_in_R3[0][i] = win_transp1 * q_solar_out_1[i][0] * self.f[2][6]  # EAST WINDOW SOLAR

            angle4 = solar_angle(n, r, longitude, ws, 90, 180)
            [win_transp4, win_absorp4, win_refelect4, e_all4, tao_all4] = solar_win2(angle4)
            self.Q_solar_in_R4[0][i] = win_transp4 * q_solar_out_4[i][0] * self.f[3][6]  # NORTH WINDOW SOLAR

            # 通过玻璃渗透获得内部太阳热量
            self.Q_solar_in_wall[0][i] = self.Q_solar_in_R1[0][i] * 0.1
            self.Q_solar_in_ceil[0][i] = self.Q_solar_in_R1[0][i] * 0.1
            self.Q_solar_in_floor[0][i] = self.Q_solar_in_R1[0][i] * 0.5
            self.Q_solar_in_wall[1][i] = self.Q_solar_in_R2[0][i] * 0.1
            self.Q_solar_in_ceil[1][i] = self.Q_solar_in_R2[0][i] * 0.1
            self.Q_solar_in_floor[1][i] = self.Q_solar_in_R2[0][i] * 0.5
            self.Q_solar_in_wall[2][i] = self.Q_solar_in_R3[0][i] * 0.1
            self.Q_solar_in_ceil[2][i] = self.Q_solar_in_R3[0][i] * 0.1
            self.Q_solar_in_floor[2][i] = self.Q_solar_in_R3[0][i] * 0.5
            self.Q_solar_in_wall[3][i] = self.Q_solar_in_R4[0][i] * 0.1
            self.Q_solar_in_ceil[3][i] = self.Q_solar_in_R4[0][i] * 0.1
            self.Q_solar_in_floor[3][i] = self.Q_solar_in_R4[0][i] * 0.5
            self.Q_solar_in_wall[3][i] = self.Q_solar_in_R5[0][i] * 0.1
            self.Q_solar_in_ceil[4][i] = self.Q_solar_in_R5[0][i] * 0.1
            self.Q_solar_in_floor[4][i] = self.Q_solar_in_R5[0][i] * 0.5

            w_e_out[i][0] = Humidity_Ratio(ta_out[i][0], RH[i][0], p_tot)  # HUMIDITY RATIO OF ATMOSHPHERIC STATE POING

            # b向量元素-修正热负荷计算矩阵
            # 房间1 - REVISED(修改)
            B_wall1_R1 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0], self.Q_solar_in_wall[0][i],
                                        self.Q_INTrad_wall1_R1[0][i], ta_out[i][0], self.f[0][0], self.h_in_wall,
                                        self.tR1_air[i - 1][0])
            B_wall2_R1 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[0][i],
                                           self.Q_INTrad_wall2_R1[0][i],
                                           self.Q_solar_in_wall[4][i], self.Q_INTrad_wall1_R5[0][i], self.f[4][0],
                                           self.tR5_air[i - 1][0],
                                           self.tR5_wall2_in[i - 1][0], self.tR5_wall3_in[i - 1][0], self.tR5_wall4_in[i - 1][0],
                                           self.tR5_ceil_in[i - 1][0],
                                           self.tR5_floor_in[i - 1][0], self.tR5_window_in[i - 1][0], self.hr[4][1],
                                           self.hr[4][2], self.hr[4][3],
                                           self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[0][1], self.tR1_air[i - 1][0])
            B_wall3_R1 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_3[i][0], self.Q_solar_in_wall[0][i],
                                        self.Q_INTrad_wall3_R1[0][i], ta_out[i][0], self.f[0][2], self.h_in_wall,
                                        self.tR1_air[i - 1][0])
            B_wall4_R1 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[0][i],
                                           self.Q_INTrad_wall2_R1[0][i],
                                           self.Q_solar_in_wall[1][i], 0, self.f[1][2], self.tR2_air[i - 1][0],
                                           self.tR2_wall1_in[i - 1][0],
                                           self.tR2_wall2_in[i - 1][0], self.tR2_wall4_in[i - 1][0], self.tR2_ceil_in[i - 1][0],
                                           self.tR2_floor_in[i - 1][0], self.tR2_window_in[i - 1][0], self.hr[1][0],
                                           self.hr[1][1], self.hr[1][3],
                                           self.hr[1][4], self.hr[1][5], self.hr[1][6], self.f[0][3], self.tR1_air[i - 1][0])
            B_ceil_R1 = setB_ceiling_load(self.N_ceil, self.eb, self.Q_solar_in_floor[0][i], self.Q_INTrad_floor_R1[0][i],
                                          self.f[0][5],
                                          self.Q_solar_in_ceil[0][i], self.Q_INTrad_ceil_R1[0][i], self.f[0][4], self.h_in_ceil,
                                          self.h_in_floor,
                                          self.tR1_air[i - 1][0])
            B_floor_R1 = setB_ceiling_load(self.N_floor, self.eb, self.Q_solar_in_ceil[0][i], self.Q_INTrad_ceil_R1[0][i],
                                           self.f[0][4],
                                           self.Q_solar_in_floor[0][i], self.Q_INTrad_floor_R1[0][i], self.f[0][5],
                                           self.h_in_floor, self.h_in_ceil,
                                           self.tR1_air[i - 1][0])
            B_window_R1 = setB_window_load(self.N_window, self.h_out, e_all3[0][1], e_all3[0][3], q_solar_out_3[i][0],
                                           self.Q_INTrad_win_R1[0][i],
                                           ta_out[i][0], self.f[0][6], self.h_in_wall,
                                           self.tR1_air[i - 1][0])  # SOUTH WINDWOW
            # B_air_R1 = self.Cp_air * self.Rou_air * n_air / 3600 * V[0] * ta_out[i][0] + Q_conv1[0][i] + Q_hvac1[0][i]
            B_main = np.vstack(
                (B_wall1_R1, B_wall2_R1, B_wall3_R1, B_wall4_R1, B_ceil_R1, B_floor_R1, B_window_R1))  # B_air_R1
            # SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            # [m1,temp_R1]=ode15s(@fun,[(i-1)*dt,i*dt],t1(:,i-1))
            # t1(:, i)=temp_R1(length(m1),:)
            a, b = function(self.Construct_heat_flux_flow_relationship_room1(), B_main, self.C_main)
            # def deriv(y, t):
            #     dy = np.zeros((44, 1))
            #     temp = np.ones((44, 44))
            #     for i in range(44):
            #         for j in range(44):
            #             temp[i][j] = a[i][j] * y[j]
            #         dy[i][0] = np.sum(temp[i, :]) + b[i][0]
            #     return dy.squeeze(axis=1)
            def deriv(y, t):
                return (np.dot(a, y.reshape(self.NN_main, 1)) + b)[:, 0]
            time1 = np.linspace((i - 1) * self.dt, i * self.dt)
            temp_R1 = odeint(deriv, self.t1[:, i - 1], time1)
            self.t1[:, i] = temp_R1[-1, :].T
            self.tR1_wall1_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int)][i]
            self.tR1_wall2_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int) + 1][i]
            self.tR1_wall3_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int) + 2][i]
            self.tR1_wall4_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int) + 3][i]
            self.tR1_ceil_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2
                                   + np.sum(self.N_ceil[:], dtype=int) + 4][i]
            self.tR1_floor_in[i][0] = self.t1[self.NN_main - np.sum(self.N_window, dtype=int) - 1][i]
            self.tR1_window_in[i][0] = self.t1[self.NN_main - 1][i]
            self.tR1_window_out[i][0] = self.t1[self.NN_main - 2][i]
            self.tR1_air[i][0] = self.t1[self.NN_main - 1][i]
            self.w_e_in_R1[i][0] = self.w_e_in_R1[i - 1][0] + self.n_air * self.dt / 3600 * (w_e_out[i][0] - self.w_e_in_R1[i - 1][0]) \
                              + self.m_w_gp * 2 * self.OCCUPIED_BED[i][0] / self.Rou_air / self.V[0]
            B_wall1_R2 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0], self.Q_solar_in_wall[1][i], 0,
                                   ta_out[i][0], self.f[1][0])
            B_wall2_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[1][i], 0,
                                      self.Q_solar_in_wall[4][i],
                                      self.Q_INTrad_wall1_R5[0][i], self.f[4][0], self.tR5_air[i - 1][0], self.tR5_wall2_in[i - 1][0],
                                      self.tR5_wall3_in[i - 1][0], self.tR5_wall4_in[i - 1][0], self.tR5_ceil_in[i - 1][0],
                                      self.tR5_floor_in[i - 1][0],
                                      self.tR5_window_in[i - 1][0], self.hr[4][1], self.hr[4][2], self.hr[4][3],
                                      self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[1][1])
            B_wall3_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[1][i], 0,
                                      self.Q_solar_in_wall[0][i],
                                      self.Q_INTrad_wall1_R1[0][i], self.f[0][3], self.tR1_air[i][0], self.tR1_wall2_in[i][0],
                                      self.tR1_wall3_in[i][0], self.tR1_wall4_in[i][0], self.tR1_ceil_in[i][0], self.tR1_floor_in[i][0],
                                      self.tR1_window_in[i][0], self.hr[0][0], self.hr[0][1], self.hr[0][2],
                                      self.hr[0][4], self.hr[0][5], self.hr[0][6], self.f[1][2])
            B_wall4_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[1][i], 0,
                                      self.Q_solar_in_wall[2][i],
                                      self.Q_INTrad_wall1_R3[0][i], self.f[2][2], self.tR3_air[i - 1][0], self.tR3_wall2_in[i - 1][0],
                                      self.tR3_wall3_in[i - 1][0], self.tR3_wall4_in[i - 1][0], self.tR3_ceil_in[i - 1][0],
                                      self.tR3_floor_in[i - 1][0],
                                      self.tR3_window_in[i - 1][0], self.hr[2][0], self.hr[2][1], self.hr[2][3],
                                      self.hr[2][4], self.hr[2][5], self.hr[2][6], self.f[1][3])
            B_ceil_R2 = setB_ceiling(self.N_ceil, self.eb, self.Q_solar_in_floor[1][i], 0, self.f[1][5],
                                     self.Q_solar_in_ceil[1][i], 0, self.f[1][4])
            B_floor_R2 = setB_ceiling(self.N_floor, self.eb, self.Q_solar_in_ceil[1][i], 0, self.f[1][4],
                                      self.Q_solar_in_floor[1][i], 0, f[1][5])
            B_window_R2 = setB_window(self.N_window, self.h_out, e_all1[0][1], e_all1[0][3], q_solar_out_1[i][0], 0,
                                      ta_out[i][0], self.f[1][6])  # EAST WINDOW
            B_air_R2 = self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[1] * ta_out[i][0] + self.Q_conv2[0][i] + \
                       self.Q_hvac2[0][i]
            B_main2 = np.vstack(
                (B_wall1_R2, B_wall2_R2, B_wall3_R2, B_wall4_R2, B_ceil_R2, B_floor_R2, B_window_R2, B_air_R2))
            #  % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m2,temp_R2]=ode15s(@fun2,[(i-1)*dt,i*dt],t2(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t2(:,i)=temp_R2(length(m2),:)';
            a2, b2 = function(self.Construct_heat_flux_flow_relationship_room2(), B_main2, self.C_main2)

            def deriv2(y, t):
                return (np.dot(a2, y.reshape(self.NN_main2, 1)) + b2)[:, 0]

            temp_R2 = odeint(deriv2, self.t2[:, i - 1], time1)
            self.t2[:, i] = temp_R2[-1, :].T
            self.tR2_wall1_in[i][0] = self.t2[np.sum(self.N_EXTwall[:], dtype=int)][i]
            self.tR2_wall2_in[i][0] = self.t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            self.tR2_wall3_in[i][0] = \
            self.t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
            self.tR2_wall4_in[i][0] = \
            self.t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 3 + 3][i]
            self.tR2_ceil_in[i][0] = self.t2[
                np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 3 + np.sum(self.N_ceil[:],
                                                                                                         dtype=int) + 4][
                i]
            self.tR2_floor_in[i][0] = self.t2[self.NN_main2 - np.sum(self.N_window, dtype=int) - 2][i]
            self.tR2_window_in[i][0] = self.t2[self.NN_main2 - 2][i]
            self.tR2_window_out[i][0] = self.t2[self.NN_main2 - 3][i]
            self.tR2_air[i][0] = self.t2[self.NN_main2 - 1][i]
            # room3
            B_wall1_R3 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0], self.Q_solar_in_wall[2][i],
                                        self.Q_INTrad_wall1_R3[0][i], ta_out[i][0], self.f[2][0], self.h_in_wall,
                                        self.tR3_air[i][0])
            B_wall2_R3 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * (
                        0.667 * np.float(self.tR4_air[i - 1][0]) + 0.333 * np.float(self.tR5_air[i - 1][0])),
                                            self.eb, self.Q_solar_in_wall[2][i], self.Q_INTrad_wall1_R3[0][i],
                                            [self.Q_solar_in_wall[3][i] + (1 / 7.55) * self.Q_solar_in_wall[4][i]],
                                            (1.0 / 7.55) * self.Q_INTrad_wall1_R5[0][i],
                                            [0.667 * (self.hr[3][1] * self.tR4_wall2_in[i - 1][0] + self.hr[3][2] *
                                                      self.tR4_wall3_in[i - 1][0] +
                                                      self.hr[3][3] * self.tR4_wall4_in[i - 1][0] + self.hr[3][4] *
                                                      self.tR4_ceil_in[i - 1][0] +
                                                      self.hr[3][5] * self.tR4_floor_in[i - 1][0] + self.hr[3][6] *
                                                      self.tR4_window_in[i - 1][0]) +
                                             0.333 * (self.hr[4][1] * self.tR5_wall2_in[i - 1][0] + self.hr[4][2] *
                                                      self.tR5_wall3_in[i - 1][0] +
                                                      self.hr[4][3] * self.tR5_wall4_in[i - 1][0] + self.hr[4][4] *
                                                      self.tR5_ceil_in[i - 1][0] +
                                                      self.hr[4][5] * self.tR5_floor_in[i - 1][0] + self.hr[4][6] *
                                                      self.tR5_window_in[i - 1][0])],
                                            self.f[2][1], self.h_in_wall * self.tR3_air[i - 1][0])
            B_wall3_R3 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[2][i],
                                           self.Q_INTrad_wall3_R3[0][i],
                                           self.Q_solar_in_wall[1][i], 0, self.f[1][3], self.tR2_air[i][0], self.tR2_wall1_in[i][0],
                                           self.tR2_wall2_in[i][0], self.tR2_wall3_in[i][0], self.tR2_ceil_in[i][0],
                                           self.tR2_floor_in[i][0],
                                           self.tR2_window_in[i][0], self.hr[1][0], self.hr[1][1], self.hr[1][2],
                                           self.hr[1][4],
                                           self.hr[1][5], self.hr[1][6], self.f[2][2], self.tR3_air[i - 1][0])
            B_wall4_R3 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_4[i][0], self.Q_solar_in_wall[2][i],
                                        self.Q_INTrad_wall4_R3[0][i], ta_out[i][0], self.f[2][3], self.h_in_wall,
                                        self.tR3_air[i - 1][0])
            B_ceil_R3 = setB_ceiling_load(self.N_ceil, self.eb, self.Q_solar_in_floor[2][i], self.Q_INTrad_floor_R3[0][i],
                                          self.f[2][5],
                                          self.Q_solar_in_ceil[2][i], self.Q_INTrad_ceil_R3[0][i], self.f[2][4], self.h_in_ceil,
                                          self.h_in_floor,
                                          self.tR3_air[i - 1][0])
            B_floor_R3 = setB_ceiling_load(self.N_floor, self.eb, self.Q_solar_in_ceil[2][i], self.Q_INTrad_ceil_R3[0][i],
                                           self.f[2][4],
                                           self.Q_solar_in_floor[2][i], self.Q_INTrad_floor_R3[0][i], self.f[2][5],
                                           self.h_in_floor, self.h_in_ceil,
                                           self.tR3_air[i - 1][0])
            B_window_R3 = setB_window_load(self.N_window, self.h_out, e_all1[0][1], e_all1[0][3], q_solar_out_1[i][0],
                                           self.Q_INTrad_win_R3[0][i],
                                           ta_out[i][0], self.f[2][6], self.h_in_wall, self.tR3_air[i - 1][0])  # EAST WINDOW
            B_main3 = np.vstack((B_wall1_R3, B_wall2_R3, B_wall3_R3, B_wall4_R3, B_ceil_R3, B_floor_R3, B_window_R3))
            # SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m3,temp_R3]=ode15s(@fun3,[(i-1)*dt,i*dt],t3(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t3(:,i)=temp_R3(length(m3),:)';
            a3, b3 = function(self.Construct_heat_flux_flow_relationship_room3(), B_main3, self.C_main3)

            def deriv3(y, t):
                return (np.dot(a3, y.reshape(self.NN_main3, 1)) + b3)[:, 0]

            temp_R3 = odeint(deriv3, self.t3[:, i - 1], time1)
            self.t3[:, i] = temp_R3[-1, :].T
            self.tR3_wall1_in[i][0] = self.t3[np.sum(self.N_EXTwall[:], dtype=int)][i]
            self.tR3_wall2_in[i][0] = self.t3[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            self.tR3_wall3_in[i][0] = \
            self.t3[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
            self.tR3_wall4_in[i][0] = \
            self.t3[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
            self.tR3_ceil_in[i][0] = self.t3[
                np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(
                    self.N_ceil[:], dtype=int) + 4][i]
            self.tR3_floor_in[i][0] = self.t3[self.NN_main3 - np.sum(self.N_window, dtype=int) - 2][i]
            self.tR3_window_in[i][0] = self.t3[self.NN_main3 - 2][i]
            self.tR3_window_out[i][0] = self.t3[self.NN_main3 - 3][i]
            self.tR3_air[i][0] = self.t3[self.NN_main3 - 1][i]
            self.w_e_in_R3[i][0] = self.w_e_in_R3[i - 1][0] + self.n_air * self.dt / 3600 * (
                        w_e_out[i][0] - self.w_e_in_R3[i - 1][0]) + \
                              self.m_w_gp * 1 * self.OCCUPIED_BED[i][0] / self.Rou_air / self.V[2]
            B_wall1_R4 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[3][i], 0,
                                      self.Q_solar_in_wall[2][i],
                                      self.Q_INTrad_wall2_R3[0][i], self.f[2][1], self.tR3_air[i][0], self.tR3_wall2_in[i][0],
                                      self.tR3_wall3_in[i][0],
                                      self.tR3_wall4_in[i][0], self.tR3_ceil_in[i][0], self.tR3_floor_in[i][0], self.tR3_window_in[i][0],
                                      self.hr[2][0],
                                      self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6],
                                      self.f[3][0])
            B_wall2_R4 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_2[i][0], self.Q_solar_in_wall[3][i], 0,
                                   ta_out[i][0], self.f[3][1])
            B_wall3_R4 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[3][i], 0,
                                      self.Q_solar_in_wall[4][i],
                                      self.Q_INTrad_wall4_R5[0][i], self.f[4][3], self.tR5_air[i - 1][0], self.tR5_wall1_in[i - 1][0],
                                      self.tR5_wall2_in[i - 1][0], self.tR5_wall3_in[i - 1][0], self.tR5_ceil_in[i - 1][0],
                                      self.tR5_floor_in[i - 1][0],
                                      self.tR5_window_in[i - 1][0], self.hr[4][0], self.hr[4][1], self.hr[4][2],
                                      self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[3][2])
            B_wall4_R4 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_4[i][0], self.Q_solar_in_wall[3][i], 0,
                                   ta_out[i][0], self.f[3][3])
            B_ceil_R4 = setB_ceiling(self.N_ceil, self.eb, self.Q_solar_in_floor[3][i], 0, self.f[3][5],
                                     self.Q_solar_in_ceil[3][i], 0, self.f[3][4])
            B_floor_R4 = setB_ceiling(self.N_floor, self.eb, self.Q_solar_in_ceil[3][i], 0, self.f[3][4],
                                      self.Q_solar_in_floor[3][i], 0, self.f[3][5])
            B_window_R4 = setB_window(self.N_window, self.h_out, e_all4[0][1], e_all4[0][3], q_solar_out_4[i][0], 0,
                                      ta_out[i][0], self.f[3][6])
            B_air_R4 = self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[3] * ta_out[i][0] + self.Q_conv4[0][i] + \
                       self.Q_hvac4[0][i]
            B_main4 = np.vstack(
                (B_wall1_R4, B_wall2_R4, B_wall3_R4, B_wall4_R4, B_ceil_R4, B_floor_R4, B_window_R4, B_air_R4))
            # % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m4,temp_R4]=ode15s(@fun4,[(i-1)*dt,i*dt],t4(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t4(:,i)=temp_R4(length(m4),:)';
            a4, b4 = function(self.Construct_heat_flux_flow_relationship_room4(), B_main4, self.C_main4)
            def deriv4(y, t):
                return (np.dot(a4, y.reshape(self.NN_main4, 1)) + b4)[:, 0]
            temp_R4 = odeint(deriv4, self.t4[:, i - 1], time1)
            self.t4[:, i] = temp_R4[-1, :].T
            self.tR4_wall1_in[i][0] = self.t4[np.sum(self.N_EXTwall[:], dtype=int)][i]
            self.tR4_wall2_in[i][0] = self.t4[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            self.tR4_wall3_in[i][0] = \
            self.t4[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
            self.tR4_wall4_in[i][0] = \
            self.t4[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
            self.tR4_ceil_in[i][0] = self.t4[
                np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(
                    self.N_ceil[:], dtype=int) + 4][i]
            self.tR4_floor_in[i][0] = self.t4[self.NN_main4 - np.sum(self.N_window, dtype=int) - 2][i]
            self.tR4_window_in[i][0] = self.t4[self.NN_main4 - 2][i]
            self.tR4_window_out[i][0] = self.t4[self.NN_main4 - 3][i]
            self.tR4_air[i][0] = self.t4[self.NN_main4 - 1][i]
            # room5
            B_wall1_R5 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * (
                        0.53 * self.tR1_air[i][0] + 0.34 * self.tR2_air[i][0] + 0.13 * self.tR3_air[i][0]),
                                            self.eb,self.Q_solar_in_wall[4][i], self.Q_INTrad_wall1_R5[0][i],
                                            self.Q_solar_in_wall[0][i] + self.Q_solar_in_wall[1][i] + 0.333 * self.Q_solar_in_wall[2][
                                                i],
                                            self.Q_INTrad_wall1_R1[0][i] + 0 + 0.333 * self.Q_INTrad_wall1_R3[0][i],
                                            0.53 * (self.hr[0][0] * self.tR1_wall1_in[i][0] + self.hr[0][2] *
                                                    self.tR1_wall3_in[i][0] +
                                                    self.hr[0][3] * self.tR1_wall4_in[i][0] + self.hr[0][4] * self.tR1_ceil_in[i][
                                                        0] +
                                                    self.hr[0][5] * self.tR1_floor_in[i][0] + self.hr[0][6] *
                                                    self.tR1_window_in[i][0]) +
                                            0.34 * (self.hr[1][0] * self.tR2_wall1_in[i][0] + self.hr[1][2] *
                                                    self.tR2_wall3_in[i][0] +
                                                    self.hr[1][3] * self.tR2_wall4_in[i][0] + self.hr[1][4] * self.tR2_ceil_in[i][
                                                        0] +
                                                    self.hr[1][5] * self.tR2_floor_in[i][0] + self.hr[1][6] *
                                                    self.tR2_window_in[i][0]) +
                                            0.13 * (self.hr[2][0] * self.tR3_wall1_in[i][0] + self.hr[2][2] *
                                                    self.tR3_wall3_in[i][0] +
                                                    self.hr[2][3] * self.tR3_wall4_in[i][0] + self.hr[2][4] * self.tR3_ceil_in[i][
                                                        0] +
                                                    self.hr[2][5] * self.tR3_floor_in[i][0] + self.hr[2][6] *
                                                    self.tR3_window_in[i][0]),
                                            self.f[4][0], self.h_in_wall * self.tR5_air[i - 1][0])
            B_wall2_R5 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_2[i][0], self.Q_solar_in_wall[4][i],
                                        self.Q_INTrad_wall2_R5[0][i], ta_out[i][0], self.f[4][1], self.h_in_wall,
                                        self.tR5_air[i - 1][0])
            B_wall3_R5 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_3[i][0], self.Q_solar_in_wall[4][i],
                                        self.Q_INTrad_wall3_R5[0][i], ta_out[i][0], self.f[4][2], self.h_in_wall,
                                        self.tR5_air[i - 1][0])
            B_wall4_R5 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * self.tR4_air[i][0], eb, self.Q_solar_in_wall[4][i],
                                            self.Q_INTrad_wall4_R5[0][i], self.Q_solar_in_wall[3][i], 0,
                                            [self.hr[3][0] * self.tR4_wall1_in[i][0] + self.hr[3][1] * self.tR4_wall2_in[i][0] +
                                             self.hr[3][3] * self.tR4_wall4_in[i][0] + self.hr[3][4] * self.tR4_ceil_in[i][0] +
                                             self.hr[3][5] * self.tR4_floor_in[i][0] + self.hr[3][6] * self.tR4_window_in[i][0]],
                                            self.f[4][3], self.h_in_wall * self.tR5_air[i - 1][0])
            B_ceil_R5 = setB_ceiling_load(self.N_ceil, self.eb, self.Q_solar_in_floor[4][i], self.Q_INTrad_floor_R5[0][i],
                                          self.f[4][5],
                                          self.Q_solar_in_ceil[4][i], self.Q_INTrad_ceil_R5[0][i], self.f[4][4], self.h_in_ceil,
                                          self.h_in_floor,
                                          self.tR5_air[i - 1][0])
            B_floor_R5 = setB_ceiling_load(self.N_floor, self.eb, self.Q_solar_in_ceil[4][i], self.Q_INTrad_ceil_R5[0][i],
                                           self.f[4][4],
                                           self.Q_solar_in_floor[4][i], self.Q_INTrad_floor_R5[0][i], self.f[4][5],
                                           self.h_in_floor, self.h_in_ceil,
                                           self.tR5_air[i - 1][0])
            B_window_R5 = setB_window_load(self.N_window, self.h_out, e_all3[0][1], e_all3[0][3], q_solar_out_3[i][0],
                                           self.Q_INTrad_win_R5[0][i],
                                           ta_out[i][0], self.f[4][6], self.h_in_wall, self.tR5_air[i - 1][0])
            B_main5 = np.vstack((B_wall1_R5, B_wall2_R5, B_wall3_R5, B_wall4_R5, B_ceil_R5, B_floor_R5, B_window_R5))
            #  % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
            #      [m5,temp_R5]=ode15s(@fun5,[(i-1)*dt,i*dt],t5(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
            #       t5(:,i)=temp_R5(length(m5),:)';
            a5, b5 = function(self.Construct_heat_flux_flow_relationship_room5(), B_main5, self.C_main5)
            def deriv5(y, t):
                return (np.dot(a5, y.reshape(self.NN_main5, 1)) + b5)[:, 0]
            temp_R5 = odeint(deriv5, self.t5[:, i - 1], time1)
            self.t5[:, i] = temp_R5[-1, :].T
            self.tR5_wall1_in[i][0] = self.t5[np.sum(self.N_EXTwall[:], dtype=int)][i]
            self.tR5_wall2_in[i][0] = self.t5[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
            self.tR5_wall3_in[i][0] = \
            self.t5[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) + 2][i]
            self.tR5_wall4_in[i][0] = \
            self.t5[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
            self.tR5_ceil_in[i][0] = self.t5[
                np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(
                    self.N_ceil[:], dtype=int) + 4][i]
            self.tR5_floor_in[i][0] = self.t5[self.NN_main5 - np.sum(self.N_window, dtype=int) - 2][i]
            self.tR5_window_in[i][0] = self.t5[self.NN_main5 - 2][i]
            self.tR5_window_out[i][0] = self.t5[self.NN_main5 - 3][i]
            self.tR5_air[i][0] = self.t5[self.NN_main5 - 1][i]
            self.w_e_in_R5[i][0] = self.w_e_in_R5[i - 1][0] + self.n_air * self.dt / 3600 * (
                        w_e_out[i][0] - self.w_e_in_R5[i - 1][0]) + \
                              self.m_w_gp * 3 * self.OCCUPIED_SITT[i][0] / self.Rou_air / self.V[4]

        if 0 <= time <= 7:
            M1 = 0.15
            M3 = 0.2
            M5 = 0.35
        elif 8 <= time <= 20:
            M1 = 0.35
            M3 = 0.35
            M5 = - 0.1
        elif 21 <= time <= 24:
            M1 = 0.1
            M3 = 0.15
            M5 = 0.35
        state_init = np.array([self.tR1_air[i][0], self.tR3_air[i][0], self.tR5_air[i][0]])
        state_init = self.compute_comfortable_terp(state_init, i)
        state_init = np.array([state_init[0], state_init[1], state_init[2], ta_out[i][0], RH[i][0], M1, M3, M5])
        return state_init

    def step(self, action, i):
        w_e_in_set_2_room1 = Humidity_Ratio(action[0], 60, p_tot)
        w_e_in_set_2_room3 = Humidity_Ratio(action[1], 60, p_tot)
        w_e_in_set_2_room5 = Humidity_Ratio(action[2], 60, p_tot)
        time = nh[i][0] % 24
        if (time >= 0 and time <= 7) or (
                time > 20 and time <= 24):  # STAYING IN BEDROOMS AT TIMEFRAME 0:00-6:00 am; 20:00-24:00 pm
            self.OCCUPIED_BED[i][0] = 1
        elif (time >= 6 and time <= 8) or (time >= 20 and time <= 22):  # LIGHTS & EQUIPMENT ON IN THE BEDCHAMBER
            self.LIGHT_ON_BED[i][0] = 1
        # if (((nh[i][0] / 24) % 7) >= 1) and (((nh[i][0] / 24) % 7) <= 5):  # WEEKDAYS(MONDAY TO FRIDAY)
        #     if (time > 6 and time <= 8) or (
        #             time >= 18 and time <= 20):  # TIMEFRAME 6:00-20:00 pm IN SITTING ROOM AND LIGHTS ON WHEN OCCUPYING
        #         self.OCCUPIED_SITT[i][0] = 1  # 时间范围:下午6:00-20:00在客厅，占用时灯亮着
        #         self.LIGHT_ON_SITT[i][0] = 1
        # else:
        #     if (time > 6 and time <= 20):
        #         self.OCCUPIED_SITT[i][0] = 1
        #         self.LIGHT_ON_SITT[i][0] = 1
        if (time > 7 and time <= 20):
            self.OCCUPIED_SITT[i][0] = 1
            self.LIGHT_ON_SITT[i][0] = 1

        self.Q_internal_rad_R1[i][0] = self.OCCUPIED_BED[i][0] * 2 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][
            0] * \
                                       (4.3 * self.f[0][5] * 0.20 + 6 * self.f[0][5] * 0.52)  # 2 OCCUPANTS
        self.Q_internal_cov_R1[i][0] = self.OCCUPIED_BED[i][0] * 2 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][
            0] * \
                                       (4.3 * self.f[0][5] * 0.80 + 6 * self.f[0][5] * 0.48)
        self.Q_internal_rad_R3[i][0] = self.OCCUPIED_BED[i][0] * 1 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][
            0] * \
                                       (4.3 * self.f[2][5] * 0.20 + 6 * self.f[2][5] * 0.52)  # 1 OCCUPANT
        self.Q_internal_cov_R3[i][0] = self.OCCUPIED_BED[i][0] * 1 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_BED[i][
            0] * \
                                       (4.3 * self.f[2][5] * 0.80 + 6 * self.f[2][5] * 0.48)
        self.Q_internal_rad_R5[i][0] = self.OCCUPIED_SITT[i][0] * 3 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_SITT[i][
            0] * \
                                       (4.3 * self.f[4][5] * 0.20 + 6 * self.f[4][5] * 0.52)  # 3 OCCUPANTS
        self.Q_internal_cov_R5[i][0] = self.OCCUPIED_SITT[i][0] * 3 * self.q_heatemss_p * 0.50 + self.LIGHT_ON_SITT[i][
            0] * \
                                       (4.3 * self.f[4][5] * 0.80 + 6 * self.f[4][5] * 0.48)
        self.Q_INTrad_wall1_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][0] / self.f_tol[
            0]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall2_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][1] / self.f_tol[
            0]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall3_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][2] / self.f_tol[
            0]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall4_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][3] / self.f_tol[
            0]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_ceil_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][4] / self.f_tol[
            0]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_floor_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][5] / self.f_tol[
            0]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_win_R1[0][i] = self.Q_internal_rad_R1[i][0] * self.f[0][6] / self.f_tol[
            0]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_conv1[0][i] = self.Q_internal_cov_R1[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
        self.Q_hvac1[0][i] = 0  # 房间1空调供热 w

        self.Q_conv2[0][i] = 0  # IT'S ASSUMED NO INTERNAL HEAT GAIN FROM ROOM 2 - TOILET AND ROOM 4 - STORAGE AND KITCHEN
        self.Q_hvac2[0][i] = 0

        self.Q_INTrad_wall1_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][0] / self.f_tol[
            2]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall2_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][1] / self.f_tol[
            2]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall3_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][2] / self.f_tol[
            2]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall4_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][3] / self.f_tol[
            2]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_ceil_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][4] / self.f_tol[
            2]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_floor_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][5] / self.f_tol[
            2]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_win_R3[0][i] = self.Q_internal_rad_R3[i][0] * self.f[2][6] / self.f_tol[
            2]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_conv3[0][i] = self.Q_internal_cov_R3[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
        self.Q_hvac3[0][i] = 0  # 房间3空调供热 w

        self.Q_conv4[0][i] = 0  # IT'S ASSUMED NO INTERNAL HEAT GAIN FROM ROOM 4 - TOILET AND ROOM 4 - STORAGE AND KITCHEN
        self.Q_hvac4[0][i] = 0

        self.Q_INTrad_wall1_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][0] / self.f_tol[
            4]  # 东墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall2_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][1] / self.f_tol[
            4]  # 西墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall3_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][2] / self.f_tol[
            4]  # 南墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_wall4_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][3] / self.f_tol[
            4]  # 北墙辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_ceil_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][4] / self.f_tol[
            4]  # 天花板辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_floor_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][5] / self.f_tol[
            4]  # 地板辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_INTrad_win_R5[0][i] = self.Q_internal_rad_R5[i][0] * self.f[4][6] / self.f_tol[
            4]  # 窗户辐射换热，w / m2(可设为内产热的辐射换热部分)
        self.Q_conv5[0][i] = self.Q_internal_cov_R5[i][0]  # 从室内空气产生的内部热量中获得的对流热量，w
        self.Q_hvac5[0][i] = 0  # 房间5空调供热 w

        if i < 143 * (3600 / self.dt):
            n = n0
        else:
            n = n0 + int((i * (self.dt / 3600) - 144) / 24)
        ws = i * (self.dt / 3600) - 24 * int(i * (self.dt / 3600) / 24)
        angle3 = solar_angle(n, r, longitude, ws, 90, 0)
        [win_transp3, win_absorp3, win_refelect3, e_all3, tao_all3] = solar_win2(angle3)
        self.Q_solar_in_R1[0][i] = win_transp3 * q_solar_out_3[i][0] * self.f[0][6]  # SOUTH WINDOW SOLAR
        self.Q_solar_in_R5[0][i] = win_transp3 * q_solar_out_3[i][0] * self.f[4][6]  # SOUTH WINDOW SOLAR

        angle1 = solar_angle(n, r, longitude, ws, 90, -90)
        [win_transp1, win_absorp1, win_refelect1, e_all1, tao_all1] = solar_win2(angle1)
        self.Q_solar_in_R2[0][i] = win_transp1 * q_solar_out_1[i][0] * self.f[1][6]  # EAST WINDOW SOLAR
        self.Q_solar_in_R3[0][i] = win_transp1 * q_solar_out_1[i][0] * self.f[2][6]  # EAST WINDOW SOLAR

        angle4 = solar_angle(n, r, longitude, ws, 90, 180)
        [win_transp4, win_absorp4, win_refelect4, e_all4, tao_all4] = solar_win2(angle4)
        self.Q_solar_in_R4[0][i] = win_transp4 * q_solar_out_4[i][0] * self.f[3][6]  # NORTH WINDOW SOLAR

        # 通过玻璃渗透获得内部太阳热量
        self.Q_solar_in_wall[0][i] = self.Q_solar_in_R1[0][i] * 0.1
        self.Q_solar_in_ceil[0][i] = self.Q_solar_in_R1[0][i] * 0.1
        self.Q_solar_in_floor[0][i] = self.Q_solar_in_R1[0][i] * 0.5
        self.Q_solar_in_wall[1][i] = self.Q_solar_in_R2[0][i] * 0.1
        self.Q_solar_in_ceil[1][i] = self.Q_solar_in_R2[0][i] * 0.1
        self.Q_solar_in_floor[1][i] = self.Q_solar_in_R2[0][i] * 0.5
        self.Q_solar_in_wall[2][i] = self.Q_solar_in_R3[0][i] * 0.1
        self.Q_solar_in_ceil[2][i] = self.Q_solar_in_R3[0][i] * 0.1
        self.Q_solar_in_floor[2][i] = self.Q_solar_in_R3[0][i] * 0.5
        self.Q_solar_in_wall[3][i] = self.Q_solar_in_R4[0][i] * 0.1
        self.Q_solar_in_ceil[3][i] = self.Q_solar_in_R4[0][i] * 0.1
        self.Q_solar_in_floor[3][i] = self.Q_solar_in_R4[0][i] * 0.5
        self.Q_solar_in_wall[3][i] = self.Q_solar_in_R5[0][i] * 0.1
        self.Q_solar_in_ceil[4][i] = self.Q_solar_in_R5[0][i] * 0.1
        self.Q_solar_in_floor[4][i] = self.Q_solar_in_R5[0][i] * 0.5

        w_e_out[i][0] = Humidity_Ratio(ta_out[i][0], RH[i][0], p_tot)  # HUMIDITY RATIO OF ATMOSHPHERIC STATE POING
        # b向量元素-修正热负荷计算矩阵
        # 房间1 - REVISED(修改)
        B_wall1_R1 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0],
                                    self.Q_solar_in_wall[0][i],
                                    self.Q_INTrad_wall1_R1[0][i], ta_out[i][0], self.f[0][0], self.h_in_wall,
                                    self.tR1_air[i - 1][0])
        B_wall2_R1 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[0][i],
                                       self.Q_INTrad_wall2_R1[0][i],
                                       self.Q_solar_in_wall[4][i], self.Q_INTrad_wall1_R5[0][i], self.f[4][0],
                                       self.tR5_air[i - 1][0],
                                       self.tR5_wall2_in[i - 1][0], self.tR5_wall3_in[i - 1][0],
                                       self.tR5_wall4_in[i - 1][0],
                                       self.tR5_ceil_in[i - 1][0],
                                       self.tR5_floor_in[i - 1][0], self.tR5_window_in[i - 1][0], self.hr[4][1],
                                       self.hr[4][2], self.hr[4][3],
                                       self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[0][1],
                                       self.tR1_air[i - 1][0])
        B_wall3_R1 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_3[i][0],
                                    self.Q_solar_in_wall[0][i],
                                    self.Q_INTrad_wall3_R1[0][i], ta_out[i][0], self.f[0][2], self.h_in_wall,
                                    self.tR1_air[i - 1][0])
        B_wall4_R1 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[0][i],
                                       self.Q_INTrad_wall2_R1[0][i],
                                       self.Q_solar_in_wall[1][i], 0, self.f[1][2], self.tR2_air[i - 1][0],
                                       self.tR2_wall1_in[i - 1][0],
                                       self.tR2_wall2_in[i - 1][0], self.tR2_wall4_in[i - 1][0],
                                       self.tR2_ceil_in[i - 1][0],
                                       self.tR2_floor_in[i - 1][0], self.tR2_window_in[i - 1][0], self.hr[1][0],
                                       self.hr[1][1], self.hr[1][3],
                                       self.hr[1][4], self.hr[1][5], self.hr[1][6], self.f[0][3],
                                       self.tR1_air[i - 1][0])
        B_ceil_R1 = setB_ceiling_load(self.N_ceil, self.eb, self.Q_solar_in_floor[0][i], self.Q_INTrad_floor_R1[0][i],
                                      self.f[0][5],
                                      self.Q_solar_in_ceil[0][i], self.Q_INTrad_ceil_R1[0][i], self.f[0][4],
                                      self.h_in_ceil,
                                      self.h_in_floor,
                                      self.tR1_air[i - 1][0])
        B_floor_R1 = setB_ceiling_load(self.N_floor, self.eb, self.Q_solar_in_ceil[0][i], self.Q_INTrad_ceil_R1[0][i],
                                       self.f[0][4],
                                       self.Q_solar_in_floor[0][i], self.Q_INTrad_floor_R1[0][i], self.f[0][5],
                                       self.h_in_floor, self.h_in_ceil,
                                       self.tR1_air[i - 1][0])
        B_window_R1 = setB_window_load(self.N_window, self.h_out, e_all3[0][1], e_all3[0][3], q_solar_out_3[i][0],
                                       self.Q_INTrad_win_R1[0][i],
                                       ta_out[i][0], self.f[0][6], self.h_in_wall,
                                       self.tR1_air[i - 1][0])  # SOUTH WINDWOW
        # B_air_R1 = self.Cp_air * self.Rou_air * n_air / 3600 * V[0] * ta_out[i][0] + Q_conv1[0][i] + Q_hvac1[0][i]
        B_main = np.vstack(
            (B_wall1_R1, B_wall2_R1, B_wall3_R1, B_wall4_R1, B_ceil_R1, B_floor_R1, B_window_R1))  # B_air_R1
        # SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
        # [m1,temp_R1]=ode15s(@fun,[(i-1)*dt,i*dt],t1(:,i-1))
        # t1(:, i)=temp_R1(length(m1),:)
        a, b = function(self.Construct_heat_flux_flow_relationship_room1(), B_main, self.C_main)

        def deriv(y, t):
            return (np.dot(a, y.reshape(self.NN_main, 1)) + b)[:, 0]

        time1 = np.linspace((i - 1) * self.dt, i * self.dt, 1000)
        temp_R1 = odeint(deriv, self.t1[:, i - 1], time1)
        # print(temp_R1.shape)
        self.t1[:, i] = temp_R1[-1, :].T
        self.tR1_wall1_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int)][i]
        self.tR1_wall2_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int) + 1][i]
        self.tR1_wall3_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int) + 2][i]
        self.tR1_wall4_in[i][0] = self.t1[np.sum(self.N_EXTwall[:], dtype=int) + 3][i]
        self.tR1_ceil_in[i][0] = \
        self.t1[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2
                + np.sum(self.N_ceil[:], dtype=int) + 4][i]
        self.tR1_floor_in[i][0] = self.t1[self.NN_main - np.sum(self.N_window, dtype=int) - 1][i]
        self.tR1_window_in[i][0] = self.t1[self.NN_main - 1][i]
        self.tR1_window_out[i][0] = self.t1[self.NN_main - 2][i]
        dQair_change_x1 = self.h_in_wall * self.f[0][0] * (
                self.tR1_wall1_in[i][0] - self.tR1_air[i - 1][0]) + self.h_in_wall * \
                          self.f[0][1] * (self.tR1_wall2_in[i][0] - self.tR1_air[i - 1][0]) + self.h_in_wall * \
                          self.f[0][2] * \
                          (self.tR1_wall3_in[i][0] - self.tR1_air[i - 1][0]) + self.h_in_wall * self.f[0][3] * \
                          (self.tR1_wall4_in[i][0] - self.tR1_air[i - 1][0]) + self.h_in_ceil * self.f[0][4] * \
                          (self.tR1_ceil_in[i][0] - self.tR1_air[i - 1][0]) + self.h_in_floor * self.f[0][5] * \
                          (self.tR1_ceil_in[i][0] - self.tR1_air[i - 1][0]) + self.h_in_wall * self.f[0][6] * \
                          np.float(self.tR1_window_in[i][0] - self.tR1_air[i - 1][0]) + np.float(
            self.Cp_air) * np.float(
            self.Rou_air) * np.float(self.n_air) * \
                          np.float(self.V[0]) / np.float(3600) * np.float(ta_out[i][0] - self.tR1_air[i - 1][0]) + \
                          self.Q_conv1[0][i]
        denominator1 = (self.h_in_wall * np.sum(self.f[0, 0:4]) + self.h_in_ceil * self.f[0][4] + self.h_in_floor *
                        self.f[0][5] + self.h_in_wall * self.f[0][6]) \
                       * self.dt / self.Cp_air / self.Rou_air / self.V[0] + (self.n_air * self.dt / 3600 + 1)
        dQair_change1 = dQair_change_x1 / denominator1
        # SOLVING ROOM 1 TEMPERATURE
        self.tR1_air[i][0] = dQair_change1 * self.dt / self.Cp_air / self.Rou_air / self.V[0] + self.tR1_air[i - 1][0]
        # self.tR1_air_time[i][0] = self.tR1_air[i][0]
        # print(self.tR1_air_time[i][0])
        # HIMIDITY MIXING CALCULATION OF ROOM 1; 2 OCCUPANTS IN MAIN BEDCHAMBER
        self.w_e_in_R1[i][0] = self.w_e_in_R1[i - 1][0] + self.n_air * self.dt / 3600 * (
                    w_e_out[i][0] - self.w_e_in_R1[i - 1][0]) \
                               + self.m_w_gp * 2 * self.OCCUPIED_BED[i][0] / self.Rou_air / self.V[0]
        # HEATING & COOLING LOADS CALCULATION ENCOMPASSING DEHUMIDIFICATION/LATENT HEAT
        if self.HC_mode == 1:  # FULL TIME SPACE HEATING AND COOLING - 24 H AVAILABLE IN THE PERIODS
            self.HEATING_BEDROOM[i][0] = 0
            self.HEATING_SITTROOM[i][0] = 0
            self.COOLING_BEDROOM[i][0] = float(Hour[i][0] >= 0 and Hour[i][0] <= 3818)
            self.COOLING_SITTROOM[i][0] = float(Hour[i][0] >= 0 and Hour[i][0] <= 3818)
        elif self.HC_mode == 2:  # PART TIME SPACE HEATING AND COOLING AS PER SPECIFIED OCCUPANTS BEHAVIOUR PROFILE
            self.HEATING_BEDROOM[i][0] = 0
            self.HEATING_SITTROOM[i][0] = 0
            self.COOLING_BEDROOM[i][0] = float(OCCUPIED_BED[i][0] and (Hour[i][0] >= 0 and Hour[i][0] <= 3818))
            self.COOLING_SITTROOM[i][0] = float(OCCUPIED_SITT[i][0] and (Hour[i][0] >= 0 and Hour[i][0] <= 3818))
        # TARGETED ROOM TEMPERATURE 26 C FOR SPACE COOLIN
        if self.COOLING_BEDROOM[i][0]:
            if self.w_e_in_R1[i][0] > w_e_in_set_2_room1:
                self.Q_c_latent_R1[i - 1][0] = 2260 * (self.w_e_in_R1[i][0] - w_e_in_set_2_room1) * self.Rou_air * self.V[
                    0] / self.dt
                self.w_e_in_R1[i][0] = w_e_in_set_2_room1
            if self.tR1_air[i][0] >= action[0] and dQair_change_x1 > 0:
                self.HVAC1[i][0] = 1
                self.Q_c_sens_R1[i - 1][0] = self.Cp_air * self.Rou_air * self.V[0] * (
                        self.tR1_air[i - 1][0] - action[0]) / self.dt + dQair_change_x1
                self.tR1_air[i][0] = action[0]
        # room2
        B_wall1_R2 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0], self.Q_solar_in_wall[1][i], 0,
                               ta_out[i][0], self.f[1][0])
        B_wall2_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[1][i], 0,
                                  self.Q_solar_in_wall[4][i],
                                  self.Q_INTrad_wall1_R5[0][i], self.f[4][0], self.tR5_air[i - 1][0],
                                  self.tR5_wall2_in[i - 1][0],
                                  self.tR5_wall3_in[i - 1][0], self.tR5_wall4_in[i - 1][0], self.tR5_ceil_in[i - 1][0],
                                  self.tR5_floor_in[i - 1][0],
                                  self.tR5_window_in[i - 1][0], self.hr[4][1], self.hr[4][2], self.hr[4][3],
                                  self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[1][1])
        B_wall3_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[1][i], 0,
                                  self.Q_solar_in_wall[0][i],
                                  self.Q_INTrad_wall1_R1[0][i], self.f[0][3], self.tR1_air[i][0],
                                  self.tR1_wall2_in[i][0],
                                  self.tR1_wall3_in[i][0], self.tR1_wall4_in[i][0], self.tR1_ceil_in[i][0],
                                  self.tR1_floor_in[i][0],
                                  self.tR1_window_in[i][0], self.hr[0][0], self.hr[0][1], self.hr[0][2],
                                  self.hr[0][4], self.hr[0][5], self.hr[0][6], self.f[1][2])
        B_wall4_R2 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[1][i], 0,
                                  self.Q_solar_in_wall[2][i],
                                  self.Q_INTrad_wall1_R3[0][i], self.f[2][2], self.tR3_air[i - 1][0],
                                  self.tR3_wall2_in[i - 1][0],
                                  self.tR3_wall3_in[i - 1][0], self.tR3_wall4_in[i - 1][0], self.tR3_ceil_in[i - 1][0],
                                  self.tR3_floor_in[i - 1][0],
                                  self.tR3_window_in[i - 1][0], self.hr[2][0], self.hr[2][1], self.hr[2][3],
                                  self.hr[2][4], self.hr[2][5], self.hr[2][6], self.f[1][3])
        B_ceil_R2 = setB_ceiling(self.N_ceil, self.eb, self.Q_solar_in_floor[1][i], 0, self.f[1][5],
                                 self.Q_solar_in_ceil[1][i], 0, self.f[1][4])
        B_floor_R2 = setB_ceiling(self.N_floor, self.eb, self.Q_solar_in_ceil[1][i], 0, self.f[1][4],
                                  self.Q_solar_in_floor[1][i], 0, f[1][5])
        B_window_R2 = setB_window(self.N_window, self.h_out, e_all1[0][1], e_all1[0][3], q_solar_out_1[i][0], 0,
                                  ta_out[i][0], self.f[1][6])  # EAST WINDOW
        B_air_R2 = self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[1] * ta_out[i][0] + self.Q_conv2[0][i] + \
                   self.Q_hvac2[0][i]
        B_main2 = np.vstack(
            (B_wall1_R2, B_wall2_R2, B_wall3_R2, B_wall4_R2, B_ceil_R2, B_floor_R2, B_window_R2, B_air_R2))
        #  % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
        #      [m2,temp_R2]=ode15s(@fun2,[(i-1)*dt,i*dt],t2(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
        #       t2(:,i)=temp_R2(length(m2),:)';
        a2, b2 = function(self.Construct_heat_flux_flow_relationship_room2(), B_main2, self.C_main2)

        def deriv2(y, t):
            return (np.dot(a2, y.reshape(self.NN_main2, 1)) + b2)[:, 0]

        temp_R2 = odeint(deriv2, self.t2[:, i - 1], time1)
        self.t2[:, i] = temp_R2[-1, :].T
        self.tR2_wall1_in[i][0] = self.t2[np.sum(self.N_EXTwall[:], dtype=int)][i]
        self.tR2_wall2_in[i][0] = \
        self.t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
        self.tR2_wall3_in[i][0] = \
            self.t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
        self.tR2_wall4_in[i][0] = \
            self.t2[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 3 + 3][i]
        self.tR2_ceil_in[i][0] = self.t2[
            np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 3 + np.sum(self.N_ceil[:],
                                                                                                     dtype=int) + 4][i]
        self.tR2_floor_in[i][0] = self.t2[self.NN_main2 - np.sum(self.N_window, dtype=int) - 2][i]
        self.tR2_window_in[i][0] = self.t2[self.NN_main2 - 2][i]
        self.tR2_window_out[i][0] = self.t2[self.NN_main2 - 3][i]
        self.tR2_air[i][0] = self.t2[self.NN_main2 - 1][i]
        # room3
        B_wall1_R3 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_1[i][0],
                                    self.Q_solar_in_wall[2][i],
                                    self.Q_INTrad_wall1_R3[0][i], ta_out[i][0], self.f[2][0], self.h_in_wall,
                                    self.tR3_air[i][0])
        B_wall2_R3 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * (
                0.667 * np.float(self.tR4_air[i - 1][0]) + 0.333 * np.float(self.tR5_air[i - 1][0])),
                                        self.eb, self.Q_solar_in_wall[2][i], self.Q_INTrad_wall1_R3[0][i],
                                        [self.Q_solar_in_wall[3][i] + (1 / 7.55) * self.Q_solar_in_wall[4][i]],
                                        (1.0 / 7.55) * self.Q_INTrad_wall1_R5[0][i],
                                        [0.667 * (self.hr[3][1] * self.tR4_wall2_in[i - 1][0] + self.hr[3][2] *
                                                  self.tR4_wall3_in[i - 1][0] +
                                                  self.hr[3][3] * self.tR4_wall4_in[i - 1][0] + self.hr[3][4] *
                                                  self.tR4_ceil_in[i - 1][0] +
                                                  self.hr[3][5] * self.tR4_floor_in[i - 1][0] + self.hr[3][6] *
                                                  self.tR4_window_in[i - 1][0]) +
                                         0.333 * (self.hr[4][1] * self.tR5_wall2_in[i - 1][0] + self.hr[4][2] *
                                                  self.tR5_wall3_in[i - 1][0] +
                                                  self.hr[4][3] * self.tR5_wall4_in[i - 1][0] + self.hr[4][4] *
                                                  self.tR5_ceil_in[i - 1][0] +
                                                  self.hr[4][5] * self.tR5_floor_in[i - 1][0] + self.hr[4][6] *
                                                  self.tR5_window_in[i - 1][0])],
                                        self.f[2][1], self.h_in_wall * self.tR3_air[i - 1][0])
        B_wall3_R3 = setB_INTwall_load(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[2][i],
                                       self.Q_INTrad_wall3_R3[0][i],
                                       self.Q_solar_in_wall[1][i], 0, self.f[1][3], self.tR2_air[i][0],
                                       self.tR2_wall1_in[i][0],
                                       self.tR2_wall2_in[i][0], self.tR2_wall3_in[i][0], self.tR2_ceil_in[i][0],
                                       self.tR2_floor_in[i][0],
                                       self.tR2_window_in[i][0], self.hr[1][0], self.hr[1][1], self.hr[1][2],
                                       self.hr[1][4],
                                       self.hr[1][5], self.hr[1][6], self.f[2][2], self.tR3_air[i - 1][0])
        B_wall4_R3 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_4[i][0],
                                    self.Q_solar_in_wall[2][i],
                                    self.Q_INTrad_wall4_R3[0][i], ta_out[i][0], self.f[2][3], self.h_in_wall,
                                    self.tR3_air[i - 1][0])
        B_ceil_R3 = setB_ceiling_load(self.N_ceil, self.eb, self.Q_solar_in_floor[2][i], self.Q_INTrad_floor_R3[0][i],
                                      self.f[2][5],
                                      self.Q_solar_in_ceil[2][i], self.Q_INTrad_ceil_R3[0][i], self.f[2][4],
                                      self.h_in_ceil,
                                      self.h_in_floor,
                                      self.tR3_air[i - 1][0])
        B_floor_R3 = setB_ceiling_load(self.N_floor, self.eb, self.Q_solar_in_ceil[2][i], self.Q_INTrad_ceil_R3[0][i],
                                       self.f[2][4],
                                       self.Q_solar_in_floor[2][i], self.Q_INTrad_floor_R3[0][i], self.f[2][5],
                                       self.h_in_floor, self.h_in_ceil,
                                       self.tR3_air[i - 1][0])
        B_window_R3 = setB_window_load(self.N_window, self.h_out, e_all1[0][1], e_all1[0][3], q_solar_out_1[i][0],
                                       self.Q_INTrad_win_R3[0][i],
                                       ta_out[i][0], self.f[2][6], self.h_in_wall,
                                       self.tR3_air[i - 1][0])  # EAST WINDOW
        B_main3 = np.vstack((B_wall1_R3, B_wall2_R3, B_wall3_R3, B_wall4_R3, B_ceil_R3, B_floor_R3, B_window_R3))
        # SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
        #      [m3,temp_R3]=ode15s(@fun3,[(i-1)*dt,i*dt],t3(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
        #       t3(:,i)=temp_R3(length(m3),:)';
        a3, b3 = function(self.Construct_heat_flux_flow_relationship_room3(), B_main3, self.C_main3)

        def deriv3(y, t):
            return (np.dot(a3, y.reshape(self.NN_main3, 1)) + b3)[:, 0]

        temp_R3 = odeint(deriv3, self.t3[:, i - 1], time1)
        self.t3[:, i] = temp_R3[-1, :].T
        self.tR3_wall1_in[i][0] = self.t3[np.sum(self.N_EXTwall[:], dtype=int)][i]
        self.tR3_wall2_in[i][0] = \
        self.t3[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
        self.tR3_wall3_in[i][0] = \
            self.t3[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
        self.tR3_wall4_in[i][0] = \
            self.t3[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
        self.tR3_ceil_in[i][0] = self.t3[
            np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(
                self.N_ceil[:], dtype=int) + 4][i]
        self.tR3_floor_in[i][0] = self.t3[self.NN_main3 - np.sum(self.N_window, dtype=int) - 2][i]
        self.tR3_window_in[i][0] = self.t3[self.NN_main3 - 2][i]
        self.tR3_window_out[i][0] = self.t3[self.NN_main3 - 3][i]
        # self.tR3_air[i][0] = self.t3[self.NN_main3 - 1][i]
        dQair_change_x3 = self.h_in_wall * self.f[2][0] * (
                self.tR3_wall1_in[i][0] - self.tR3_air[i - 1][0]) + self.h_in_wall * \
                          self.f[2][1] * (self.tR3_wall2_in[i][0] - self.tR3_air[i - 1][0]) + self.h_in_wall * \
                          self.f[2][2] * \
                          (self.tR3_wall3_in[i][0] - self.tR3_air[i - 1][0]) + self.h_in_wall * self.f[2][3] * (
                                  self.tR3_wall4_in[i][0] - self.tR3_air[i - 1][0]) \
                          + self.h_in_ceil * self.f[2][4] * (
                                  self.tR3_ceil_in[i][0] - self.tR3_air[i - 1][0]) + self.h_in_floor * \
                          self.f[2][5] * np.float(self.tR3_ceil_in[i][0] - self.tR3_air[i - 1][0]) + self.h_in_wall * \
                          self.f[2][6] * \
                          np.float(self.tR3_window_in[i][0] - self.tR3_air[i - 1][0]) + np.float(
            self.Cp_air) * np.float(
            self.Rou_air) * \
                          np.float(self.n_air) * np.float(self.V[2]) / 3600 * np.float(
            ta_out[i][0] - self.tR1_air[i - 1][0]) + self.Q_conv3[0][i]
        denominator3 = (self.h_in_wall * np.sum(self.f[2, 0:4]) + self.h_in_ceil * self.f[2][4] + self.h_in_floor *
                        self.f[2][5] + self.h_in_wall * self.f[2][6]) * self.dt / self.Cp_air / self.Rou_air / \
                       self.V[2] \
                       + (self.n_air * self.dt / 3600 + 1)
        dQair_change3 = dQair_change_x3 / denominator3
        self.tR3_air[i][0] = dQair_change3 * self.dt / self.Cp_air / self.Rou_air / self.V[2] + self.tR3_air[i - 1][0]  # SOLVING ROOM 3 TEMPERATURE
        self.w_e_in_R3[i][0] = self.w_e_in_R3[i - 1][0] + self.n_air * self.dt / 3600 * (
                w_e_out[i][0] - self.w_e_in_R3[i - 1][0]) + \
                               self.m_w_gp * 1 * self.OCCUPIED_BED[i][0] / self.Rou_air / self.V[2]
        if self.COOLING_BEDROOM[i][0]:
            if self.w_e_in_R3[i][0] > w_e_in_set_2_room3:
                self.Q_c_latent_R3[i - 1][0] = 2260 * (self.w_e_in_R3[i][0] - w_e_in_set_2_room3) * self.Rou_air * self.V[
                    2] / self.dt
                self.w_e_in_R3[i][0] = w_e_in_set_2_room3
            if self.tR3_air[i][0] >= action[1] and dQair_change_x3 > 0:
                self.HVAC3[i][0] = 1
                self.Q_c_sens_R3[i - 1][0] = self.Cp_air * self.Rou_air * self.V[2] * (
                        self.tR3_air[i - 1][0] - action[1]) / self.dt + dQair_change_x3
                self.tR3_air[i][0] = action[1]

        # room4
        B_wall1_R4 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[3][i], 0,
                                  self.Q_solar_in_wall[2][i],
                                  self.Q_INTrad_wall2_R3[0][i], self.f[2][1], self.tR3_air[i][0],
                                  self.tR3_wall2_in[i][0],
                                  self.tR3_wall3_in[i][0],
                                  self.tR3_wall4_in[i][0], self.tR3_ceil_in[i][0], self.tR3_floor_in[i][0],
                                  self.tR3_window_in[i][0],
                                  self.hr[2][0],
                                  self.hr[2][2], self.hr[2][3], self.hr[2][4], self.hr[2][5], self.hr[2][6],
                                  self.f[3][0])
        B_wall2_R4 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_2[i][0], self.Q_solar_in_wall[3][i], 0,
                               ta_out[i][0], self.f[3][1])
        B_wall3_R4 = setB_INTwall(self.N_INTwall, self.h_in_wall, self.eb, self.Q_solar_in_wall[3][i], 0,
                                  self.Q_solar_in_wall[4][i],
                                  self.Q_INTrad_wall4_R5[0][i], self.f[4][3], self.tR5_air[i - 1][0],
                                  self.tR5_wall1_in[i - 1][0],
                                  self.tR5_wall2_in[i - 1][0], self.tR5_wall3_in[i - 1][0], self.tR5_ceil_in[i - 1][0],
                                  self.tR5_floor_in[i - 1][0],
                                  self.tR5_window_in[i - 1][0], self.hr[4][0], self.hr[4][1], self.hr[4][2],
                                  self.hr[4][4], self.hr[4][5], self.hr[4][6], self.f[3][2])
        B_wall4_R4 = setB_wall(self.N_EXTwall, self.h_out, self.eb, q_solar_out_4[i][0], self.Q_solar_in_wall[3][i], 0,
                               ta_out[i][0], self.f[3][3])
        B_ceil_R4 = setB_ceiling(self.N_ceil, self.eb, self.Q_solar_in_floor[3][i], 0, self.f[3][5],
                                 self.Q_solar_in_ceil[3][i], 0, self.f[3][4])
        B_floor_R4 = setB_ceiling(self.N_floor, self.eb, self.Q_solar_in_ceil[3][i], 0, self.f[3][4],
                                  self.Q_solar_in_floor[3][i], 0, self.f[3][5])
        B_window_R4 = setB_window(self.N_window, self.h_out, e_all4[0][1], e_all4[0][3], q_solar_out_4[i][0], 0,
                                  ta_out[i][0], self.f[3][6])
        B_air_R4 = self.Cp_air * self.Rou_air * self.n_air / 3600 * self.V[3] * ta_out[i][0] + self.Q_conv4[0][i] + \
                   self.Q_hvac4[0][i]
        B_main4 = np.vstack(
            (B_wall1_R4, B_wall2_R4, B_wall3_R4, B_wall4_R4, B_ceil_R4, B_floor_R4, B_window_R4, B_air_R4))
        # % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
        #      [m4,temp_R4]=ode15s(@fun4,[(i-1)*dt,i*dt],t4(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
        #       t4(:,i)=temp_R4(length(m4),:)';
        a4, b4 = function(self.Construct_heat_flux_flow_relationship_room4(), B_main4, self.C_main4)

        def deriv4(y, t):
            return (np.dot(a4, y.reshape(self.NN_main4, 1)) + b4)[:, 0]

        temp_R4 = odeint(deriv4, self.t4[:, i - 1], time1)
        self.t4[:, i] = temp_R4[-1, :].T
        self.tR4_wall1_in[i][0] = self.t4[np.sum(self.N_EXTwall[:], dtype=int)][i]
        self.tR4_wall2_in[i][0] = \
        self.t4[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
        self.tR4_wall3_in[i][0] = \
            self.t4[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) * 2 + 2][i]
        self.tR4_wall4_in[i][0] = \
            self.t4[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
        self.tR4_ceil_in[i][0] = self.t4[
            np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(
                self.N_ceil[:], dtype=int) + 4][i]
        self.tR4_floor_in[i][0] = self.t4[self.NN_main4 - np.sum(self.N_window, dtype=int) - 2][i]
        self.tR4_window_in[i][0] = self.t4[self.NN_main4 - 2][i]
        self.tR4_window_out[i][0] = self.t4[self.NN_main4 - 3][i]
        self.tR4_air[i][0] = self.t4[self.NN_main4 - 1][i]
        # room5
        B_wall1_R5 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * (
                0.53 * self.tR1_air[i][0] + 0.34 * self.tR2_air[i][0] + 0.13 * self.tR3_air[i][0]),
                                        self.eb, self.Q_solar_in_wall[4][i], self.Q_INTrad_wall1_R5[0][i],
                                        self.Q_solar_in_wall[0][i] + self.Q_solar_in_wall[1][i] + 0.333 *
                                        self.Q_solar_in_wall[2][
                                            i],
                                        self.Q_INTrad_wall1_R1[0][i] + 0 + 0.333 * self.Q_INTrad_wall1_R3[0][i],
                                        0.53 * (self.hr[0][0] * self.tR1_wall1_in[i][0] + self.hr[0][2] *
                                                self.tR1_wall3_in[i][0] +
                                                self.hr[0][3] * self.tR1_wall4_in[i][0] + self.hr[0][4] *
                                                self.tR1_ceil_in[i][
                                                    0] +
                                                self.hr[0][5] * self.tR1_floor_in[i][0] + self.hr[0][6] *
                                                self.tR1_window_in[i][0]) +
                                        0.34 * (self.hr[1][0] * self.tR2_wall1_in[i][0] + self.hr[1][2] *
                                                self.tR2_wall3_in[i][0] +
                                                self.hr[1][3] * self.tR2_wall4_in[i][0] + self.hr[1][4] *
                                                self.tR2_ceil_in[i][
                                                    0] +
                                                self.hr[1][5] * self.tR2_floor_in[i][0] + self.hr[1][6] *
                                                self.tR2_window_in[i][0]) +
                                        0.13 * (self.hr[2][0] * self.tR3_wall1_in[i][0] + self.hr[2][2] *
                                                self.tR3_wall3_in[i][0] +
                                                self.hr[2][3] * self.tR3_wall4_in[i][0] + self.hr[2][4] *
                                                self.tR3_ceil_in[i][
                                                    0] +
                                                self.hr[2][5] * self.tR3_floor_in[i][0] + self.hr[2][6] *
                                                self.tR3_window_in[i][0]),
                                        self.f[4][0], self.h_in_wall * self.tR5_air[i - 1][0])
        B_wall2_R5 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_2[i][0],
                                    self.Q_solar_in_wall[4][i],
                                    self.Q_INTrad_wall2_R5[0][i], ta_out[i][0], self.f[4][1], self.h_in_wall,
                                    self.tR5_air[i - 1][0])
        B_wall3_R5 = setB_wall_load(self.N_EXTwall, self.h_out, self.eb, q_solar_out_3[i][0],
                                    self.Q_solar_in_wall[4][i],
                                    self.Q_INTrad_wall3_R5[0][i], ta_out[i][0], self.f[4][2], self.h_in_wall,
                                    self.tR5_air[i - 1][0])
        B_wall4_R5 = setB_INT3wall_load(self.N_INTwall, self.h_in_wall * self.tR4_air[i][0], eb,
                                        self.Q_solar_in_wall[4][i],
                                        self.Q_INTrad_wall4_R5[0][i], self.Q_solar_in_wall[3][i], 0,
                                        [self.hr[3][0] * self.tR4_wall1_in[i][0] + self.hr[3][1] * self.tR4_wall2_in[i][
                                            0] +
                                         self.hr[3][3] * self.tR4_wall4_in[i][0] + self.hr[3][4] * self.tR4_ceil_in[i][
                                             0] +
                                         self.hr[3][5] * self.tR4_floor_in[i][0] + self.hr[3][6] *
                                         self.tR4_window_in[i][0]],
                                        self.f[4][3], self.h_in_wall * self.tR5_air[i - 1][0])
        B_ceil_R5 = setB_ceiling_load(self.N_ceil, self.eb, self.Q_solar_in_floor[4][i], self.Q_INTrad_floor_R5[0][i],
                                      self.f[4][5],
                                      self.Q_solar_in_ceil[4][i], self.Q_INTrad_ceil_R5[0][i], self.f[4][4],
                                      self.h_in_ceil,
                                      self.h_in_floor,
                                      self.tR5_air[i - 1][0])
        B_floor_R5 = setB_ceiling_load(self.N_floor, self.eb, self.Q_solar_in_ceil[4][i], self.Q_INTrad_ceil_R5[0][i],
                                       self.f[4][4],
                                       self.Q_solar_in_floor[4][i], self.Q_INTrad_floor_R5[0][i], self.f[4][5],
                                       self.h_in_floor, self.h_in_ceil,
                                       self.tR5_air[i - 1][0])
        B_window_R5 = setB_window_load(self.N_window, self.h_out, e_all3[0][1], e_all3[0][3], q_solar_out_3[i][0],
                                       self.Q_INTrad_win_R5[0][i],
                                       ta_out[i][0], self.f[4][6], self.h_in_wall, self.tR5_air[i - 1][0])
        B_main5 = np.vstack((B_wall1_R5, B_wall2_R5, B_wall3_R5, B_wall4_R5, B_ceil_R5, B_floor_R5, B_window_R5))
        #  % SOLVE THE THERMAL BALANCE MATRIX C(DT) = AT + B
        #      [m5,temp_R5]=ode15s(@fun5,[(i-1)*dt,i*dt],t5(:,i-1)); % ADOPT ODE45 or ODE15s SOLVER
        #       t5(:,i)=temp_R5(length(m5),:)';
        a5, b5 = function(self.Construct_heat_flux_flow_relationship_room5(), B_main5, self.C_main5)

        def deriv5(y, t):
            return (np.dot(a5, y.reshape(self.NN_main5, 1)) + b5)[:, 0]

        temp_R5 = odeint(deriv5, self.t5[:, i - 1], time1)
        self.t5[:, i] = temp_R5[-1, :].T
        self.tR5_wall1_in[i][0] = self.t5[np.sum(self.N_EXTwall[:], dtype=int)][i]
        self.tR5_wall2_in[i][0] = \
        self.t5[np.sum(self.N_EXTwall[:], dtype=int) + np.sum(self.N_INTwall[:], dtype=int) + 1][i]
        self.tR5_wall3_in[i][0] = \
            self.t5[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) + 2][i]
        self.tR5_wall4_in[i][0] = \
            self.t5[np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + 3][i]
        self.tR5_ceil_in[i][0] = self.t5[
            np.sum(self.N_EXTwall[:], dtype=int) * 2 + np.sum(self.N_INTwall[:], dtype=int) * 2 + np.sum(
                self.N_ceil[:], dtype=int) + 4][i]
        self.tR5_floor_in[i][0] = self.t5[self.NN_main5 - np.sum(self.N_window, dtype=int) - 2][i]
        self.tR5_window_in[i][0] = self.t5[self.NN_main5 - 2][i]
        self.tR5_window_out[i][0] = self.t5[self.NN_main5 - 3][i]
        # self.tR5_air[i][0] = self.t5[self.NN_main5 - 1][i]
        dQair_change_x5 = self.h_in_wall * self.f[4][0] * (
                self.tR5_wall1_in[i][0] - self.tR5_air[i - 1][0]) + self.h_in_wall * \
                          self.f[4][1] * (self.tR5_wall2_in[i][0] - self.tR5_air[i - 1][0]) + self.h_in_wall * \
                          self.f[4][2] * \
                          (self.tR5_wall3_in[i][0] - self.tR5_air[i - 1][0]) + self.h_in_wall * self.f[4][3] * \
                          (self.tR5_wall4_in[i][0] - self.tR5_air[i - 1][0]) + self.h_in_ceil * self.f[4][4] * \
                          (self.tR5_ceil_in[i][0] - self.tR5_air[i - 1][0]) + self.h_in_floor * self.f[4][5] * \
                          (self.tR5_ceil_in[i][0] - self.tR5_air[i - 1][0]) + self.h_in_wall * self.f[4][6] * \
                          (self.tR5_window_in[i][0] - self.tR5_air[i - 1][
                              0]) + self.Cp_air * self.Rou_air * self.n_air * \
                          self.V[4] \
                          / 3600 * (ta_out[i][0] - self.tR5_air[i - 1][0]) + self.Q_conv5[0][i]
        denominator5 = (self.h_in_wall * np.sum(self.f[4, 0:4]) + self.h_in_ceil * self.f[4][4] + self.h_in_floor *
                        self.f[4][5] + self.h_in_wall * self.f[4][6]) * self.dt / self.Cp_air / self.Rou_air / \
                       self.V[4] + \
                       (self.n_air * self.dt / 3600 + 1)
        dQair_change5 = dQair_change_x5 / denominator5
        self.tR5_air[i][0] = dQair_change5 * self.dt / self.Cp_air / self.Rou_air / self.V[4] + self.tR5_air[i - 1][0]
        self.w_e_in_R5[i][0] = self.w_e_in_R5[i - 1][0] + self.n_air * self.dt / 3600 * (
                w_e_out[i][0] - self.w_e_in_R5[i - 1][0]) + \
                               self.m_w_gp * 3 * self.OCCUPIED_SITT[i][0] / self.Rou_air / self.V[4]

        if self.COOLING_SITTROOM[i][0]:
            if self.w_e_in_R5[i][0] > w_e_in_set_2_room5:
                self.Q_c_latent_R5[i - 1][0] = 2260 * (self.w_e_in_R5[i][0] - w_e_in_set_2_room5) * self.Rou_air * self.V[
                    4] / self.dt
                self.w_e_in_R5[i][0] = w_e_in_set_2_room5
            if self.tR5_air[i][0] >= action[2] and dQair_change_x5 > 0:
                self.HVAC5[i][0] = 1
                self.Q_c_sens_R5[i - 1][0] = self.Cp_air * self.Rou_air * self.V[4] * (
                        self.tR5_air[i - 1][0] - action[2]) / self.dt + dQair_change_x5
                self.tR5_air[i][0] = action[2]
        if 0 <= time <= 7:
            M1 = 0.15
            M3 = 0.2
            M5 = 0.35
            state = np.array([self.tR1_air[i][0], self.tR3_air[i][0], self.tR5_air[i][0]])
            state = self.compute_comfortable_terp(state, i)
            state = np.array([state[0], state[1], state[2], ta_out[i][0], RH[i][0], M1, M3, M5])
        if 8 <= time <= 20:
            M1 = 0.35
            M3 = 0.35
            M5 = - 0.1
            state = np.array([self.tR1_air[i][0], self.tR3_air[i][0], self.tR5_air[i][0]])
            state = self.compute_comfortable_terp(state, i)
            state = np.array([state[0], state[1], state[2], ta_out[i][0], RH[i][0], M1, M3, M5])
        if 21 <= time <= 24:
            M1 = 0.1
            M3 = 0.15
            M5 = 0.35
            state = np.array([self.tR1_air[i][0], self.tR3_air[i][0], self.tR5_air[i][0]])
            state = self.compute_comfortable_terp(state, i)
            state = np.array([state[0], state[1], state[2], ta_out[i][0], RH[i][0], M1, M3, M5])
        return state, self.Q_c_sens_R1[i - 1][0], self.Q_c_sens_R3[i - 1][0], self.Q_c_sens_R5[i - 1][0]
    # def comupte_total_load(self):
    #
    #     BSC = (np.sum(self.f[0:4, 0]) + np.sum(self.f[3:5, 3]) + np.sum(self.f[3:5, 1]) + self.f[4][2] + self.f[0][2] +
    #            np.sum(self.f[0:5, 4]) + np.sum(self.f[0:5, 5])) / np.sum(self.V[:])  # BUILDING SHAPE FACTOR
    #     U_value_w = 1 / (0.13 + np.sum(self.D_EXTwall / 1000. / self.Lamda_EXTwall) + 0.04)  # U-VALUE OF EXTERNAL WALLS
    #     Q_h_SENS = self.Q_hload_R1 + self.Q_hload_R3 + self.Q_hload_R5  # WINTER SENSIBLE HEATING LOAD
    #     Q_h_dehumid = self.Q_h_dehumid_R1 + self.Q_h_dehumid_R3 + self.Q_h_dehumid_R5  # DEHUMIDIFICATION LOAD IN WINTER
    #     Q_hload_tot = Q_h_SENS + Q_h_dehumid
    #     Dev = 24 * 6 * 3600 / self.dt
    #     Q_h_SENS_sum = np.sum(Q_h_SENS[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(
    #         self.f[0:5, 5])  # WINTER SENSIBLE HEATING LOAD IN kWh / m ^ 2
    #     q_hload = Q_h_SENS_sum * 1000 / 24 / 121
    #     Q_h_LATT_sum = np.sum(Q_h_dehumid[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0:5, 5])
    #     Q_hload_sum = Q_h_SENS_sum + Q_h_LATT_sum
    #     # %PRINT SUM OF HEATING LOAD PER MONTH
    #     # Q_h_Jan = (np.sum(Q_h_SENS[int(Dev - 1):int(888 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(Dev - 1): int(888 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
    #     # Q_h_Feb = (np.sum(Q_h_SENS[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Mar = (np.sum(Q_h_SENS[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(1560 * 3600 / self.dt) - 1: int(2304 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Apr = (np.sum(Q_h_SENS[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(2304 * 3600 / self.dt) - 1: int(3024 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_May = (np.sum(Q_h_SENS[int(Dev):int(889), 0]) + np.sum(
    #     #     Q_h_dehumid[int(Dev): int(889), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Jun = (np.sum(Q_h_SENS[int(Dev * 3600 / self.dt):int(865 * 3600 / self.dt), 0]) + np.sum(
    #     #     Q_h_dehumid[int(Dev * 3600 / self.dt):int(865 * 3600 / self.dt),
    #     #     0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Jul = (np.sum(Q_h_SENS[int(865 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0]) + np.sum(
    #     #     Q_h_dehumid[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_h_Aug = (np.sum(Q_h_SENS[int(2353 * 3600 / self.dt):int(3097 * 3600 / self.dt), 0]) + np.sum(
    #     #     Q_h_dehumid[int(2353 * 3600 / self.dt):int(3097 * 3600 / self.dt),
    #     #     0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Sep = (np.sum(Q_h_SENS[int(3097 * 3600 / self.dt):int(3817 * 3600 / self.dt), 0]) +
    #     #            np.sum(Q_h_dehumid[int(3097 * 3600 / self.dt): int(3817 * 3600 / self.dt),
    #     #                   0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Oct = (np.sum(Q_h_SENS[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(6696 * 3600 / self.dt) - 1: int(7440 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Nov = (np.sum(Q_h_SENS[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(7440 * 3600 / self.dt) - 1: int(8160 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_h_Dec = (np.sum(Q_h_SENS[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(8160 * 3600 / self.dt) - 1: int(8904 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     Q_h_MONTH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     Q_c_SENS = self.Q_c_sens_R1 + self.Q_c_sens_R3 + self.Q_c_sens_R5
    #     Q_c_LATENT = self.Q_c_latent_R1 + self.Q_c_latent_R3 + self.Q_c_latent_R5
    #     Q_cload_tot = Q_c_SENS + Q_c_LATENT
    #     Q_c_SENS_sum = np.sum(Q_c_SENS[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(f[0:5, 5])
    #     Q_c_LATENT_sum = np.sum(Q_c_LATENT[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(f[0:5, 5])
    #     Q_cload_sum = Q_c_SENS_sum + Q_c_LATENT_sum
    #     Qload_sum = Q_hload_sum + Q_cload_sum
    #     # Q_c_Jan = (np.sum(Q_c_SENS[int(Dev - 1):int(888 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(Dev - 1): int(888 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_c_Feb = (np.sum(Q_c_SENS[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0])) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_c_Mar = (np.sum(Q_c_SENS[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(1560 * 3600 / self.dt) - 1: int(2304 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_c_Apr = (np.sum(Q_c_SENS[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(2304 * 3600 / self.dt) - 1: int(3024 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_c_May = (np.sum(Q_c_SENS[int(Dev):int(889), 0]) +
    #     #            np.sum(Q_c_LATENT[int(Dev):int(889), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     Q_c_Jun = (np.sum(Q_c_SENS[int(Dev * 3600 / self.dt):int(865 * 3600 / self.dt), 0]) + np.sum(
    #         Q_c_LATENT[int(Dev * 3600 / self.dt):int(865 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
    #         self.f[0: 5, 5])
    #     Q_c_Jul = (np.sum(Q_c_SENS[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt), 0]) + np.sum(
    #         Q_c_LATENT[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
    #         self.f[0: 5, 5])
    #     Q_c_Aug = (np.sum(Q_c_SENS[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0]) + np.sum(
    #         Q_c_LATENT[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
    #         self.f[0: 5, 5])
    #     # Q_c_Sep = (np.sum(Q_c_SENS[int(3097 * 3600 / self.dt):int(3817 * 3600 / self.dt), 0]) +
    #     #            np.sum(Q_c_LATENT[int(3097 * 3600 / self.dt):int(3817 * 3600 / self.dt),
    #     #                   0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_c_Oct = (np.sum(Q_c_SENS[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(6696 * 3600 / self.dt) - 1: int(7440 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_c_Nov = (np.sum(Q_c_SENS[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(7440 * 3600 / self.dt) - 1: int(8160 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_c_Dec = (np.sum(Q_c_SENS[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(8160 * 3600 / self.dt) - 1: int(8904 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     Q_c_MONTH = [0, 0, 0, 0, 0, Q_c_Jun, Q_c_Jul, Q_c_Aug, 0, 0, 0, 0]
    #     # 通过窗户获得内部太阳热量的统计 - full time?
    #     # Q_solar_in_win = self.Q_solar_in_R1 + self.Q_solar_in_R2 + self.Q_solar_in_R3 + self.Q_solar_in_R4 + self.Q_solar_in_R5
    #     # Q_s_win_Jan = np.sum(Q_solar_in_win[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
    #     # Q_s_win_Feb = np.sum(Q_solar_in_win[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Mar = np.sum(Q_solar_in_win[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Apr = np.sum(Q_solar_in_win[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_May = np.sum(Q_solar_in_win[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Jun = np.sum(
    #     #     Q_solar_in_win[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_s_win_Jul = np.sum(Q_solar_in_win[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
    #     #                      0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Aug = np.sum(Q_solar_in_win[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
    #     #                      0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Sep = np.sum(Q_solar_in_win[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Oct = np.sum(Q_solar_in_win[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Nov = np.sum(Q_solar_in_win[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_Dec = np.sum(Q_solar_in_win[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_s_win_MONTH = [0, 0, 0, 0, 0, Q_s_win_Jun, Q_s_win_Jul, Q_s_win_Aug, 0, 0, 0, 0]
    #     # # 空气渗透热损益统计——本质上应分为加热和冷却考虑部分时间目标
    #     # Q_air_infiltra = self.Cp_air * self.Rou_air * self.n_air * (self.V[0] / 3600 * (ta_out - self.tR1_air) +
    #     #                                                             self.V[2] / 3600 * (ta_out - self.tR3_air) + self.V[
    #     #                                                                 4] / 3600 * (ta_out - self.tR5_air))
    #     # Q_air_inf_Jan = np.sum(Q_air_infiltra[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
    #     # Q_air_inf_Feb = np.sum(Q_air_infiltra[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Mar = np.sum(Q_air_infiltra[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Apr = np.sum(Q_air_infiltra[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_May = np.sum(Q_air_infiltra[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Jun = np.sum(
    #     #     Q_air_infiltra[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_air_inf_Jul = np.sum(Q_air_infiltra[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
    #     #                        0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Aug = np.sum(Q_air_infiltra[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
    #     #                        0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Sep = np.sum(Q_air_infiltra[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Oct = np.sum(Q_air_infiltra[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Nov = np.sum(Q_air_infiltra[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_Dec = np.sum(Q_air_infiltra[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_air_inf_MONTH = [0, 0, 0, 0, 0, Q_air_inf_Jun, Q_air_inf_Jul, Q_air_inf_Aug, 0, 0, 0, 0]
    #     # # 统计从人员、设备和照明中获得的内热
    #     # Q_int_hgain = self.Q_internal_rad_R1 + self.Q_internal_cov_R1 + self.Q_internal_rad_R3 + self.Q_internal_cov_R3 + self.Q_internal_rad_R5 + self.Q_internal_cov_R5
    #     # # Q_int_hg_Jan = np.sum(Q_int_hgain[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_int_hg_Feb = np.sum(Q_int_hgain[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_int_hg_Mar = np.sum(Q_int_hgain[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_int_hg_Apr = np.sum(Q_int_hgain[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_int_hg_May = np.sum(Q_int_hgain[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_int_hg_Jun = np.sum(
    #     #     Q_int_hgain[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_int_hg_Jul = np.sum(
    #     #     Q_int_hgain[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
    #     #     0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_int_hg_Aug = np.sum(
    #     #     Q_int_hgain[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
    #     #     0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # # Q_int_hg_Sep = np.sum(Q_int_hgain[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_int_hg_Oct = np.sum(Q_int_hgain[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_int_hg_Nov = np.sum(Q_int_hgain[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_int_hg_Dec = np.sum(Q_int_hgain[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_int_hg_MONTH = [0, 0, 0, 0, 0, Q_int_hg_Jun, Q_int_hg_Jul, Q_int_hg_Aug, 0, 0, 0, 0]
    #     # # 窗热传输的统计
    #     # Q_winTRANs_h = self.h_window_12 * self.f[0][6] * (
    #     #         self.tR1_window_in - self.tR1_window_out) * self.HEATING_BEDROOM + self.h_window_12 * \
    #     #                self.f[2][6] * (self.tR3_window_in - self.tR3_window_out) * self.HEATING_BEDROOM + self.h_window_12 * \
    #     #                self.f[4][6] * (self.tR5_window_in - self.tR5_window_out) * self.HEATING_SITTROOM
    #     # Q_winTRANs_c = self.h_window_12 * self.f[0][6] * (
    #     #         self.tR1_window_out - self.tR1_window_in) * self.COOLING_BEDROOM + self.h_window_12 * \
    #     #                self.f[2][6] * (self.tR3_window_out - self.tR3_window_in) * self.COOLING_BEDROOM + self.h_window_12 * \
    #     #                self.f[4][6] * (self.tR5_window_out - self.tR5_window_in) * self.COOLING_SITTROOM
    #     # Q_winTranh_Jan = np.sum(Q_winTRANs_h[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_winTranh_Feb = np.sum(Q_winTRANs_h[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / np.sum(self.f[0: 5, 5])
    #     # Q_winTranh_Mar = np.sum(Q_winTRANs_h[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_winTranh_Apr = np.sum(Q_winTRANs_h[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_winTranh_May = np.sum(Q_winTRANs_h[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_winTranh_Jun = np.sum(
    #     #     Q_winTRANs_h[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_winTranh_Jul = np.sum(
    #     #     Q_winTRANs_h[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
    #     #     0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_winTranh_Aug = np.sum(Q_winTRANs_h[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
    #     #                         0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranh_Sep = np.sum(Q_winTRANs_h[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranh_Oct = np.sum(Q_winTRANs_h[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranh_Nov = np.sum(Q_winTRANs_h[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranh_Dec = np.sum(Q_winTRANs_h[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_winTranh_MONTH = [0, 0, 0, 0, 0, Q_winTranh_Jun, Q_winTranh_Jul, Q_winTranh_Aug, 0, 0, 0, 0]
    #     # # Q_winTranc_Jan = sum(Q_winTRANs_c[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_Feb = sum(Q_winTRANs_c[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_Mar = sum(Q_winTRANs_c[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_Apr = sum(Q_winTRANs_c[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_May = sum(Q_winTRANs_c[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_winTranc_Jun = np.sum(
    #     #     Q_winTRANs_c[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_winTranc_Jul = np.sum(
    #     #     Q_winTRANs_c[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
    #     #     0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_winTranc_Aug = np.sum(Q_winTRANs_c[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
    #     #                      0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_Sep = sum(Q_winTRANs_c[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_Oct = sum(Q_winTRANs_c[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_Nov = sum(Q_winTRANs_c[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_winTranc_Dec = sum(Q_winTRANs_c[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_winTranc_MONTH = [0, 0, 0, 0, 0, Q_winTranc_Jun, Q_winTranc_Jul, Q_winTranc_Aug, 0, 0, 0, 0]
    #     # # 外墙吸收太阳辐射的统计
    #     # Q_ext_solarr = np.sum(self.f[0:3, 0]) * self.eb * q_solar_out_1 + np.sum(f[3:5, 1]) * self.eb * q_solar_out_2 + \
    #     #                (self.f[0][2] + self.f[4][2]) * self.eb * q_solar_out_3 + np.sum(
    #     #     self.f[2:4, 3]) * self.eb * q_solar_out_4
    #     # # Q_ext_sol_Jan = sum(Q_ext_solarr[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_Feb = sum(Q_ext_solarr[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_Mar = sum(Q_ext_solarr[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_Apr = sum(Q_ext_solarr[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_May = sum(Q_ext_solarr[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_ext_sol_Jun = np.sum(
    #     #     Q_ext_solarr[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_ext_sol_Jul = np.sum(
    #     #     Q_ext_solarr[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
    #     #     0]) / 1000 * self.dt / 3600 / np.sum(
    #     #     self.f[0: 5, 5])
    #     # Q_ext_sol_Aug = np.sum(Q_ext_solarr[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
    #     #                     0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_Sep = sum(Q_ext_solarr[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_Oct = sum(Q_ext_solarr[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_Nov = sum(Q_ext_solarr[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # # Q_ext_sol_Dec = sum(Q_ext_solarr[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
    #     # Q_ext_sol_MONTH = [0, 0, 0, 0, 0, Q_ext_sol_Jun, Q_ext_sol_Jul, Q_ext_sol_Aug, 0, 0, 0, 0]
    #     # outcome = [Mon[int(Dev - 1):int(T), 0], Day[int(Dev - 1):int(T), 0], nh[int(Dev - 1):int(T), 0], Min[int(Dev - 1):int(T), 0], ta_out[int(Dev - 1):int(T), 0],
    #     #            tR1_air[int(Dev - 1):int(T), 0], tR2_air[int(Dev - 1):int(T), 0], tR3_air[int(Dev - 1):int(T), 0], tR4_air[int(Dev - 1):int(T), 0],
    #     #            tR5_air[int(Dev - 1):int(T), 0], Q_hload_R1[int(Dev - 1):int(T), 0], Q_c_sens_R1[int(Dev - 1):int(T), 0], Q_hload_R3[int(Dev - 1):int(T), 0],
    #     #            Q_c_sens_R3[int(Dev - 1):int(T), 0], Q_hload_R5[int(Dev - 1):int(T), 0], Q_c_sens_R5[int(Dev - 1):int(T), 0], Q_hload_tot[int(Dev - 1):int(T), 0],
    #     #            Q_cload_tot[int(Dev - 1):int(T), 0], Q_h_dehumid[int(Dev - 1):int(T), 0], Q_c_LATENT[int(Dev - 1):int(T), 0]]
    #     summary = [D_insulation[0][0], U_value_w, 0, Q_h_SENS_sum, q_hload, Q_h_LATT_sum, Q_hload_sum, 0,
    #                Q_c_SENS_sum, Q_c_LATENT_sum, Q_cload_sum, Qload_sum]
    #
    #     load = np.array([summary, Q_h_MONTH, Q_c_MONTH]).T
    #     load_data = pd.DataFrame(load, columns=['summary', 'Q_h_MONTH', 'Q_c_MONTH'])
    #     temperatures = np.hstack((Mon[int(Dev):int(T), 0:1], Day[int(Dev):int(T), 0:1],
    #                               nh[int(Dev):int(T), 0:1], Min[int(Dev):int(T), 0:1],
    #                               ta_out[int(Dev):int(T), 0:1],
    #                               self.tR1_air[int(Dev):int(T), 0:1], self.tR2_air[int(Dev):int(T), 0:1],
    #                               self.tR3_air[int(Dev):int(T), 0:1], self.tR4_air[int(Dev):int(T), 0:1],
    #                               self.tR5_air[int(Dev):int(T), 0:1], self.Q_hload_R1[int(Dev):int(T), 0:1],
    #                               self.Q_c_sens_R1[int(Dev):int(T), 0:1], self.Q_hload_R3[int(Dev):int(T), 0:1],
    #                               self.Q_c_sens_R3[int(Dev):int(T), 0:1], self.Q_hload_R5[int(Dev):int(T), 0:1],
    #                               self.Q_c_sens_R5[int(Dev):int(T), 0:1], Q_hload_tot[int(Dev):int(T), 0:1],
    #                               Q_cload_tot[int(Dev):int(T), 0:1], Q_h_dehumid[int(Dev):int(T), 0:1],
    #                               Q_c_LATENT[int(Dev):int(T), 0:1]))
    #     temperatures_data = pd.DataFrame(temperatures)
    #     load_data.to_csv("cooling_load_data_DDPG1_fixed.csv")
    #     temperatures_data.to_csv("temperatures_data_cooling_DDPG1_fixed.csv")
    def comupte_total_load(self):

        BSC = (np.sum(self.f[0:4, 0]) + np.sum(self.f[3:5, 3]) + np.sum(self.f[3:5, 1]) + self.f[4][2] + self.f[0][2] +
               np.sum(self.f[0:5, 4]) + np.sum(self.f[0:5, 5])) / np.sum(self.V[:])  # BUILDING SHAPE FACTOR
        U_value_w = 1 / (0.13 + np.sum(self.D_EXTwall / 1000. / self.Lamda_EXTwall) + 0.04)  # U-VALUE OF EXTERNAL WALLS
        Q_h_SENS = self.Q_hload_R1 + self.Q_hload_R3 + self.Q_hload_R5  # WINTER SENSIBLE HEATING LOAD
        Q_h_dehumid = self.Q_h_dehumid_R1 + self.Q_h_dehumid_R3 + self.Q_h_dehumid_R5  # DEHUMIDIFICATION LOAD IN WINTER
        Q_hload_tot = Q_h_SENS + Q_h_dehumid
        Dev = 24 * 6 * 3600 / self.dt
        Q_h_SENS_sum = np.sum(Q_h_SENS[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(
            self.f[0:5, 5])  # WINTER SENSIBLE HEATING LOAD IN kWh / m ^ 2
        q_hload = Q_h_SENS_sum * 1000 / 24 / 121
        Q_h_LATT_sum = np.sum(Q_h_dehumid[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0:5, 5])
        Q_hload_sum = Q_h_SENS_sum + Q_h_LATT_sum
        # %PRINT SUM OF HEATING LOAD PER MONTH
        # Q_h_Jan = (np.sum(Q_h_SENS[int(Dev - 1):int(888 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(Dev - 1): int(888 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
        # Q_h_Feb = (np.sum(Q_h_SENS[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_h_Mar = (np.sum(Q_h_SENS[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(1560 * 3600 / self.dt) - 1: int(2304 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_h_Apr = (np.sum(Q_h_SENS[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(2304 * 3600 / self.dt) - 1: int(3024 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_May = (np.sum(Q_h_SENS[int(Dev):int(889), 0]) + np.sum(
            Q_h_dehumid[int(Dev): int(889), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Jun = (np.sum(Q_h_SENS[int(889 * 3600 / self.dt):int(1609 * 3600 / self.dt), 0]) + np.sum(
            Q_h_dehumid[int(889 * 3600 / self.dt):int(1609 * 3600 / self.dt),
            0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Jul = (np.sum(Q_h_SENS[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0]) + np.sum(
            Q_h_dehumid[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
            self.f[0: 5, 5])
        Q_h_Aug = (np.sum(Q_h_SENS[int(2353 * 3600 / self.dt):int(3097 * 3600 / self.dt), 0]) + np.sum(
            Q_h_dehumid[int(2353 * 3600 / self.dt):int(3097 * 3600 / self.dt),
            0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_Sep = (np.sum(Q_h_SENS[int(3097 * 3600 / self.dt):int(3817 * 3600 / self.dt), 0]) +
                   np.sum(Q_h_dehumid[int(3097 * 3600 / self.dt): int(3817 * 3600 / self.dt),
                          0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_h_Oct = (np.sum(Q_h_SENS[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(6696 * 3600 / self.dt) - 1: int(7440 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_h_Nov = (np.sum(Q_h_SENS[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(7440 * 3600 / self.dt) - 1: int(8160 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_h_Dec = (np.sum(Q_h_SENS[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) + np.sum(Q_h_dehumid[int(8160 * 3600 / self.dt) - 1: int(8904 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_h_MONTH = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Q_c_SENS = self.Q_c_sens_R1 + self.Q_c_sens_R3 + self.Q_c_sens_R5
        Q_c_LATENT = self.Q_c_latent_R1 + self.Q_c_latent_R3 + self.Q_c_latent_R5
        Q_cload_tot = Q_c_SENS + Q_c_LATENT
        Q_c_SENS_sum = np.sum(Q_c_SENS[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(f[0:5, 5])
        Q_c_LATENT_sum = np.sum(Q_c_LATENT[int(Dev):, 0]) / 1000 * self.dt / 3600 / np.sum(f[0:5, 5])
        Q_cload_sum = Q_c_SENS_sum + Q_c_LATENT_sum
        Qload_sum = Q_hload_sum + Q_cload_sum
        # Q_c_Jan = (np.sum(Q_c_SENS[int(Dev - 1):int(888 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(Dev - 1): int(888 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_c_Feb = (np.sum(Q_c_SENS[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0])) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_c_Mar = (np.sum(Q_c_SENS[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(1560 * 3600 / self.dt) - 1: int(2304 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_c_Apr = (np.sum(Q_c_SENS[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(2304 * 3600 / self.dt) - 1: int(3024 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_May = (np.sum(Q_c_SENS[int(Dev):int(889), 0]) +
                   np.sum(Q_c_LATENT[int(Dev):int(889), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_Jun = (np.sum(Q_c_SENS[int(889 * 3600 / self.dt):int(1609 * 3600 / self.dt), 0]) + np.sum(
            Q_c_LATENT[int(889 * 3600 / self.dt):int(1609 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
            self.f[0: 5, 5])
        Q_c_Jul = (np.sum(Q_c_SENS[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0]) + np.sum(
            Q_c_LATENT[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
            self.f[0: 5, 5])
        Q_c_Aug = (np.sum(Q_c_SENS[int(2353 * 3600 / self.dt):int(3097 * 3600 / self.dt), 0]) + np.sum(
            Q_c_LATENT[int(2353 * 3600 / self.dt):int(3097 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(
            self.f[0: 5, 5])
        Q_c_Sep = (np.sum(Q_c_SENS[int(3097 * 3600 / self.dt):int(3817 * 3600 / self.dt), 0]) +
                   np.sum(Q_c_LATENT[int(3097 * 3600 / self.dt):int(3817 * 3600 / self.dt),
                          0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_c_Oct = (np.sum(Q_c_SENS[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(6696 * 3600 / self.dt) - 1: int(7440 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_c_Nov = (np.sum(Q_c_SENS[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(7440 * 3600 / self.dt) - 1: int(8160 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_c_Dec = (np.sum(Q_c_SENS[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) + np.sum(Q_c_LATENT[int(8160 * 3600 / self.dt) - 1: int(8904 * 3600 / self.dt), 0])) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        Q_c_MONTH = [0, 0, 0, 0, Q_c_May, Q_c_Jun, Q_c_Jul, Q_c_Aug, Q_c_Sep, 0, 0, 0]
        # 通过窗户获得内部太阳热量的统计 - full time?
        # Q_solar_in_win = self.Q_solar_in_R1 + self.Q_solar_in_R2 + self.Q_solar_in_R3 + self.Q_solar_in_R4 + self.Q_solar_in_R5
        # Q_s_win_Jan = np.sum(Q_solar_in_win[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
        # Q_s_win_Feb = np.sum(Q_solar_in_win[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Mar = np.sum(Q_solar_in_win[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Apr = np.sum(Q_solar_in_win[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_May = np.sum(Q_solar_in_win[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Jun = np.sum(
        #     Q_solar_in_win[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_s_win_Jul = np.sum(Q_solar_in_win[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
        #                      0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Aug = np.sum(Q_solar_in_win[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
        #                      0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Sep = np.sum(Q_solar_in_win[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Oct = np.sum(Q_solar_in_win[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Nov = np.sum(Q_solar_in_win[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_Dec = np.sum(Q_solar_in_win[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_s_win_MONTH = [0, 0, 0, 0, 0, Q_s_win_Jun, Q_s_win_Jul, Q_s_win_Aug, 0, 0, 0, 0]
        # # 空气渗透热损益统计——本质上应分为加热和冷却考虑部分时间目标
        # Q_air_infiltra = self.Cp_air * self.Rou_air * self.n_air * (self.V[0] / 3600 * (ta_out - self.tR1_air) +
        #                                                             self.V[2] / 3600 * (ta_out - self.tR3_air) + self.V[
        #                                                                 4] / 3600 * (ta_out - self.tR5_air))
        # Q_air_inf_Jan = np.sum(Q_air_infiltra[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(f[0: 5, 5])
        # Q_air_inf_Feb = np.sum(Q_air_infiltra[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Mar = np.sum(Q_air_infiltra[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Apr = np.sum(Q_air_infiltra[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_May = np.sum(Q_air_infiltra[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Jun = np.sum(
        #     Q_air_infiltra[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_air_inf_Jul = np.sum(Q_air_infiltra[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
        #                        0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Aug = np.sum(Q_air_infiltra[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
        #                        0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Sep = np.sum(Q_air_infiltra[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Oct = np.sum(Q_air_infiltra[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Nov = np.sum(Q_air_infiltra[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_Dec = np.sum(Q_air_infiltra[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_air_inf_MONTH = [0, 0, 0, 0, 0, Q_air_inf_Jun, Q_air_inf_Jul, Q_air_inf_Aug, 0, 0, 0, 0]
        # # 统计从人员、设备和照明中获得的内热
        # Q_int_hgain = self.Q_internal_rad_R1 + self.Q_internal_cov_R1 + self.Q_internal_rad_R3 + self.Q_internal_cov_R3 + self.Q_internal_rad_R5 + self.Q_internal_cov_R5
        # # Q_int_hg_Jan = np.sum(Q_int_hgain[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_int_hg_Feb = np.sum(Q_int_hgain[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_int_hg_Mar = np.sum(Q_int_hgain[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_int_hg_Apr = np.sum(Q_int_hgain[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_int_hg_May = np.sum(Q_int_hgain[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_int_hg_Jun = np.sum(
        #     Q_int_hgain[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_int_hg_Jul = np.sum(
        #     Q_int_hgain[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
        #     0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_int_hg_Aug = np.sum(
        #     Q_int_hgain[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
        #     0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # # Q_int_hg_Sep = np.sum(Q_int_hgain[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_int_hg_Oct = np.sum(Q_int_hgain[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_int_hg_Nov = np.sum(Q_int_hgain[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_int_hg_Dec = np.sum(Q_int_hgain[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_int_hg_MONTH = [0, 0, 0, 0, 0, Q_int_hg_Jun, Q_int_hg_Jul, Q_int_hg_Aug, 0, 0, 0, 0]
        # # 窗热传输的统计
        # Q_winTRANs_h = self.h_window_12 * self.f[0][6] * (
        #         self.tR1_window_in - self.tR1_window_out) * self.HEATING_BEDROOM + self.h_window_12 * \
        #                self.f[2][6] * (self.tR3_window_in - self.tR3_window_out) * self.HEATING_BEDROOM + self.h_window_12 * \
        #                self.f[4][6] * (self.tR5_window_in - self.tR5_window_out) * self.HEATING_SITTROOM
        # Q_winTRANs_c = self.h_window_12 * self.f[0][6] * (
        #         self.tR1_window_out - self.tR1_window_in) * self.COOLING_BEDROOM + self.h_window_12 * \
        #                self.f[2][6] * (self.tR3_window_out - self.tR3_window_in) * self.COOLING_BEDROOM + self.h_window_12 * \
        #                self.f[4][6] * (self.tR5_window_out - self.tR5_window_in) * self.COOLING_SITTROOM
        # Q_winTranh_Jan = np.sum(Q_winTRANs_h[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_winTranh_Feb = np.sum(Q_winTRANs_h[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / np.sum(self.f[0: 5, 5])
        # Q_winTranh_Mar = np.sum(Q_winTRANs_h[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_winTranh_Apr = np.sum(Q_winTRANs_h[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_winTranh_May = np.sum(Q_winTRANs_h[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_winTranh_Jun = np.sum(
        #     Q_winTRANs_h[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_winTranh_Jul = np.sum(
        #     Q_winTRANs_h[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
        #     0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_winTranh_Aug = np.sum(Q_winTRANs_h[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
        #                         0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranh_Sep = np.sum(Q_winTRANs_h[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranh_Oct = np.sum(Q_winTRANs_h[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranh_Nov = np.sum(Q_winTRANs_h[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranh_Dec = np.sum(Q_winTRANs_h[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_winTranh_MONTH = [0, 0, 0, 0, 0, Q_winTranh_Jun, Q_winTranh_Jul, Q_winTranh_Aug, 0, 0, 0, 0]
        # # Q_winTranc_Jan = sum(Q_winTRANs_c[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_Feb = sum(Q_winTRANs_c[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_Mar = sum(Q_winTRANs_c[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_Apr = sum(Q_winTRANs_c[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_May = sum(Q_winTRANs_c[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_winTranc_Jun = np.sum(
        #     Q_winTRANs_c[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_winTranc_Jul = np.sum(
        #     Q_winTRANs_c[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
        #     0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_winTranc_Aug = np.sum(Q_winTRANs_c[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
        #                      0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_Sep = sum(Q_winTRANs_c[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_Oct = sum(Q_winTRANs_c[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_Nov = sum(Q_winTRANs_c[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_winTranc_Dec = sum(Q_winTRANs_c[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_winTranc_MONTH = [0, 0, 0, 0, 0, Q_winTranc_Jun, Q_winTranc_Jul, Q_winTranc_Aug, 0, 0, 0, 0]
        # # 外墙吸收太阳辐射的统计
        # Q_ext_solarr = np.sum(self.f[0:3, 0]) * self.eb * q_solar_out_1 + np.sum(f[3:5, 1]) * self.eb * q_solar_out_2 + \
        #                (self.f[0][2] + self.f[4][2]) * self.eb * q_solar_out_3 + np.sum(
        #     self.f[2:4, 3]) * self.eb * q_solar_out_4
        # # Q_ext_sol_Jan = sum(Q_ext_solarr[int(Dev - 1):int(888 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_Feb = sum(Q_ext_solarr[int(888 * 3600 / self.dt) - 1:int(1560 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_Mar = sum(Q_ext_solarr[int(1560 * 3600 / self.dt) - 1:int(2304 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_Apr = sum(Q_ext_solarr[int(2304 * 3600 / self.dt) - 1:int(3024 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_May = sum(Q_ext_solarr[int(3024 * 3600 / self.dt) - 1:int(3768 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_ext_sol_Jun = np.sum(
        #     Q_ext_solarr[int(Dev):int(865 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_ext_sol_Jul = np.sum(
        #     Q_ext_solarr[int(865 * 3600 / self.dt):int(1609 * 3600 / self.dt),
        #     0]) / 1000 * self.dt / 3600 / np.sum(
        #     self.f[0: 5, 5])
        # Q_ext_sol_Aug = np.sum(Q_ext_solarr[int(1609 * 3600 / self.dt):int(2353 * 3600 / self.dt),
        #                     0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_Sep = sum(Q_ext_solarr[int(5976 * 3600 / self.dt) - 1:int(6696 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_Oct = sum(Q_ext_solarr[int(6696 * 3600 / self.dt) - 1:int(7440 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_Nov = sum(Q_ext_solarr[int(7440 * 3600 / self.dt) - 1:int(8160 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # # Q_ext_sol_Dec = sum(Q_ext_solarr[int(8160 * 3600 / self.dt) - 1:int(8904 * 3600 / self.dt), 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0: 5, 5])
        # Q_ext_sol_MONTH = [0, 0, 0, 0, 0, Q_ext_sol_Jun, Q_ext_sol_Jul, Q_ext_sol_Aug, 0, 0, 0, 0]
        # outcome = [Mon[int(Dev - 1):int(T), 0], Day[int(Dev - 1):int(T), 0], nh[int(Dev - 1):int(T), 0], Min[int(Dev - 1):int(T), 0], ta_out[int(Dev - 1):int(T), 0],
        #            tR1_air[int(Dev - 1):int(T), 0], tR2_air[int(Dev - 1):int(T), 0], tR3_air[int(Dev - 1):int(T), 0], tR4_air[int(Dev - 1):int(T), 0],
        #            tR5_air[int(Dev - 1):int(T), 0], Q_hload_R1[int(Dev - 1):int(T), 0], Q_c_sens_R1[int(Dev - 1):int(T), 0], Q_hload_R3[int(Dev - 1):int(T), 0],
        #            Q_c_sens_R3[int(Dev - 1):int(T), 0], Q_hload_R5[int(Dev - 1):int(T), 0], Q_c_sens_R5[int(Dev - 1):int(T), 0], Q_hload_tot[int(Dev - 1):int(T), 0],
        #            Q_cload_tot[int(Dev - 1):int(T), 0], Q_h_dehumid[int(Dev - 1):int(T), 0], Q_c_LATENT[int(Dev - 1):int(T), 0]]
        summary = [D_insulation[0][0], U_value_w, 0, Q_h_SENS_sum, q_hload, Q_h_LATT_sum, Q_hload_sum, 0,
                   Q_c_SENS_sum, Q_c_LATENT_sum, Q_cload_sum, Qload_sum]

        load = np.array([summary, Q_h_MONTH, Q_c_MONTH]).T
        load_data = pd.DataFrame(load, columns=['summary', 'Q_h_MONTH', 'Q_c_MONTH'])
        temperatures = np.hstack((Mon[int(Dev):int(T), 0:1], Day[int(Dev):int(T), 0:1],
                                  nh[int(Dev):int(T), 0:1], Min[int(Dev):int(T), 0:1],
                                  ta_out[int(Dev):int(T), 0:1],
                                  self.tR1_air[int(Dev):int(T), 0:1], self.tR2_air[int(Dev):int(T), 0:1],
                                  self.tR3_air[int(Dev):int(T), 0:1], self.tR4_air[int(Dev):int(T), 0:1],
                                  self.tR5_air[int(Dev):int(T), 0:1], self.Q_hload_R1[int(Dev):int(T), 0:1],
                                  self.Q_c_sens_R1[int(Dev):int(T), 0:1], self.Q_hload_R3[int(Dev):int(T), 0:1],
                                  self.Q_c_sens_R3[int(Dev):int(T), 0:1], self.Q_hload_R5[int(Dev):int(T), 0:1],
                                  self.Q_c_sens_R5[int(Dev):int(T), 0:1], Q_hload_tot[int(Dev):int(T), 0:1],
                                  Q_cload_tot[int(Dev):int(T), 0:1], Q_h_dehumid[int(Dev):int(T), 0:1],
                                  Q_c_LATENT[int(Dev):int(T), 0:1]))
        temperatures_data = pd.DataFrame(temperatures)
        load_data.to_csv("cooling_load_data_DDPG1_fixed.csv")
        temperatures_data.to_csv("temperatures_data_cooling_DDPG1_fixed.csv")

    def reset(self):
        self.HVAC1 = np.zeros((T, 1))
        self.HVAC2 = np.zeros((T, 1))
        self.HVAC3 = np.zeros((T, 1))
        self.terp_vio = 0
        self.t1 = 10 * np.ones((self.NN_main, int(T)))
        self.tR1_wall1_in = ta_out.copy()
        self.tR1_wall2_in = ta_out.copy()
        self.tR1_wall3_in = ta_out.copy()
        self.tR1_wall4_in = ta_out.copy()
        self.tR1_ceil_in = ta_out.copy()
        self.tR1_floor_in = ta_out.copy()
        self.tR1_window_in = ta_out.copy()
        self.tR1_air = ta_out.copy()
        self.t2 = 10 * np.ones((self.NN_main2, int(T)))
        self.tR2_wall1_in = ta_out.copy()
        self.tR2_wall2_in = ta_out.copy()
        self.tR2_wall3_in = ta_out.copy()
        self.tR2_wall4_in = ta_out.copy()
        self.tR2_ceil_in = ta_out.copy()
        self.tR2_floor_in = ta_out.copy()
        self.tR2_window_in = ta_out.copy()
        self.tR2_air = ta_out.copy()
        self.t3 = 10 * np.ones((self.NN_main3, int(T)))
        self.tR3_wall1_in = ta_out.copy()
        self.tR3_wall2_in = ta_out.copy()
        self.tR3_wall3_in = ta_out.copy()
        self.tR3_wall4_in = ta_out.copy()
        self.tR3_ceil_in = ta_out.copy()
        self.tR3_floor_in = ta_out.copy()
        self.tR3_window_in = ta_out.copy()
        self.tR3_air = ta_out.copy()
        self.t4 = 10 * np.ones((self.NN_main4, int(T)))
        self.tR4_wall1_in = ta_out.copy()
        self.tR4_wall2_in = ta_out.copy()
        self.tR4_wall3_in = ta_out.copy()
        self.tR4_wall4_in = ta_out.copy()
        self.tR4_ceil_in = ta_out.copy()
        self.tR4_floor_in = ta_out.copy()
        self.tR4_window_in = ta_out.copy()
        self.tR4_air = ta_out.copy()
        self.t5 = 10 * np.ones((self.NN_main5, int(T)))
        self.tR5_wall1_in = ta_out.copy()
        self.tR5_wall2_in = ta_out.copy()
        self.tR5_wall3_in = ta_out.copy()
        self.tR5_wall4_in = ta_out.copy()
        self.tR5_ceil_in = ta_out.copy()
        self.tR5_floor_in = ta_out.copy()
        self.tR5_window_in = ta_out.copy()
        self.tR5_air = ta_out.copy()
        self.tR1_window_out = ta_out.copy()
        self.tR2_window_out = ta_out.copy()
        self.tR3_window_out = ta_out.copy()
        self.tR4_window_out = ta_out.copy()
        self.tR5_window_out = ta_out.copy()
        self.Q_hload_R1 = np.zeros((int(T), 1))  # heating load of room1
        self.Q_hload_R3 = np.zeros((int(T), 1))  # heating load of room3
        self.Q_hload_R5 = np.zeros((int(T), 1))  # heating load of room5
        self.Q_h_dehumid_R1 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 1(1室潜热除湿)
        self.Q_h_dehumid_R3 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 3
        self.Q_h_dehumid_R5 = np.zeros((int(T), 1))  # dehumidification of latent heat for room 5
        self.Q_c_sens_R1 = np.zeros((int(T), 1))
        self.Q_c_latent_R1 = np.zeros(
            (int(T), 1))  # sensible and latent cooling loads of room 1,respectively (1室的显冷负荷和潜冷负荷)
        self.Q_c_sens_R3 = np.zeros((int(T), 1))
        self.Q_c_latent_R3 = np.zeros(
            (int(T), 1))  # sensible and latent cooling loads of room 3,respectively (3室的显冷负荷和潜冷负荷)
        self.Q_c_sens_R5 = np.zeros((int(T), 1))
        self.Q_c_latent_R5 = np.zeros(
            (int(T), 1))  # sensible and latent cooling loads of room 5,respectively (5室的显冷负荷和潜冷负荷)
        self.w_e_in_R1 = np.zeros((int(T), 1))
        self.w_e_in_R1[0][0] = 6
        self.w_e_in_R3 = np.zeros((int(T), 1))
        self.w_e_in_R3[0][0] = 6
        self.w_e_in_R5 = np.zeros((int(T), 1))
        self.w_e_in_R5[0][0] = 6
        self.q_heatemss_p = 70 + 60  # 居住者对房间的感热和潜热散发量，70瓦特/人
        self.m_w_gp = 50 / 3600 * self.dt  # 人的吸湿量(增湿量)，50克/(h人)
        self.n_p = 3  # 公寓里住了三个人
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
        self.OCCUPIED_BED = np.zeros((int(T), 1))  # BEDROOM OCCUPATION
        self.LIGHT_ON_BED = np.zeros((int(T), 1))  # LIGHTS & EQUIPMENT ON
        self.OCCUPIED_SITT = np.zeros((int(T), 1))  # SITTING ROOM OCCUPATION
        self.LIGHT_ON_SITT = np.zeros((int(T), 1))
        self.HEATING_BEDROOM = np.zeros((int(T), 1))
        self.HEATING_SITTROOM = np.zeros((int(T), 1))
        self.COOLING_BEDROOM = np.zeros((int(T), 1))
        self.COOLING_SITTROOM = np.zeros((int(T), 1))
        self.Q_internal_rad_R1 = np.zeros((int(T), 1))
        self.Q_internal_cov_R1 = np.zeros((int(T), 1))
        self.Q_internal_rad_R3 = np.zeros((int(T), 1))
        self.Q_internal_cov_R3 = np.zeros((int(T), 1))
        self.Q_internal_rad_R5 = np.zeros((int(T), 1))
        self.Q_internal_cov_R5 = np.zeros((int(T), 1))
        self.Q_INTrad_wall1_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_wall2_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_wall3_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_wall4_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_ceil_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_floor_R1 = np.zeros((1, int(T)))
        self.Q_INTrad_win_R1 = np.zeros((1, int(T)))
        self.Q_conv1 = np.zeros((1, int(T)))
        self.Q_hvac1 = np.zeros((1, int(T)))
        self.Q_conv2 = np.zeros((1, int(T)))
        self.Q_hvac2 = np.zeros((1, int(T)))
        self.Q_INTrad_wall1_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_wall2_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_wall3_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_wall4_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_ceil_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_floor_R3 = np.zeros((1, int(T)))
        self.Q_INTrad_win_R3 = np.zeros((1, int(T)))
        self.Q_conv3 = np.zeros((1, int(T)))
        self.Q_hvac3 = np.zeros((1, int(T)))
        self.Q_conv4 = np.zeros((1, int(T)))
        self.Q_hvac4 = np.zeros((1, int(T)))
        self.Q_INTrad_wall1_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_wall2_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_wall3_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_wall4_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_ceil_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_floor_R5 = np.zeros((1, int(T)))
        self.Q_INTrad_win_R5 = np.zeros((1, int(T)))
        self.Q_conv5 = np.zeros((1, int(T)))
        self.Q_hvac5 = np.zeros((1, int(T)))
        # Q_internal_rad_R1 = 0.03 * self.f[0][5] * 70.0 * 0.50 + 4.3 * self.f[0][5] * 0.20 + 6 * self.f[0][5] * 0.52
        # Q_internal_cov_R1 = 0.03 * self.f[0][5] * 70.0 * 0.50 + 4.3 * self.f[0][5] * 0.80 + 6 * self.f[0][5] * 0.48
        self.Q_internal_rad_R2 = 0  # NO INTERNAL HEAT GAIN FOR TOILET
        self.Q_internal_cov_R2 = 0  # NO INTERNAL HEAT GAIN FOR TOILET
        # Q_internal_rad_R3 = 0.03 * self.f[2][4] * 70.0 * 0.50 + 4.3 * self.f[2][4] * 0.20 + 6 * self.f[2][4] * 0.52
        # Q_internal_cov_R3 = 0.03 * self.f[2][4] * 70.0 * 0.50 + 4.3 * self.f[2][4] * 0.80 + 6 * self.f[2][4] * 0.48
        # Q_internal_rad_R5 = 0.03 * self.f[4][5] * 70.0 * 0.50 + 4.3 * self.f[4][5] * 0.20 + 6 * self.f[4][5] * 0.52
        # Q_internal_cov_R5 = 0.03 * self.f[4][5] * 70.0 * 0.50 + 4.3 * self.f[4][5] * 0.80 + 6 * self.f[4][5] * 0.48
        self.Q_solar_in_R1 = np.zeros((1, int(T)))
        self.Q_solar_in_R2 = np.zeros((1, int(T)))
        self.Q_solar_in_R3 = np.zeros((1, int(T)))
        self.Q_solar_in_R4 = np.zeros((1, int(T)))
        self.Q_solar_in_R5 = np.zeros((1, int(T)))
        self.Q_solar_in_wall = np.zeros((5, int(T)))
        self.Q_solar_in_ceil = np.zeros((5, int(T)))
        self.Q_solar_in_floor = np.zeros((5, int(T)))

    def q_load(self):
        Q_c_SENS_ = self.Q_c_sens_R1 + self.Q_c_sens_R3 + self.Q_c_sens_R5
        Q_c_LATENT_ = self.Q_c_latent_R1 + self.Q_c_latent_R3 + self.Q_c_latent_R5
        Q_cload_tot_ = Q_c_SENS_ + Q_c_LATENT_
        Q_c_SENS_sum_ = np.sum(Q_c_SENS_[144:, 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0:5, 5])
        # Q_c_LATENT_sum_ = np.sum(Q_c_LATENT_[144:, 0]) / 1000 * self.dt / 3600 / np.sum(self.f[0:5, 5])
        # Q_cload_sum_ = Q_c_SENS_sum_ + Q_c_LATENT_sum_
        return Q_c_SENS_sum_

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
             dx_ceil, dx_floor, N_EXTwall, N_INTwall, N_ceil, N_floor, V, Rou_air, float(Cp_air), Lamda_EXTwall, D_EXTwall,
             h_out, h_in_wall, hr, Lamda_INTwall, N_window, Lamda_ceil, h_in_ceil, h_in_floor, h_window_12, int(dt), f,
             f_tol, eb, Ta_targ_min, Ta_targ_max, Lamda_floor, n_air, HC_mode=1)

# model.fixed_compute()
# model.comupte_total_load()


