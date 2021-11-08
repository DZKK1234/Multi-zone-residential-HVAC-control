import numpy as np
import pandas as pd
from scipy.integrate import odeint
from function_building import *
import math

r = 28.22
longitude = 112.92
p_tot = 1008
dt = 3600
a = pd.read_csv("five_to_nine_cooling_weather.csv")
changsha_tmy = np.array(a)
nhour = len(changsha_tmy)
T = int(nhour * 3600 / dt)
ta_out = np.zeros((int(T), 1))
q_solar_out_1 = np.zeros((int(T), 1))
q_solar_out_2 = np.zeros((int(T), 1))
q_solar_out_3 = np.zeros((int(T), 1))
q_solar_out_4 = np.zeros((int(T), 1))
q_solar_out_5 = np.zeros((int(T), 1))
Mon = np.zeros((int(T), 1))
Day = np.zeros((int(T), 1))
Hour = np.zeros((int(T), 1))
Min = np.zeros((int(T), 1))
RH = np.zeros((int(T), 1))
w_e_out = np.zeros((int(T), 1))
nh = np.zeros((int(T), 1))
for i in range(int(T - 3600 / dt)):
    nx = int(i * dt / 3600)
    ta_out[i][0] = changsha_tmy[nx][4] + (i * dt / 3600 + 1 - nx) * (changsha_tmy[nx + 1][4] - changsha_tmy[nx][4])
    q_solar_out_1[i][0] = changsha_tmy[nx][6] + (i * dt / 3600 + 1 - nx) * \
                          (changsha_tmy[nx + 1][6] - changsha_tmy[nx][
                              6])  # EXTERNAL WALL SOLAR RADIATION ON EAST WALL, W / m2
    q_solar_out_2[i][0] = changsha_tmy[nx][8] + (i * dt / 3600 + 1 - nx) * \
                          (changsha_tmy[nx + 1][8] - changsha_tmy[nx][
                              8])  # EXTERNAL WALL SOLAR RADIATION ON WEST WALL, W / m2
    q_solar_out_3[i][0] = changsha_tmy[nx][7] + (i * dt / 3600 + 1 - nx) * \
                          (changsha_tmy[nx + 1][7] - changsha_tmy[nx][
                              7])  # EXTERNAL WALL SOLAR RADIATION ON SOUTH WALL, W / m2
    q_solar_out_4[i][0] = changsha_tmy[nx][9] + (i * dt / 3600 + 1 - nx) * \
                          (changsha_tmy[nx + 1][9] - changsha_tmy[nx][
                              9])  # EXTERNAL WALL SOLAR RADIATION ON NORTH WALL, W / m2
    q_solar_out_5[i][0] = changsha_tmy[nx][5] + (i * dt / 3600 + 1 - nx) * \
                          (changsha_tmy[nx + 1][5] - changsha_tmy[nx][
                              5])  # EXTERNAL SOLAR RADIATION OF ROOF(HORIZONTAL), W / m2
    RH[i][0] = changsha_tmy[nx][10] + (i * dt / 3600 + 1 - nx) * \
               (changsha_tmy[nx + 1][10] - changsha_tmy[nx][10])  # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
    nh[i][0] = changsha_tmy[nx][2] + (i * dt / 3600 + 1 - nx) * \
               (changsha_tmy[nx + 1][2] - changsha_tmy[nx][
                   2])  # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR(用于安排乘员行为的每日小时数)
    Mon[i][0] = changsha_tmy[nx][0]  # MONTHS
    Day[i][0] = changsha_tmy[nx][1]  # DATE NUMBER
    Hour[i][0] = changsha_tmy[0][2] + i * dt / 3600  # HOURLY TIME
    Min[i][0] = changsha_tmy[nx][3]
q_solar_out_6 = 0
for i in range(int(T - 3600 / dt), int(T)):
    nx = int(i * dt / 3600)
    ta_out[i][0] = changsha_tmy[nx][4]
    q_solar_out_1[i][0] = changsha_tmy[nx][6]  # EXTERNAL WALL SOLAR RADIATION OF EAST WALL, W / m2
    q_solar_out_2[i][0] = changsha_tmy[nx][8]  # EXTERNAL WALL SOLAR RADIATION OF WEST WALL, W / m2
    q_solar_out_3[i][0] = changsha_tmy[nx][7]  # EXTERNAL WALL SOLAR RADIATION OF SOUTH WALL, W / m2
    q_solar_out_4[i][0] = changsha_tmy[nx][9]  # EXTERNAL WALL SOLAR RADIATION OF NORTH WALL, W / m2
    q_solar_out_5[i][0] = changsha_tmy[nx][5]  # EXTERNAL WALL SOLAR RADIATION OF FLOOR, W / m2
    RH[i][0] = changsha_tmy[nx][10]  # OUTDOOR RELATIVE HUMIDITY, 0 - 100( %)
    nh[i][0] = changsha_tmy[nx][2]  # DAILY HOUR NUMBER USED FOR SCHEDULING OCCUPANT BEHAVIOUR
    Mon[i][0] = changsha_tmy[nx][0]  # MONTHS
    Day[i][0] = changsha_tmy[nx][1]  # DATE NUMBER IN A YEAR
    Hour[i][0] = changsha_tmy[0][2] + i * dt / 3600  # HOURLY NUMBER
    Min[i][0] = changsha_tmy[nx][3]  # MINUTES
mon_nd = [0, 30, 61, 92]  # 6-8months
n0 = mon_nd[int(Mon[0][0] - 5)] + Day[0][0]
