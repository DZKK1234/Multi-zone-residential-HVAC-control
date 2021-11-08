import numpy as np
from scipy.integrate import odeint
# long wave radiation heat transfer coefficients
# 长波辐射传热系数
def h_lwrr(fi, f_tol, e_brick, e_glass):

    hr = np.zeros((5, 7))
    for i in range(0, 5):
        for j in range(0, 6):

            hr[i][j] = e_brick * (fi[i][j] / f_tol[i][0]) * 5.67 * 10**(-8) * \
                       (2 * np.power(293, 2) * 2 * 293)

        hr[i][6] = e_glass  * (fi[i][6] / f_tol[i][0]) * 5.67 * 10**(-8) * \
                   (2 * np.power(293, 2) * 2 * 293)

    return hr

# % SET THERMAL CAPACITANCE MATRIX FOR EXTERNAL WALLS
def setC_wall(N, Cp, Rou, dx):
    C = np.zeros((np.sum(N[:], dtype=int) + 1, np.sum(N[:], dtype=int) + 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            C[i][i] = 0.5 * dx[0] * Rou[0] * Cp[0]
        elif i > 0 and i < (N[0]):
            C[i][i] = Cp[0] * Rou[0] * dx[0]
        elif i == N[0]:
            C[i][i] = 0.5 * Cp[0] * Rou[0] * dx[0] + 0.5 * Cp[1] * Rou[1] * dx[1]
        elif i > (N[0]) and i < (np.sum(N[0 : 2], dtype=int)):
            C[i][i] = Cp[1] * Rou[1] * dx[1]
        elif i == (np.sum(N[0 : 2], dtype=int)):
            C[i][i] = 0.5 * Cp[1] * Rou[1] * dx[1] + 0.5 * Cp[2] * Rou[2] * dx[2]
        elif i > (np.sum(N[0:2], dtype=int)) and i < (np.sum(N[0:3], dtype=int)):
            C[i][i] = Cp[2] * Rou[2] * dx[2]
        elif i == (np.sum(N[0:3], dtype=int)):
            C[i][i] = 0.5 * Cp[2] * Rou[2] * dx[2] + 0.5 * Cp[3] * Rou[3] * dx[3]
        elif i > (np.sum(N[0:3], dtype=int)) and i < (np.sum(N[0:4], dtype=int)):
            C[i][i] = Cp[3] * Rou[3] * dx[3]
        elif i == (np.sum(N[0:4], dtype=int)):
            C[i][i] = 0.5 * Cp[3] * Rou[3] * dx[3]

    return C

# % 不透明围护结构物性矩阵C
# function C=setC_INTwall(N,Cp,Rou,dx) %N：围护结构被分的份数；Cp：围护结构比热；Rou：围护结构密度；dx：围护结构每份的长度，m
def setC_INTwall(N, Cp, Rou, dx):

    C = np.zeros((np.sum(N[:], dtype=int) + 1, np.sum(N[:], dtype=int) + 1))

    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            C[i][i] = 0.5 * dx[0] * Rou[0] * Cp[0]
        elif i > 0 and i < (N[0]):
            C[i][i] = Cp[0] * Rou[0] * dx[0]
        elif i == N[0]:
            C[i][i] = 0.5 * Cp[0] * Rou[0] * dx[0] + 0.5 * Cp[1] * Rou[1] * dx[1]
        elif i > (N[0]) and i < (np.sum(N[0 : 2], dtype=int)):
            C[i][i] = Cp[1] * Rou[1] * dx[1]
        elif i == (np.sum(N[0 : 2], dtype=int)):
            C[i][i] = 0.5 * Cp[1] * Rou[1] * dx[1] + 0.5 * Cp[2] * Rou[2] * dx[2]
        elif i > (np.sum(N[0:2], dtype=int)) and i < (np.sum(N[0:3], dtype=int)):
            C[i][i] = Cp[2] * Rou[2] * dx[2]
        elif i == (np.sum(N[0 : 3], dtype=int)):
            C[i][i] = 0.5 * Cp[2] * Rou[2] * dx[2] + 0.5 * Cp[2] * Rou[2] * dx[2]

    return C

# function C=setC_window(N,Cp,Rou,dx)%N：玻璃层数；Cp：围护结构比热；Rou：围护结构密度；dx：围护结构每份的长度，m
def setC_window(N, Cp, Rou, dx):
    C = np.zeros((N, N))
    for i in range(N):
        C[i][i] = 0.5 * dx * Rou * Cp
    return C

# SET HEAT FLUX MATRICES FOR EXTERNAL WALLS functionA = setA_wall(N, Lamda, dx, ha_out, ha_in, hr1, hr2, hr3, hr4, hr5, hr6)
# 为外墙设置热流矩阵
# FOUR LAYERS
def setA_wall(N, Lamda, dx, ha_out, ha_in, hr1, hr2, hr3, hr4, hr5, hr6):
    A = np.zeros((np.sum(N[:], dtype=int) + 1, np.sum(N[:], dtype=int) + 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            A[i][i] = - ha_out - Lamda[0] / dx[0]
        elif i > 0 and i < (N[0]):
            A[i][i] = -2 * Lamda[0] / dx[0]
        elif i == N[0]:
            A[i][i] = -(Lamda[0] / dx[0] + Lamda[1] / dx[1])
        elif i > N[0] and i < (np.sum(N[0 : 2], dtype=int)):
            A[i][i] = -2 * Lamda[1] / dx[1]
        elif i == (np.sum(N[0 : 2], dtype=int)):
            A[i][i] = -(Lamda[1] / dx[1] + Lamda[2] / dx[2])
        elif i > (np.sum(N[0 : 2], dtype=int)) and i < (np.sum(N[0 : 3], dtype=int)):
            A[i][i] = -2 * Lamda[2] / dx[2]
        elif i == (np.sum(N[0 : 3], dtype=int)):
            A[i][i] = -(Lamda[2] / dx[2] + Lamda[3] / dx[3])
        elif i > (np.sum(N[0 : 3], dtype=int)) and i < (np.sum(N[0 : 4], dtype=int)):
            A[i][i] = -2 * Lamda[3] / dx[3]
        elif i == (np.sum(N[0 : 4], dtype=int)):
            A[i][i] = -ha_in - Lamda[3] / dx[3] - hr1 - hr2 - hr3 - hr4 - hr5 - hr6
    for i in range(np.sum(N[:], dtype=int) + 1):
        if (i) <= (N[0] - 1):
            A[i][i + 1] = Lamda[0] / dx[0]
            A[i + 1][i] = A[i][i + 1]
        elif i > N[0] - 1 and i <= (np.sum(N[0 : 2], dtype=int) - 1):
            A[i][i + 1] = Lamda[1] / dx[1]
            A[i + 1][i] = A[i][i + 1]
        elif i > (np.sum(N[0 : 2], dtype=int)- 1) and i <= (np.sum(N[0 : 3], dtype=int)- 1):
            A[i][i + 1] = Lamda[2] / dx[2]
            A[i + 1][i] = A[i][i + 1]
        elif i > (np.sum(N[0 : 3], dtype=int)- 1) and i <= (np.sum(N[0 : 4], dtype=int)- 1):
            A[i][i + 1] = Lamda[3] / dx[3]
            A[i + 1][i] = A[i][i + 1]
    return A

# SET HEAT DISTURBANCE VECTOR FOR INTERNAL WALLS
# function A=setA_INTwall(N,Lamda,dx,ha_in,ha_aj,hr1,hr2,hr3,hr4,hr5,hr6,hr_aj1,hr_aj2,hr_aj3,hr_aj4,hr_aj5,hr_aj6)
# % three layers
def setA_INTwall(N, Lamda, dx, ha_in, ha_aj, hr1, hr2, hr3, hr4, hr5, hr6, hr_aj1, hr_aj2, hr_aj3,
                 hr_aj4, hr_aj5, hr_aj6):
    A = np.zeros((np.sum(N[:], dtype=int) + 1, np.sum(N[:], dtype=int) + 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            A[i][i] = -ha_aj - Lamda[0]/dx[0] - hr_aj1 - hr_aj2 - hr_aj3 - hr_aj4 - hr_aj5 - hr_aj6
        elif i > 0 and i < (N[0]):
            A[i][i] = -2 * Lamda[0] / dx[0]
        elif i == N[0]:
            A[i][i] = -(Lamda[0] / dx[0] + Lamda[1] / dx[1])
        elif i > N[0] and i < (np.sum(N[0: 2], dtype=int)):
            A[i][i] = -2 * Lamda[1] / dx[1]
        elif i == (np.sum(N[0: 2], dtype=int)):
            A[i][i] = -(Lamda[1] / dx[1] + Lamda[2] / dx[2])
        elif i > (np.sum(N[0: 2], dtype=int)) and i < (np.sum(N[0: 3], dtype=int)):
            A[i][i] = -2 * Lamda[2] / dx[2]
        elif i == (np.sum(N[0: 3], dtype=int)):
            A[i][i] = -ha_in - Lamda[2]/dx[2]- hr1 - hr2 - hr3 - hr4 - hr5 - hr6
    for i in range(np.sum(N[:], dtype=int) + 1):
        if (i) <= (N[0] - 1):
            A[i][i + 1] = Lamda[0] / dx[0]
            A[i + 1][i] = A[i][i + 1]
        elif i > N[0] - 1 and i <= (np.sum(N[0: 2], dtype=int) - 1):
            A[i][i + 1] = Lamda[1] / dx[1]
            A[i + 1][i] = A[i][i + 1]
        elif i > (np.sum(N[0: 2], dtype=int) - 1) and i <= (np.sum(N[0: 3], dtype=int) - 1):
            A[i][i + 1] = Lamda[2] / dx[2]
            A[i + 1][i] = A[i][i + 1]
    return A
# SET LONG WAVE RADIATION MATRIX
# function A_lwr_i_j=A_lwr(ni,nj,h_lwr_i_j)
def A_lwr(ni,nj,h_lwr_i_j):
    A_lwr_i_j = np.zeros((nj,ni))
    A_lwr_i_j[nj - 1][ni - 1] = h_lwr_i_j
    return A_lwr_i_j

# SET LONG WAVE RADIATION MATRIX
# function A_lwr_i_j=A_lwr_cyc(ni,nj,h_lwr_i_j)     % CYCLIC BOUNDARY CONDITIONS
def A_lwr_cyc(ni,nj,h_lwr_i_j):
    A_lwr_i_j = np.zeros((nj, ni))
    A_lwr_i_j[0][ni - 1] = h_lwr_i_j
    A_lwr_i_j[nj - 1][ni - 1] = h_lwr_i_j
    return A_lwr_i_j
#  CONVECTIVE TERMS OF AIR NODE(空气节点对流项)
# function Aconv_air=A_conv_air_wall(n,ha_in)
def A_conv_air_wall(n,ha_in):
    Aconv_air = np.zeros((n, 1))
    Aconv_air[n - 1][0] = ha_in
    return Aconv_air

# CONVECTIVE TERMS OF AIR NODE(空气节点对流项)
# function Aconv_air=A_conv_air_CYCwall(n,ha_adj,ha_in)       % CYCLIC BOUNDARY CONDITIONS
def A_conv_air_CYCwall(n,ha_adj,ha_in):
    Aconv_air = np.zeros((n, 1))
    Aconv_air[0][0] = ha_adj
    Aconv_air[n - 1][0] = ha_in
    return Aconv_air
# CALCULATION OF HUMIDITY RATIO (MIXING RATIO OF MASS OF WATER VAPOUR TO
# MASS OF DRY GAS) ACCORDING TO PSYCHROMETER RELATIONSHIP
#  function w_e=Humidity_Ratio(t,RH,p_tot)
# temp-temperature in Celsuis Degree;
# RH - relative humidity: 0-100
# p_tot - total atmospheric pressure
# p_ws=6.116441*power(10,7.591386*t/(t+240.7263));
# % Water vapour saturation pressure over water in hPa(i.e. 100 Pascals)
# %The formula is suitable for temperature range of -20 C to+50 C; for
# %a temperature exceeds the range, the constants should be changed.
#
# p_w=p_ws*RH/100; % water (surface) vapour pressure
# w_e= 622*p_w/(p_tot-p_w); % unit in g/kg;
def Humidity_Ratio(t, RH, p_tot):
    p_ws = 6.116441 * np.power(10, 7.591386 * t / (t + 240.7263))
    p_w = p_ws * RH / 100
    w_e = 622 * p_w / (p_tot - p_w)
    return w_e

# % SOLAR INCIDENCE ANGLE
# function angle=solar_angle(n,r,longitude,ws,s,t)
# %delta=23.45*sin((n-80)/370*360)*pi/180;
def solar_angle(n, r, longitude, ws, s, t):

    delta=23.45 * np.sin(360 * (284 + n) / 365 * np.pi / 180) * np.pi / 180  # DECLINATION
    w = 15 * (ws + (longitude-120) * 4 / 60 - 12) * np.pi / 180 # Revised local mean solar time (close to the true solar time)
    r = r * np.pi / 180
    h = np.arcsin(np.sin(r) * np.sin(delta) + np.cos(r) * np.cos(delta) * np.cos(w))
    # %只有太阳高度角大于0时，太阳入射角才不为零
    if h > 0:
        A = np.cos(delta) * np.sin(w) / np.cos(h)
        if A <= 1:
            a = np.arcsin(A)
        else:
            a = np.arcsin((np.sin(h) * np.sin(r) - np.sin(delta)) / np.cos(h) / np.cos(r))
        s = s * np.pi / 180
        t = t * np.pi / 180
        angle = 180 * np.arccos(np.cos(s) * np.sin(h) + np.sin(s) * np.cos(h) * np.cos(a - t)) / np.pi
    else:
        angle = 90

    if angle > 90:
        angle =180 - angle
    return angle

# OPTICAL PROPERTIES CALCULATION OF WINDOW GLAZING
# function [win_transparent,win_absorption,win_refelection,e_all,tao_all]=solar_win2(angle1)    %angle窗户外层的太阳辐射入射角,°；
# 窗户为双层玻璃窗，无镀膜，因此有5个界面层（4个玻璃-空气界面层，1个等效界面层）和3个空气层
def solar_win2(angle1):
    #（1）定义界面及介质属性
    n = [1, 1.526, 1, 1.526, 1] # 不同的介质的折射率，空气为1，玻璃为1.526
    k = [0, 0.16, 0, 0.16, 0]
    d = [0, 0.003, 0.009, 0.003, 0] # 双层玻璃窗和中间空气层的厚度，m, 窗户厚度3mm
    angle = np.zeros((1, 5)) #计算不同界面的入射角
    angle[0][0] = angle1
    angle[0][1] = np.arcsin(n[0] / n[1] * np.sin(angle1 * np.pi / 180)) * 180 / np.pi
    angle[0][2] = np.arcsin(n[1] / n[2] * np.sin(angle[0][1] * np.pi / 180)) * 180 / np.pi
    angle[0][3] = np.arcsin(n[2] / n[3] * np.sin(angle[0][2] * np.pi / 180)) * 180 / np.pi
    angle[0][4] = np.arcsin(n[3] / n[4] * np.sin(angle[0][3] * np.pi / 180)) * 180 / np.pi
    rou = np.zeros((1, 5))
    for i in range(4):
        if angle[0][i] >= 30:
            rou[0][i] = 0.5 * ((np.sin((angle[0][i + 1] - angle[0][i]) * np.pi / 180)) ** 2 /
                               (np.sin((angle[0][i + 1] + angle[0][i]) * np.pi / 180)) ** 2 +
                               (np.tan((angle[0][i + 1] - angle[0][i]) * np.pi / 180)) ** 2 /
                               (np.tan((angle[0][i + 1] + angle[0][i]) * np.pi / 180)) ** 2)
        else:
            rou[0][i] = ((n[i] - n[i + 1]) / (n[i] + n[i + 1])) ** 2
    L = np.zeros((1, 5)) # 计算光线路程
    for i in range(1, 4):
        L[0][i] = d[i] / np.cos(angle[0][i] * np.pi / 180)
    tao = np.ones((1, 5)) # 计算不同界面的透过率，tao(i)为i - 1界面到i界面的透过率
    for i in range(5):
        tao[0][i] = np.exp(- k[i] * 100 * L[0][i])
    beta = np.zeros((1, 5)) # 每一层界面的等效反射率
    alpha = np.zeros((1, 5)) # 每一层界面的等效透过率
    alpha[0][4] = 1
    A = np.zeros((1, 5))
    for i in reversed(range(4)):          # 通过递推公式，求出各截面的等效反射率、吸收率
        A[0][i] = beta[0][i + 1] * (tao[0][i + 1]) ** 2     # A - 反向反射率的倒数
        alpha[0][i] = (1 - rou[0][i]) / (1 - rou[0][i] * A[0][i])
        beta[0][i] = 1 - alpha[0][i] * (1 - A[0][i])
    tao_all = np.zeros((1, 5)) # 每层介质相对于最外层输入的辐射能的透过率
    tao_all[0][1] = alpha[0][0] * tao[0][1]
    tao_all[0][2] = alpha[0][0] * alpha[0][1] * tao[0][1] * tao[0][2]
    tao_all[0][3] = alpha[0][0] * alpha[0][1] * alpha[0][2] * tao[0][1] * tao[0][2] * tao[0][3]
    tao_all[0][4] = alpha[0][0] * alpha[0][1] * alpha[0][2] * alpha[0][3] * tao[0][1] * tao[0][2] * tao[0][3] * tao[0][4] # 窗户整体透过率
    e = np.zeros((1, 5)) # 各层介质相对于此介质辐射能的吸收系数，e(i)为i - 1界面到i界面之间包含的介质的吸收系数
    for i in range(1, 4):
        A[0][i] = beta[0][i + 1] * (tao[0][i + 1]) ** 2
        e[0][i] = alpha[0][i - 1] * (1 - tao[0][i] + A[0][i] * (1 - tao[0][i]) / tao[0][i])
    e_all = np.zeros((1, 5)) # 各层介质相对于最外层输入能量的系数，e_all(i)为i - 1界面到i界面之间包含的介质的吸收系数
    e_all[0][1] = e[0][1] * alpha[0][0] * tao[0][1]
    e_all[0][2] = e[0][2] * alpha[0][1] * tao[0][2] * alpha[0][0] * tao[0][1]
    e_all[0][3] = e[0][3] * alpha[0][2] * tao[0][3] * alpha[0][1] * tao[0][2] * alpha[0][0] * tao[0][1]
    e_all[0][4] = e[0][4] * alpha[0][3] * tao[0][4] * alpha[0][2] * tao[0][3] * alpha[0][1] * tao[0][2] * alpha[0][0] * tao[0][1]  # 窗户整体吸收率（相对于最外层界面输入的辐射能的吸收系数）
    win_transparent = tao_all[0][4] # 窗户整体的透过率
    win_absorption = np.sum(e_all[0, :]) # 窗户整体的吸收率
    win_refelection = beta[0][0] # 窗户整体的发射率（外层表面的反射率）

    return [win_transparent, win_absorption, win_refelection, e_all, tao_all]

# set heat disturbance vector for external walls
# four layers
# function B=setB_wall_load(N,ha_out,e,q_solar_out,Q_solar_in_wall,Q_radiation,ta_out,f,h_in,tR_air_prev)
def setB_wall_load(N, ha_out, e, q_solar_out, Q_solar_in_wall, Q_radiation, ta_out, f, h_in, tR_air_prev):
    B = np.zeros((np.sum(N[:], dtype=int) + 1, 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            B[i][0] = ha_out * ta_out + e * q_solar_out
        elif i > 0 and i < (np.sum(N[0:4], dtype=int)):
            B[i][0] = 0
        elif i == (np.sum(N[0:4], dtype=int)):
            B[i][0] = (e * Q_radiation + Q_solar_in_wall) / f + h_in * tR_air_prev

    return B

# INTERNAL WALL
# function B=setB_INTwall_load(N,h_in_wall,e,Q_solar_in,Q_radiation,Q_solar_in_adj,Q_radiation_adj,f_adj,ta_adjacent,taj_w1,taj_w2,taj_w3,taj_w4,taj_w5,taj_w6,hr_aj1,hr_aj2,hr_aj3,hr_aj4,hr_aj5,hr_aj6,f,tRair_prev)
#  three layers
def setB_INTwall_load(N, h_in_wall, e, Q_solar_in, Q_radiation, Q_solar_in_adj, Q_radiation_adj, f_adj,
                      ta_adjacent, taj_w1, taj_w2, taj_w3, taj_w4, taj_w5, taj_w6, hr_aj1, hr_aj2, hr_aj3,
                      hr_aj4, hr_aj5, hr_aj6, f, tRair_prev):
    B = np.zeros((np.sum(N[:], dtype=int) + 1, 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            B[i][0] = h_in_wall * ta_adjacent + (Q_solar_in_adj + e * Q_radiation_adj) / f_adj + hr_aj1 * taj_w1 + \
                      hr_aj2 * taj_w2 + hr_aj3 * taj_w3 + hr_aj4 * taj_w4 + hr_aj5 * taj_w5 + hr_aj6 * taj_w6
           # convective and radiative heat transfer from adjacent room as well as internal heat gains
        elif i > 0 and i < (np.sum(N[0:3], dtype=int)):
            B[i][0] = 0
        elif i == (np.sum(N[0:3], dtype=int)):
            # internal solar heat gain from windows and internal radiative heat gain
            B[i][0] = (Q_solar_in + e * Q_radiation) / f + h_in_wall * tRair_prev
    return B

# INTERNAL WALL
# function B=setB_ceiling_load(N,e,Q_solar_in_cyc,Q_radiation_cyc,f_cyc,Q_solar_in,Q_radiation,f,h_in_ceil,h_in_floor,tR_air_prev)
# Three layers
def setB_ceiling_load(N, e, Q_solar_in_cyc, Q_radiation_cyc, f_cyc, Q_solar_in,
                      Q_radiation, f, h_in_ceil, h_in_floor, tR_air_prev):
    B = np.zeros((np.sum(N[:], dtype=int) + 1, 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            B[i][0] = h_in_floor * tR_air_prev + (e * Q_solar_in_cyc + e * Q_radiation_cyc) / f_cyc
        elif i > 0 and i < (np.sum(N[0:3], dtype=int)):
            B[i][0] = 0
        elif i == (np.sum(N[0:3], dtype=int)):
            B[i][0] = (e * Q_solar_in + e * Q_radiation) / f + h_in_ceil * tR_air_prev
    return B

# 透明围护结构热扰B矩阵
# function B=setB_window_load(N,ha_out,e1,e2,q_solar_out,Q_radiation,ta_out,f,h_in_wall,tR_air_prev)
def setB_window_load(N, ha_out, e1, e2, q_solar_out, Q_radiation, ta_out, f, h_in_wall, tR_air_prev):
    B = np.zeros((N, 1))
    for i in range(N):
        if i == 0:
            B[i][0] = ha_out * ta_out + e1 * q_solar_out + e2 * Q_radiation / f
        elif i == 1:
            B[i][0] = e2 * q_solar_out + e1 * Q_radiation / f + h_in_wall * tR_air_prev # e * Q_radiation为室内产热的辐射部分
    return B
# SET HEAT DISTURBANCE VECTORS FOR EXTERNAL WALLS
# four layers
# function B=setB_wall(N,ha_out,e,q_solar_out,Q_solar_in,Q_radiation,ta_out,f)
def setB_wall(N, ha_out, e, q_solar_out, Q_solar_in, Q_radiation, ta_out, f):
    B = np.zeros((np.sum(N[:], dtype=int) + 1, 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            B[i][0] = ha_out * ta_out + e * q_solar_out   # ha_out*ta_out为围护结构外表面节点与室外对流换热项、e*Q_solar_out为维护外表面获得的太阳辐射热量份额
        elif i > 0 and i < (np.sum(N[0:4], dtype=int)):
            B[i][0] = 0
        elif i == np.sum(N[0:4], dtype=int):
            B[i][0] = (e * Q_radiation + Q_solar_in) / f  # e*Q_solar_in/f为当前围护结构内表面获得的单位面积辐射热量（过窗太阳辐射热量，包括直射和散射）
    return B

# % INTERNAL WALL
# function B=setB_INTwall(N,ha_adj,e,Q_solar_in,Q_radiation,Q_solar_in_adj,Q_radiation_adj,f_adj,ta_adjacent,taj_w1,taj_w2,taj_w3,taj_w4,taj_w5,taj_w6,hr_aj1,hr_aj2,hr_aj3,hr_aj4,hr_aj5,hr_aj6,f)
# % three layers
def setB_INTwall(N, ha_adj, e, Q_solar_in, Q_radiation, Q_solar_in_adj, Q_radiation_adj, f_adj, ta_adjacent, taj_w1,
                 taj_w2, taj_w3, taj_w4, taj_w5, taj_w6, hr_aj1, hr_aj2, hr_aj3, hr_aj4, hr_aj5, hr_aj6, f):
    B = np.zeros(((np.sum(N[:], dtype=int) + 1, 1)))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            # convective and radiative heat transfer from adjacent room as well as internal heat gains
            B[i][0] = ha_adj * ta_adjacent + (Q_solar_in_adj + e * Q_radiation_adj) / f_adj + hr_aj1 * taj_w1 + hr_aj2 * taj_w2 \
                      + hr_aj3 * taj_w3 + hr_aj4 * taj_w4 + hr_aj5 * taj_w5 + hr_aj6 * taj_w6
        elif i > 0 and i < (np.sum(N[0:3], dtype=int)):
            B[i][0] = 0
        elif i == np.sum(N[0:3], dtype=int):
            # internal solar heat gain from windows and internal radiative heat gain
            B[i][0] = (Q_solar_in + e * Q_radiation) / f
    return B
# % INTERNAL WALL
# function B=setB_ceiling(N,e,Q_solar_in_cyc,Q_radiation_cyc,f_cyc,Q_solar_in,Q_radiation,f)
#  % Three layers
def setB_ceiling(N, e, Q_solar_in_cyc, Q_radiation_cyc, f_cyc, Q_solar_in, Q_radiation, f):
    B = np.zeros(((np.sum(N[:], dtype=int) + 1, 1)))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            B[i][0] = (e * Q_solar_in_cyc + e * Q_radiation_cyc) / f_cyc
        elif i > 0 and i < (np.sum(N[0:3], dtype=int)):
            B[i][0] = 0
        elif i == np.sum(N[0:3], dtype=int):
            B[i][0] = (e * Q_solar_in + e * Q_radiation) / f

    return B
# %透明围护结构热扰B矩阵
# function B=setB_window(N,ha_out,e1,e2,q_solar_out,Q_radiation,ta_out,f)
def setB_window(N, ha_out, e1, e2, q_solar_out, Q_radiation, ta_out, f):
    B = np.zeros((N, 1))
    for i in range(N):
        if i == 0:
            B[i][0] = ha_out * ta_out + e1 * q_solar_out + e2 * Q_radiation / f
        elif i == 1:
            B[i][0] = e2 * q_solar_out + e1 * Q_radiation / f   # e*Q_radiation为室内产热的辐射部分
    return B
# % INTERNAL WALL
# function B=setB_INT3wall(N,ha_ta_adj,e,Q_solar_in,Q_radiation,Q_solar_in_adj,Q_radiation_adj,hr_twall_adj,f)
# % three layers
def setB_INT3wall(N, ha_ta_adj, e, Q_solar_in, Q_radiation, Q_solar_in_adj, Q_radiation_adj, hr_twall_adj, f):
    B = np.zeros((np.sum(N[:], dtype=int) + 1, 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            B[i][0] = ha_ta_adj + (Q_solar_in_adj + e * Q_radiation_adj) / f + hr_twall_adj
        elif i > 0 and i < (np.sum(N[:], dtype=int)):
            B[i][0] = 0
        elif i == np.sum(N[:], dtype=int):
            B[i][0] = (Q_solar_in + e * Q_radiation) / f
    return B
# % INTERNAL WALL
# function B=setB_INT3wall_load(N,ha_ta_adj,e,Q_solar_in,Q_radiation,Q_solar_in_adj,Q_radiation_adj,hr_twall_adj,f,ha_ta)
# % three layers
def setB_INT3wall_load(N, ha_ta_adj, e, Q_solar_in, Q_radiation, Q_solar_in_adj, Q_radiation_adj, hr_twall_adj, f, ha_ta):
    B = np.zeros((np.sum(N[:], dtype=int) + 1, 1))
    for i in range(np.sum(N[:], dtype=int) + 1):
        if i == 0:
            B[i][0] = ha_ta_adj + (Q_solar_in_adj + e * Q_radiation_adj) / f + hr_twall_adj
        elif i > 0 and i < (np.sum(N[:], dtype=int)):
            B[i][0] = 0
        elif i == np.sum(N[:], dtype=int):
            B[i][0] = (Q_solar_in + e * Q_radiation) / f + ha_ta
    return B

# % SOLVING PDEs
# function dt=fun(time,t)          % PEDS: dT/d_time=fun(time,t)
# [m1,temp_R1]=ode15s(@fun,[(i-1)*dt,i*dt],t1(:,i-1))
def function(A, B, C):
    a = np.dot(np.linalg.inv(C), A)
    b = np.dot(np.linalg.inv(C), B)
    return a, b

# % WINDOW HEAT FLOW MATRIX
# function A=setA_window(N,ha_out,ha_in,h12,hr1,hr2,hr3,hr4,hr5,hr6)
def setA_window(N, ha_out, ha_in, h12, hr1, hr2, hr3, hr4, hr5, hr6):
    A = np.zeros((N, N))
    if N == 2:   # DOUBLE GLAZING
        A[0][0] = -ha_out - h12
        A[0][1] = h12
        A[1][0] = h12
        A[1][1] = -ha_in - h12 - hr1 - hr2 - hr3 - hr4 - hr5 - hr6
    elif N == 1:   # SINGLE GLAZING
        A[0][0] = -ha_out - ha_in - hr1 - hr2 - hr3 - hr4 - hr5 - hr6
    return A

# % CONVECTIVE TERMS OF AIR NODE
# function Aconv_air=A_conv_wall_air(n,ha_in,f)
def A_conv_wall_air(n, ha_in, f):
    Aconv_air = np.zeros((1, n))
    Aconv_air[0][n - 1] = ha_in * f
    return Aconv_air
