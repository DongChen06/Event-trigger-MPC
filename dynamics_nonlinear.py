import casadi as ca
import numpy as np
from casadi import sin, cos, pi


def veh_rhs_nonlinear(x0, u0):
    dxdt = ca.SX.zeros(6, 1)

    rho_aero = 1.225
    mveh = 1500
    Cd = 0.389
    radwhl = 0.2159
    Af = 4
    Lxf = 1.2
    Lxr = 1.4
    mu = 1
    Iw = 3.8782
    I = 4192
    Calpha = -0.08 * 180 / pi
    sa_max = 10 / 180 * pi
    g = 9.8
    sigma = 0.0
    # small_pos_th = 1e-6

    lx = x0[0, :]
    vx = x0[1, :]
    ly = x0[2, :]
    vy = x0[3, :]
    psi = x0[4, :]
    r = x0[5, :]

    Tf = u0[0, :]  # u0[0]
    Tr = 0.0  # u0[1]
    dmf = 0.0  # what is dmf ??
    dmr = 0.0  # what is dmr ??
    betaf = u0[1, :]  # u0[2]
    betar = 0.0  # u0[3]

    # Define rotation matrix
    rotM_00 = ca.cos(betaf)
    rotM_01 = -ca.sin(betaf)
    rotM_10 = ca.sin(betaf)
    rotM_11 = ca.cos(betaf)
    rotM_20 = ca.cos(betar)
    rotM_21 = -ca.sin(betar)
    rotM_30 = ca.sin(betar)
    rotM_31 = ca.cos(betar)

    # Calculating corner and wheel velocity
    v_c_00 = vx  # v_c_00 = v_xr
    v_c_01 = vy + Lxf * r  # = v_yr
    v_c_10 = vx  # = v_xf
    v_c_11 = vy - Lxr * r  # = v_yf

    v_w_00 = v_c_00 * rotM_00 + v_c_01 * rotM_10  # \bar{v}_xr
    v_w_01 = v_c_00 * rotM_01 + v_c_01 * rotM_11  # \bar{v}_yr
    v_w_10 = v_c_10 * rotM_20 + v_c_11 * rotM_30  # \bar{v}_xf
    v_w_11 = v_c_10 * rotM_21 + v_c_11 * rotM_31  # \bar{v}_yf

    # Calculating tire force at each wheel
    Fz_0 = Lxr * mveh * g / 2 / (Lxf + Lxr)
    Fz_1 = Lxf * mveh * g / 2 / (Lxf + Lxr)

    F_w00 = Tf / 2 / radwhl
    F_w10 = Tr / 2 / radwhl

    sa0 = ca.arctan(v_w_01 / v_w_00)
    sa1 = ca.arctan(v_w_11 / v_w_10)

    F_w01 = Calpha * mu * sa0 * Fz_0
    F_w11 = Calpha * mu * sa1 * Fz_1

    F_c_00 = F_w00 * rotM_00 + F_w01 * rotM_01
    F_c_01 = F_w00 * rotM_10 + F_w01 * rotM_11
    F_c_10 = F_w10 * rotM_20 + F_w11 * rotM_21
    F_c_11 = F_w10 * rotM_30 + F_w11 * rotM_31

    # calculating the xdot
    # xdot = np.zeros(6)
    dxdt[0, :] = vx * ca.cos(psi) - vy * ca.sin(psi)
    dxdt[1, :] = vy * r + 2 * (F_c_00 + F_c_10) / mveh - g * np.sin(sigma) - 0.5 * rho_aero * Cd * Af * vx * vx / mveh
    dxdt[2, :] = vx * ca.sin(psi) + vy * ca.cos(psi)
    dxdt[3, :] = -vx * r + 2 * (F_c_01 + F_c_11) / mveh
    dxdt[4, :] = r
    dxdt[5, :] = (2 * F_c_01 * Lxf - 2 * F_c_11 * Lxr + dmf + dmr) / I  # TODO: difference with Eqn21-f

    return dxdt
