# Continuous-time nonlinear dynamic model of a pendulum on a cart
#
# 4 states (x):
#   cart position (z)
#   cart velocity (z_dot): when positive, cart moves to right
#  angle (theta): when 0, pendulum is at upright position
#   angular velocity (theta_dot): when positive, pendulum moves anti-clockwisely
#
# 1 inputs: (u)
#   force (F): when positive, force pushes cart to right
#
# Copyright 2018 The MathWorks, Inc.

import casadi as ca
import numpy as np
from casadi import sin, cos, pi


def linearize(x0, u0):
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


    A = ca.vertcat(
        ca.horzcat( 0, 0, cos(psi),  -sin(psi), - vy*cos(psi) - vx*sin(psi), 0),
        ca.horzcat(0, 0, (Calpha*Lxr*g*mu*sin(betaf)*(sin(betaf)/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) + (cos(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)) - (Af*Cd*rho_aero*vx)/mveh, r - (Calpha*Lxr*g*mu*sin(betaf)*(cos(betaf)/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) - (sin(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)), 0, vy - (Calpha*Lxr*g*mu*sin(betaf)*((Lxf*cos(betaf))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) - (Lxf*sin(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr))),
        ca.horzcat(0, 0, sin(psi), cos(psi), vx*cos(psi) - vy*sin(psi), 0),
        ca.horzcat(0, 0, - r - ((Calpha*Lxr*g*mu*mveh*cos(betaf)*(sin(betaf)/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) + (cos(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)) + (Calpha*Lxf*g*mu*mveh*(vy - Lxr*r))/(vx**2*((vy - Lxr*r)**2/vx**2 + 1)*(Lxf + Lxr)))/mveh,       ((Calpha*Lxf*g*mu*mveh)/(vx*((vy - Lxr*r)**2/vx**2 + 1)*(Lxf + Lxr)) + (Calpha*Lxr*g*mu*mveh*cos(betaf)*(cos(betaf)/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) - (sin(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)))/mveh, 0, ((Calpha*Lxr*g*mu*mveh*cos(betaf)*((Lxf*cos(betaf))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) - (Lxf*sin(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)) - (Calpha*Lxf*Lxr*g*mu*mveh)/(vx*((vy - Lxr*r)**2/vx**2 + 1)*(Lxf + Lxr)))/mveh - vx),
        ca.horzcat(0, 0, 0, 0, 0, 1),
        ca.horzcat( 0, 0, -((Calpha*Lxf*Lxr*g*mu*mveh*cos(betaf)*(sin(betaf)/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) + (cos(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)) - (Calpha*Lxf*Lxr*g*mu*mveh*(vy - Lxr*r))/(vx**2*((vy - Lxr*r)**2/vx**2 + 1)*(Lxf + Lxr)))/I, -((Calpha*Lxf*Lxr*g*mu*mveh)/(vx*((vy - Lxr*r)**2/vx**2 + 1)*(Lxf + Lxr)) - (Calpha*Lxf*Lxr*g*mu*mveh*cos(betaf)*(cos(betaf)/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) - (sin(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)))/I,0,((Calpha*Lxf*Lxr**2*g*mu*mveh)/(vx*((vy - Lxr*r)**2/vx**2 + 1)*(Lxf + Lxr)) + (Calpha*Lxf*Lxr*g*mu*mveh*cos(betaf)*((Lxf*cos(betaf))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)) - (Lxf*sin(betaf)*(cos(betaf)*(vy + Lxf*r) - vx*sin(betaf)))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2))/(((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))**2/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf))**2 + 1)*(Lxf + Lxr)))/I)
    )
    B = ca.vertcat(
        ca.horzcat(0, 0),
        ca.horzcat(cos(betaf)/(mveh*radwhl), -((Tf*sin(betaf))/radwhl - (Calpha*Lxr*g*mu*mveh*sin(betaf))/(Lxf + Lxr) + (Calpha*Lxr*g*mu*mveh*ca.atan((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)))*cos(betaf))/(Lxf + Lxr))/mveh),
        ca.horzcat(0, 0),
        ca.horzcat(sin(betaf)/(mveh*radwhl), -((Calpha*Lxr*g*mu*mveh*cos(betaf))/(Lxf + Lxr) - (Tf*cos(betaf))/radwhl + (Calpha*Lxr*g*mu*mveh*ca.atan((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)))*sin(betaf))/(Lxf + Lxr))/mveh),
        ca.horzcat(0, 0),
        ca.horzcat((Lxf*sin(betaf))/(I*radwhl), -(Lxf*((Calpha*Lxr*g*mu*mveh*cos(betaf))/(Lxf + Lxr) - (Tf*cos(betaf))/radwhl + (Calpha*Lxr*g*mu*mveh*ca.atan((cos(betaf)*(vy + Lxf*r) - vx*sin(betaf))/(sin(betaf)*(vy + Lxf*r) + vx*cos(betaf)))*sin(betaf))/(Lxf + Lxr)))/I)
    )
    return A, B
