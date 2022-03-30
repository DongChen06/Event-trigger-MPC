from time import time
import casadi as ca
import numpy as np
import math
from casadi import sin, cos, pi


def newargs_new(horizon, specs):
    n_states = specs[2]
    n_controls = specs[3]
    lbx = ca.DM.zeros((n_states * (horizon) + n_controls * horizon, 1))
    ubx = ca.DM.zeros((n_states * (horizon) + n_controls * horizon, 1))

    lbx[0: n_states * (horizon): n_states] = -ca.inf  # x1 lower bound
    lbx[1: n_states * (horizon): n_states] = -ca.inf  # x2 lower bound
    lbx[2: n_states * (horizon): n_states] = -ca.inf  # x3 lower bound
    lbx[3: n_states * (horizon): n_states] = -ca.inf  # x4 lower bound
    lbx[4: n_states * (horizon): n_states] = -ca.inf  # x5 lower bound
    lbx[5: n_states * (horizon): n_states] = -ca.inf  # x6 lower bound

    ubx[0: n_states * (horizon): n_states] = ca.inf  # x1 upper bound
    ubx[1: n_states * (horizon): n_states] = ca.inf  # x2 upper bound
    ubx[2: n_states * (horizon): n_states] = ca.inf  # x3 upper bound
    ubx[3: n_states * (horizon): n_states] = ca.inf  # x4 upper bound
    ubx[4: n_states * (horizon): n_states] = ca.inf  # x5 upper bound
    ubx[5: n_states * (horizon): n_states] = ca.inf  # x6 upper bound

    u1_min = -500
    u1_max = 500

    u2_min = -0.54105
    u2_max = 0.54105

    lbx[n_states * (horizon):: n_controls] = u1_min
    lbx[n_states * (horizon) + 1:: n_controls] = u2_min

    ubx[n_states * (horizon):: n_controls] = u1_max
    ubx[n_states * (horizon) + 1:: n_controls] = u2_max

    args = {
        'lbg': ca.DM.zeros((n_states * (horizon), 1)),  # constraints lower bound
        'ubg': ca.DM.zeros((n_states * (horizon), 1)),  # constraints upper bound
        'lbx': lbx,
        'ubx': ubx}

    return args
