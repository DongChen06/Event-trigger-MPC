from time import time
import casadi as ca
import numpy as np
import math
from casadi import sin, cos, pi


def shift_timestep(Ts, t0, state_init, u, f):
    z1 = f(state_init, u[:, 0])
    z2 = f(state_init + Ts / 2 * z1, u[:, 0])
    z3 = f(state_init + Ts / 2 * z2, u[:, 0])
    z4 = f(state_init + Ts * z3, u[:, 0])
    next_state = ca.DM.full(state_init + (Ts / 6) * (z1 + 2 * z2 + 2 * z3 + z4))
    t0 = t0 + Ts
    return t0, next_state