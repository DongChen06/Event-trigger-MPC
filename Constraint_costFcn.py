####

####

from time import time
import casadi as ca
import numpy as np
import math
from casadi import sin, cos, pi


def const_and_CostFcn_new(horizon, X, U, P, specs):

    Ts = specs[0]
    f = specs[1]
    n_states = specs[2]

    cost_fn = 0  # cost function

    x_0 = P[:n_states]
    u_0 = U[:, 0]
    k1 = f(x_0, u_0)
    k2 = f(x_0 + Ts / 2 * k1, u_0)
    k3 = f(x_0 + Ts / 2 * k2, u_0)
    k4 = f(x_0 + Ts * k3, u_0)
    st_next_RK41 = x_0 + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = X[:, 0] - st_next_RK41
    # runge kutta
    for k in range(horizon - 1):
        st = X[:, k]
        con = U[:, k + 1]
        cost_fn = cost_fn \
                  + 2.0 * (st[2] - 4 * sin((2 * pi * st[0]) / 50 )) ** 2\
                  + 0.001 * con[0] ** 2 \
                  + 0.001 * con[1] ** 2
                  # + ((st - P[n_states:]).T @ Q @ (st - P[n_states:])) \
                  # + (R * (con ** 2))

        # 2.0 * (x[1] - 4 * np.sin(2 * pi / 50 * x[0])) ** 2 + 0.001 * u[0] ** 2
        st_next = X[:, k + 1]
        k1 = f(st, con)
        k2 = f(st + Ts / 2 * k1, con)
        k3 = f(st + Ts / 2 * k2, con)
        k4 = f(st + Ts * k3, con)
        st_next_RK4 = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        g = ca.vertcat(g, st_next - st_next_RK4)
    return cost_fn, g
