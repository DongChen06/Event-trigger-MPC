# -------------------------------------------------------------#
# This is cloud MPC
#
#
# -------------------------------------------------------------#
from time import time
import casadi as ca
import numpy as np
import math
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from dynamics_linear import *
from dynamics_nonlinear import *
import random


def cloud_mpc(N, state_init, u_init):
    Ts = 0.2  # time between steps in seconds

    # dly_dist = (state_init / 5000) * np.random.rand(1)
    # state_init = state_init + dly_dist

    # state_init += 0 * np.random.rand(4, 1)

    x1_init = state_init[0]
    x2_init = state_init[1]
    x3_init = state_init[2]
    x4_init = state_init[3]
    x5_init = state_init[4]
    x6_init = state_init[5]

    # state symbolic variables
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')
    x5 = ca.SX.sym('x5')
    x6 = ca.SX.sym('x6')

    states = ca.vertcat(
        x1,
        x2,
        x3,
        x4,
        x5,
        x6
    )
    n_states = states.numel()

    # control symbolic variables
    u1 = ca.SX.sym('u1')
    u2 = ca.SX.sym('u2')
    controls = ca.vertcat(
        u1,
        u2
    )
    n_controls = controls.numel()

    # matrix containing all states over all time steps +1 (each column is a state vector)
    X = ca.SX.sym('X', n_states, N + 1)

    # matrix containing all control actions over all time steps (each column is an action vector)
    U = ca.SX.sym('U', n_controls, N)

    # column vector for storing initial state and target state
    P = ca.SX.sym('P', n_states)

    # nonlinear mapping function f(x,u)
    f = ca.Function('f', [states, controls], [veh_rhs_nonlinear(states, controls)])

    cost_fn = 0  # cost function
    g = X[:, 0] - P[:n_states]  # constraints in the equation
    # runge kutta

    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        cost_fn = cost_fn \
                  + 2.0 * (st[2] - 4 * sin((2 * pi * st[0]) / 50)) ** 2 \
                  + 1e-6 * con[0] ** 2 \
                  + con[1] ** 2

        st_next = X[:, k + 1]
        k1 = f(st, con)
        k2 = f(st + Ts / 2 * k1, con)
        k3 = f(st + Ts / 2 * k2, con)
        k4 = f(st + Ts * k3, con)
        st_next_RK4 = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        g = ca.vertcat(g, st_next - st_next_RK4)

    cost_fn = cost_fn + 2.0 * (X[2, N] - 4 * sin((2 * pi * X[0, N]) / 50)) ** 2

    OPT_variables = ca.vertcat(
        X.reshape((-1, 1)),  # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
        U.reshape((-1, 1))
    )

    nlp_prob = {
        'f': cost_fn,
        'x': OPT_variables,
        'g': g,
        'p': P
    }

    opts = {
        'ipopt': {
            # 'max_iter': 2000,
            'print_level': 1,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6
        },
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
    ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

    lbx[0: n_states * (N + 1): n_states] = -ca.inf  # x1 lower bound
    lbx[1: n_states * (N + 1): n_states] = -ca.inf  # x2 lower bound
    lbx[2: n_states * (N + 1): n_states] = -ca.inf  # x3 lower bound
    lbx[3: n_states * (N + 1): n_states] = -ca.inf  # x4 lower bound
    lbx[4: n_states * (N + 1): n_states] = -ca.inf  # x5 lower bound
    lbx[5: n_states * (N + 1): n_states] = -ca.inf  # x6 lower bound

    ubx[0: n_states * (N + 1): n_states] = ca.inf  # x1 upper bound
    ubx[1: n_states * (N + 1): n_states] = ca.inf  # x2 upper bound
    ubx[2: n_states * (N + 1): n_states] = ca.inf  # x3 upper bound
    ubx[3: n_states * (N + 1): n_states] = ca.inf  # x4 upper bound
    ubx[4: n_states * (N + 1): n_states] = ca.inf  # x5 upper bound
    ubx[5: n_states * (N + 1): n_states] = ca.inf  # x6 upper bound

    u1_min = -500
    u1_max = 500

    u2_min = -0.54105
    u2_max = 0.54105

    lbx[n_states * (N + 1):: n_controls] = u1_min
    lbx[n_states * (N + 1) + 1:: n_controls] = u2_min

    ubx[n_states * (N + 1):: n_controls] = u1_max
    ubx[n_states * (N + 1) + 1:: n_controls] = u2_max

    args = {
        'lbg': ca.DM.zeros((n_states * (N + 1), 1)),  # constraints lower bound
        'ubg': ca.DM.zeros((n_states * (N + 1), 1)),  # constraints upper bound
        'lbx': lbx,
        'ubx': ubx}

    state_init = ca.DM([x1_init, x2_init, x3_init, x4_init, x5_init, x6_init])  # initial state
    # state_target = ca.DM([x1_target, x2_target, x3_target, x4_target])  # target state
    # u = 1e-4 * ca.DM.ones((n_controls, N))  # initial control

    args['p'] = ca.vertcat(state_init)
    X0 = ca.repmat(state_init, 1, N + 1)
    u0 = u_init * ca.DM.ones((n_controls, N))  # initial control

    # optimization variable current state
    args['x0'] = ca.vertcat(
        ca.reshape(X0, n_states * (N + 1), 1),
        ca.reshape(u0, n_controls * N, 1)
    )
    sol = solver(
        x0=args['x0'],
        lbx=args['lbx'],
        ubx=args['ubx'],
        lbg=args['lbg'],
        ubg=args['ubg'],
        p=args['p']
    )
    u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
    X_sol = ca.reshape(sol['x'][: n_states * (N + 1)], n_states, N + 1)

    u_sol = u

    return u_sol, X_sol
