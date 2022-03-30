from time import time
import casadi as ca
import numpy as np
import math
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from pendulumCT0_LinearModel import pendulum_Linear_Model
from RealModel import pendulumCT0

ct = 0
N = 20
Ts = 0.1  # time between steps in seconds
# horizon = N - (ct % N)

x1_target = 0
x2_target = 0
x3_target = 0
x4_target = 0

Q1 = Q3 = 3
Q2 = Q4 = 0.4

# state symbolic variables
x1 = ca.SX.sym('x1')
x2 = ca.SX.sym('x2')
x3 = ca.SX.sym('x3')
x4 = ca.SX.sym('x4')

states = ca.vertcat(
    x1,
    x2,
    x3,
    x4
)
n_states = states.numel()
u = ca.SX.sym('u')
controls = u
n_controls = controls.numel()

# column vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)
X = ca.SX.sym('X', n_states, horizon)
U = ca.SX.sym('U', n_controls, horizon)

OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),  # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)

# state weights matrix
Q = ca.diagcat(Q1, Q2, Q3, Q4)
# controls weights matrix
R = 1e-5
# Casadi symbolic functions
f = ca.Function('f', [states, controls], [pendulum_Linear_Model(states, controls)])  # pendulum_Linear_Model
f_nonlinear = ca.Function('f1', [states, controls], [pendulumCT0(states, controls)])  # nonlinear model
plant = ca.Function('plant', [states, controls], [pendulumCT0(states, controls)])  # Actual plant
specs = [Q, R, Ts, f, n_states, n_controls]
cost_fn, g = const_and_CostFcn_new(horizon, X, U, P, specs)
args = newargs_new(horizon, specs)