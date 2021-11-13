# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):
    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.
        self.gains = np.matrix([50, 0.1, 5])
        self.e_previous = 0
        self.e_sum = 0
        self.e_der = 0

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        closest_distance, closest_idx = closestNode(X, Y, trajectory)

        # * Lookahead point
        lookahead = 150

        lookahead_idx = closest_idx + lookahead
        X_d = trajectory[min(lookahead_idx, len(trajectory) - 1)][0]
        Y_d = trajectory[min(lookahead_idx, len(trajectory) - 1)][1]
        psi_d = np.arctan2(Y_d - Y, X_d - X)

        # ---------------|Lateral Controller|-------------------------
        # * State Space
        A = np.zeros((4, 4))
        A[0, 1] = 1
        A[2, 3] = 1
        A[1, 1] = -(4 * Ca) / (m * xdot)
        A[1, 2] = (4 * Ca) / m
        A[1, 3] = -(2 * Ca * (lf - lr)) / (m * xdot)
        A[3, 1] = -(2 * Ca * (lf - lr)) / (Iz * xdot)
        A[3, 2] = (2 * Ca * (lf - lr)) / Iz
        A[3, 3] = -(2 * Ca * (lf ** 2 + lr ** 2)) / (Iz * xdot)
        B = np.zeros((4, 1))
        B[1, 0] = (2 * Ca) / m
        B[3, 0] = (2 * Ca * lf) / Iz
        C = np.zeros((1, 4))
        C[0, 3] = 1
        D = np.zeros((1, 1))

        discrete_system = signal.cont2discrete((A, B, C, D), delT, method="zoh")
        Ad = discrete_system[0]
        Bd = discrete_system[1]

        # * LQR Controller
        Q = np.array([[10, 0, 0, 0], [0, 1, 0, 0], [0, 0, 100, 0], [0, 0, 0, 10]])
        R = np.array([1])

        S = linalg.solve_discrete_are(Ad, Bd, Q, R)
        K = -linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A

        e1 = (Y - Y_d) * np.cos(psi_d) + (X - X_d) * np.sin(psi_d)
        e2 = wrapToPi(psi - psi_d)
        e1dot = ydot * np.cos(e2) + xdot * np.sin(e2)
        e2dot = psidot
        lateral_errors = [[e1], [e1dot], [e2], [e2dot]]

        delta = float(K @ lateral_errors)

        # ---------------|Longitudinal Controller|-------------------------
        # * PID Controller
        e = np.sqrt((X_d - X) ** 2 + (Y_d - Y) ** 2)
        self.e_sum += e
        self.e_der = e - self.e_previous
        self.e_previous = e
        longitudinal_errors = np.matrix([[e], [self.e_sum * delT], [self.e_der / delT]])

        F = float(self.gains @ longitudinal_errors)

        return X, Y, xdot, ydot, psi, psidot, F, delta
