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
        self.e1_last = 0
        self.e2_last = 0

        self.ex_last = 0
        self.ex_sum = 0

        self.kpx = 20
        self.kix = 0
        self.kdx = 1

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        lookahead = 50
        max_node = len(trajectory)

        closest_distance, closest_idx = closestNode(X, Y, trajectory)
        lookahead_idx = closest_idx + lookahead

        X_d = trajectory[min(lookahead_idx, max_node - 1)][0]
        Y_d = trajectory[min(lookahead_idx, max_node - 1)][1]
        psi_d = np.arctan2(Y_d - Y, X_d - X)

        velocity_d = 40

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

        # * Pole Placement
        e1 = np.sqrt((X_d - X) ** 2 + (Y_d - Y) ** 2)
        e1dot = 0
        e2 = wrapToPi(psi - psi_d)
        e2dot = psidot - (psi_d - self.e2_last) / delT

        self.e1_last = X_d
        self.e2_last = psi_d

        poles = np.array([-0.1, -0.01, -0.02, -0.001])

        lateral_place = signal.place_poles(A, B, poles, method="YT")
        K = lateral_place.gain_matrix
        delta = -K[0, 0] * e1 - K[0, 1] * e1dot - K[0, 2] * e2 - K[0, 3] * e2dot

        # ---------------|Longitudinal Controller|-------------------------
        # * PID Controller
        ex = velocity_d * abs(np.cos(psi)) - xdot
        ex_der = ex - self.ex_last
        self.ex_sum += ex
        self.ex_last = ex
        F = (self.kpx * self.ex_last) + (self.kix * self.ex_sum) * delT + (self.kdx * ex_der) / delT

        # print("___________________________________________")
        # print(f"Error = \t{e1:.2f} \t{e1dot:.2f} \t{e2:.2f} \t{e2dot:.2f}")
        # print(f"Gains = \t{K[0, 0]:.2f} \t{K[0, 1]:.2f} \t{K[0, 2]:.2f} \t{K[0, 3]:.2f}")
        # print(f"Prop = \t{K[0, 0] * e1/delta:.2f} \t{K[0, 1] * e1dot/delta:.2f} \t{K[0, 2] * e2/delta:.2f} \t{K[0, 3] * e2dot/delta:.2f}")
        # print(f"Steering = {delta:.3f}")

        # print(f"Error = \t{e1:.2f} \t{e1dot:.2f} \t{e2:.2f} \t{e2dot:.2f}")
        # print(f"Force = {F:.3f}")

        return X, Y, xdot, ydot, psi, psidot, F, delta
