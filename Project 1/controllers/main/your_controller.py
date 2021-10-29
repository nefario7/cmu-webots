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
        self.f = 0.019

        # Add additional member variables according to your need here.
        self.kpy = 0.6
        self.kiy = 0.01
        self.kdy = 1

        self.kpx = 20
        self.kix = 0.1
        self.kdx = 5

        self.eyp_sum = 0
        self.eyp_last = 0

        self.ex_sum = 0
        self.ex_last = 0

        self.max_dev = -100
        self.dev = []

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        f = self.f

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).

        dist, closest_index = closestNode(X, Y, trajectory)
        look_ahead = 100

        X_d = trajectory[min(closest_index + look_ahead, len(trajectory) - 1), 0]
        Y_d = trajectory[min(closest_index + look_ahead, len(trajectory) - 1), 1]
        psi_d = np.arctan2(Y_d - Y, X_d - X)

        # T---------------|Lateral Controller|-------------------------
        epsi = wrapToPi(psi_d - psi)
        # print("Lateral Error = {:.2f}".format(epsi))

        # * PID Controller
        eyp = epsi
        eyp_der = (eyp - self.eyp_last) * delT
        self.eyp_sum += eyp * delT
        self.eyp_last = eyp
        # print(f"Lateral Components - P {(self.kpy * self.eyp_last):.2f}, I {(self.kiy * self.eyp_sum):.2f}, D{(self.kdy * eyp_der):.4f}")
        delta = (self.kpy * self.eyp_last) + (self.kiy * self.eyp_sum) + (self.kdy * eyp_der)

        # ---------------|Longitudinal Controller|-------------------------
        ex = np.sqrt((X_d - X) ** 2 + (Y_d - Y) ** 2)
        # print("Longitudinal Error = {:.2f}".format(ex))

        # * PID Controller
        ex_der = (ex - self.ex_last) / delT
        self.ex_sum += ex * delT
        self.ex_last = ex
        # print(f"Longitudinal Components - P {(self.kpx * self.ex_last):.2f}, I {(self.kix * self.ex_sum):.2f}, D{(self.kdx * ex_der):.2f}")
        F = (self.kpx * self.ex_last) + (self.kix * self.ex_sum) + (self.kdx * ex_der)

        # * Constraint Check
        if abs(delta) > np.pi / 6:
            print("Limiting Steering!")
            delta = max(delta, np.pi / 6)
        if F < 0 or F > 15736:
            F = clamp(F, 0, 15736)
        if xdot < 10 ** -5:
            xdot = min(xdot, 10 ** -5)

        return X, Y, xdot, ydot, psi, psidot, F, delta
