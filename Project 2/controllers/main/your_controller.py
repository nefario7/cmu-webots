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

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).

        # ---------------|Lateral Controller|-------------------------
        A1 = np.zeros((4, 4))
        A1[1, 2] = 1
        A1[2, 2] = (-4 * Ca) / (m * xdot)
        A1[2, 3] = (4 * Ca) / m
        A1[2, 4] = (-2 * Ca * (lf - lr)) / (m * xdot)
        A1[3, 3] = 1
        A1[4, 2] = (-2 * Ca * (lf - lr)) / (Iz * xdot)
        A1[4, 3] = (2 * Ca * (lf - lr)) / Iz
        A1[4, 4] = (-2 * Ca * (lf ^ 2 + lr ^ 2)) / (Iz * xdot)
        B1 = np.zeros((4, 2))
        B1[2, 1] = (2 * Ca) / m
        B1[4, 1] = (2 * Ca * lf) / Iz
        C1 = np.eye(4)
        D1 = np.zeros((4, 2))

        poles1 = np.array([0, 0, 0, 0])
        lateral_place = signal.place_poles(A1, B1, poles1, method="YT")
        K1 = lateral_place.gain_matrix

        delta = -K1[1, 1] * 

        # ---------------|Longitudinal Controller|-------------------------

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
