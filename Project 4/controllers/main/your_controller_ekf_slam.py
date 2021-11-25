# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

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

        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.
        self.gains = np.matrix([20, 0.1, 5])
        self.e_previous = 0
        self.e_sum = 0
        self.e_der = 0

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120.0, 450.0, -500.0, 50.0
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1, 1)
            map_Y = map_Y.reshape(-1, 1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))

            # Parameters for EKF SLAM
            self.n = int(len(self.map) / 2)
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3 + 2 * self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1 * np.eye(3 + 2 * self.n)
            W = np.zeros((3 + 2 * self.n, 3 + 2 * self.n))
            W[0:3, 0:3] = delT ** 2 * 0.1 * np.eye(3)
            V = 0.1 * np.eye(2 * self.n)
            V[self.n :, self.n :] = 0.01 * np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)

            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3 + 2 * self.n)
            mu[0:3] = np.array([X, Y, psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")

        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3 + 2 * self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map

        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1, 2))

        y = np.zeros(2 * self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n + i] = wrapToPi(np.arctan2(m[i, 1] - p[1], m[i, 0] - p[0]) - psi)

        y = y + np.random.multivariate_normal(np.zeros(2 * self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        # You must not use true_X, true_Y and true_psi since they are for plotting purpose
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

        # You are free to reuse or refine your code from P3 in the spaces below.
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

        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
