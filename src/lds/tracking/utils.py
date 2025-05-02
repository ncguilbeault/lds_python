import math
import numpy as np


def getLDSmatricesForKinematics_np(dt, sigma_a, sigma_x, sigma_y):
    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.3.3

    # Eq. 6.3.3-2
    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, .5*dt**2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]],
                 dtype=np.double)
    Z = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]],
                 dtype=np.double)
    # Eq. 6.3.3-4
    Qe = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)
    R = np.diag([sigma_x**2, sigma_y**2])
    Q = buildQfromQe_np(Qe=Qe, sigma_ax=sigma_a, sigma_ay=sigma_a)

    return B, Q, Z, R, Qe


def buildQfromQe_np(Qe, sigma_ax, sigma_ay):
    Q = np.zeros_like(Qe)
    lower_slice = slice(0, int(Q.shape[0]/2))
    upper_slice = slice(int(Q.shape[0]/2), Q.shape[0])
    Q[lower_slice, lower_slice] = sigma_ax**2*Qe[lower_slice, lower_slice]
    Q[upper_slice, upper_slice] = sigma_ay**2*Qe[upper_slice, upper_slice]
    return Q


def getLDSmatricesForKinematics_torch(dt, sigma_a, sigma_x, sigma_y):
    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.3.3

    # Eq. 6.3.3-2
    import torch
    B = torch.array([[1, dt, .5*dt**2, 0, 0, 0],
                     [0, 1, dt, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, dt, .5*dt**2],
                     [0, 0, 0, 0, 1, dt],
                     [0, 0, 0, 0, 0, 1]],
                    dtype=torch.double)
    Z = torch.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0]],
                    dtype=torch.double)
    # Eq. 6.3.3-4
    Qe = torch.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                      [dt**3/2, dt**2,   dt,      0, 0, 0],
                      [dt**2/2, dt,      1,       0, 0, 0],
                      [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                      [0, 0, 0, dt**3/2, dt**2,   dt],
                      [0, 0, 0, dt**2/2, dt,      1]],
                     dtype=torch.double)
    R = torch.diag([sigma_x**2, sigma_y**2])
    Q = buildQfromQe_torch(Qe=Qe, sigma_ax=sigma_a, sigma_ay=sigma_a)

    return B, Q, Z, R, Qe


def buildQfromQe_torch(Qe, sigma_ax, sigma_ay):
    import torch
    Q = torch.zeros_like(Qe)
    lower_slice = slice(0, int(Q.shape[0]/2))
    upper_slice = slice(int(Q.shape[0]/2), Q.shape[0])
    Q[lower_slice, lower_slice] = sigma_ax**2*Qe[lower_slice, lower_slice]
    Q[upper_slice, upper_slice] = sigma_ay**2*Qe[upper_slice, upper_slice]
    return Q


def getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
        dt, sigma_a, sigma_x, sigma_y, alpha,
        sigma_cos_theta_state, sigma_sin_theta_state,
        sigma_omega,
        sigma_cos_theta_measurement, sigma_sin_theta_measurement):
    import torch

    def getB(dt, alpha):
        def B(x):
            pos_x = x[0]
            vel_x = x[1]
            acc_x = x[2]
            pos_y = x[3]
            vel_y = x[4]
            acc_y = x[5]
            cos_theta = x[6]
            sin_theta = x[7]
            omega = x[8]
            v = math.sqrt(vel_x**2 + vel_y**2)
            answer = torch.empty(size=(9,), dtype=torch.double)
            answer[0] = pos_x + dt * vel_x + .5 * dt**2 * acc_x
            answer[1] = vel_x + dt * acc_x
            answer[2] = acc_x
            answer[3] = pos_y + dt * vel_y + .5 * dt**2 * acc_y
            answer[4] = vel_y + dt * acc_y
            answer[5] = acc_y
            answer[6] = (alpha * dt * vel_x +
                         (1 - alpha * v * dt) * cos_theta -
                         omega * dt * sin_theta)
            answer[7] = (alpha * dt * vel_y +
                         omega * dt * cos_theta +
                         (1 - alpha * v * dt) * sin_theta)
            answer[8] = omega
            return answer
        return B

    def getZ():
        def Z(x):
            Zmat = torch.zeros(size=(4, 9), dtype=torch.double)
            Zmat[0, 0] = 1
            Zmat[1, 3] = 1
            Zmat[2, 6] = 1
            Zmat[3, 7] = 1
            answer = Zmat @ x
            return answer
        return Z

    def getQ(dt, sigma_a, sigma_cos_theta_state, sigma_sin_theta_state,
             sigma_omega):
        def Q(x):
            Qt = torch.tensor([[dt**4/4, dt**3/2, dt**2/2],
                               [dt**3/2, dt**2, dt],
                               [dt**2/2, dt, 1]], dtype=torch.double)
            Q = torch.zeros(size=(9, 9), dtype=torch.double)
            Q[:3, :3] = sigma_a**2 * Qt
            Q[3:6, 3:6] = sigma_a**2 * Qt
            Q[6, 6] = sigma_cos_theta_state**2
            Q[7, 7] = sigma_sin_theta_state**2
            Q[8, 8] = sigma_omega**2
            return Q
        return Q

    def getR(sigma_x, sigma_y,
             sigma_cos_theta_measurement,
             sigma_sin_theta_measurement):
        def R(x):
            R = torch.diag(torch.tensor([sigma_x**2, sigma_y**2,
                                         sigma_cos_theta_measurement**2,
                                         sigma_sin_theta_measurement**2]))
            return R
        return R

    B = getB(dt=dt, alpha=alpha)
    Z = getZ()
    Q = getQ(dt=dt, sigma_a=sigma_a,
             sigma_cos_theta_state=sigma_cos_theta_state,
             sigma_sin_theta_state=sigma_sin_theta_state,
             sigma_omega=sigma_omega)
    R = getR(sigma_x=sigma_x, sigma_y=sigma_y,
             sigma_cos_theta_measurement=sigma_cos_theta_measurement,
             sigma_sin_theta_measurement=sigma_sin_theta_measurement)
    return B, Z, Q, R
