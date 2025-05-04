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
        dt, alpha, sigma_a,
        cos_theta_Q_sigma, sin_theta_Q_sigma,
        omega_Q_sigma,
        pos_x_R_sigma, pos_y_R_sigma,
        cos_theta_R_sigma, sin_theta_R_sigma):
    import torch

    def getBdot(dt, alpha, epsilon=1e-10):
        def Bdot(x):
            def pAWRTxdot(dt, alpha, cos_theta, vel_x, v):
                answer = -alpha * dt * cos_theta * vel_x / v + alpha * dt
                return answer

            def pBWRTxdot(dt, alpha, cos_theta, vel_x, v):
                answer = -alpha * dt * sin_theta * vel_x / v
                return answer

            def pAWRTydot(dt, alpha, cos_theta, vel_x, v):
                answer = -alpha * dt * cos_theta * vel_y / v
                return answer

            def pBWRTydot(dt, alpha, cos_theta, vel_x, v):
                answer = -alpha * dt * sin_theta * vel_y / v + alpha * dt
                return answer

            pos_x = x[0]
            vel_x = x[1]
            acc_x = x[2]
            pos_y = x[3]
            vel_y = x[4]
            acc_y = x[5]
            cos_theta = x[6]
            sin_theta = x[7]
            omega = x[8]
            v = math.sqrt(vel_x**2 + vel_y**2 + epsilon)
            A = ((1 - alpha * v * dt) * cos_theta - omega * dt * sin_theta +
                 alpha * dt * vel_x)
            B = ((1 - alpha * v * dt) * sin_theta + omega * dt * cos_theta +
                 alpha * dt * vel_y)

            pAWRTxdot_val = pAWRTxdot(dt=dt, alpha=alpha, cos_theta=cos_theta,
                                      vel_x=vel_x, v=v)
            pBWRTxdot_val = pBWRTxdot(dt=dt, alpha=alpha, cos_theta=cos_theta,
                                      vel_x=vel_x, v=v)
            pSpeedWRTxdot_val = A * pAWRTxdot_val + B * pBWRTxdot_val / (A**2 + B**2)
            pCosWRTxdot = (pAWRTxdot_val * math.sqrt(A**2 + B**2) -
                           A * pSpeedWRTxdot_val) / (A**2 + B**2)
            pSinWRTxdot = (pBWRTxdot_val * math.sqrt(A**2 + B**2) -
                           B * pSpeedWRTxdot_val) / (A**2 + B**2)

            pAWRTydot_val = pAWRTydot(dt=dt, alpha=alpha, cos_theta=cos_theta,
                                      vel_x=vel_x, v=v)
            pBWRTydot_val = pBWRTydot(dt=dt, alpha=alpha, cos_theta=cos_theta,
                                      vel_x=vel_x, v=v)
            pSpeedWRTydot_val = A * pAWRTydot_val + B * pBWRTydot_val / (A**2 + B**2)
            pCosWRTydot = (pAWRTydot_val * math.sqrt(A**2 + B**2) -
                           A * pSpeedWRTydot_val) / (A**2 + B**2)
            pSinWRTydot = (pBWRTydot_val * math.sqrt(A**2 + B**2) -
                           B * pSpeedWRTydot_val) / (A**2 + B**2)

            pAWRTcosTheta_val = 1 - alpha * v * dt
            pBWRTcosTheta_val = omega * dt
            pSpeedWRTcosTheta_val = A * pAWRTcosTheta_val + B * pBWRTcosTheta_val / (A**2 + B**2)
            pCosWRTcosTheta = (pAWRTcosTheta_val * math.sqrt(A**2 + B**2) -
                               A * pSpeedWRTcosTheta_val) / (A**2 + B**2)
            pSinWRTcosTheta = (pBWRTcosTheta_val * math.sqrt(A**2 + B**2) -
                               B * pSpeedWRTcosTheta_val) / (A**2 + B**2)

            pAWRTsinTheta_val = -omega * dt
            pBWRTsinTheta_val = 1 - alpha * v * dt
            pSpeedWRTsinTheta_val = A * pAWRTsinTheta_val + B * pBWRTsinTheta_val / (A**2 + B**2)
            pCosWRTsinTheta = (pAWRTsinTheta_val * math.sqrt(A**2 + B**2) -
                               A * pSpeedWRTsinTheta_val) / (A**2 + B**2)
            pSinWRTsinTheta = (pBWRTsinTheta_val * math.sqrt(A**2 + B**2) -
                               B * pSpeedWRTsinTheta_val) / (A**2 + B**2)

            pAWRTomega_val = -dt * sin_theta
            pBWRTomega_val = dt * cos_theta
            pSpeedWRTomega_val = A * pAWRTomega_val + B * pBWRTomega_val / (A**2 + B**2)
            pCosWRTomega = (pAWRTomega_val * math.sqrt(A**2 + B**2) -
                            A * pSpeedWRTomega_val) / (A**2 + B**2)
            pSinWRTomega = (pBWRTomega_val * math.sqrt(A**2 + B**2) -
                            B * pSpeedWRTomega_val) / (A**2 + B**2)

            aBdot = torch.zeros(size=(9, 9), dtype=torch.double)
            aBdot[0, 0] = 1; aBdot[0, 1] = dt; aBdot[0, 2] = .5*dt**2
            aBdot[1, 1] = 1; aBdot[1, 2] = dt
            aBdot[2, 2] = 1
            aBdot[3, 3] = 1; aBdot[3, 4] = dt; aBdot[3, 5] = .5*dt**2
            aBdot[4, 4] = 1; aBdot[4, 5] = dt
            aBdot[5, 5] = 1
            aBdot[6, 1] = pCosWRTxdot; aBdot[6, 4] = pCosWRTydot; aBdot[6, 6] = pCosWRTcosTheta; aBdot[6, 7] = pCosWRTsinTheta
            aBdot[7, 1] = pSinWRTxdot; aBdot[7, 4] = pSinWRTydot; aBdot[7, 6] = pSinWRTcosTheta; aBdot[7, 7] = pSinWRTsinTheta
            aBdot[8, 8] = 1
            return aBdot
        return Bdot

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
            A = (alpha * dt * vel_x +
                 (1 - alpha * v * dt) * cos_theta -
                 omega * dt * sin_theta)
            B = (alpha * dt * vel_y +
                 omega * dt * cos_theta +
                 (1 - alpha * v * dt) * sin_theta)
            answer[6] = A / math.sqrt(A**2 + B**2)
            answer[7] = B / math.sqrt(A**2 + B**2)
            answer[8] = omega
            return answer
        return B

    def getZdot():
        def Zdot(x):
            Zmat = torch.zeros(size=(4, 9), dtype=torch.double)
            Zmat[0, 0] = 1
            Zmat[1, 3] = 1
            Zmat[2, 6] = 1
            Zmat[3, 7] = 1
            return Zmat
        return Zdot

    def getZ():
        def Z(x):
            Zmat = getZdot()(x=None)
            answer = Zmat @ x
            return answer
        return Z

    def getQ(dt, sigma_a, cos_theta_Q_sigma, sin_theta_Q_sigma,
             omega_Q_sigma):
        def Q(x):
            Qt = torch.tensor([[dt**4/4, dt**3/2, dt**2/2],
                               [dt**3/2, dt**2, dt],
                               [dt**2/2, dt, 1]], dtype=torch.double)
            Q = torch.zeros(size=(9, 9), dtype=torch.double)
            Q[:3, :3] = sigma_a**2 * Qt
            Q[3:6, 3:6] = sigma_a**2 * Qt
            Q[6, 6] = cos_theta_Q_sigma**2
            Q[7, 7] = sin_theta_Q_sigma**2
            Q[8, 8] = omega_Q_sigma**2
            return Q
        return Q

    def getR(pos_x_R_sigma, pos_y_R_sigma,
             cos_theta_R_sigma,
             sin_theta_R_sigma):
        def R(x):
            R = torch.diag(torch.tensor([pos_x_R_sigma**2, pos_y_R_sigma**2,
                                         cos_theta_R_sigma**2,
                                         sin_theta_R_sigma**2]))
            return R
        return R

    B = getB(dt=dt, alpha=alpha)
    Bdot = getBdot(dt=dt, alpha=alpha)
    Z = getZ()
    Zdot = getZdot()
    Q = getQ(dt=dt, sigma_a=sigma_a,
             cos_theta_Q_sigma=cos_theta_Q_sigma,
             sin_theta_Q_sigma=sin_theta_Q_sigma,
             omega_Q_sigma=omega_Q_sigma)
    R = getR(pos_x_R_sigma=pos_x_R_sigma, pos_y_R_sigma=pos_y_R_sigma,
             cos_theta_R_sigma=cos_theta_R_sigma,
             sin_theta_R_sigma=sin_theta_R_sigma)
    return B, Bdot, Z, Zdot, Q, R
