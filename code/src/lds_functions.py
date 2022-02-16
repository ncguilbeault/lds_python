
import numpy as np


def filterLDS_SS_withMissingValues(y, B, Q, m0, V0, Z, R):
    """ Kalman filter implementation of the algorithm described in Shumway and
    Stoffer 2006.

    :param: y: time series to be smoothed
    :type: y: numpy array (NxT)

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)

    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)

    :return:  {xnn1, Vnn1, xnn, Vnn, innov, K, Sn, logLike}: xnn1 and Vnn1 (predicted means, MxT, and covariances, MxMxT), xnn and Vnn (filtered means, MxT, and covariances, MxMxT), innov (innovations, NxT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    # N: number of observations
    # M: dim state space
    # P: dim observations
    M = B.shape[0]
    N = y.shape[1]
    P = y.shape[0]
    xnn1 = np.empty(shape=[M, 1, N])
    Vnn1 = np.empty(shape=[M, M, N])
    xnn = np.empty(shape=[M, 1, N])
    Vnn = np.empty(shape=[M, M, N])
    innov = np.empty(shape=[P, 1, N])
    Sn = np.empty(shape=[P, P, N])

    # k==0
    xnn1[:, 0, 0] = np.dot(B, m0)
    Vnn1[:, :, 0] = np.matmul(B, np.matmul(V0, np.transpose(B)))+Q
    Stmp = np.matmul(Z, np.matmul(Vnn1[:, :, 0], np.transpose(Z)))+R
    Sn[:, :, 0] = (Stmp+np.transpose(Stmp))/2
    Sinv = np.linalg.inv(Sn[:, :, 0])
    K = np.matmul(Vnn1[:, :, 0], np.matmul(np.transpose(Z), Sinv))
    innov[:, 0, 0] = y[:, 0]-(np.matmul(Z, xnn1[:, :, 0]).squeeze())
    xnn[:, :, 0] = xnn1[:, :, 0]+np.matmul(K, innov[:, :, 0])
    Vnn[:, :, 0] = Vnn1[:, :, 0]-np.matmul(K, np.matmul(Z, Vnn1[:, :, 0]))
    logLike = -N*P*np.log(2*np.pi)-np.linalg.slogdet(Sn[:, :, 0])[1] - \
        np.matmul(np.transpose(innov[:, :, 0]),
                  np.matmul(Sinv, innov[:, :, 0]))

    # k>1
    for k in range(1, N):
        xnn1[:, :, k] = np.matmul(B, xnn[:, :, k-1])
        Vnn1[:, :, k] = np.matmul(B, np.matmul(Vnn[:, :, k-1],
                                               np.transpose(B))) + Q
        if(np.any(np.isnan(y[:, k]))):
            xnn[:, :, k] = xnn1[:, :, k]
            Vnn[:, :, k] = Vnn1[:, :, k]
        else:
            Stmp = np.matmul(Z, np.matmul(Vnn1[:, :, k], np.transpose(Z))) + R
            Sn[:, :, k] = (Stmp+np.transpose(Stmp))/2
            Sinv = np.linalg.inv(Sn[:, :, k])
            K = np.matmul(Vnn1[:, :, k], np.matmul(np.transpose(Z), Sinv))
            innov[:, 0, k] = y[:, k]-(np.matmul(Z, xnn1[:, :, k]).squeeze())
            xnn[:, :, k] = xnn1[:, :, k] + np.matmul(K, innov[:, :, k])
            Vnn[:, :, k] = Vnn1[:, :, k] - \
                np.matmul(K, np.matmul(Z, Vnn1[:, :, k]))
        logLike = logLike-np.linalg.slogdet(Sn[:, :, k])[1] -\
            np.matmul(np.transpose(innov[:, :, k]),
                      np.matmul(Sinv, innov[:, :, k]))
    logLike = 0.5*logLike
    answer = {"xnn1": xnn1, "Vnn1": Vnn1, "xnn": xnn, "Vnn": Vnn,
              "innov": innov, "KN": K, "Sn": Sn, "logLike": logLike}
    return answer


def smoothLDS_SS(B, xnn, Vnn, xnn1, Vnn1, m0, V0):
    """ Kalman smoother implementation

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: xnn: filtered means (from Kalman filter)
    :type: xnn: numpy array (MxT)

    :param: Vnn: filtered covariances (from Kalman filter)
    :type: Vnn: numpy array (MxMXT)

    :param: xnn1: predicted means (from Kalman filter)
    :type: xnn1: numpy array (MxT)

    :param: Vnn1: predicted covariances (from Kalman filter)
    :type: Vnn1: numpy array (MxMXT)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :return:  {xnN, VnN, Jn, x0N, V0N, J0}: xnn1 and Vnn1 (smoothed means, MxT, and covariances, MxMxT), Jn (smoothing gain matrix, MxMxT), x0N and V0N (smoothed initial state mean, M, and covariance, MxM), J0 (initial smoothing gain matrix, MxN).

    """
    N = xnn.shape[2]
    M = B.shape[0]
    xnN = np.empty(shape=[M, 1, N])
    VnN = np.empty(shape=[M, M, N])
    Jn = np.empty(shape=[M, M, N])

    xnN[:, :, -1] = xnn[:, :, -1]
    VnN[:, :, -1] = Vnn[:, :, -1]
    for n in reversed(range(1, N)):
        Jn[:, :, n-1] = np.matmul(Vnn[:, :, n-1],
                                  np.matmul(np.transpose(B),
                                            np.linalg.inv(Vnn1[:, :, n])))
        xnN[:, :, n-1] = xnn[:, :, n-1] + \
            np.matmul(Jn[:, :, n-1], xnN[:, :, n]-xnn1[:, :, n])
        VnN[:, :, n-1] = Vnn[:, :, n-1] + \
            np.matmul(Jn[:, :, n-1], np.matmul(VnN[:, :, n]-Vnn1[:, :, n],
                                               np.transpose(Jn[:, :, n-1])))
    # initial state x00 and V00
    # return the smooth estimates of the state at time 0: x0N and V0N
    J0 = np.matmul(V0, np.matmul(np.transpose(B),
                                 np.linalg.inv(Vnn1[:, :, 0])))
    x0N = m0 + np.matmul(J0, (xnN[:, :, 0]-xnn1[:, :, 0]))
    V0N = V0 + np.matmul(J0, np.matmul(VnN[:, :, 0]-Vnn1[:, :, 0],
                                       np.transpose(J0)))
    answer = {"xnN": xnN, "VnN": VnN, "Jn": Jn, "x0N": x0N, "V0N": V0N,
              "J0": J0}
    return answer
