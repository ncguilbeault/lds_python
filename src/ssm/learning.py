
import math
import time
import numpy as np
import scipy.optimize
import warnings

from . import inference
from .tracking import utils

iteration = 0


def scipy_optimize_SS_tracking_fullV0(y, B, sigma_a0, Qe, Z, diag_R_0,
                                      m0_0, V0_0, max_iter, disp=True):
    iL_V0 = np.tril_indices(V0_0.shape[0])

    def get_coefs_from_params(sigma_a, diag_R, m0, V0, iL_V0=iL_V0):
        V0_chol = np.linalg.cholesky(V0)
        L_coefs = V0_chol[iL_V0]
        x = np.insert(np.concatenate([diag_R, m0, L_coefs]), 0, sigma_a)
        return x

    def get_params_from_coefs(x, iL_V0=iL_V0, sigma_a0=sigma_a0,
                              diag_R_0=diag_R_0, m0_0=m0_0, V0_0=V0_0):
        cur = 0
        sigma_a = x[slice(cur, cur+1)]
        cur += len(sigma_a)
        diag_R = x[slice(cur, cur+len(diag_R_0))]
        cur += len(diag_R)
        m0 = x[slice(cur, cur+len(m0_0))]
        cur += len(m0)
        M = V0_0.shape[0]
        L_coefs = x[slice(cur, cur+int(M*(M+1)/2))]
        L_V0 = np.zeros(shape=V0_0.shape)
        L_V0[iL_V0] = L_coefs
        V0 = L_V0 @ L_V0.T

        return sigma_a, diag_R, m0, V0

    def optim_criterion(x):
        sigma_a, diag_R, m0, V0 = get_params_from_coefs(x)
        R = np.diag(diag_R)
        kf = inference.filterLDS_SS_withMissingValues(y=y, B=B, Q=sigma_a*Qe,
                                                      m0=m0, V0=V0, Z=Z, R=R)
        answer = 0
        N = kf["Sn"].shape[2]
        for n in range(N):
            innov = kf["innov"][:, :, n]
            Sn = kf["Sn"][:,:,n] 

            Sn_inv = np.linalg.inv(Sn)
            answer += np.linalg.slogdet(Sn)[1]
            answer += innov.T @ Sn_inv @ innov
        return answer

    def callback(x):
        global iteration
        iteration += 1

        sigma_a, diag_R, m0, V0 = get_params_from_coefs(x)
        optim_value = optim_criterion(x=x)
        print("Iteration: {:d}".format(iteration))
        print("optim criterion: {:f}".format(optim_value.item()))
        print("sigma_a={:f}".format(sigma_a.item()))
        print("diag_R:")
        print(diag_R)
        print("m0:")
        print(m0)
        print("V0:")
        print(V0)

    x0 = get_coefs_from_params(sigma_a=sigma_a0, diag_R=diag_R_0,
                               m0=m0_0, V0=V0_0)
    options={"disp": disp, "maxiter": max_iter}
    opt_res = scipy.optimize.minimize(optim_criterion, x0, method="Nelder-Mead",
                                      callback=callback, options=options)


def torch_lbfgs_optimize_kinematicsHO_logLikeEKF_diagV0(
    dt, y, sigma_a0,
    cos_theta_Q_std0,
    sin_theta_Q_std0,
    omega_Q_std0,
    pos_x_R_std0,
    pos_y_R_std0,
    cos_theta_R_std0,
    sin_theta_R_std0,
    alpha0,
    m0_kinematics_0,
    m0_HO_0,
    sqrt_diag_V0_kinematics_0,
    sqrt_diag_V0_HO_0,
    max_iter=20, lr=1.0,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe",
    n_epochs = 100, tol=1e-6,
    vars_to_estimate={
        "sigma_a": False,
        "cos_theta_Q_std": True,
        "sin_theta_Q_std": True,
        "omega_Q_std": True,
        "sqrt_pos_x_R_std": False,
        "sqrt_pos_y_R_std": False,
        "sqrt_cos_theta_R_std": True,
        "sqrt_sin_theta_R_std": True,
        "alpha": True,
        "m0_kinematics": False,
        "m0_HO": True,
        "sqrt_diag_V0_kinematics": False,
        "sqrt_diag_V0_HO": True,
    },
    disp=True):

    import torch
    def log_likelihood_fn():
        B, Bdot, Z, Zdot, Q, R = utils.getNDSwithGaussianNoiseFunctionsForKinematicsAndHO_torch(
            dt=dt, sigma_a=sigma_a,
            cos_theta_Q_std=cos_theta_Q_std,
            sin_theta_Q_std=sin_theta_Q_std,
            omega_Q_std=omega_Q_std,
            pos_x_R_std=pos_x_R_std,
            pos_y_R_std=pos_y_R_std,
            cos_theta_R_std=cos_theta_R_std,
            sin_theta_R_std=sin_theta_R_std,
            alpha=alpha,
        )
        log_like = inference.logLikeEKF_withMissingValues_torch(
            y=y, B=B, Bdot=Bdot, Q=Q, m0=m0, V0=V0, Z=Z, Zdot=Zdot, R=R)
        return log_like

    optim_params = {"max_iter": max_iter, "lr": lr,
                    "tolerance_grad": tolerance_grad,
                    "tolerance_change": tolerance_change,
                    "line_search_fn": line_search_fn}
    m0 = torch.cat([m0_kinematics_0, m0_HO_0])
    sqrt_diag_V0 = torch.cat([sqrt_diag_V0_kinematics_0, sqrt_diag_V0_HO_0])
    sigma_a = torch.tensor([sigma_a0], dtype=torch.double)
    cos_theta_Q_std = torch.tensor([cos_theta_Q_std0], dtype=torch.double)
    sin_theta_Q_std = torch.tensor([sin_theta_Q_std0], dtype=torch.double)
    omega_Q_std = torch.tensor([omega_Q_std0], dtype=torch.double)

    pos_x_R_std = torch.tensor([pos_x_R_std0], dtype=torch.double)
    pos_y_R_std = torch.tensor([pos_y_R_std0], dtype=torch.double)
    cos_theta_R_std = torch.tensor([cos_theta_R_std0], dtype=torch.double)
    sin_theta_R_std = torch.tensor([sin_theta_R_std0], dtype=torch.double)

    alpha = torch.tensor([alpha0], dtype=torch.double)

    x = []
    if vars_to_estimate["sigma_a"]:
        x.append(sigma_a)
    if vars_to_estimate["cos_theta_Q_std"]:
        x.append(cos_theta_Q_std)
    if vars_to_estimate["sin_theta_Q_std"]:
        x.append(sin_theta_Q_std)
    if vars_to_estimate["omega_Q_std"]:
        x.append(omega_Q_std)
    if vars_to_estimate["pos_x_R_std"]:
        x.append(pos_x_R_std)
    if vars_to_estimate["pos_y_R_std"]:
        x.append(pos_y_R_std)
    if vars_to_estimate["cos_theta_R_std"]:
        x.append(cos_theta_R_std)
    if vars_to_estimate["sin_theta_R_std"]:
        x.append(sin_theta_R_std)
    if vars_to_estimate["alpha"]:
        x.append(alpha)
    if vars_to_estimate["m0_kinematics"] or vars_to_estimate["m0_HO"]:
        x.append(m0)
    if vars_to_estimate["sqrt_diag_V0_kinematics"] or vars_to_estimate["sqrt_diag_V0_HO"]:
        x.append(sqrt_diag_V0)
    if len(x) == 0:
        raise RuntimeError("No variable to estimate. Please set one element "
                           "of vars_to_estimate to True")
    optimizer = torch.optim.LBFGS(x, **optim_params)
    for i in range(len(x)):
        x[i].requires_grad = True

    def closure():
        optimizer.zero_grad()
        curEval = -log_likelihood_fn()
        curEval.backward()
        print_string = f"ll={-curEval}"
        if vars_to_estimate["sigma_a"]:
            print_string += f", sigma_a={sigma_a.item()}"
        if vars_to_estimate["cos_theta_Q_std"]:
            print_string += f", cos_theta_Q_std={cos_theta_Q_std.item()}"
        if vars_to_estimate["sin_theta_Q_std"]:
            print_string += f", sin_theta_Q_std={sin_theta_Q_std.item()}"
        if vars_to_estimate["omega_Q_std"]:
            print_string += f", omega_Q_std={omega_Q_std.item()}"
        if vars_to_estimate["pos_x_R_std"]:
            print_string += f", pos_x_R_std={pos_x_R_std.item()}"
        if vars_to_estimate["pos_y_R_std"]:
            print_string += f", pos_y_R_std={pos_y_R_std.item()}"
        if vars_to_estimate["cos_theta_R_std"]:
            print_string += f", cos_theta_R_std={cos_theta_R_std.item()}"
        if vars_to_estimate["sin_theta_R_std"]:
            print_string += f", sin_theta_R_std={sin_theta_R_std.item()}"
        if vars_to_estimate["alpha"]:
            print_string += f", alpha={alpha.item()}"
        if vars_to_estimate["m0_kinematics"]:
            m0.grad[6:] = 0.0
            print_string += f", m0={m0.tolist()}"
        if vars_to_estimate["m0_HO"]:
            m0.grad[:6] = 0.0
            print_string += f", m0={m0.tolist()}"
        if vars_to_estimate["sqrt_diag_V0_kinematics"]:
            sqrt_diag_V0.grad[6:] = 0.0
            print_string += f", sqrt_diag_V0={sqrt_diag_V0.tolist()}"
        if vars_to_estimate["sqrt_diag_V0_HO"]:
            sqrt_diag_V0.grad[:6] = 0.0
            print_string += f", sqrt_diag_V0={sqrt_diag_V0.tolist()}"
        print(print_string)
        return curEval

    termination_info = "success: reached maximum number of iterations"
    log_like = []
    elapsed_time = []
    start_time = time.time()
    V0 = torch.diag(sqrt_diag_V0)
    curEval = -log_likelihood_fn()
    log_like.append(-curEval.item())
    elapsed_time.append(time.time() - start_time)
    print("--------------------------------------------------------------------------------")
    print(f"startup")
    print(f"likelihood: {log_like[-1]}")
    for epoch in range(n_epochs):
        optimizer.step(closure)
        curEval = -log_likelihood_fn()
        log_like.append(-curEval.item())
        elapsed_time.append(time.time() - start_time)
        print("--------------------------------------------------------------------------------")
        print(f"epoch: {epoch}")
        print(f"likelihood: {log_like[-1]}")
        if vars_to_estimate["sigma_a"]:
            print("sigma_a: ")
            print(sigma_a.item())
        if vars_to_estimate["cos_theta_Q_std"]:
            print("cos_theta_Q_std: ")
            print(cos_theta_Q_std.item())
        if vars_to_estimate["sin_theta_Q_std"]:
            print("sin_theta_Q_std: ")
            print(sin_theta_Q_std.item())
        if vars_to_estimate["omega_Q_std"]:
            print("omega_Q_std: ")
            print(omega_Q_std.item())
        if vars_to_estimate["pos_x_R_std"]:
            print("pos_x_R_std: ")
            print(pos_x_R_std.item())
        if vars_to_estimate["pos_y_R_std"]:
            print("pos_y_R_std: ")
            print(pos_y_R_std.item())
        if vars_to_estimate["cos_theta_R_std"]:
            print("cos_theta_R_std: ")
            print(cos_theta_R_std.item())
        if vars_to_estimate["sin_theta_R_std"]:
            print("sin_theta_R_std: ")
            print(sin_theta_R_std.item())
        if vars_to_estimate["alpha"]:
            print("alpha: ")
            print(alpha.item())
        if vars_to_estimate["m0_kinematics"] or vars_to_estimate["m0_HO"]:
            print("m0: ")
            print(m0)
        if vars_to_estimate["sqrt_diag_V0_kinematics"] or vars_to_estimate["sqrt_diag_V0_HO"]:
            print("sqrt_diag_V0: ")
            print(sqrt_diag_V0)
        if epoch > 0 and log_like[-1] - log_like[-2] < tol:
            termination_info = "success: converged"
            break
    for i in range(len(x)):
        x[i].requires_grad = False

    estimates = {}
    if vars_to_estimate["sigma_a"]:
        estimates["sigma_a"] = sigma_a
    if vars_to_estimate["cos_theta_Q_std"]:
        estimates["cos_theta_Q_std"] = cos_theta_Q_std
    if vars_to_estimate["sin_theta_Q_std"]:
        estimates["sin_theta_Q_std"] = sin_theta_Q_std
    if vars_to_estimate["omega_Q_std"]:
        estimates["omega_Q_std"] = omega_Q_std
    if vars_to_estimate["pos_x_R_std"]:
        estimates["pos_x_R_std"] = pos_x_R_std
    if vars_to_estimate["pos_y_R_std"]:
        estimates["pos_y_R_std"] = pos_y_R_std
    if vars_to_estimate["cos_theta_R_std"]:
        estimates["cos_theta_R_std"] = cos_theta_R_std
    if vars_to_estimate["sin_theta_R_std"]:
        estimates["sin_theta_R_std"] = sin_theta_R_std
    if vars_to_estimate["alpha"]:
        estimates["alpha"] = alpha
    if vars_to_estimate["m0_kinematics"] or vars_to_estimate["m0_HO"]:
        estimates["m0"] = m0
    if vars_to_estimate["sqrt_diag_V0_kinematics"] or vars_to_estimate["sqrt_diag_V0_HO"]:
        estimates["sqrt_diag_V0"] = sqrt_diag_V0
    answer = {"estimates": estimates,
              "log_like": log_like,
              "elapsed_time": elapsed_time,
              "termination_info": termination_info}
    return answer


def scipy_optimize_SS_tracking_diagV0(y, B, sigma_ax0, sigma_ay0, Qe, Z,
                                      sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                      max_iter=50, disp=True):

    def get_coefs_from_params(sigma_ax, sigma_ay, sqrt_diag_R, m0,
                              sqrt_diag_V0):
        x = np.concatenate([[sigma_ax, sigma_ay], sqrt_diag_R, m0,
                            sqrt_diag_V0])
        return x

    def get_params_from_coefs(x, sigma_ax0=sigma_ax0, sigma_ay0=sigma_ay0,
                              sqrt_diag_R_0=sqrt_diag_R_0, m0_0=m0_0,
                              sqrt_diag_V0_0=sqrt_diag_V0_0):
        cur = 0
        sigma_ax = x[slice(cur, cur+1)]
        cur += len(sigma_ax)
        sigma_ay = x[slice(cur, cur+1)]
        cur += len(sigma_ay)
        sqrt_diag_R = x[slice(cur, cur+len(sqrt_diag_R_0))]
        cur += len(sqrt_diag_R)
        m0 = x[slice(cur, cur+len(m0_0))]
        cur += len(m0)
        sqrt_diag_V0 = x[slice(cur, cur+len(sqrt_diag_V0_0))]

        return sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0

    def optim_criterion(x):
        sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
            get_params_from_coefs(x)
        V0 = np.diag(sqrt_diag_V0**2)
        R = np.diag(sqrt_diag_R**2)
        # build Q from Qe, sigma_ax, sigma_ay
        Q = utils.buildQfromQe_np(Qe=Qe, sigma_ax=sigma_ax, sigma_ay=sigma_ay)

        kf = inference.filterLDS_SS_withMissingValues_np(y=y, B=B, Q=Q,
                                                         m0=m0, V0=V0, Z=Z, R=R)
        answer = 0
        N = kf["Sn"].shape[2]
        for n in range(N):
            innov = kf["innov"][:, :, n]
            Sn = kf["Sn"][:, :, n]

            Sn_inv = np.linalg.inv(Sn)
            answer += np.linalg.slogdet(Sn)[1]
            answer += innov.T @ Sn_inv @ innov
        return answer

    def callback(x):
        global iteration
        iteration += 1

        sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
            get_params_from_coefs(x)
        optim_value = optim_criterion(x=x)
        print("Iteration: {:d}".format(iteration))
        print("optim criterion: {:f}".format(optim_value.item()))
        print("sigma_ax={:f}".format(sigma_ax.item()))
        print("sigma_ay={:f}".format(sigma_ay.item()))
        print("sqrt_diag_R:")
        print(sqrt_diag_R)
        print("m0:")
        print(m0)
        print("sqrt_diag_V0:")
        print(sqrt_diag_V0)

    x0 = get_coefs_from_params(sigma_ax=sigma_ax0, sigma_ay=sigma_ay0,
                               sqrt_diag_R=sqrt_diag_R_0, m0=m0_0,
                               sqrt_diag_V0=sqrt_diag_V0_0)
    options = {"disp": disp, "maxiter": max_iter}
    opt_res = scipy.optimize.minimize(optim_criterion, x0,
                                      method="Nelder-Mead", callback=callback,
                                      options=options)
    sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
        get_params_from_coefs(opt_res["x"])
    x = {"sigma_ax": sigma_ax, "sigma_ay": sigma_ay,
         "sqrt_diag_R": sqrt_diag_R, "m0": m0,
         "sqrt_diag_V0": sqrt_diag_V0}
    answer = {"fun": opt_res["fun"], "message": opt_res["message"],
              "nfev": opt_res["nfev"], "nit": opt_res["nit"],
              "status": opt_res["status"], "success": opt_res["success"],
              "x": x}
    return answer


def torch_lbfgs_optimize_SS_tracking_diagV0(y, B, Qe, Z,
                                            sigma_a0, pos_x_R_std0,
                                            pos_y_R_std0,
                                            m0_0, sqrt_diag_V0_0,
                                            max_iter=20, lr=1.0,
                                            tolerance_grad=1e-7,
                                            tolerance_change=1e-9,
                                            n_epochs = 100, tol=1e-6,
                                            line_search_fn="strong_wolfe",
                                            vars_to_estimate={
                                                "sigma_a": True,
                                                "pos_x_R_std": True,
                                                "pos_y_R_std": True,
                                                "m0": True,
                                                "sqrt_diag_V0": True}):

    import torch
    def log_likelihood_fn():
        V0 = torch.diag(sqrt_diag_V0**2)
        e1 = torch.tensor([1, 0], dtype=torch.double)
        e2 = torch.tensor([0, 1], dtype=torch.double)
        R = (pos_x_R_std**2 * torch.outer(e1, e1) +
             pos_y_R_std**2 * torch.outer(e2, e2))
        Q = Qe * sigma_a**2
        log_like = inference.logLikeLDS_withMissingValues_torch(
            y=y, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
        return log_like

    optim_params = {"max_iter": max_iter, "lr": lr,
                    "tolerance_grad": tolerance_grad,
                    "tolerance_change": tolerance_change,
                    "line_search_fn": line_search_fn}
    sigma_a = torch.tensor([sigma_a0], dtype=torch.double)
    pos_x_R_std = torch.tensor([pos_x_R_std0], dtype=torch.double)
    pos_y_R_std = torch.tensor([pos_y_R_std0], dtype=torch.double)
    m0 = m0_0
    sqrt_diag_V0 = sqrt_diag_V0_0
    x = []
    if vars_to_estimate["sigma_a"]:
        x.append(sigma_a)
    if vars_to_estimate["pos_x_R_std"]:
        x.append(pos_x_R_std)
    if vars_to_estimate["pos_y_R_std"]:
        x.append(pos_y_R_std)
    if vars_to_estimate["m0"]:
        x.append(m0)
    if vars_to_estimate["sqrt_diag_V0"]:
        x.append(sqrt_diag_V0)
    if len(x) == 0:
        raise RuntimeError("No variable to estimate. Please set one element "
                           "of vars_to_estimate to True")
    optimizer = torch.optim.LBFGS(x, **optim_params)
    for i in range(len(x)):
        x[i].requires_grad = True

    def closure():
        optimizer.zero_grad()
        curEval = -log_likelihood_fn()
        curEval.backward()
        print_string = f"ll={-curEval}"
        if vars_to_estimate["sigma_a"]:
            print_string += f", sigma_a={sigma_a.item()}"
        if vars_to_estimate["pos_x_R_std"]:
            print_string += f", pos_x_R_std={pos_x_R_std.item()}"
        if vars_to_estimate["pos_y_R_std"]:
            print_string += f", pos_y_R_std={pos_y_R_std.item()}"
        if vars_to_estimate["m0"]:
            print_string += f", m0={m0}"
        if vars_to_estimate["sqrt_diag_V0"]:
            print_string += f", sqrt_diag_V0={sqrt_diag_V0}"
        print(print_string)
        return curEval

    termination_info = "success: reached maximum number of iterations"
    log_like = []
    elapsed_time = []
    start_time = time.time()
    curEval = -log_likelihood_fn()
    log_like.append(-curEval.item())
    elapsed_time.append(time.time() - start_time)
    print("--------------------------------------------------------------------------------")
    print(f"startup")
    print(f"likelihood: {log_like[-1]}")
    for epoch in range(n_epochs):
        curEval = optimizer.step(closure)
        curEval = -log_likelihood_fn()
        log_like.append(-curEval.item())
        elapsed_time.append(time.time() - start_time)
        print("--------------------------------------------------------------------------------")
        print(f"epoch: {epoch}")
        print(f"likelihood: {log_like[-1]}")
        if vars_to_estimate["sigma_a"]:
            print("sigma_a: ")
            print(sigma_a)
        if vars_to_estimate["pos_x_R_std"]:
            print("pos_x_R_std: ")
            print(pos_x_R_std)
        if vars_to_estimate["pos_y_R_std"]:
            print("pos_y_R_std: ")
            print(pos_y_R_std)
        if vars_to_estimate["m0"]:
            print("m0: ")
            print(m0)
        if vars_to_estimate["sqrt_diag_V0"]:
            print("sqrt_diag_V0: ")
            print(sqrt_diag_V0)
        if epoch > 0 and log_like[-1] - log_like[-2] < tol:
            termination_info = "success: converged"
            break
    for i in range(len(x)):
        x[i].requires_grad = False

    estimates = {}
    initial_conditions = {}
    if vars_to_estimate["sigma_a"]:
        initial_conditions["sigma_a"] = sigma_a0
        estimates["sigma_a"] = sigma_a
    if vars_to_estimate["pos_x_R_std"]:
        initial_conditions["pos_x_R_std"] = pos_x_R_std0
        estimates["pos_x_R_std"] = pos_x_R_std
    if vars_to_estimate["pos_y_R_std"]:
        initial_conditions["pos_y_R_std"] = pos_y_R_std0
        estimates["pos_y_R_std"] = pos_y_R_std
    if vars_to_estimate["m0"]:
        initial_conditions["m0"] = m0_0
        estimates["m0"] = m0
    if vars_to_estimate["sqrt_diag_V0"]:
        initial_conditions["sqrt_diag_V0"] = sqrt_diag_V0_0
        estimates["sqrt_diag_V0"] = sqrt_diag_V0
    answer = {"initial_conditions": initial_conditions,
              "estimates": estimates,
              "log_like": log_like,
              "elapsed_time": elapsed_time,
              "termination_info": termination_info}
    return answer


def torch_adam_optimize_SS_tracking_diagV0(y, B, sigma_a0, Qe, Z,
                                           sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                           max_iter=50, lr=1e-3, eps=1e-8,
                                           vars_to_estimate={
                                               "sigma_a": True,
                                               "R": True, "m0": True,
                                               "V0": True},
                                          ):

    import torch
    def log_likelihood_fn():
        V0 = torch.diag(sqrt_diag_V0**2)
        R = torch.diag(sqrt_diag_R**2)
        Q = Qe * sigma_a**2
        kf = inference.filterLDS_SS_withMissingValues_torch(y=y, B=B, Q=Q,
                                                            m0=m0, V0=V0, Z=Z,
                                                            R=R)
        log_like = kf["logLike"]
        return log_like

    optim_params = {"lr": lr, "eps": eps}
    sigma_a = torch.Tensor([sigma_a0])
    sqrt_diag_R = sqrt_diag_R_0
    m0 = m0_0
    sqrt_diag_V0 = sqrt_diag_V0_0
    x = []
    if vars_to_estimate["sigma_a"]:
        x.append(sigma_a)
    if vars_to_estimate["sqrt_diag_R"]:
        x.append(sqrt_diag_R)
    if vars_to_estimate["m0"]:
        x.append(m0)
    if vars_to_estimate["sqrt_diag_V0"]:
        x.append(sqrt_diag_V0)
    if len(x) == 0:
        raise RuntimeError("No variable to estimate. Please set one element "
                           "of vars_to_estimate to True")
    optimizer = torch.optim.Adam(x, **optim_params)
    for i in range(len(x)):
        x[i].requires_grad = True

    log_like = []
    elapsed_time = []
    start_time = time.time()
    for i in range(max_iter):
        optimizer.zero_grad()
        curEval = -log_likelihood_fn()
        curEval.backward()
        optimizer.step()
        log_like.append(-curEval.item())
        elapsed_time.append(time.time() - start_time)
        print("--------------------------------------------------------------------------------")
        print(f"iteration: {i} ({max_iter})")
        print(f"logLike: {-curEval}")
        print("sigma_a: ")
        print(sigma_a)
        print("sqrt_diag_R: ")
        print(sqrt_diag_R)
        print("m0: ")
        print(m0)
        print("sqrt_diag_V0: ")
        print(sqrt_diag_V0)

    for i in range(len(x)):
        x[i].requires_grad = False

    estimates = {}
    if vars_to_estimate["sigma_a"]:
        e_sigma_a = x.pop(0)[0].item()
        estimates["sigma_a"] = e_sigma_a
    if vars_to_estimate["sqrt_diag_R"]:
        e_sqrt_diag_R = x.pop(0)
        estimates["sqrt_diag_R"] = e_sqrt_diag_R
    if vars_to_estimate["m0"]:
        e_m0 = x.pop(0)
        estimates["m0"] = e_m0
    if vars_to_estimate["sqrt_diag_V0"]:
        e_sqrt_diag_V0 = x.pop(0).numpy()
        estimates["sqrt_diag_V0"] = e_sqrt_diag_V0
    answer = {"estimates": estimates,
              "log_like": log_like,
              "elapsed_time": elapsed_time,
             }
    return answer


def em_SS_tracking(y, B, sigma_a0, Qe, Z, R_0, m0_0, V0_0,
                   vars_to_estimate={"sigma_a": True, "R": True,
                                     "m0": True, "V0": True},
                   max_iter=50, tolerance_change=1e-9):
    sigma_a = sigma_a0
    R = R_0
    m0 = m0_0
    V0 = V0_0

    Qe_inv = np.linalg.inv(Qe)
    N = y.shape[1]
    M = Qe.shape[0]
    termination_info = "success: reached maximum number of iterations"
    log_like = []
    elapsed_time = []
    start_time = time.time()
    for iter in range(max_iter):
        # E step
        Q = Qe * sigma_a**2
        kf = inference.filterLDS_SS_withMissingValues_np(y=y, B=B,
                                                         Q=Q, m0=m0, V0=V0,
                                                         Z=Z, R=R)
        print("LogLike[{:04d}]={:f}".format(iter, kf["logLike"].item()))
        print("sigma_a={:f}".format(sigma_a))
        print("R:")
        print(R)
        print("m0:")
        print(m0)
        print("V0:")
        print(V0)
        log_like.append(kf["logLike"].item())
        if math.isnan(log_like[-1]) or iter > 0 and log_like[-1] < log_like[-2]:
            if math.isnan(log_like[-1]):
                termination_info = "nan detected"
            else:
                termination_info = "log likelihood decreased"
            # begin backtrack
            if vars_to_estimate["sigma_a"]:
                sigma_a = sigma_a_prev
            if vars_to_estimate["R"]:
                R = R_prev
            if vars_to_estimate["m0"]:
                m0 = m0_prev
            if vars_to_estimate["V0"]:
                V0 = V0_prev
            del log_like[-1]
            # end backtrack
            break
        elif iter > 0 and log_like[-1] - log_like[-2] < tolerance_change:
            termination_info = "converged"
            break
        elapsed_time.append(time.time() - start_time)
        ks = inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Pnn=kf["Pnn"],
                                    xnn1=kf["xnn1"], Pnn1=kf["Pnn1"],
                                    m0=m0, V0=V0)
        # M step
        if vars_to_estimate["sigma_a"]:
            sigma_a_prev = sigma_a
            S11, S10, S00 = posteriorCorrelationMatrices(Z=Z, B=B, KN=kf["KN"],
                                                         Pnn=kf["Pnn"], xnN=ks["xnN"],
                                                         PnN=ks["PnN"], x0N=ks["x0N"],
                                                         V0N=ks["V0N"], Jn=ks["Jn"],
                                                         J0=ks["J0"])
            # sigma_a
            W = S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T
            U = W @ Qe_inv
            K = np.trace(U)
            sigma_a = np.sqrt(K/(N*M))
        # R
        if vars_to_estimate["R"]:
            R_prev = R
            u = y[:, 0] - (Z @ ks["xnN"][:, :, 0]).squeeze()
            R = np.outer(u, u) + Z @ ks["PnN"][:, :, 0] @ Z.T
            for i in range(1, N):
                u = y[:, i] - (Z @ ks["xnN"][:, :, i]).squeeze()
                R = R + np.outer(u, u) + Z @ ks["PnN"][:, :, i] @ Z.T
            R = R/N

        # m0, V0
        if vars_to_estimate["m0"]:
            m0_prev = m0
            m0 = ks["x0N"].squeeze()

        if vars_to_estimate["V0"]:
            V0_prev = V0
            V0 = ks["V0N"]
    estimates = {}
    if vars_to_estimate["sigma_a"]:
        estimates["sigma_a"] = sigma_a
    if vars_to_estimate["R"]:
        estimates["R"] = R
    if vars_to_estimate["m0"]:
        estimates["m0"] = m0
    if vars_to_estimate["V0"]:
        estimates["V0"] = V0

    optim_res = {"estimates": estimates,
                 "log_like": log_like,
                 "elapsed_time": elapsed_time,
                 "termination_info": termination_info,
                }
    return optim_res


def em_SS_LDS(y, B0, Q0, Z0, R0, m0_0, V0_0, max_iter=50, tol=1e-4,
              vars_to_estimate=dict(m0=True, V0=True, B=True, Q=True, Z=True,
                                    R=True)):
    B  = B0
    Q  = Q0
    Z  = Z0
    R  = R0
    m0 = m0_0
    V0 = V0_0

    M = B0.shape[0]
    N = y.shape[1]
    log_like = np.empty(max_iter)
    prev_log_like = -np.inf
    for iter in range(max_iter):
        kf = inference.filterLDS_SS_withMissingValues_np(y=y, B=B,
                                                         Q=Q, m0=m0, V0=V0,
                                                         Z=Z, R=R)
        print("LogLike[{:04d}]={:f}".format(iter, kf["logLike"].item()))
        log_like[iter] = kf["logLike"]
        assert(kf["logLike"] > prev_log_like)
        if (kf["logLike"] - prev_log_like) < tol:
            break
        prev_log_like = kf["logLike"]
        ks = inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Pnn=kf["Pnn"],
                                    xnn1=kf["xnn1"], Pnn1=kf["Pnn1"],
                                    m0=m0, V0=V0)

        S11, S10, S00 = posteriorCorrelationMatrices(Z=Z, B=B, KN=kf["KN"],
                                                     Pnn=kf["Pnn"], xnN=ks["xnN"],
                                                     PnN=ks["PnN"], x0N=ks["x0N"],
                                                     V0N=ks["V0N"], Jn=ks["Jn"],
                                                     J0=ks["J0"])
        if vars_to_estimate["Z"]:
            Z = np.outer(y[:,0], ks["xnN"][:, :, 0])
            for i in range(1, N):
                Z = Z + np.outer(y[:, i], ks["xnN"][:, :, i])
            Z = Z @ np.linalg.inv(S11)

        if vars_to_estimate["B"]:
            B = S10 @ np.linalg.inv(S00)

        if vars_to_estimate["Q"]:
            Q = (S11 - S10 @ np.linalg.inv(S00) @ S10.T)/N
            Q = (Q.T + Q)/2

        # Now that Z is estimated, lets estimate R, if requested
        if vars_to_estimate["R"]:
            u = y[:, 0] - (Z @ ks["xnN"][:, :, 0]).squeeze()
            R = np.outer(u, u) + Z @ ks["PnN"][:, :, 0] @ Z.T
            for i in range(1, N):
                u = y[:, i] - (Z @ ks["xnN"][:, :, i]).squeeze()
                R = R + np.outer(u, u) + Z @ ks["PnN"][:, :, i] @ Z.T
            R = R/N

        if vars_to_estimate["m0"]:
            m0 = ks["x0N"].squeeze()

        if vars_to_estimate["V0"]:
            V0 = ks["V0N"]

    answer = dict(B=B, Q=Q, Z=Z, R=R, m0=m0, V0=V0, log_like=log_like[:(iter+1)],
                  niter=iter)
    return answer

def posteriorCorrelationMatrices(Z, B, KN, Pnn, xnN, PnN, x0N, V0N, Jn, J0):
    # We want to first estimate Z and then R, because R depends on Z
    Pnn1N = lag1CovSmootherLDS_SS(Z=Z, KN=KN, B=B, Pnn=Pnn, Jn=Jn, J0=J0)
    S11 = np.outer(xnN[:,:,0], xnN[:,:,0]) + PnN[:,:,0]
    S10 = np.outer(xnN[:,:,0], x0N) + Pnn1N[:,:,0]
    S00 = np.outer(x0N, x0N) + V0N
    N = xnN.shape[2]
    for i in range(1, N):
        S11 = S11 + np.outer(xnN[:, :, i], xnN[:, :, i]) + PnN[:, :, i]
        S10 = S10 + np.outer(xnN[:, :, i], xnN[:, :, i-1]) + Pnn1N[:, :, i]
        S00 = S00 + np.outer(xnN[:, :, i-1], xnN[:, :, i-1]) + PnN[:, :, i-1]
    return S11, S10, S00

def lag1CovSmootherLDS_SS(Z, KN, B, Pnn, Jn, J0):
    #SS16, Property 6.3
    M = KN.shape[0]
    N = Pnn.shape[2]
    Pnn1N = np.empty(shape=(M, M, N))
    eye = np.eye(M)
    Pnn1N[:, :, N-1] = (eye - KN @ Z) @ B @ Pnn[:, :, N-2]
    for k in range(N-1, 1, -1):
        Pnn1N[:, :, k-1] = Pnn[:, :, k-1] @ Jn[:, :, k-2].T + \
                           Jn[:, :, k-1] @ \
                           (Pnn1N[:, :, k] - B @ Pnn[:, :, k-1]) @ Jn[:, :, k-2].T
    Pnn1N[:, :, 0] = Pnn[:, :, 0] @ J0.T + Jn[:, :, 0] @ (Pnn1N[:, :, 1] - B @ Pnn[:, :, 0]) @ J0.T
    return Pnn1N
