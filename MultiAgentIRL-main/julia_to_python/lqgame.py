import numpy as np

def lqgame(dynamics, costs):
    # assume planning horizon T
    # return values:
    #   Ns: list len = T
    #   alphas: list len = T

    # dynamics matrices lists, each list should be of length T
    As = dynamics["A"]
    B1s = dynamics["B1"]
    B2s = dynamics["B2"]
    
    # cost matrices lists, Qs and ls should be of length T + 1, while Rs should be of length T
    Q1s = costs["Q1"]
    l1s = costs["l1"]
    Q2s = costs["Q2"]
    l2s = costs["l2"]
    R11s = costs["R11"]
    R12s = costs["R12"]
    R21s = costs["R21"]
    R22s = costs["R22"]
    
    # infer dimensions
    T = len(As)    # planning horizon
    n = B1s[0].shape[0]     # state dimension
    
    m1 = 1 if len(B1s[0].shape) == 1 else B1s[0].shape[1]    # control 1 dimension
    m2 = 1 if len(B1s[0].shape) == 1 else B2s[0].shape[1]    # control 2 dimension
    
    # return containers
    P1s, alpha1s = [], []
    P2s, alpha2s = [], []
    # intermediate computation + boundary conditions
    Z1s = [Q1s[-1]]
    Z2s = [Q2s[-1]]
    zeta1s = [l1s[-1]]
    zeta2s = [l2s[-1]]
    Fs, betas = [], []
    
    # initialize containers
    for t in range(T-1, -1, -1):
        # ##########################################
        # 1. solve for P_t^1 and P_t^2
        # ##########################################
        Z1_n = Z1s[-1]
        Z2_n = Z2s[-1]

        S = np.zeros((m1 + m2, m1 + m2))
        S[:m1, :m1] = R11s[t] + B1s[t].T @ Z1_n @ B1s[t]
        S[:m1, m1:] = B1s[t].T @ Z1_n @ B2s[t]
        S[m1:, :m1] = B2s[t].T @ Z2_n @ B1s[t]
        S[m1:, m1:] = R22s[t] + B2s[t].T @ Z2_n @ B2s[t]

        YN = np.zeros((m1 + m2, n))
        YN[:m1, :] = B1s[t].T @ Z1_n @ As[t]
        YN[m1:, :] = B2s[t].T @ Z2_n @ As[t]

        P = np.linalg.solve(S, YN)     # least square computation
        P1 = P[:m1, :]
        P2 = P[m1:, :]
        P1s.append(P1)
        P2s.append(P2)

        # ##########################################
        # 2. solve for the zero order term in FB strategy
        # ##########################################
        zeta1_n = zeta1s[-1]
        zeta2_n = zeta2s[-1]
        # use least square to solve for alphas, notice that the coefficient matrix (S) is the same as before
        YA = np.zeros((m1 + m2, 1))
        YA[:m1, 0] = B1s[t].T @ zeta1_n
        YA[m1:, 0] = B2s[t].T @ zeta2_n
        alpha = np.linalg.solve(S, YA)
        alpha1 = alpha[:m1]
        alpha2 = alpha[m1:]
        alpha1s.append(alpha1)
        alpha2s.append(alpha2)

        # ##########################################
        # 3. compute for next iteration
        # ##########################################
        # solve F for next iteration
        F = As[t] - B1s[t] @ P1 - B2s[t] @ P2
        beta = -B1s[t] @ alpha1 - B2s[t] @ alpha2
        Fs.append(F)
        betas.append(beta)

        zeta1 = (F.T @ Z1_n @ beta +
                 F.T @ zeta1_n +
                 P1.T @ R11s[t] @ alpha1 +
                 P2.T @ R12s[t] @ alpha2 +
                 l1s[t])
        zeta2 = (F.T @ Z2_n @ beta +
                 F.T @ zeta2_n +
                 P1.T @ R21s[t] @ alpha1 +
                 P2.T @ R22s[t] @ alpha2 +
                 l2s[t])
        zeta1s.append(zeta1)
        zeta2s.append(zeta2)

        Z1 = (F.T @ Z1_n @ F +
              P1.T @ R11s[t] @ P1 +
              P2.T @ R12s[t] @ P2 +
              Q1s[t])
        Z2 = (F.T @ Z2_n @ F +
              P2.T @ R22s[t] @ P2 +
              P1.T @ R21s[t] @ P1 +
              Q2s[t])
        Z1s.append(Z1)
        Z2s.append(Z2)

    return P1s, P2s, alpha1s, alpha2s

def lqgame_QRE(dynamics, costs):
    # Quantal Response Equilibria for Dynamic Markov Games
    # assume planning horizon T
    # return values:
    #   Ns: list len = T
    #   alphas: list len = T
    
    # dynamics matrices lists, each list should be of length T
    As = dynamics["A"]
    B1s = dynamics["B1"]
    B2s = dynamics["B2"]
    
    # cost matrices lists, Qs and ls should be of length T + 1, while Rs should be of length T
    Q1s = costs["Q1"]
    l1s = costs["l1"]
    Q2s = costs["Q2"]
    l2s = costs["l2"]
    R11s = costs["R11"]
    R12s = costs["R12"]
    R21s = costs["R21"]
    R22s = costs["R22"]
    
    
    # infer dimensions
    T = len(As)    # planning horizon
    n = B1s[0].shape[0]     # state dimension
    m1 = 1 if len(B1s[0].shape) == 1 else B1s[0].shape[1]    # control 1 dimension
    m2 = 1 if len(B2s[0].shape) == 1 else B2s[0].shape[1]    # control 2 dimension
    
    # return containers
    P1s, alpha1s, cov1s = [], [], []
    P2s, alpha2s, cov2s = [], [], []

    # intermediate computation
    Z1s, Z2s = [], []
    Fs = []
    zeta1s, zeta2s = [], []
    betas = []

    # boundary conditions
    Z1s.append(Q1s[-1])
    Z2s.append(Q2s[-1])
    zeta1s.append(l1s[-1])
    zeta2s.append(l2s[-1])
    
    # initialize containers
    for t in range(T-1, -1, -1):
        # ##########################################
        # 1. solve for P_t^1 and P_t^2
        # ##########################################
        print(t)
        Z1_n = Z1s[-1]
        Z2_n = Z2s[-1]

        S = np.zeros((m1 + m2, m1 + m2))
        S[:m1, :m1] = R11s[t] + B1s[t].T @ Z1_n @ B1s[t]
        S[:m1, m1:] = B1s[t].T @ Z1_n @ B2s[t]
        S[m1:, :m1] = B2s[t].T @ Z2_n @ B1s[t]
        S[m1:, m1:] = R22s[t] + B2s[t].T @ Z2_n @ B2s[t]

        YN = np.zeros((m1 + m2, n))
        YN[:m1, :] = B1s[t].T @ Z1_n @ As[t]
        YN[m1:, :] = B2s[t].T @ Z2_n @ As[t]
        
        P = np.linalg.solve(S, YN)    # least square computation
        P1 = P[:m1, :]
        P2 = P[m1:, :]
        P1s.append(P1)
        P2s.append(P2)

        cov1s.append(np.linalg.inv(S[:m1, :m1]))
        cov2s.append(np.linalg.inv(S[m1:, m1:]))

        # ##########################################
        # 2. solve for the zero order term in FB strategy
        # ##########################################

        zeta1_n = zeta1s[-1]
        zeta2_n = zeta2s[-1]
        # use least square to solve for alphas, notice that the coefficient matrix (S) is the same as before
        YA = np.zeros((m1 + m2, 1))
        YA[:m1, :] = (B1s[t].T @ zeta1_n).reshape((m1,1))
        YA[m1:, :] = (B2s[t].T @ zeta2_n).reshape((m2,1))
        alpha = np.linalg.solve(S, YA)
        alpha1 = alpha[:m1].reshape(-1)
        alpha2 = alpha[m1:].reshape(-1)
        alpha1s.append(alpha1)
        alpha2s.append(alpha2)
        
        # ##########################################
        # 3. compute for next iteration
        # ##########################################
        # solve F for next iteration
        F = As[t] - B1s[t] @ P1 - B2s[t] @ P2
        Fs.append(F)

        beta = -B1s[t] @ alpha1 - B2s[t] @ alpha2
        betas.append(beta)

        zeta1 = F.T @ Z1_n @ beta + F.T @ zeta1_n + P1.T @ R11s[t] @ alpha1 + P2.T @ R12s[t] @ alpha2 + l1s[t]
        zeta2 = F.T @ Z2_n @ beta + F.T @ zeta2_n + P1.T @ R21s[t] @ alpha1 + P2.T @ R22s[t] @ alpha2 + l2s[t]
        zeta1s.append(zeta1)
        zeta2s.append(zeta2)

        Z1 = F.T @ Z1_n @ F + P1.T @ R11s[t] @ P1 + P2.T @ R12s[t] @ P2 + Q1s[t]
        Z2 = F.T @ Z2_n @ F + P2.T @ R22s[t] @ P2 + P1.T @ R21s[t] @ P1 + Q2s[t]
        Z1s.append(Z1)
        Z2s.append(Z2)

    return P1s, P2s, alpha1s, alpha2s, cov1s, cov2s

def lqgame_QRE_3player(dynamics, costs):
    """
    Quantal Response Equilibrium solver for 3-player linear-quadratic dynamic games.

    Inputs:
        dynamics: dict with keys "A", "B1", "B2", "B3"
        costs: dict with keys "Q1", "l1", ..., "R33"
    Returns:
        Tuple of lists: P1s, P2s, P3s, alpha1s, alpha2s, alpha3s, cov1s, cov2s, cov3s
    """
    As = dynamics["A"]
    B1s = dynamics["B1"]
    B2s = dynamics["B2"]
    B3s = dynamics["B3"]

    Q1s, l1s = costs["Q1"], costs["l1"]
    Q2s, l2s = costs["Q2"], costs["l2"]
    Q3s, l3s = costs["Q3"], costs["l3"]
    R11s, R12s, R13s = costs["R11"], costs["R12"], costs["R13"]
    R21s, R22s, R23s = costs["R21"], costs["R22"], costs["R23"]
    R31s, R32s, R33s = costs["R31"], costs["R32"], costs["R33"]

    T = len(As)
    n = B1s[0].shape[0]
    m1 = 1 if B1s[0].ndim == 1 else B1s[0].shape[1]
    m2 = 1 if B2s[0].ndim == 1 else B2s[0].shape[1]
    m3 = 1 if B3s[0].ndim == 1 else B3s[0].shape[1]

    P1s, alpha1s, cov1s = [], [], []
    P2s, alpha2s, cov2s = [], [], []
    P3s, alpha3s, cov3s = [], [], []

    Z1s, zeta1s = [Q1s[-1]], [l1s[-1]]
    Z2s, zeta2s = [Q2s[-1]], [l2s[-1]]
    Z3s, zeta3s = [Q3s[-1]], [l3s[-1]]

    for t in reversed(range(T)):
        Z1_n, Z2_n, Z3_n = Z1s[-1], Z2s[-1], Z3s[-1]

        S = np.zeros((m1 + m2 + m3, m1 + m2 + m3))
        S[:m1, :m1] = R11s[t] + B1s[t].T @ Z1_n @ B1s[t]
        S[:m1, m1:m1+m2] = B1s[t].T @ Z1_n @ B2s[t]
        S[:m1, m1+m2:] = B1s[t].T @ Z1_n @ B3s[t]

        S[m1:m1+m2, :m1] = B2s[t].T @ Z2_n @ B1s[t]
        S[m1:m1+m2, m1:m1+m2] = R22s[t] + B2s[t].T @ Z2_n @ B2s[t]
        S[m1:m1+m2, m1+m2:] = B2s[t].T @ Z2_n @ B3s[t]

        S[m1+m2:, :m1] = B3s[t].T @ Z3_n @ B1s[t]
        S[m1+m2:, m1:m1+m2] = B3s[t].T @ Z3_n @ B2s[t]
        S[m1+m2:, m1+m2:] = R33s[t] + B3s[t].T @ Z3_n @ B3s[t]

        YN = np.zeros((m1 + m2 + m3, n))
        YN[:m1, :] = B1s[t].T @ Z1_n @ As[t]
        YN[m1:m1+m2, :] = B2s[t].T @ Z2_n @ As[t]
        YN[m1+m2:, :] = B3s[t].T @ Z3_n @ As[t]

        P = np.linalg.solve(S, YN)
        P1, P2, P3 = P[:m1, :], P[m1:m1+m2, :], P[m1+m2:, :]
        P1s.append(P1)
        P2s.append(P2)
        P3s.append(P3)

        cov1s.append(np.linalg.inv(S[:m1, :m1]))
        cov2s.append(np.linalg.inv(S[m1:m1+m2, m1:m1+m2]))
        cov3s.append(np.linalg.inv(S[m1+m2:, m1+m2:]))

        zeta1_n, zeta2_n, zeta3_n = zeta1s[-1], zeta2s[-1], zeta3s[-1]

        YA = np.zeros((m1 + m2 + m3, 1))
        YA[:m1, :] = B1s[t].T @ zeta1_n
        YA[m1:m1+m2, :] = B2s[t].T @ zeta2_n
        YA[m1+m2:, :] = B3s[t].T @ zeta3_n

        alpha = np.linalg.solve(S, YA)
        alpha1, alpha2, alpha3 = alpha[:m1], alpha[m1:m1+m2], alpha[m1+m2:]
        alpha1s.append(alpha1)
        alpha2s.append(alpha2)
        alpha3s.append(alpha3)

        F = As[t] - B1s[t] @ P1 - B2s[t] @ P2 - B3s[t] @ P3
        beta = - B1s[t] @ alpha1 - B2s[t] @ alpha2 - B3s[t] @ alpha3

        zeta1 = F.T @ Z1_n @ beta + F.T @ zeta1_n + P1.T @ R11s[t] @ alpha1 + P2.T @ R12s[t] @ alpha2 + P3.T @ R13s[t] @ alpha3 + l1s[t]
        zeta2 = F.T @ Z2_n @ beta + F.T @ zeta2_n + P1.T @ R21s[t] @ alpha1 + P2.T @ R22s[t] @ alpha2 + P3.T @ R23s[t] @ alpha3 + l2s[t]
        zeta3 = F.T @ Z3_n @ beta + F.T @ zeta3_n + P1.T @ R31s[t] @ alpha1 + P2.T @ R32s[t] @ alpha2 + P3.T @ R33s[t] @ alpha3 + l3s[t]

        Z1 = F.T @ Z1_n @ F + P1.T @ R11s[t] @ P1 + P2.T @ R12s[t] @ P2 + P3.T @ R13s[t] @ P3 + Q1s[t]
        Z2 = F.T @ Z2_n @ F + P1.T @ R21s[t] @ P1 + P2.T @ R22s[t] @ P2 + P3.T @ R23s[t] @ P3 + Q2s[t]
        Z3 = F.T @ Z3_n @ F + P1.T @ R31s[t] @ P1 + P2.T @ R32s[t] @ P2 + P3.T @ R33s[t] @ P3 + Q3s[t]

        Z1s.append(Z1)
        Z2s.append(Z2)
        Z3s.append(Z3)
        zeta1s.append(zeta1)
        zeta2s.append(zeta2)
        zeta3s.append(zeta3)

    return P1s, P2s, P3s, alpha1s, alpha2s, alpha3s, cov1s, cov2s, cov3s