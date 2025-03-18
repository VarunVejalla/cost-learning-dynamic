################################
# lqgame.jl      solve finite horizon , discrete time linear quadratic game
# author:     mingyuw@stanford.edu
################################
function lqgame(dynamics::Dict{String, Array{Array{Float64}}},
                 costs::Dict{String, Array{Array{Float64}}})
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
    T = length(As)    # planning horizon
    n = size(B1s[1])[1]     # state dimension
    if(length(size(B1s[1])) == 1)
        m1 = 1
    else
        m1 = size(B1s[1])[2]    # control 1 dimension
    end
    if(length(size(B2s[1])) == 1)
        m2 = 1
    else
        m2 = size(B2s[1])[2]    # control 2 dimension
    end


    # return containers
    P1s, alpha1s = [], []
    P2s, alpha2s = [], []
    # intermediate computation
    Z1s, Z2s = [], []
    Fs = []
    zeta1s, zeta2s = [], []
    betas = []

    # boundary conditions
    push!(Z1s, Q1s[end])
    push!(Z2s, Q2s[end])
    push!(zeta1s, l1s[end])
    push!(zeta2s, l2s[end])

    # initialize containers
    for t = T:-1:1
        # note that in zero-based time index, the actual time index should be t-1
        # ##########################################
        # 1. solve for P_t^1 and P_t^2
        # ##########################################
        Z1_n = Z1s[end]
        Z2_n = Z2s[end]

        S = zeros(m1+m2, m1+m2)
        S[1:m1, 1:m1]             = R11s[t] + B1s[t]' * Z1_n * B1s[t]
        S[1:m1, m1+1:m1+m2]       = B1s[t]' * Z1_n * B2s[t]
        S[m1+1:m1+m2, 1:m1]       = B2s[t]' * Z2_n * B1s[t]
        S[m1+1:m1+m2, m1+1:m1+m2] = R22s[t] + B2s[t]' * Z2_n * B2s[t]
        YN = zeros(m1+m2, n)
        YN[1:m1, :]        = B1s[t]' * Z1_n * As[t]
        YN[m1+1:m1+m2, :] = B2s[t]' * Z2_n * As[t]
        P = S\YN    # least square computation
        P1 = P[1:m1,:]
        P2 = P[m1+1:m1+m2,:]
        push!(P1s, P1)
        push!(P2s, P2)

        # ##########################################
        # 2. solve for the zero order term in FB strategy
        # ##########################################
        zeta1_n = zeta1s[end]
        zeta2_n = zeta2s[end]
        # use least square to solve for alphas, notice that the coefficient matrix (S) is the same as before
        YA = zeros(m1+m2, 1)
        YA[1:m1,:] = B1s[t]'  * zeta1_n
        YA[m1+1:m1+m2,:] = B2s[t]' * zeta2_n
        alpha = S\YA
        alpha1 = alpha[1:m1]
        alpha2 = alpha[m1+1:m1+m2]
        push!(alpha1s, alpha1)
        push!(alpha2s, alpha2)


        # ##########################################
        # 3. compute for next iteration
        # ##########################################
        # solve F for next iteration
        F = As[t] - B1s[t] * P1 - B2s[t] * P2
        push!(Fs, F)

        beta = - B1s[t] * alpha1 - B2s[t] * alpha2
        push!(betas, beta)

        zeta1 = F' * Z1_n * beta + F' * zeta1_n + P1' * R11s[t] * alpha1 + P2' * R12s[t] * alpha2 + l1s[t]
        zeta2 = F' * Z2_n * beta + F' * zeta2_n + P1' * R21s[t] * alpha1 + P2' * R22s[t] * alpha2 + l2s[t]
        push!(zeta1s, zeta1)
        push!(zeta2s, zeta2)

        Z1 = F' * Z1_n * F + P1' * R11s[t] * P1 +  P2' * R12s[t] * P2 + Q1s[t]
        Z2 = F' * Z2_n * F + P2' * R22s[t] * P2 +  P1' * R21s[t] * P1 + Q2s[t]
        push!(Z1s, Z1)
        push!(Z2s, Z2)
    end

    return P1s, P2s, alpha1s, alpha2s

end




function lqgame_QRE(dynamics::Dict{String, Array{Array{Float64}}},
                    costs::Dict{String, Array{Array{Float64}}})
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
    T = length(As)    # planning horizon
    n = size(B1s[1])[1]     # state dimension
    if(length(size(B1s[1])) == 1)
        m1 = 1
    else
        m1 = size(B1s[1])[2]    # control 1 dimension
    end
    if(length(size(B2s[1])) == 1)
        m2 = 1
    else
        m2 = size(B2s[1])[2]    # control 2 dimension
    end


    # return containers
    P1s, alpha1s, cov1s = [], [], []
    P2s, alpha2s, cov2s = [], [], []

    # intermediate computation
    Z1s, Z2s = [], []
    Fs = []
    zeta1s, zeta2s = [], []
    betas = []

    # boundary conditions
    push!(Z1s, Q1s[end])
    push!(Z2s, Q2s[end])
    push!(zeta1s, l1s[end])
    push!(zeta2s, l2s[end])

    # initialize containers
    for t = T:-1:1
        # note that in zero-based time index, the actual time index should be t-1
        # ##########################################
        # 1. solve for P_t^1 and P_t^2
        # ##########################################
        Z1_n = Z1s[end]
        Z2_n = Z2s[end]

        S = zeros(m1+m2, m1+m2)
        S[1:m1, 1:m1]             = R11s[t] + B1s[t]' * Z1_n * B1s[t]
        S[1:m1, m1+1:m1+m2]       = B1s[t]' * Z1_n * B2s[t]
        S[m1+1:m1+m2, 1:m1]       = B2s[t]' * Z2_n * B1s[t]
        S[m1+1:m1+m2, m1+1:m1+m2] = R22s[t] + B2s[t]' * Z2_n * B2s[t]


        YN = zeros(m1+m2, n)
        YN[1:m1, :]        = B1s[t]' * Z1_n * As[t]
        YN[m1+1:m1+m2, :] = B2s[t]' * Z2_n * As[t]
        P = S\YN    # least square computation
        P1 = P[1:m1,:]
        P2 = P[m1+1:m1+m2,:]
        push!(P1s, P1)
        push!(P2s, P2)
        push!(cov1s, inv(S[1:m1, 1:m1]))
        push!(cov2s, inv(S[m1+1:m1+m2, m1+1:m1+m2]))

        # ##########################################
        # 2. solve for the zero order term in FB strategy
        # ##########################################
        zeta1_n = zeta1s[end]
        zeta2_n = zeta2s[end]
        # use least square to solve for alphas, notice that the coefficient matrix (S) is the same as before
        YA = zeros(m1+m2, 1)
        YA[1:m1,:] = B1s[t]'  * zeta1_n
        YA[m1+1:m1+m2,:] = B2s[t]' * zeta2_n
        alpha = S\YA
        alpha1 = alpha[1:m1]
        alpha2 = alpha[m1+1:m1+m2]
        push!(alpha1s, alpha1)
        push!(alpha2s, alpha2)


        # ##########################################
        # 3. compute for next iteration
        # ##########################################
        # solve F for next iteration
        F = As[t] - B1s[t] * P1 - B2s[t] * P2
        push!(Fs, F)

        beta = - B1s[t] * alpha1 - B2s[t] * alpha2
        push!(betas, beta)

        zeta1 = F' * Z1_n * beta + F' * zeta1_n + P1' * R11s[t] * alpha1 + P2' * R12s[t] * alpha2 + l1s[t]
        zeta2 = F' * Z2_n * beta + F' * zeta2_n + P1' * R21s[t] * alpha1 + P2' * R22s[t] * alpha2 + l2s[t]
        push!(zeta1s, zeta1)
        push!(zeta2s, zeta2)

        Z1 = F' * Z1_n * F + P1' * R11s[t] * P1 +  P2' * R12s[t] * P2 + Q1s[t]
        Z2 = F' * Z2_n * F + P2' * R22s[t] * P2 +  P1' * R21s[t] * P1 + Q2s[t]
        push!(Z1s, Z1)
        push!(Z2s, Z2)
    end

    return P1s, P2s, alpha1s, alpha2s, cov1s, cov2s

end


#======================
Quantal equilibrium for 3 player games
=======================#
function lqgame_QRE_3player(dynamics::Dict{String, Array{Array{Float64}}},
                    costs::Dict{String, Array{Array{Float64}}})
    # Quantal Response Equilibria for Dynamic Markov Games

    # assume planning horizon T
    # return values:
    #   Ns: list len = T
    #   alphas: list len = T

    # dynamics matrices lists, each list should be of length T

    As = dynamics["A"]
    B1s = dynamics["B1"]
    B2s = dynamics["B2"]
    B3s = dynamics["B3"]

    # cost matrices lists, Qs and ls should be of length T + 1, while Rs should be of length T
    Q1s = costs["Q1"]
    l1s = costs["l1"]
    Q2s = costs["Q2"]
    l2s = costs["l2"]
    Q3s = costs["Q3"]
    l3s = costs["l3"]
    R11s = costs["R11"]
    R12s = costs["R12"]
    R13s = costs["R13"]
    R21s = costs["R21"]
    R22s = costs["R22"]
    R23s = costs["R23"]
    R31s = costs["R31"]
    R32s = costs["R32"]
    R33s = costs["R33"]

    # infer dimensions
    T = length(As)    # planning horizon
    n = size(B1s[1])[1]     # state dimension
    if(length(size(B1s[1])) == 1)
        m1 = 1
    else
        m1 = size(B1s[1])[2]    # control 1 dimension
    end
    if(length(size(B2s[1])) == 1)
        m2 = 1
    else
        m2 = size(B2s[1])[2]    # control 2 dimension
    end
    if(length(size(B3s[1])) == 1)
        m3 = 1
    else
        m3 = size(B3s[1])[2]    # control 3 dimension
    end


    # return containers
    P1s, alpha1s, cov1s = [], [], []
    P2s, alpha2s, cov2s = [], [], []
    P3s, alpha3s, cov3s = [], [], []

    # intermediate computation
    Z1s, Z2s, Z3s = [], [], []
    Fs = []
    zeta1s, zeta2s, zeta3s = [], [], []
    betas = []

    # boundary conditions
    push!(Z1s, Q1s[end])
    push!(Z2s, Q2s[end])
    push!(Z3s, Q3s[end])
    push!(zeta1s, l1s[end])
    push!(zeta2s, l2s[end])
    push!(zeta3s, l3s[end])

    # initialize containers
    for t = T:-1:1
        # note that in zero-based time index, the actual time index should be t-1
        # ##########################################
        # 1. solve for P_t^1 and P_t^2
        # ##########################################
        Z1_n = Z1s[end]
        Z2_n = Z2s[end]
        Z3_n = Z3s[end]

        S = zeros(m1+m2+m3, m1+m2+m3)
        S[1:m1, 1:m1]             = R11s[t] + B1s[t]' * Z1_n * B1s[t]
        S[1:m1, m1+1:m1+m2]       = B1s[t]' * Z1_n * B2s[t]
        S[1:m1, m1+m2+1:m1+m2+m3] = B1s[t]' * Z1_n * B3s[t]

        S[m1+1:m1+m2, 1:m1]             = B2s[t]' * Z2_n * B1s[t]
        S[m1+1:m1+m2, m1+1:m1+m2]       = R22s[t] + B2s[t]' * Z2_n * B2s[t]
        S[m1+1:m1+m2, m1+m2+1:m1+m2+m3] = B2s[t]' * Z2_n * B3s[t]

        S[m1+m2+1:end, 1:m1]             = B3s[t]' * Z3_n * B1s[t]
        S[m1+m2+1:end, m1+1:m1+m2]       = B3s[t]' * Z3_n * B2s[t]
        # println(" what is this ", size(R33s[t]), " and ", size(B3s[t]' * Z3_n * B3s[t]))
        S[m1+m2+1:end, m1+m2+1:m1+m2+m3] = R33s[t] + B3s[t]' * Z3_n * B3s[t]


        YN = zeros(m1+m2+m3, n)
        YN[1:m1, :]             = B1s[t]' * Z1_n * As[t]
        YN[m1+1:m1+m2, :]       = B2s[t]' * Z2_n * As[t]
        YN[m1+m2+1:m1+m2+m3, :] = B3s[t]' * Z3_n * As[t]
        P = S\YN    # least square computation
        P1 = P[1:m1,:]
        P2 = P[m1+1:m1+m2,:]
        P3 = P[m1+m2+1:m1+m2+m3,:]
        push!(P1s, P1)
        push!(P2s, P2)
        push!(P3s, P3)
        push!(cov1s, inv(S[1:m1, 1:m1]))
        push!(cov2s, inv(S[m1+1:m1+m2, m1+1:m1+m2]))
        push!(cov3s, inv(S[m1+m2+1:m1+m2+m3, m1+m2+1:m1+m2+m3]))

        # ##########################################
        # 2. solve for the zero order term in FB strategy
        # ##########################################
        zeta1_n = zeta1s[end]
        zeta2_n = zeta2s[end]
        zeta3_n = zeta3s[end]
        # use least square to solve for alphas, notice that the coefficient matrix (S) is the same as before
        YA = zeros(m1+m2+m3, 1)
        YA[1:m1,:]             = B1s[t]' * zeta1_n
        YA[m1+1:m1+m2,:]       = B2s[t]' * zeta2_n
        YA[m1+m2+1:m1+m2+m3,:] = B3s[t]' * zeta3_n
        alpha = S\YA
        alpha1 = alpha[1:m1]
        alpha2 = alpha[m1+1:m1+m2]
        alpha3 = alpha[m1+m2+1:m1+m2+m3]
        push!(alpha1s, alpha1)
        push!(alpha2s, alpha2)
        push!(alpha3s, alpha3)


        # ##########################################
        # 3. compute for next iteration
        # ##########################################
        # solve F for next iteration
        F = As[t] - B1s[t] * P1 - B2s[t] * P2 - B3s[t] * P3
        push!(Fs, F)

        beta = - B1s[t] * alpha1 - B2s[t] * alpha2 - B3s[t] * alpha3
        push!(betas, beta)

        zeta1 = F' * Z1_n * beta + F' * zeta1_n + P1' * R11s[t] * alpha1 + P2' * R12s[t] * alpha2 + P3' * R13s[t] * alpha3 + l1s[t]
        zeta2 = F' * Z2_n * beta + F' * zeta2_n + P1' * R21s[t] * alpha1 + P2' * R22s[t] * alpha2 + P3' * R23s[t] * alpha3 + l2s[t]
        zeta3 = F' * Z3_n * beta + F' * zeta3_n + P1' * R31s[t] * alpha1 + P2' * R32s[t] * alpha2 + P3' * R33s[t] * alpha3 + l3s[t]
        push!(zeta1s, zeta1)
        push!(zeta2s, zeta2)
        push!(zeta3s, zeta3)

        Z1 = F' * Z1_n * F + P1' * R11s[t] * P1 +  P2' * R12s[t] * P2 + P3' * R13s[t] * P3 + Q1s[t]
        Z2 = F' * Z2_n * F + P1' * R21s[t] * P1 +  P2' * R22s[t] * P2 + P3' * R23s[t] * P3 + Q2s[t]
        Z3 = F' * Z3_n * F + P1' * R31s[t] * P1 +  P2' * R32s[t] * P2 + P3' * R33s[t] * P3 + Q3s[t]
        push!(Z1s, Z1)
        push!(Z2s, Z2)
        push!(Z3s, Z3)
    end

    return P1s, P2s, P3s, alpha1s, alpha2s, alpha3s, cov1s, cov2s, cov3s

end
