import numpy as np
import torch
from lqgame import *

# =======================
# define simulation parameters
# =======================
class SimulationParams:
    def __init__(self, steps=60, horizon=6.0, plan_steps=10):
        self.steps = steps
        self.horizon = horizon
        self.plan_steps = plan_steps


# =======================
# define a nonlinear game
# =======================
class NonlinearGame:
    def __init__(self, dynamics_func, cost_funcs, state_dims, state_dim, ctrl_dims, ctrl_dim, radius):
        # dynamics and cost function
        self.dynamics_func = dynamics_func
        self.cost_funcs = cost_funcs

        # dimensions
        self.state_dims = state_dims
        self.state_dim = state_dim  # total dimension of states
        self.ctrl_dims = ctrl_dims
        self.ctrl_dim = ctrl_dim

        self.radius = radius

    @classmethod
    def from_dt_theta(cls, state_dims, ctrl_dims, DT, theta):
        """
        Constructor that generates dynamics and cost functions from parameters.
        """

        state_dim = sum(state_dims)
        ctrl_dim = sum(ctrl_dims)
        radius = 0.25

        def cost1(s):
            # s total dimension: state_dim * 2 + ctrl_dim
            assert len(s) == state_dim * 2 + ctrl_dim

            state = s[:state_dim]
            ref = s[state_dim:state_dim*2]
            control1 = s[state_dim*2: state_dim*2 + ctrl_dims[0]]
            control2 = s[state_dim*2 + ctrl_dims[0]: state_dim*2 + ctrl_dim]

            dist = np.sqrt((state[0] - state[4])**2 + (state[1] - state[5])**2) - (2 * radius)

            return (
                theta[0] * np.dot(state[0:4] - ref[0:4], state[0:4] - ref[0:4]) +
                theta[1] * np.dot(control1, control1) +
                theta[2] / ((0.2 * dist + 1)**10)
            )

        def cost2(s):
            # s total dimension: state_dim * 2 + ctrl_dim
            assert len(s) == state_dim * 2 + ctrl_dim

            state = s[:state_dim]
            ref = s[state_dim:state_dim*2]
            control1 = s[state_dim*2: state_dim*2 + ctrl_dims[0]]
            control2 = s[state_dim*2 + ctrl_dims[0]: state_dim*2 + ctrl_dim]

            dist = np.sqrt((state[0] - state[4])**2 + (state[1] - state[5])**2) - (2 * radius)

            return (
                theta[3] * np.dot(state[4:] - ref[4:], state[4:] - ref[4:]) +
                theta[4] * np.dot(control2, control2) +
                theta[5] / ((0.2 * dist + 1)**10)
            )

        def dynamics_func(s):
            assert len(s) == state_dim + ctrl_dim
            x1, y1, v1, theta1, x2, y2, v2, theta2 = s[:state_dim]
            acc1, yr1, acc2, yr2 = s[state_dim:]

            x1_new = x1 + v1 * np.cos(theta1) * DT
            y1_new = y1 + v1 * np.sin(theta1) * DT
            v1_new = v1 + acc1 * DT
            theta1_new = theta1 + yr1 * DT

            x2_new = x2 + v2 * np.cos(theta2) * DT
            y2_new = y2 + v2 * np.sin(theta2) * DT
            v2_new = v2 + acc2 * DT
            theta2_new = theta2 + yr2 * DT

            return np.array([x1_new, y1_new, v1_new, theta1_new,
                             x2_new, y2_new, v2_new, theta2_new])

        return cls(dynamics_func, [cost1, cost2], state_dims, state_dim, ctrl_dims, ctrl_dim, radius)

    @classmethod
    def from_custom_functions(cls, state_dims, ctrl_dims, dynamics_func, cost_funcs):
        """
        Constructor that takes user-defined dynamics and cost functions.
        """
        assert len(cost_funcs) == len(state_dims)
        assert len(cost_funcs) == len(ctrl_dims)

        state_dim = sum(state_dims)
        ctrl_dim = sum(ctrl_dims)
        radius = 0.25

        return cls(dynamics_func, cost_funcs, state_dims, state_dim, ctrl_dims, ctrl_dim, radius)

def define_game(state_dims, ctrl_dims, DT, theta):
    """
    Return a NonlinearGame instance for the three-player case.
    """

    import numpy as np

    state_dim = sum(state_dims)
    ctrl_dim = sum(ctrl_dims)
    radius = 0.25

    def dynamics_func(s):
        assert len(s) == state_dim + ctrl_dim
        x1, y1, v1, theta1, x2, y2, v2, theta2, x3, y3, v3, theta3 = s[:state_dim]
        acc1, yr1, acc2, yr2, acc3, yr3 = s[state_dim:]

        x1_new = x1 + v1 * np.cos(theta1) * DT
        y1_new = y1 + v1 * np.sin(theta1) * DT
        v1_new = v1 + acc1 * DT
        theta1_new = theta1 + yr1 * DT

        x2_new = x2 + v2 * np.cos(theta2) * DT
        y2_new = y2 + v2 * np.sin(theta2) * DT
        v2_new = v2 + acc2 * DT
        theta2_new = theta2 + yr2 * DT

        x3_new = x3 + v3 * np.cos(theta3) * DT
        y3_new = y3 + v3 * np.sin(theta3) * DT
        v3_new = v3 + acc3 * DT
        theta3_new = theta3 + yr3 * DT

        return np.array([
            x1_new, y1_new, v1_new, theta1_new,
            x2_new, y2_new, v2_new, theta2_new,
            x3_new, y3_new, v3_new, theta3_new
        ])

    def cost1(s):
        assert len(s) == state_dim * 2 + ctrl_dim
        state_dim1 = state_dims[0]
        state_dim2 = state_dims[1]
        ctrl_dim1 = ctrl_dims[0]

        state = s[:state_dim]
        ref = s[state_dim:state_dim*2]

        ego_state = state[:state_dim1]
        ego_ref = ref[:state_dim1]

        control1 = s[state_dim*2:state_dim*2 + ctrl_dim1]

        dist12 = np.sqrt((ego_state[0] - state[state_dim1])**2 + (ego_state[1] - state[state_dim1+1])**2) - (2 * radius)
        dist13 = np.sqrt((ego_state[0] - state[state_dim1 + state_dim2])**2 + (ego_state[1] - state[state_dim1 + state_dim2 + 1])**2) - (2 * radius)

        return (
            theta[0] * np.dot(ego_state - ego_ref, ego_state - ego_ref) +
            theta[1] * np.dot(control1, control1) +
            theta[2] / ((0.2 * dist12 + 1)**10) +
            theta[2] / ((0.2 * dist13 + 1)**10)
        )

    def cost2(s):
        assert len(s) == state_dim * 2 + ctrl_dim
        state_dim1 = state_dims[0]
        state_dim2 = state_dims[1]
        ctrl_dim1 = ctrl_dims[0]
        ctrl_dim2 = ctrl_dims[1]

        state = s[:state_dim]
        ref = s[state_dim:state_dim*2]

        ego_state = state[state_dim1:state_dim1 + state_dim2]
        ego_ref = ref[state_dim1:state_dim1 + state_dim2]

        control2 = s[state_dim*2 + ctrl_dim1:state_dim*2 + ctrl_dim1 + ctrl_dim2]

        dist21 = np.sqrt((ego_state[0] - state[0])**2 + (ego_state[1] - state[1])**2) - (2 * radius)
        dist23 = np.sqrt((ego_state[0] - state[state_dim1 + state_dim2])**2 + (ego_state[1] - state[state_dim1 + state_dim2 + 1])**2) - (2 * radius)

        return (
            theta[3] * np.dot(ego_state - ego_ref, ego_state - ego_ref) +
            theta[4] * np.dot(control2, control2) +
            theta[5] / ((0.2 * dist21 + 1)**10) +
            theta[5] / ((0.2 * dist23 + 1)**10)
        )

    def cost3(s):
        assert len(s) == state_dim * 2 + ctrl_dim
        state_dim1 = state_dims[0]
        state_dim2 = state_dims[1]
        ctrl_dim1 = ctrl_dims[0]
        ctrl_dim2 = ctrl_dims[1]

        state = s[:state_dim]
        ref = s[state_dim:state_dim*2]

        ego_state = state[state_dim1 + state_dim2:]
        ego_ref = ref[state_dim1 + state_dim2:]

        control3 = s[state_dim*2 + ctrl_dim1 + ctrl_dim2:]

        dist31 = np.sqrt((ego_state[0] - state[0])**2 + (ego_state[1] - state[1])**2) - (2 * radius)
        dist32 = np.sqrt((ego_state[0] - state[state_dim1])**2 + (ego_state[1] - state[state_dim1 + 1])**2) - (2 * radius)

        return (
            theta[6] * np.dot(ego_state - ego_ref, ego_state - ego_ref) +
            theta[7] * np.dot(control3, control3) +
            theta[8] / ((0.2 * dist31 + 1)**10) +
            theta[8] / ((0.2 * dist32 + 1)**10)
        )

    # Construct and return the game instance using the alternate constructor
    return NonlinearGame.from_custom_functions(
        state_dims, ctrl_dims,
        dynamics_func=dynamics_func,
        cost_funcs=[cost1, cost2, cost3]
    )

def get_feature_counts_three(game, x_trajectories, u_trajectories, x_ref, feature_k):
    feature_counts = np.zeros(feature_k)
    num = len(x_trajectories)

    state_dim1 = game.state_dims[0]
    state_dim2 = game.state_dims[1]
    ctrl_dim1 = game.ctrl_dims[0]
    ctrl_dim2 = game.ctrl_dims[1]
    radius = game.radius

    for i in range(num):
        xtraj = x_trajectories[i]
        utraj = u_trajectories[i]
        steps = utraj.shape[0]

        for t in range(steps + 1):
            state = xtraj[t]
            ref = x_ref[t]

            state1 = state[:state_dim1]
            state2 = state[state_dim1:state_dim1 + state_dim2]
            state3 = state[state_dim1 + state_dim2:]
            ref1 = ref[:state_dim1]
            ref2 = ref[state_dim1:state_dim1 + state_dim2]
            ref3 = ref[state_dim1 + state_dim2:]

            if t < steps:
                control = utraj[t]
                control1 = control[:ctrl_dim1]
                control2 = control[ctrl_dim1:ctrl_dim1 + ctrl_dim2]
                control3 = control[ctrl_dim1 + ctrl_dim2:]

            dist12 = np.linalg.norm(state1[:2] - state2[:2]) - (2 * radius)
            dist13 = np.linalg.norm(state1[:2] - state3[:2]) - (2 * radius)
            dist23 = np.linalg.norm(state2[:2] - state3[:2]) - (2 * radius)

            feature_counts[0] += np.dot(state1 - ref1, state1 - ref1)
            feature_counts[2] += 1.0 / ((0.2 * dist12 + 1)**10) + 1.0 / ((0.2 * dist13 + 1)**10)
            feature_counts[3] += np.dot(state2 - ref2, state2 - ref2)
            feature_counts[5] += 1.0 / ((0.2 * dist12 + 1)**10) + 1.0 / ((0.2 * dist23 + 1)**10)
            feature_counts[6] += np.dot(state3 - ref3, state3 - ref3)
            feature_counts[8] += 1.0 / ((0.2 * dist13 + 1)**10) + 1.0 / ((0.2 * dist23 + 1)**10)

            if t < steps:
                feature_counts[1] += np.dot(control1, control1)
                feature_counts[4] += np.dot(control2, control2)
                feature_counts[7] += np.dot(control3, control3)

    avg_feature_counts = feature_counts / num
    return avg_feature_counts

def get_feature_counts(game, x_trajectories, u_trajectories, x_ref, feature_k):
    num = len(x_trajectories)
    state_dim1 = game.state_dims[0]
    ctrl_dim1 = game.ctrl_dims[0]
    radius = game.radius

    feature_counts = np.zeros((feature_k, num))

    for i in range(num):
        xtraj = x_trajectories[i]
        utraj = u_trajectories[i]
        steps = utraj.shape[0]

        for t in range(steps + 1):
            state = xtraj[t]
            ref = x_ref[t]

            state1 = state[:state_dim1]
            state2 = state[state_dim1:]
            ref1 = ref[:state_dim1]
            ref2 = ref[state_dim1:]

            if t < steps:
                control = utraj[t]
                control1 = control[:ctrl_dim1]
                control2 = control[ctrl_dim1:]

            dist = np.linalg.norm(state1[:2] - state2[:2]) - (2 * radius)

            feature_counts[0, i] += np.dot(state1 - ref1, state1 - ref1)
            feature_counts[2, i] += 1.0 / ((0.2 * dist + 1)**10)
            feature_counts[3, i] += np.dot(state2 - ref2, state2 - ref2)
            feature_counts[5, i] += 1.0 / ((0.2 * dist + 1)**10)

            if t < steps:
                feature_counts[1, i] += np.dot(control1, control1)
                feature_counts[4, i] += np.dot(control2, control2)

    avg_feature_counts = np.sum(feature_counts, axis=1) / num
    return avg_feature_counts, feature_counts

# =======================
# an iterative LQ game solver
# input: sim_param::SimulationParams,
#        nl_game::NonlinearGame,
#        x_init::Array{Float64},
#        traj_ref::Array{Float64}
# ========================

def solve_iLQGame(sim_param=None, nl_game=None, x_init=None, traj_ref=None):
    
    num_player = len(nl_game.ctrl_dims)

    plan_steps = sim_param.plan_steps
    state_dim = nl_game.state_dim
    ctrl_dim = nl_game.ctrl_dim
    ctrl_dim1 = nl_game.ctrl_dims[0]
    ctrl_dim2 = nl_game.ctrl_dims[1]

    # ###########################
    # initialize the iteration (forward simulation)
    # ###########################
    x_trajectory = np.zeros((plan_steps + 1, state_dim))
    x_trajectory[0, :] = x_init
    u_trajectory = np.zeros((plan_steps, ctrl_dim))

    # forward simulation
    for t in range(plan_steps):
        x_trajectory[t + 1, :] = nl_game.dynamics_func(np.concatenate([x_trajectory[t, :], u_trajectory[t, :]]))
    
    ##################
    # iteration
    ################
    if num_player == 2:
        tol = 1.0
        itr = 1
        err = 10
        max_itr = 50

        x_trajectory_prev = x_trajectory
        u_trajectory_prev = u_trajectory
        N1, N2 = [], []
        alpha1, alpha2 = [], []
        cov1, cov2 = [], []

        while abs(err) > tol and itr < max_itr:
            # if plot_flag
            #     ax.plot(x_trajectory[:,1], x_trajectory[:,2], "yellow")
            #     ax.plot(x_trajectory[:,5], x_trajectory[:,6], "yellow")
            # end

            # println(" at iteration ", itr, " and error ", err)

            ###########################
            #  linear quadratic approximation
            ###########################
            Dynamics = {"A": [], "B1": [], "B2": []}
            Costs = {"Q1": [], "l1": [], "Q2": [], "l2": [], "R11": [], "R12": [], "R21": [], "R22": []}

            for t in range(plan_steps):
                # Compute jacobian
                jac = torch.autograd.functional.jacobian(nl_game.dynamics_func, torch.cat([x_trajectory_prev[t,:], u_trajectory_prev[t,:]]))
                A = jac[:, :state_dim]
                B1 = jac[:, state_dim:state_dim + nl_game.ctrl_dims[0]]
                B2 = jac[:, state_dim + nl_game.ctrl_dims[0]:]

                Dynamics["A"].append(A)
                Dynamics["B1"].append(B1)
                Dynamics["B2"].append(B2)

                # Compute gradients and Hessians
                grads1 = torch.autograd.grad(nl_game.cost_funcs[0]([x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]]), [x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]], create_graph=True)
                hess1 = torch.autograd.functional.hessian(nl_game.cost_funcs[0], torch.cat([x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]]))
                
                grads2 = torch.autograd.grad(nl_game.cost_funcs[1]([x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]]), [x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]], create_graph=True)
                hess2 = torch.autograd.functional.hessian(nl_game.cost_funcs[1], torch.cat([x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]]))

                Q1 = hess1[0][:state_dim, :state_dim]
                min_eig_Q1 = torch.min(torch.linalg.eigvals(Q1))
                if min_eig_Q1 <= 0.0:
                    Q1 += (abs(min_eig_Q1) + 1e-3) * torch.eye(state_dim)
                
                Q2 = hess2[0][:state_dim, :state_dim]
                min_eig_Q2 = torch.min(torch.linalg.eigvals(Q2))
                if min_eig_Q2 <= 0.0:
                    Q2 += (abs(min_eig_Q2) + 1e-3) * torch.eye(state_dim)

                Costs["Q1"].append(Q1)
                Costs["Q2"].append(Q2)
                Costs["l1"].append(grads1[0][:state_dim])
                Costs["l2"].append(grads2[0][:state_dim])

                Costs["R11"].append(hess1[0][state_dim*2:state_dim*2+ctrl_dim1, state_dim*2:state_dim*2+ctrl_dim1])
                Costs["R12"].append(hess1[0][state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim, state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim])
                Costs["R21"].append(hess2[0][state_dim*2:state_dim*2+ctrl_dim1, state_dim*2:state_dim*2+ctrl_dim1])
                Costs["R22"].append(hess2[0][state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim, state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim])

            grads1 = torch.autograd.grad(nl_game.cost_funcs[0]([x_trajectory_prev[-1,:], traj_ref[-1,:], u_trajectory[-1,:]]), [x_trajectory_prev[-1,:], traj_ref[-1,:], u_trajectory[-1,:]], create_graph=True)
            hess1 = torch.autograd.functional.hessian(nl_game.cost_funcs[0], torch.cat([x_trajectory_prev[-1,:], traj_ref[-1,:], u_trajectory[-1,:]]))
            
            grads2 = torch.autograd.grad(nl_game.cost_funcs[1]([x_trajectory_prev[-1,:], traj_ref[-1,:], u_trajectory[-1,:]]), [x_trajectory_prev[-1,:], traj_ref[-1,:], u_trajectory[-1,:]], create_graph=True)
            hess2 = torch.autograd.functional.hessian(nl_game.cost_funcs[1], torch.cat([x_trajectory_prev[-1,:], traj_ref[-1,:], u_trajectory[-1,:]]))

            Q1 = hess1[0][:state_dim, :state_dim]
            min_eig_Q1 = torch.min(torch.linalg.eigvals(Q1))
            if min_eig_Q1 <= 0.0:
                Q1 += (abs(min_eig_Q1) + 1e-3) * torch.eye(state_dim)
            
            Q2 = hess2[0][:state_dim, :state_dim]
            min_eig_Q2 = torch.min(torch.linalg.eigvals(Q2))
            if min_eig_Q2 <= 0.0:
                Q2 += (abs(min_eig_Q2) + 1e-3) * torch.eye(state_dim)

            Costs["Q1"].append(Q1)
            Costs["Q2"].append(Q2)
            Costs["l1"].append(grads1[0][:state_dim])
            Costs["l2"].append(grads2[0][:state_dim])

            ###########################
            # backward computation
            ###########################
            # backtrack on gamma parameters
            N1, N2, alpha1, alpha2, cov1, cov2 = lqgame_QRE(Dynamics, Costs)

            ###########################
            # forward simulation
            ###########################
            step_size = 1.0
            done = False
            while not done:
                u_trajectory = torch.zeros(plan_steps, ctrl_dim)
                x_trajectory = torch.zeros(plan_steps + 1, state_dim)
                x_trajectory[0,:] = x_init
                for t in range(plan_steps):
                    u_trajectory[t,:] = torch.cat([
                        (-N1[-(t+1)] * (x_trajectory[t, :] - x_trajectory_prev[t,:]) - alpha1[-(t+1)] * step_size),
                        (-N2[-(t+1)] * (x_trajectory[t, :] - x_trajectory_prev[t,:]) - alpha2[-(t+1)] * step_size)
                    ]) + u_trajectory_prev[t, :]
                    x_trajectory[t+1, :] = nl_game.dynamics_func(torch.cat([x_trajectory[t, :], u_trajectory[t,:]]))

                if torch.max(torch.abs(x_trajectory - x_trajectory_prev)) > 1.0:
                    step_size /= 2
                else:
                    done = True

            ###########################
            # book keeping and convergence test
            ###########################
            err = torch.sum(torch.abs(x_trajectory_prev - x_trajectory))
            itr += 1
            x_trajectory_prev = x_trajectory
            u_trajectory_prev = u_trajectory

        return [N1, N2], [alpha1, alpha2], [cov1, cov2], x_trajectory_prev, u_trajectory_prev
    else:
        tol = 1.0
        itr = 1
        err = 10
        max_itr = 50

        x_trajectory_prev = x_trajectory
        u_trajectory_prev = u_trajectory
        N1, N2, N3 = [], [], []
        alpha1, alpha2, alpha3 = [], [], []
        cov1, cov2, cov3 = [], [], []
        
        while abs(err) > tol and itr < max_itr:
            # if plot_flag
            #     ax.plot(x_trajectory[:,1], x_trajectory[:,2], "yellow")
            #     ax.plot(x_trajectory[:,5], x_trajectory[:,6], "yellow")
            # end

            # println(" at iteration ", itr, " and error ", err)

            ###########################
            #  linear quadratic approximation
            # ###########################
            Dynamics = {
                "A": [],
                "B1": [],
                "B2": [],
                "B3": []
            }

            Costs = {
                "Q1": [],
                "l1": [],
                "Q2": [],
                "l2": [],
                "Q3": [],
                "l3": [],
                "R11": [],
                "R12": [],
                "R13": [],
                "R21": [],
                "R22": [],
                "R23": [],
                "R31": [],
                "R32": [],
                "R33": []
            }

            for t in range(plan_steps):
                jac = torch.autograd.functional.jacobian(nl_game.dynamics_func, torch.cat((x_trajectory_prev[t,:], u_trajectory_prev[t,:]), dim=0))
                A = jac[:, :state_dim]
                B1 = jac[:, state_dim:state_dim+ctrl_dim1]
                B2 = jac[:, state_dim+ctrl_dim1:state_dim+ctrl_dim1+ctrl_dim2]
                B3 = jac[:, state_dim+ctrl_dim1+ctrl_dim2:]

                Dynamics["A"].append(A)
                Dynamics["B1"].append(B1)
                Dynamics["B2"].append(B2)
                Dynamics["B3"].append(B3)

                grads1 = torch.autograd.grad(nl_game.cost_funcs[1](torch.cat((x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]), dim=0)), x_trajectory_prev[t,:], create_graph=True)[0]
                hess1 = torch.autograd.functional.hessian(nl_game.cost_funcs[1], torch.cat((x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]), dim=0))
                grads2 = torch.autograd.grad(nl_game.cost_funcs[2](torch.cat((x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]), dim=0)), x_trajectory_prev[t,:], create_graph=True)[0]
                hess2 = torch.autograd.functional.hessian(nl_game.cost_funcs[2], torch.cat((x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]), dim=0))
                grads3 = torch.autograd.grad(nl_game.cost_funcs[3](torch.cat((x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]), dim=0)), x_trajectory_prev[t,:], create_graph=True)[0]
                hess3 = torch.autograd.functional.hessian(nl_game.cost_funcs[3], torch.cat((x_trajectory_prev[t,:], traj_ref[t,:], u_trajectory[t,:]), dim=0))

                Q1 = hess1[:state_dim, :state_dim]
                min_eig_Q1 = torch.min(torch.eig(Q1, eigenvectors=False)[0][:, 0])
                if min_eig_Q1 <= 0.0:
                    Q1 += (abs(min_eig_Q1) + 1e-3) * torch.eye(state_dim)
                
                Q2 = hess2[:state_dim, :state_dim]
                min_eig_Q2 = torch.min(torch.eig(Q2, eigenvectors=False)[0][:, 0])
                if min_eig_Q2 <= 0.0:
                    Q2 += (abs(min_eig_Q2) + 1e-3) * torch.eye(state_dim)
                
                Q3 = hess3[:state_dim, :state_dim]
                min_eig_Q3 = torch.min(torch.eig(Q3, eigenvectors=False)[0][:, 0])
                if min_eig_Q3 <= 0.0:
                    Q3 += (abs(min_eig_Q3) + 1e-3) * torch.eye(state_dim)

                Costs["Q1"].append(Q1)
                Costs["Q2"].append(Q2)
                Costs["Q3"].append(Q3)
                Costs["l1"].append(grads1[:state_dim])
                Costs["l2"].append(grads2[:state_dim])
                Costs["l3"].append(grads3[:state_dim])

                Costs["R11"].append(hess1[state_dim*2:state_dim*2+ctrl_dim1, state_dim*2:state_dim*2+ctrl_dim1])
                Costs["R12"].append(hess1[state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim1+ctrl_dim2, state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim1+ctrl_dim2])
                Costs["R13"].append(hess1[state_dim*2+ctrl_dim1+ctrl_dim2:state_dim*2+ctrl_dim, state_dim*2+ctrl_dim1+ctrl_dim2:state_dim*2+ctrl_dim])
                
                Costs["R21"].append(hess2[state_dim*2:state_dim*2+ctrl_dim1, state_dim*2:state_dim*2+ctrl_dim1])
                Costs["R22"].append(hess2[state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim1+ctrl_dim2, state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim1+ctrl_dim2])
                Costs["R23"].append(hess2[state_dim*2+ctrl_dim1+ctrl_dim2:state_dim*2+ctrl_dim, state_dim*2+ctrl_dim1+ctrl_dim2:state_dim*2+ctrl_dim])
                
                Costs["R31"].append(hess3[state_dim*2:state_dim*2+ctrl_dim1, state_dim*2:state_dim*2+ctrl_dim1])
                Costs["R32"].append(hess3[state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim1+ctrl_dim2, state_dim*2+ctrl_dim1:state_dim*2+ctrl_dim1+ctrl_dim2])
                Costs["R33"].append(hess3[state_dim*2+ctrl_dim1+ctrl_dim2:state_dim*2+ctrl_dim, state_dim*2+ctrl_dim1+ctrl_dim2:state_dim*2+ctrl_dim])
            

            # Assuming `nl_game.cost_funcs[i]` are callable PyTorch functions and Costs is a dictionary
            # Concatenate the necessary tensors
            x_u_concat = torch.cat((x_trajectory_prev[-1, :], traj_ref[-1, :], u_trajectory[-1, :]), dim=0)
            x_u_concat.requires_grad = True  # Enable autograd

            # Compute gradients and Hessians using PyTorch's autograd
            grads1 = torch.autograd.grad(nl_game.cost_funcs[1](x_u_concat), x_u_concat, create_graph=True)[0]
            hess1 = torch.autograd.functional.hessian(nl_game.cost_funcs[1], x_u_concat)

            grads2 = torch.autograd.grad(nl_game.cost_funcs[2](x_u_concat), x_u_concat, create_graph=True)[0]
            hess2 = torch.autograd.functional.hessian(nl_game.cost_funcs[2], x_u_concat)

            grads3 = torch.autograd.grad(nl_game.cost_funcs[3](x_u_concat), x_u_concat, create_graph=True)[0]
            hess3 = torch.autograd.functional.hessian(nl_game.cost_funcs[3], x_u_concat)

            # Extract Q1, Q2, Q3 from Hessians (assuming state_dim is defined)
            Q1 = hess1[:state_dim, :state_dim]
            min_eig_Q1 = torch.min(torch.eig(Q1, eigenvectors=False)[0][:, 0])
            if min_eig_Q1 <= 0.0:
                Q1 += (abs(min_eig_Q1) + 1e-3) * torch.eye(state_dim)

            Q2 = hess2[:state_dim, :state_dim]
            min_eig_Q2 = torch.min(torch.eig(Q2, eigenvectors=False)[0][:, 0])
            if min_eig_Q2 <= 0.0:
                Q2 += (abs(min_eig_Q2) + 1e-3) * torch.eye(state_dim)

            Q3 = hess3[:state_dim, :state_dim]
            min_eig_Q3 = torch.min(torch.eig(Q3, eigenvectors=False)[0][:, 0])
            if min_eig_Q3 <= 0.0:
                Q3 += (abs(min_eig_Q3) + 1e-3) * torch.eye(state_dim)

            # Update Costs dictionary
            Costs["Q1"].append(Q1)
            Costs["Q2"].append(Q2)
            Costs["Q3"].append(Q3)
            Costs["l1"].append(grads1[:state_dim])
            Costs["l2"].append(grads2[:state_dim])
            Costs["l3"].append(grads3[:state_dim])

            # ###########################
            # backward computation
            # ###########################
            # backtrack on gamma parameters (assuming lqgame_QRE_3player function is available)
            N1, N2, N3, alpha1, alpha2, alpha3, cov1, cov2, cov3 = lqgame_QRE_3player(Dynamics, Costs)
            
            step_size = 1.0
            done = False
            while not done:

                # Initialize trajectories
                u_trajectory = torch.zeros(plan_steps, ctrl_dim)
                x_trajectory = torch.zeros(plan_steps + 1, state_dim)
                x_trajectory[0, :] = x_init

                for t in range(plan_steps):
                    delta_x = x_trajectory[t, :] - x_trajectory_prev[t, :]
                    
                    # Update the control inputs u_trajectory based on N1, N2, N3, alpha1, alpha2, alpha3, and previous u_trajectory
                    u_trajectory[t, :ctrl_dim1] = (-N1[plan_steps - t - 1] * delta_x - alpha1[plan_steps - t - 1] * step_size) + u_trajectory_prev[t, :ctrl_dim1]
                    u_trajectory[t, ctrl_dim1:ctrl_dim1+ctrl_dim2] = (-N2[plan_steps - t - 1] * delta_x - alpha2[plan_steps - t - 1] * step_size) + u_trajectory_prev[t, ctrl_dim1:ctrl_dim1+ctrl_dim2]
                    u_trajectory[t, ctrl_dim1+ctrl_dim2:] = (-N3[plan_steps - t - 1] * delta_x - alpha3[plan_steps - t - 1] * step_size) + u_trajectory_prev[t, ctrl_dim1+ctrl_dim2:]

                    # Update the state trajectory using the dynamics function
                    x_trajectory[t + 1, :] = nl_game.dynamics_func(torch.cat((x_trajectory[t, :], u_trajectory[t, :]), dim=0))

                # Check the convergence condition and adjust the step size if needed
                if torch.max(torch.abs(x_trajectory - x_trajectory_prev)) > 1.0:
                    step_size /= 2
                else:
                    done = True
            
            err = torch.sum(torch.abs(x_trajectory_prev - x_trajectory))
            itr += 1
            x_trajectory_prev = x_trajectory
            u_trajectory_prev = u_trajectory
        
        return [N1, N2, N3], [alpha1, alpha2, alpha3], [cov1, cov2, cov3], x_trajectory_prev, u_trajectory_prev

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as MvNormal

class SimulationResults:
    def __init__(self, state_trajectories, ctrl_trajectories):
        self.state_trajectories = state_trajectories
        self.ctrl_trajectories = ctrl_trajectories

def generate_simulations(sim_param, nl_game, x_init, traj_ref, num_sim, plot_flag=False):
    num_player = len(nl_game.ctrl_dims)
    steps = sim_param.steps
    plan_steps = sim_param.plan_steps
    state_dim = nl_game.state_dim
    ctrl_dim = nl_game.ctrl_dim
    
    x_trajectories, u_trajectories = [], []
    x_trajectories_data = np.zeros((num_sim, steps+1, state_dim))
    u_trajectories_data = np.zeros((num_sim, steps+1, ctrl_dim))
    
    if plot_flag:
        fig, ax = plt.subplots(1, 1)

    # receding horizon planning in one simulation
    # need to replan online because of stochastic policy
    for i in range(num_sim):
        if i % 3 == 0:
            print(f" --- generating sim number : {i} ---")
        
        x_history = np.zeros((steps+1, state_dim))
        x_history[0, :] = x_init
        u_history = np.zeros((steps, ctrl_dim))

        for t in range(steps):
            if plot_flag:
                ax.clear()
                ax.scatter(-4, 0, marker="X", s=100, c="y")
                ax.scatter(0, -4, marker="X", s=100, c="m")
                ax.scatter(traj_ref[t, 0], traj_ref[t, 1], alpha=0.1)
                ax.scatter(traj_ref[t, 4], traj_ref[t, 5], alpha=0.1)

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_xlim(-5, 5)
                ax.set_xticks([-3, -1, 1, 3])
                ax.set_ylim(-5, 5)
                ax.set_yticks([-3, -1, 1, 3])
                plt.pause(0.1)

            Nlist_all, alphalist_all, cov_all, x_nominal, u_nominal = solve_iLQGame(
                sim_param=sim_param, nl_game=nl_game, x_init=x_history[t, :], traj_ref=traj_ref[t:t+plan_steps, :]
            )

            delta_x = x_history[t, :] - x_nominal[0, :]
            u_dists = []
            for ip in range(num_player):
                u_mean = -Nlist_all[ip][-1] * delta_x - alphalist_all[ip][-1]
                # u_dist = MvNormal(u_mean, cov_all[ip][-1])
                u_dist = MvNormal(u_mean, np.eye(2))  # Replace with correct covariance matrix
                u_dists.append(u_dist)

            control = []
            for u_dist in u_dists:
                control.extend(u_dist.rvs())

            u_history[t, :] = control + u_nominal[0, :]
            x_history[t+1, :] = nl_game.dynamics_func(np.concatenate([x_history[t, :], u_history[t, :]]))

            if plot_flag:
                ax.plot(traj_ref[t:t+plan_steps, 0], traj_ref[t:t+plan_steps, 1], "purple")
                ax.plot(traj_ref[t:t+plan_steps, 4], traj_ref[t:t+plan_steps, 5], "purple")
                ax.plot(x_nominal[:, 0], x_nominal[:, 1], "red")
                ax.plot(x_nominal[:, 4], x_nominal[:, 5], "red")
                plt.pause(0.1)

        x_trajectories.append(x_history)
        u_trajectories.append(u_history)
        x_trajectories_data[i, :, :] = x_history
        u_trajectories_data[i, :steps, :] = u_history

    return SimulationResults(x_trajectories, u_trajectories), x_trajectories_data, u_trajectories_data



