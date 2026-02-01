import cvxpy as cp
import numpy as np

class MarineVehicleEstimator:
    def __init__(self, dof, n_params, n_horizon, theta_prev, x_nb_s):
        self.theta_prev = theta_prev
        self.p = n_params
        self.n = n_horizon * dof
        self.x_nb_s = x_nb_s

        # β = [m-X_du, m-Y_dv, m-Z_dw, m*z_g-X_dq, -m*z_g+Y_dp, -m*z_g+K_dv, m*z_g-M_du, I_x-K_dp, I_y-M_dq, I_z-N_dr ,
        #  W, B,
        # x_g*W - x_b*B, y_g*W - y_b*B , z_g*W - z_b*B, 
        # X_u, Y_v, Z_w, K_p, M_q, N_r, 
        # X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr]
    
        # parameter layout from your build_sys_regressor
        # beta = [inertia 10, W, B, x_gW_x_bB, y_gW_y_bB, z_gW_z_bB, linear 6, quad 6]
        # indices
        self.idx_inertia = list(range(0, 10))
        self.idx_WB = [10, 11]
        self.idx_moment = [12, 13, 14]
        self.idx_linear = list(range(15, 21))
        self.idx_quad = list(range(21, 27))

        self.x = cp.Variable(shape=(self.p,))
        # x_prev = cp.Parameter(shape=(self.p,))
        self.w = cp.Variable(shape=(self.p,))
        self.v = cp.Variable(shape=(self.n,))
        
        self.A = cp.Parameter(shape=(self.n, self.p))
        self.b = cp.Parameter(shape=(self.n,))
        self.x_hat_prev = cp.Parameter(shape=(self.p,))
        self.W_min=105.0
        self.W_max=150.0
        # tau = 2
        # rho = 0

        # eps = 1e-6
        # scales = np.maximum(np.abs(x_prev), eps)  # elementwise
        # Q = np.diag(1.0 / (scales**2))            # penalize relative change #positive definite
        # obj = cp.quad_form(w, Q)
        # obj += tau*cp.huber(cp.norm(v),rho)

        # obj = rho*cp.sum_squares(self.w)
        obj = cp.Minimize(cp.sum_squares(self.v))
        # obj = cp.Minimize(cp.norm1(self.v))

        constr = []
        
        Muv = self.M_from_uv(self.x)
        constr += [Muv >> 1e-6 * np.eye(6)]
        constr += [Muv == Muv.T]

        # Data fitting constraint
        constr += [self.b == self.A @ self.x + self.v]

        #  force linear and quadratic drag terms less than zero
        constr += [self.x[self.idx_linear] <= 0]
        constr += [self.x[self.idx_quad] <= 0]

        # weight box constraint, W is x[self.idx_WB[0]]
        W_idx = self.idx_WB[0]
        constr += [self.x[W_idx] >= self.W_min, self.x[W_idx] <= self.W_max]

        self.prob = cp.Problem(obj, constr)

    #  Helper, pseudo inertia LMI per joint
    def M_from_uv(self, th):
        m_X_du = th[0]
        m_Y_dv = th[1]
        m_Z_dw = th[2]
        mz_g_X_dq = th[3]
        mz_g_Y_dp = th[4]
        mz_g_K_dv = th[5]
        mz_g_M_du = th[6]
        I_x_K_dp = th[7]
        I_y_M_dq = th[8]
        I_z_N_dr = th[9]
        zero = cp.Constant(0)
        M = cp.bmat([[m_X_du, zero, zero, zero, mz_g_X_dq, zero],
                     [zero, m_Y_dv, zero, mz_g_Y_dp, zero, zero],
                     [zero, zero, m_Z_dw, zero, zero, zero],
                     [zero, mz_g_K_dv, zero, I_x_K_dp, zero, zero],
                     [mz_g_M_du, zero, zero, zero, I_y_M_dq, zero],
                     [zero, zero, zero, zero, zero, I_z_N_dr]])
        return M
    
    def skew(self, u):
            """
            3x3 cross product matrix for a 3x1 or (3,) CVXPY expression u.
            Changed 2D indexing u[i, 0] to 1D indexing u[i] to handle
            1D CVXPY expressions (shape (3,)).
            """
            return cp.bmat([
                [cp.Constant(0),   -u[2],        u[1]],
                [u[2],          cp.Constant(0), -u[0]],
                [-u[1],         u[0],        cp.Constant(0)]
            ])

    def coriolis_cvxpy(self, M, x_nb):
        """
        M: 6x6 CVXPY expression or Parameter
        x_nb: 6x1 CVXPY expression or Parameter or numpy array
              (In this context, it's a Python list from x_nb_s)
        Returns 6x6 CVXPY Coriolis matrix C.
        """
        x_nb_arr = np.array(x_nb).flatten()
        v_nb = x_nb_arr[:3] # Shape (3,)
        w_nb = x_nb_arr[3:] # Shape (3,)

        # Quadrants of M
        M11 = M[:3, :3]
        M12 = M[:3, 3:]
        M21 = M[3:, :3]
        M22 = M[3:, 3:]

        # Terms that feed the cross matrices
        # v_nb and w_nb are 1D numpy arrays, CVXPY handles this.
        # b and c will be CVXPY expressions with shape (3,).
        b = M11 @ v_nb + M12 @ w_nb
        c = M21 @ v_nb + M22 @ w_nb

        # 3x3 zero as a CVXPY Constant
        Z = cp.Constant(np.zeros((3, 3)))
        
        # self.skew() is now called with 1D (3,) expressions,
        # so it must use 1D indexing (which we fixed above).
        C_top = cp.hstack([Z, -self.skew(b)])
        C_bot = cp.hstack([-self.skew(b), -self.skew(c)])
        C = cp.vstack([C_top, C_bot])
        return C

    def estimate_vehicle_physical_parameters(self, Y_big, tau_big, warm_start):
        self.A.value = Y_big
        self.b.value = tau_big
        self.x_hat_prev.value = self.theta_prev
        
        self.prob.solve(solver=cp.MOSEK, verbose=False, warm_start=warm_start, ignore_dpp = True)

        x = np.array(self.x.value)
        self.theta_prev = x
        w = np.array(self.w.value)
        v = np.array(self.v.value)
        solve_time = self.prob.solver_stats.solve_time

        return x, v, w, solve_time