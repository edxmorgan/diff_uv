"""RTS pair moments checked against joint Gaussian conditioning, not a second RTS."""
import numpy as np

from diffUV.estimation import disturbance_ekf_smoothing_moment, disturbance_ekf_smoothing_full_moment


def test_pair_moment_matches_joint_gaussian_conditioning():
    rng = np.random.default_rng(783)
    f = disturbance_ekf_smoothing_moment()
    full = disturbance_ekf_smoothing_full_moment()
    for _ in range(8):
        xf = rng.normal(0, .01, 18)
        L = rng.normal(size=(18, 18)) * .02
        Pf = L @ L.T + np.eye(18) * .01
        A = np.eye(18) + rng.normal(size=(18, 18)) * .02
        Q = np.diag(rng.uniform(.001, .01, 18))
        xp = A @ xf
        Pp = A @ Pf @ A.T + Q
        joint_mean = np.r_[xf, xp]
        joint_P = np.block([[Pf, Pf @ A.T], [A @ Pf, Pp]])
        H = rng.normal(size=(9, 18))
        R = np.eye(9) * .02
        y = H @ xp + rng.normal(0, .02, 9)
        observation = np.c_[np.zeros((9, 18)), H]
        K = np.linalg.solve(observation @ joint_P @ observation.T + R,
                            observation @ joint_P).T
        mean = joint_mean + K @ (y - observation @ joint_mean)
        posterior = joint_P - K @ observation @ joint_P
        xs, Ps, moments = map(np.asarray, f(xf, Pf, xp, Pp, A,
                                           mean[18:], posterior[18:, 18:]))
        np.testing.assert_allclose(xs.ravel(), mean[:18], atol=1e-14)
        np.testing.assert_allclose(Ps, posterior[:18, :18], atol=1e-14)
        D = np.zeros((6, 36))
        D[:, 12:18] = -np.eye(6)
        D[:, 30:36] = np.eye(6)
        expected = (D @ mean)**2 + np.diag(D @ posterior @ D.T)
        np.testing.assert_allclose(moments.ravel(), expected, atol=1e-14)
        full_result = full(xf, Pf, xp, Pp, A, mean[18:], posterior[18:, 18:])
        expected_full = np.outer(D @ mean, D @ mean) + D @ posterior @ D.T
        np.testing.assert_allclose(full_result[2], expected_full, atol=1e-14)
        np.testing.assert_allclose(np.diag(np.asarray(full_result[2])), moments.ravel(), atol=1e-14)
        assert np.linalg.eigvalsh(np.asarray(full_result[2])).min() > -1e-14
        assert np.linalg.eigvalsh(Ps).min() > 0


def test_unobserved_random_walk_recovers_increment_variance():
    f = disturbance_ekf_smoothing_moment()
    x = np.zeros(18)
    Pf = np.eye(18) * .3
    Q = np.diag(np.linspace(.0001, .002, 18))
    xs, Ps, moments = map(np.asarray, f(x, Pf, x, Pf + Q, np.eye(18), x, Pf + Q))
    np.testing.assert_allclose(xs.ravel(), x, atol=0)
    np.testing.assert_allclose(Ps, Pf, atol=0)
    np.testing.assert_allclose(moments.ravel(), np.diag(Q)[12:], atol=1e-15)


def test_euler_branch_residual_does_not_invent_turn():
    f = disturbance_ekf_smoothing_moment()
    xf = np.zeros(18)
    xp = np.zeros(18)
    xn = np.zeros(18)
    xp[3:6] = np.pi - .01
    xn[3:6] = -np.pi + .01
    Pf = np.eye(18)
    xs, Ps, moments = map(np.asarray, f(xf, Pf, xp, Pf * 2, Pf, xn, Pf))
    np.testing.assert_allclose(xs.ravel()[3:6], .01, atol=1e-15)
    np.testing.assert_allclose(Ps, Pf * .75, atol=1e-15)
    assert np.isfinite(moments).all()
