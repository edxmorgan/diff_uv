import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_dof_matrix(time, vel, acc, u, τ_sim_rows, colors=None, step_inputs=True, csv_path=None):
    """
    Plot velocity, acceleration, and input for each DOF.
    If csv_path is given, save the Body Input panel data to a CSV.

    CSV columns:
      t, u.u, u.v, u.w, u.p, u.q, u.r, tau_sim.u, tau_sim.v, tau_sim.w, tau_sim.p, tau_sim.q, tau_sim.r
    """
    rows = ["u", "v", "w", "p", "q", "r"]

    unit_vel = {"u":"m s$^{-1}$","v":"m s$^{-1}$","w":"m s$^{-1}$",
                "p":"rad s$^{-1}$","q":"rad s$^{-1}$","r":"rad s$^{-1}$"}
    unit_acc = {"u":"m s$^{-2}$","v":"m s$^{-2}$","w":"m s$^{-2}$",
                "p":"rad s$^{-2}$","q":"rad s$^{-2}$","r":"rad s$^{-2}$"}
    unit_in  = {"u":"N","v":"N","w":"N","p":"N m","q":"N m","r":"N m"}

    fig, axes = plt.subplots(6, 3, figsize=(16, 18), sharex=True)
    for ri, dof in enumerate(rows):
        c = colors.get(dof) if colors else None

        # Velocity
        axes[ri, 0].plot(time, vel[dof], linewidth=1.2, color=c)
        # Optional IMU overlays if these globals exist
        if dof == 'p' and 'imu_ang_vel_x' in globals():
            axes[ri, 0].plot(time, globals()['imu_ang_vel_x'], linewidth=1.2, color=c)
        if dof == 'q' and 'imu_ang_vel_y' in globals():
            axes[ri, 0].plot(time, globals()['imu_ang_vel_y'], linewidth=1.2, color=c)
        if dof == 'r' and 'imu_ang_vel_z' in globals():
            axes[ri, 0].plot(time, globals()['imu_ang_vel_z'], linewidth=1.2, color=c)
        axes[ri, 0].set_ylabel(f"{dof} [{unit_vel[dof]}]")

        # Acceleration
        axes[ri, 1].plot(time, acc[dof], linewidth=1.2, color=c)
        axes[ri, 1].set_ylabel(f"{dof} [{unit_acc[dof]}]")

        # Input
        if step_inputs:
            axes[ri, 2].step(time, u[dof], where="post", linewidth=1.0, color=c)
            axes[ri, 2].step(time, τ_sim_rows[:, ri], where="post", linewidth=1.0)
        else:
            axes[ri, 2].plot(time, u[dof], linewidth=1.0, color=c)
            axes[ri, 2].plot(time, τ_sim_rows[:, ri], linewidth=1.0)
        axes[ri, 2].set_ylabel(f"{dof} [{unit_in[dof]}]")

        for ci in range(3):
            axes[ri, ci].grid(True, linestyle=":", linewidth=0.6)

    # Column titles
    axes[0, 0].set_title("Body Velocity")
    axes[0, 1].set_title("Body Acceleration")
    axes[0, 2].set_title("Body Input")

    # Shared x labels
    axes[-1, 0].set_xlabel("Time, s")
    axes[-1, 1].set_xlabel("Time, s")
    axes[-1, 2].set_xlabel("Time, s")

    # Save Body Input data to CSV if requested
    if csv_path:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required to write the CSV file") from e

        t = np.asarray(time)
        if τ_sim_rows.shape[1] != 6:
            raise ValueError(f"τ_sim_rows must have 6 columns for u v w p q r, got {τ_sim_rows.shape[1]}")

        data = {"t": t}
        for i, dof in enumerate(rows):
            data[f"u.{dof}"] = np.asarray(u[dof])
            data[f"tau_sim.{dof}"] = np.asarray(τ_sim_rows[:, i])

        df = pd.DataFrame(data)
        if len(df) != len(t):
            raise ValueError("Time length does not match input lengths for CSV export")
        df.to_csv(csv_path, index=False)

    # Tidy layout and show
    fig.tight_layout()
    plt.show()


def pretty_print_vehicle_params(theta_hat, title="Identified vehicle parameters"):
    """
    Pretty print and sanity check the lumped vehicle parameters used in your regressor.
    Expects the parameter order:
        [ m-X_du, m-Y_dv, m-Z_dw,
          m*z_g-X_dq, -m*z_g+Y_dp, -m*z_g+K_dv, m*z_g-M_du,
          I_x-K_dp, I_y-M_dq, I_z-N_dr,
          W, B, x_gW-x_bB, y_gW-y_bB, z_gW-z_bB,
          X_u, Y_v, Z_w, K_p, M_q, N_r,
          X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr ]
    """
    th = np.asarray(theta_hat).ravel()
    if th.size != 27:
        raise ValueError(f"theta_hat must have 27 entries, got {th.size}")

    # Unpack with explicit names for clarity
    m_Xdu, m_Ydv, m_Zdw = th[0], th[1], th[2]
    mzg_Xdq, _mzg_Ydp, _mzg_Kdv, mzg_Mdu = th[3], th[4], th[5], th[6]
    I_x_Kdp, I_y_Mdq, I_z_Ndr = th[7], th[8], th[9]

    W, B = th[10], th[11]
    xgW_xbB, ygW_ybB, zgW_zbB = th[12], th[13], th[14]

    X_u, Y_v, Z_w, K_p, M_q, N_r = th[15], th[16], th[17], th[18], th[19], th[20]
    X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr = th[21], th[22], th[23], th[24], th[25], th[26]

    # Derived and grouped views
    eff_trans_inertia = {
        "surge, m - X_du": m_Xdu,
        "sway,  m - Y_dv": m_Ydv,
        "heave, m - Z_dw": m_Zdw,
    }
    eff_rot_inertia = {
        "roll,  I_x - K_dp": I_x_Kdp,
        "pitch, I_y - M_dq": I_y_Mdq,
        "yaw,   I_z - N_dr": I_z_Ndr,
    }
    restoring = {
        "weight W [N]": W,
        "buoyancy B [N]": B,
        "net, B - W [N]": B - W,
        "x_g W - x_b B [N m]": xgW_xbB,
        "y_g W - y_b B [N m]": ygW_ybB,
        "z_g W - z_b B [N m]": zgW_zbB,
    }
    linear_drag = {
        "X_u [N s m^-1]": X_u,
        "Y_v [N s m^-1]": Y_v,
        "Z_w [N s m^-1]": Z_w,
        "K_p [N m s rad^-1]": K_p,
        "M_q [N m s rad^-1]": M_q,
        "N_r [N m s rad^-1]": N_r,
    }
    quad_drag = {
        "X_uu [N s^2 m^-2]": X_uu,
        "Y_vv [N s^2 m^-2]": Y_vv,
        "Z_ww [N s^2 m^-2]": Z_ww,
        "K_pp [N m s^2 rad^-2]": K_pp,
        "M_qq [N m s^2 rad^-2]": M_qq,
        "N_rr [N m s^2 rad^-2]": N_rr,
    }
    mzg_lumps = {
        "m*z_g - X_dq": mzg_Xdq,
        "-m*z_g + Y_dp": _mzg_Ydp,
        "-m*z_g + K_dv": _mzg_Kdv,
        "m*z_g - M_du": mzg_Mdu,
    }

    # Print, formatted
    print(f"\n{title}")
    print("----------------------------------------------------")

    print("\nEffective translational inertia terms, linear velocity axes:")
    for k, v in eff_trans_inertia.items():
        print(f"  {k}: {v:.6g}")

    print("\nEffective rotational inertia terms, angular velocity axes:")
    for k, v in eff_rot_inertia.items():
        print(f"  {k}: {v:.6g}")

    print("\nRestoring terms:")
    for k, v in restoring.items():
        print(f"  {k}: {v:.6g}")
    r_gW_minus_r_bB = np.array([xgW_xbB, ygW_ybB, zgW_zbB])
    print(f"  ||[x_g W - x_b B, y_g W - y_b B, z_g W - z_b B]||: {np.linalg.norm(r_gW_minus_r_bB):.6g}")

    print("\nLinear drag coefficients:")
    for k, v in linear_drag.items():
        print(f"  {k}: {v:.6g}")

    print("\nQuadratic drag coefficients:")
    for k, v in quad_drag.items():
        print(f"  {k}: {v:.6g}")

    print("\nLumped center of gravity and coupling terms that include m*z_g and cross added mass:")
    for k, v in mzg_lumps.items():
        print(f"  {k}: {v:.6g}")
    print("  Note, these contain products with m*z_g and cross terms, you cannot separate m, z_g, and added mass without extra information.")

    # Basic sanity checks
    warn = []
    for label, val in eff_trans_inertia.items():
        if val <= 0:
            warn.append(f"Non positive effective translational inertia, {label} = {val:.3g}")
    for label, val in eff_rot_inertia.items():
        if val <= 0:
            warn.append(f"Non positive effective rotational inertia, {label} = {val:.3g}")
    if B <= 0 or W <= 0:
        warn.append(f"W and B should be positive, got W={W:.3g}, B={B:.3g}")

    if warn:
        print("\nWarnings:")
        for w in warn:
            print(f"  - {w}")

    # Return a dict for downstream programmatic use if desired
    return {
        "eff_trans_inertia": eff_trans_inertia,
        "eff_rot_inertia": eff_rot_inertia,
        "restoring": restoring,
        "linear_drag": linear_drag,
        "quad_drag": quad_drag,
        "mzg_lumps": mzg_lumps,
        "warnings": warn,
    }

# Optional, compact visualization of the identified drag and inertia blocks
def plot_vehicle_param_overview(theta_hat):
    th = np.asarray(theta_hat).ravel()
    labels = [
        "m-X_du", "m-Y_dv", "m-Z_dw",
        "I_x-K_dp", "I_y-M_dq", "I_z-N_dr",
        "X_u", "Y_v", "Z_w", "K_p", "M_q", "N_r",
        "X_uu", "Y_vv", "Z_ww", "K_pp", "M_qq", "N_rr",
    ]
    idx = [0,1,2, 7,8,9, 15,16,17,18,19,20, 21,22,23,24,25,26]
    vals = th[idx]

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=60, ha="right")
    plt.ylabel("value")
    plt.title("Vehicle inertia and drag blocks, lumped view")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.show()