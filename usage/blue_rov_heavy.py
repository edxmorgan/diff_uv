# Copyright 2024, Edward Morgan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pandas as pd

class Params:

    # Ocean current velocities. 
    v_flow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # (m/s). Assume irrotational, constant.
    
    # Alternative thrust config matrix. 
    thrust_config = np.array([[0.707, 0.707, -0.707, -0.707, 0.0, 0.0, 0.0, 0.0],
                            [-0.707, 0.707, -0.707, 0.707, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0,-1.0],
                            [0.06, -0.06, 0.06, -0.06, -0.218, -0.218, 0.218, 0.218],
                            [0.06, 0.06, -0.06, -0.06, 0.12, -0.12, 0.12, -0.12],
                            [-0.1888, 0.1888, 0.1888, -0.1888, 0.0, 0.0, 0.0, 0.0]])
    
    # Thrust coefficient matrix
    # K = np.diag([40, 40, 40, 40, 40, 40, 40, 40])

    ### Parameters in rigid body dynamics and restoring forces
    # Based on BlueRobotics 2018b technical specs. 
    # Based on Table 5.1
    m = 11.5 #(kg)
    W = m*9.81 #(N). 112.8 N. Weight. 
    B = 114.8 #(N). bouyancy 

    rb  = np.array([0, 0, 0]) #(m). center of buoyancy (CoB) coincides with the center of origin
    rg = np.array([0, 0, -0.02]) #(m). 

    # Axis inertias. 
    # BAsed on Table 5.1.
    I_x = 0.8571 #(kg m2)
    I_y = 1.0 #(kg m2)
    I_z = 1.0 #(kg m2)
    I_xz = 0
    Io = np.array([I_x, I_y, I_z, I_xz])

    # Added mass parameters.
    # Based on Table 5.2. 
    X_du = 10.77#(kg). Surge. 
    Y_dv = 24.86 #(kg). Sway. 
    Z_dw = 34.60 #(kg). Heave.  dive --> 34.60 or rise --> 22.45
    K_dp = 0.103 #(kg m2/rad). Roll.
    M_dq = 0.120 #(kg m2/rad). Pitch. 
    N_dr = 0.120 #(kg m2/rad). Yaw. 
    added_m = np.array([X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr])

    coupl_added_m = np.array([0, 0, 0, 0]) # ASSUMING decoupling motion

    # Linear damping coeffs. 
    Xu = 38.95 #(Ns/m). Surge. 
    Yv = 60.11 #(Ns/m). Sway.
    Zw = 49.50 #(Ns/m). Heave.  rise --> 50.62
    Kp = 0.0 #(Ns/rad). Roll.
    Mq = 0.0 #(Ns/rad). Pitch.
    Nr = 0.0 #(Ns/rad). Yaw. 
    linear_dc = np.array([Xu, Yv, Zw, Kp,  Mq, Nr])

    # Quadratic damping coeffs. 
    Xuu = 31.01 #(Ns2/m2). Surge. 
    Yvv = 36.95 #(Ns2/m2). Sway. 
    Zww = 113.86 #(Ns2/m2). Heave. rise --> 76.69
    Kpp = 2.08 #(Ns2/rad2). Roll. 
    Mqq = 2.42 #(Ns2/rad2). Pitch. 
    Nrr = 2.42 #(Ns2/rad2). Yaw. 
    quadratic_dc = np.array([Xuu, Yvv, Zww, Kpp, Mqq, Nrr])


    rg = np.array([0, 0, 0.02]) #(m). 
    rb = np.array([0, 0, 0]) #(m). center of buoyancy (CoB) coincides with the center of origin

    T = 70 # time horizon in seconds
    N = 1600 # number of control intervals
    at_surface = 0.0
    below_surface = 1.0 #random
    dt_s = T/N

    kp = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ki = np.array([0, 0, 0, 0, 0, 0])
    kd = np.array([2, 2, 2, 2, 2, 2])
    
    sim_params = np.concatenate(( np.array([m]) , np.array([W]), np.array([B]), 
                                           rg, rb, Io, added_m, coupl_added_m, linear_dc, quadratic_dc, v_flow))
    

    # https://gist.github.com/edxmorgan/4d38ca349537a36214927f16359848a1


    # ----------------------------
    # User-specified parameters
    # ----------------------------
    forward_poly_order = 9    # Order for PWM -> Thrust mapping
    reverse_poly_order = 25   # Order for Thrust -> PWM mapping
    no_thrusters = 8          # Number of thrusters for CasADi mapping

    # Define bounds for the forward mapping (PWM -> Thrust)
    # (Adjust these values as appropriate for your system.)
    pwm_input_lb = 1100.0  # Lower bound for PWM input
    pwm_input_ub = 1900.0  # Upper bound for PWM input

    # ----------------------------
    # Read CSV file and convert thrust from Ibf (lbf) to Newtons (N)
    # ----------------------------
    csv = pd.read_csv('t200.csv')
    conversion_factor = 4.44822
    csv['thrust 16V'] = csv['thrust 16V'] * conversion_factor

    # ----------------------------
    # Forward mapping: PWM -> Thrust
    # ----------------------------

    # Extract data from CSV
    x_pwm = csv['pwm 16V'].tolist()
    y_thrust = csv['thrust 16V'].tolist()

    # Fit a polynomial of user-specified order: PWM -> Thrust
    coefs = np.polyfit(x_pwm, y_thrust, forward_poly_order)
    ffit = np.polyval(coefs, x_pwm)

    # # Plot the forward mapping (raw data vs. fitted curve)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_pwm, y_thrust, 'bo', label='Original Data')
    # plt.plot(x_pwm, ffit, 'r-', label=f'Fitted Polynomial (order {forward_poly_order})')
    # plt.xlabel('PWM 16V')
    # plt.ylabel('Thrust (N)')
    # plt.legend()
    # plt.title('PWM to Thrust Conversion for bluerov T200 thruster')
    # plt.show()

    # # Calculate and print the RMS error for the forward mapping
    # RMS = np.sqrt(np.mean((np.array(ffit) - np.array(y_thrust))**2))
    # print("Forward mapping RMS error: {:.3f} N".format(RMS))

    # Define output bounds for thrust based on the polynomial evaluated at the PWM bounds
    thrust_output_lb = np.polyval(coefs, pwm_input_lb)
    thrust_output_ub = np.polyval(coefs, pwm_input_ub)

    # Create a CasADi function for the forward mapping with bounds.
    # The input is first saturated (clipped) to [pwm_input_lb, pwm_input_ub],
    # then the polynomial is evaluated, and the output is clipped to [thrust_output_lb, thrust_output_ub].
    pwm = ca.SX.sym('pwm')

    # Saturate the PWM input:
    pwm_sat = ca.fmax(pwm_input_lb, ca.fmin(pwm, pwm_input_ub))

    # Build the polynomial expression using the saturated input.
    thrust_expr = 0
    for i, coef in enumerate(coefs):
        # Exponent decreases from forward_poly_order down to 0.
        exponent = forward_poly_order - i
        thrust_expr += coef * pwm_sat**exponent

    # Saturate the output thrust:
    thrust_bounded = ca.fmax(thrust_output_lb, ca.fmin(thrust_expr, thrust_output_ub))

    # Create the forward mapping CasADi function and map it to no_thrusters.
    pwm_to_thrust = ca.Function('Pwm_to_thrust', [pwm], [thrust_bounded]).map(no_thrusters)

    # ----------------------------
    # Reverse mapping: Thrust -> PWM
    # ----------------------------

    # For the inverse mapping, we fit a polynomial with thrust as the independent variable.
    inv_coefs = np.polyfit(y_thrust, x_pwm, reverse_poly_order)
    ffit_inv = np.polyval(inv_coefs, y_thrust)

    # # Plot the reverse mapping (raw data vs. fitted curve)
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_thrust, x_pwm, 'bo', label='Original Inverse Data')
    # plt.plot(y_thrust, ffit_inv, 'r-', label=f'Fitted Inverse Polynomial (order {reverse_poly_order})')
    # plt.xlabel('Thrust (N)')
    # plt.ylabel('PWM 16V')
    # plt.legend()
    # plt.title('Thrust to PWM Conversion for bluerov T200 thruster')
    # plt.show()

    # Calculate and print the RMS error for the reverse mapping
    # RMS_inv = np.sqrt(np.mean((np.array(ffit_inv) - np.array(x_pwm))**2))
    # print("Reverse mapping RMS error: {:.3f}".format(RMS_inv))

    # For the reverse mapping, set input bounds based on the thrust data.
    # Here, we use the min and max values from the original thrust data.
    thrust_input_lb = min(y_thrust)
    thrust_input_ub = max(y_thrust)

    # And output bounds for PWM based on the original PWM data.
    pwm_output_lb = min(x_pwm)
    pwm_output_ub = max(x_pwm)

    # Create a CasADi function for the reverse mapping with bounds.
    # The input thrust is clipped to [thrust_input_lb, thrust_input_ub],
    # then the inverse polynomial is evaluated, and the output PWM is clipped to [pwm_output_lb, pwm_output_ub].
    thrust = ca.SX.sym('thrust')

    # Saturate the thrust input:
    thrust_sat = ca.fmax(thrust_input_lb, ca.fmin(thrust, thrust_input_ub))

    # Build the inverse polynomial expression using the saturated thrust input.
    pwm_expr = 0
    for i, coef in enumerate(inv_coefs):
        exponent = reverse_poly_order - i
        pwm_expr += coef * thrust_sat**exponent

    # Saturate the PWM output:
    pwm_bounded = ca.fmax(pwm_output_lb, ca.fmin(pwm_expr, pwm_output_ub))

    # Create the reverse mapping CasADi function and map it to no_thrusters.
    thrust_to_pwm = ca.Function('Thrust_to_pwm', [thrust], [pwm_bounded]).map(no_thrusters)
    n_thrust = ca.SX.sym('thrust',8)
    getNpwm_func = ca.Function('getNpwm', [n_thrust], [thrust_to_pwm(n_thrust)])


    # ----------------------------
    # Construct PWM -> RPM Mapping
    # ----------------------------
    # Choose a polynomial order for PWM -> RPM
    rpm_poly_order = 11  # Adjust this as needed

    # Extract PWM and RPM data from CSV
    x_pwm_rpm = csv['pwm 16V'].values
    y_rpm = csv['rpm 16V'].values

    # Fit a polynomial of user-specified order: PWM -> RPM
    rpm_coefs = np.polyfit(x_pwm_rpm, y_rpm, rpm_poly_order)
    rpm_fit = np.polyval(rpm_coefs, x_pwm_rpm)

    # (Optional) Plot the fitted polynomial vs. original data
    # plt.figure(figsize=(10,6))
    # plt.plot(x_pwm_rpm, y_rpm, 'bo', label='Raw RPM Data')
    # plt.plot(x_pwm_rpm, rpm_fit, 'r-', label=f'Fitted Polynomial (order {rpm_poly_order})')
    # plt.xlabel('PWM 16V')
    # plt.ylabel('RPM')
    # plt.legend()
    # plt.title('PWM to RPM Conversion for bluerov T200 thruster')
    # plt.show()

    # Calculate and print RMS error for your own analysis (optional)
    # RMS_rpm = np.sqrt(np.mean((rpm_fit - y_rpm)**2))
    # print("PWM->RPM mapping RMS error: {:.3f} RPM".format(RMS_rpm))

    # Define bounds for PWM input (these may be the same as for the thrust mapping)
    pwm_input_lb = min(x_pwm_rpm)
    pwm_input_ub = max(x_pwm_rpm)

    # Optionally define known RPM bounds (if you have them), or base on polynomial extremes
    # For example, if your thruster realistically spins from 0 to 5000 RPM:
    rpm_output_lb = min(y_rpm) # or min(rpm_fit)
    rpm_output_ub = max(y_rpm)   # or max(rpm_fit) or some known maximum

    # Create the CasADi function for PWM -> RPM
    pwm_sym = ca.SX.sym('pwm')
    pwm_sat_rpm = ca.fmax(pwm_input_lb, ca.fmin(pwm_sym, pwm_input_ub))

    # Build polynomial expression
    rpm_expr = 0
    for i, coef in enumerate(rpm_coefs):
        exponent = rpm_poly_order - i
        rpm_expr += coef * (pwm_sat_rpm ** exponent)

    # (Optional) saturate output RPM if desired
    rpm_bounded = ca.fmax(rpm_output_lb, ca.fmin(rpm_expr, rpm_output_ub))

    # Convert RPM to rad/s = rpm_bounded * (2π / 60)
    rad_expr = rpm_bounded * (2 * np.pi / 60.0)

    # Create the function that maps a single PWM input to rad/s output
    pwm_to_rads_single = ca.Function('pwm_to_rads_single', [pwm_sym], [rad_expr])  
    # (Or replace [rad_expr] with [rad_bounded] if using extra saturation.)

    # Map it across multiple thrusters
    pwm_to_rads = pwm_to_rads_single.map(no_thrusters)
    n_pwm_rad = ca.SX.sym('n_pwm_rad',8)
    pwm_to_rads_func = ca.Function('pwm_to_rads', [n_pwm_rad], [pwm_to_rads(n_pwm_rad)])