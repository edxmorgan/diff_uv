"""Export SO(3) vehicle controllers with existing ABI. No plant/EKF changes."""
import argparse
from pathlib import Path
import casadi as ca
from diffUV.controller import Controller
from diffUV.utils.rotation_error import rotation_log
from diffUV.utils.symbols import Kp, Ki, Kd, sum_e_buffer, n, x_nb, nd, xb_d, dt, des_acc_b, v_c


def export(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = Controller()
    args = [model.lumped_params, Kp, Ki, Kd, sum_e_buffer, ca.vertcat(n, x_nb), nd, xb_d]
    ca.Function('pid_control_reg', args+[dt], list(model.position_pid_reg())).save(
        str(output/'uv_reg_pid_controller.casadi'))
    ca.Function('trackingpid', args+[des_acc_b, dt, v_c], list(model.trajectorytracking_pid())).save(
        str(output/'uv_trackingController.casadi'))
    rotation = ca.SX.sym('R', 3, 3)
    ca.Function('rotation_log', [rotation], [rotation_log(rotation)]).save(str(output/'rotation_log.casadi'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    export(parser.parse_args().output)
