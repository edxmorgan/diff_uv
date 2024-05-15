from casadi import SX,  vertcat, DM

eta,eps1,eps2,eps3 = SX.sym('eta'),SX.sym('eps1'),SX.sym('eps2'),SX.sym('eps3')
q = [eta,eps1,eps2,eps3] #unit quaternion

# 6 DOF states vectors in body-fixed
u = SX.sym('u')
du = SX.sym('du')
v = SX.sym('v')
dv = SX.sym('dv')
w = SX.sym('w')
dw = SX.sym('dw')
v_nb = vertcat(u, v, w)  # body-fixed linear velocity
dv_nb = vertcat(du, dv, dw)

p = SX.sym('p')
dp = SX.sym('dp')
q = SX.sym('q')
dq = SX.sym('dq')
r = SX.sym('r')
dr = SX.sym('dr')
w_nb = vertcat(p, q, r)  # body-fixed angular velocity
dw_nb = vertcat(dp, dq, dr)

x_nb = vertcat(v_nb, w_nb)
dx_nb = vertcat(dv_nb, dw_nb)
################################################

# 6 DOF states vectors in NED
x = SX.sym('x')
dx = SX.sym('dx')
ddx = SX.sym('ddx')
y = SX.sym('y')
dy = SX.sym('dy')
ddy = SX.sym('ddy')
z = SX.sym('z')
dz = SX.sym('dz')
ddz = SX.sym('ddz')
p_n = vertcat(x, y, z)  # NED linear velocity
dp_n = vertcat(dx, dy, dz)
ddp_n = vertcat(ddx, ddy, ddz)


thet = SX.sym('thet')
dthet = SX.sym('dthet')
ddthet = SX.sym('ddthet')
phi = SX.sym('phi')
dphi = SX.sym('dphi')
ddphi = SX.sym('ddphi')
psi = SX.sym('psi')
dpsi = SX.sym('dpsi')
ddpsi = SX.sym('ddpsi')

eul = vertcat(thet, phi, psi)  # NED euler angular velocity
deul = vertcat(dthet, dphi, dpsi)
ddeul = vertcat(ddthet, ddphi, ddpsi)

n = vertcat(p_n, eul)
dn  = vertcat(dp_n, deul)
ddn  = vertcat(ddp_n, ddeul)

###################################################


W = SX.sym('W')  # weight
B = SX.sym('B')  # buoyancy

m = SX.sym('m')  # Mass

I_x = SX.sym('I_x')  # moment of inertia x entry
I_y = SX.sym('I_y')  # moment of inertia y entry
I_z = SX.sym('I_z')  # moment of inertia z entry
I_zx = SX.sym('I_zx')  # product of inertia zx entry
I_yx = SX.sym('I_yx')  # product of inertia yx entry
I_zy = SX.sym('I_zy')  # product of inertia zy entry

x_g = SX.sym('x_g')  # Center of gravity, x-axis
y_g = SX.sym('y_g')  # Center of gravity, y-axis
z_g = SX.sym('z_g')  # Center of gravity, z-axis

x_b = SX.sym('x_b')  # Center of buoyancy, x-axis
y_b = SX.sym('y_b')  # Center of buoyancy, y-axis
z_b = SX.sym('z_b')  # Center of buoyancy, z-axis

X_du = SX.sym('X_du') # Added mass in surge
X_dv = SX.sym('X_dv') # coupled Added mass in surge & sway
X_dw = SX.sym('X_dw') # coupled Added mass in surge & heave
X_dp = SX.sym('X_dp') # coupled Added mass in surge & roll
X_dq = SX.sym('X_dq') # coupled Added mass in surge & pitch
X_dr = SX.sym('X_dr') # coupled Added mass in surge & pitch

Y_du = SX.sym('Y_du') # Added mass in sway & surge
Y_dv = SX.sym('Y_dv') # Added mass in sway
Y_dw = SX.sym('Y_dw') # coupled Added mass in sway & heave
Y_dp = SX.sym('Y_dp') # coupled Added mass in sway & roll
Y_dq = SX.sym('Y_dq') # coupled Added mass in sway & pitch
Y_dr = SX.sym('Y_dr') # coupled Added mass in sway & yaw

Z_du = SX.sym('Z_du') # coupled Added mass in heave & surge
Z_dv = SX.sym('Z_dv') # coupled Added mass in heave & sway
Z_dw = SX.sym('Z_dw') # Added mass in heave
Z_dp = SX.sym('Z_dp') # coupled Added mass in heave & roll
Z_dq = SX.sym('Z_dq') # coupled Added mass in heave & pitch
Z_dr = SX.sym('Z_dr') # coupled Added mass in heave & yaw

K_du = SX.sym('K_du') # coupled Added mass in roll & surge
K_dv = SX.sym('K_dv') # coupled Added mass in roll & sway
K_dw = SX.sym('K_dw') # coupled Added mass in roll & heave
K_dp = SX.sym('K_dp') # Added mass in roll
K_dq = SX.sym('K_dq') # coupled Added mass in roll & pitch
K_dr = SX.sym('K_dr') # coupled Added mass in roll & yaw

M_du = SX.sym('M_du') # coupled Added mass in pitch & surge
M_dv = SX.sym('M_dv') # coupled Added mass in pitch & sway
M_dw = SX.sym('M_dw') # coupled Added mass in pitch & heave
M_dp = SX.sym('M_dp') # coupled Added mass in pitch & roll
M_dq = SX.sym('M_dq') # Added mass in pitch
M_dr = SX.sym('M_dr') # coupled Added mass in pitch & yaw

N_du = SX.sym('N_du') # coupled Added mass in yaw & surge
N_dv = SX.sym('N_dv') # coupled Added mass in yaw & sway
N_dw = SX.sym('N_dw') # coupled Added mass in yaw & heave
N_dp = SX.sym('N_dp') # coupled Added mass in yaw & roll
N_dq = SX.sym('N_dq') # coupled Added mass in yaw & pitch
N_dr = SX.sym('N_dr') # Added mass in yaw


# Starboard–port symmetrical underwater vehicles config
star_board_config = DM([[1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1]
           ])
