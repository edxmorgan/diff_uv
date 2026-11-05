"""Shortest SO(3) logarithm without Euler-rate charts.

The half-turn has two equally short axes; choose a deterministic quaternion
branch. No globally continuous shortest-rotation error exists at that cut.
"""
import casadi as ca


def rotation_log(rotation):
    r = rotation
    # Largest quaternion component avoids dividing by sin(theta) near pi.
    squares = ca.vertcat(1+ca.trace(r), 1+2*r[0, 0]-ca.trace(r),
                        1+2*r[1, 1]-ca.trace(r), 1+2*r[2, 2]-ca.trace(r))
    numerators = [
        ca.vertcat(squares[0], r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]),
        ca.vertcat(r[2, 1]-r[1, 2], squares[1], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0]),
        ca.vertcat(r[0, 2]-r[2, 0], r[0, 1]+r[1, 0], squares[2], r[1, 2]+r[2, 1]),
        ca.vertcat(r[1, 0]-r[0, 1], r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], squares[3])]
    best = squares[0]
    q = numerators[0] / (2*ca.sqrt(ca.fmax(squares[0], 1e-16)))
    for i in range(1, 4):
        candidate = numerators[i] / (2*ca.sqrt(ca.fmax(squares[i], 1e-16)))
        q = ca.if_else(squares[i] > best, candidate, q)
        best = ca.fmax(best, squares[i])
    q = ca.if_else(q[0] < 0, -q, q)
    s2 = ca.dot(q[1:], q[1:])
    s = ca.sqrt(ca.fmax(s2, 1e-16))
    factor = ca.if_else(s2 < 1e-12, 2+s2/3, 2*ca.atan2(s, q[0])/s)
    return factor*q[1:]
