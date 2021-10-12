import numpy as np

from pydrake.all import (PiecewisePolynomial, PiecewiseQuaternionSlerp,
                         MultibodyPlant, RigidTransform,
                         RotationMatrix, BodyFrame)
from pydrake.multibody import inverse_kinematics
import pydrake.solvers.mathematicalprogram as mp


def calc_joint_trajectory(X_WE_start: RigidTransform,
                          X_WE_final: RigidTransform,
                          duration: float,
                          frame_E: BodyFrame, plant: MultibodyPlant,
                          q_initial_guess: np.ndarray,
                          n_knots: int = 15):
    R_WE_traj = PiecewiseQuaternionSlerp(
        [0, duration], [X_WE_start.rotation().ToQuaternion(),
                        X_WE_final.rotation().ToQuaternion()])
    p_WEo_traj = PiecewisePolynomial.FirstOrderHold(
        [0, duration], np.vstack([X_WE_start.translation(),
                                 X_WE_final.translation()]).T)

    position_tolerance = 0.002
    angle_tolerance = 0.01
    nq = 7

    q_knots = np.zeros((n_knots + 1, nq))
    q_knots[0] = q_initial_guess

    for i in range(1, n_knots + 1):
        t = i / n_knots * duration
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q()

        # Position constraint
        p_WQ_ref = p_WEo_traj.value(t).ravel()
        ik.AddPositionConstraint(
            frameB=frame_E, p_BQ=np.zeros(3),
            frameA=plant.world_frame(),
            p_AQ_lower=p_WQ_ref - position_tolerance,
            p_AQ_upper=p_WQ_ref + position_tolerance)

        # Orientation constraint
        R_WE_ref = RotationMatrix(R_WE_traj.value(t))
        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=R_WE_ref,
            frameBbar=frame_E,
            R_BbarB=RotationMatrix(),
            theta_bound=angle_tolerance)

        prog = ik.prog()
        # use the robot posture at the previous knot point as
        # an initial guess.
        prog.SetInitialGuess(q_variables, q_knots[i-1])
        result = mp.Solve(prog)
        assert result.is_success()
        q_knots[i] = result.GetSolution(q_variables)

    t_knots = np.linspace(0, duration, n_knots + 1)
    q_traj_forward = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        t_knots, q_knots.T, np.zeros(nq), np.zeros(nq))
    q_traj_reverse = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        t_knots, q_knots[::-1].T, np.zeros(nq), np.zeros(nq))

    return q_traj_forward, q_traj_reverse
