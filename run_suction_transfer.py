from pydrake.all import RotationMatrix

from object_pickup_suction import *
from inverse_kinematics import calc_joint_trajectory

#%%
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

'''
Boxes are arranged into layers of 4 boxes, starting from the lower-left corner. For 
example, 
8, 9, 10, 11
4, 5, 6, 7
0, 1, 2, 3
============== Bottom of the box.
'''
n_layers = 2
n_objects = n_layers * 4
env_sim = EnvSim(n_objects=n_objects, packing_mode=PackingMode.kTransfer)
X_WBin1 = env_sim.get_bin_pose("bin1")

# Move robot to q0.
q_traj_10 = get_q_traj_10()
suction_traj = PiecewisePolynomial.ZeroOrderHold([0, q_traj_10.end_time()], [[0, 0]])
env_sim.run_robot_traj(q_traj_10, suction_traj)


#%%
durations = {
    "1 to 0": 3,
    "0 to pre-suction": 1,
    "pre-suction to suction": 4,
    "suction": 2,
    "1 to pre-drop": 1,
    "pre-drop to drop": 4,
    "drop": 2,
}

q_traj_segment_names_first = [
    "0 to pre-suction",
    "pre-suction to suction",
    "suction",
    "pre-suction to suction",
    "1 to 0"]

q_traj_segment_names_second = [
    "pre-drop to drop",
    "drop",
    "pre-drop to drop",
    "1 to 0"
]

durations_list = [durations[name] for name in q_traj_segment_names_first]
t_knots = np.cumsum(np.hstack([[0], durations_list]))
suction_setpoints = np.array([[0, 0, 1, 1, 1, 0]]) * 8.0
suction_traj_first = PiecewisePolynomial.ZeroOrderHold(t_knots, suction_setpoints)

durations_list = [durations[name] for name in q_traj_segment_names_second]
t_knots = np.cumsum(np.hstack([[0], durations_list]))
suction_setpoints = np.array([[1, 0, 0, 0, 0]]) * 8.0
suction_traj_second = PiecewisePolynomial.ZeroOrderHold(t_knots, suction_setpoints)


# home EE poses for bin1 and bin2.
X_WE_bin0 = env_sim.calc_ee_pose(q_iiwa_bin0)
X_WE_bin1 = env_sim.calc_ee_pose(q_iiwa_bin1)

#%%
env_sim.viz.reset_recording()
env_sim.viz.start_recording()

d = 0.103
y_positions = np.array([d / 2,
                        0.2 - d / 2,
                        -0.2 + d / 2 * 3 + 0.004,
                        -0.2 + d / 2 + 0.003])


# Transfer
for i_layer in range(n_layers - 1, -1, -1):
    for j in range(3, -1, -1):
        i = i_layer * 4 + j
        env_sim.suc_sys.idx_box_to_suck = i
        print(f"picking up box {i}...")

        # -------------------------------------------------------------------------
        # Suction to pre-drop.
        X_WB = env_sim.get_box_pose(i)
        X_WE_suck, _ = calc_suction_ee_pose([X_WB])

        # 0 to pre-suction pose, which needs to right above the box to be picked up.
        p_WB_pre_suck = np.zeros(3)
        p_WB_pre_suck[:2] = X_WB.translation()[:2]  # set pre-suck x, y.
        p_WB_pre_suck[2] = X_WE_bin0.translation()[2] - 0.1
        X_WE_pre_suck = RigidTransform(X_WE_suck.rotation(), p_WB_pre_suck)

        q_traj_0_to_pre_suction, q_traj_pre_suction_to_0 = calc_joint_trajectory(
            X_WE_start=X_WE_bin0, X_WE_final=X_WE_pre_suck,
            duration=durations["0 to pre-suction"],
            frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
            q_initial_guess=q_iiwa_bin0, n_knots=5)

        (q_traj_pre_suction_to_suction,
         q_traj_suction_to_pre_suction) = calc_joint_trajectory(
            X_WE_start=X_WE_pre_suck, X_WE_final=X_WE_suck,
            duration=durations["pre-suction to suction"],
            frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
            q_initial_guess=q_traj_pre_suction_to_0.value(0).ravel(), n_knots=8)

        # hold at suction
        q_suction = q_traj_pre_suction_to_suction.value(
            q_traj_pre_suction_to_suction.end_time()).ravel()
        q_traj_suction = PiecewisePolynomial.ZeroOrderHold(
            [0, durations["suction"]], np.vstack([q_suction, q_suction]).T)

        # Desired box pose.
        p_WB_desired = np.array(
            [0, y_positions[j], 0.03 + 0.04 * (n_layers - 1 - i_layer)])
        p_WB_desired += X_WBin1.translation()
        X_WB_desired = RigidTransform(RotationMatrix(), p_WB_desired)

        # Pre-drop ee pose, which needs to be right above where the box is dropped.
        X_WE_pre_drop = RigidTransform(X_WE_bin1)
        p_WE_pre_drop = np.array(X_WE_pre_drop.translation())
        p_WE_pre_drop[1] = p_WB_desired[1]
        X_WE_pre_drop.set_translation(p_WE_pre_drop)

        # 1 to pre-drop.
        q_traj_1_to_pre_drop, q_traj_pre_drop_to_1 = calc_joint_trajectory(
            X_WE_start=X_WE_bin1, X_WE_final=X_WE_pre_drop,
            duration=durations["1 to pre-drop"],
            frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
            q_initial_guess=q_iiwa_bin1, n_knots=3)

        # pre_suction to pre_drop.
        q_knots = np.zeros((2, 7))
        q_knots[0] = q_traj_pre_suction_to_suction.value(0).ravel()
        q_knots[1] = q_traj_pre_drop_to_1.value(0).ravel()
        q_traj_pre_suction_to_pre_drop = \
            PiecewisePolynomial.CubicHermite(
                [0, durations["1 to 0"]], q_knots.T, np.zeros((7, 2)))

        q_traj = concatenate_traj_list([
            q_traj_0_to_pre_suction, q_traj_pre_suction_to_suction,
            q_traj_suction, q_traj_suction_to_pre_suction,
            q_traj_pre_suction_to_pre_drop])

        env_sim.run_robot_traj(q_traj, suction_traj_first)

        # -------------------------------------------------------------------------
        # Pre-drop to 0.
        q_iiwa_pre_drop = q_traj_pre_drop_to_1.value(0).ravel()
        X_WE = env_sim.calc_ee_pose(q_iiwa_pre_drop)
        X_WB = env_sim.get_box_pose(i)
        X_BE = X_WB.inverse().multiply(X_WE)

        # drop ee pose.
        X_WE_drop = X_WB_desired.multiply(X_BE)

        # pre-drop to drop
        q_traj_pre_drop_to_drop, q_traj_drop_to_pre_drop = calc_joint_trajectory(
            X_WE_start=X_WE_pre_drop, X_WE_final=X_WE_drop,
            duration=durations["pre-drop to drop"],
            frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
            q_initial_guess=q_traj_pre_drop_to_1.value(0).squeeze(), n_knots=8)

        # hold at drop
        q_drop = q_traj_drop_to_pre_drop.value(0).ravel()
        q_traj_drop = PiecewisePolynomial.ZeroOrderHold(
            [0, durations["drop"]], np.vstack([q_drop, q_drop]).T)

        # pre_drop to 0
        q_knots = np.zeros((2, 7))
        q_knots[0] = q_traj_pre_drop_to_drop.value(0).squeeze()
        q_knots[1] = q_iiwa_bin0
        q_traj_pre_drop_to_0 = \
            PiecewisePolynomial.CubicHermite(
                [0, durations["1 to 0"]], q_knots.T, np.zeros((7, 2)))

        q_traj = concatenate_traj_list([q_traj_pre_drop_to_drop, q_traj_drop,
                                        q_traj_drop_to_pre_drop, q_traj_pre_drop_to_0])
        env_sim.run_robot_traj(q_traj, suction_traj_second)


env_sim.viz.stop_recording()
env_sim.viz.publish_recording()



