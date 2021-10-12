import numpy as np
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
env_sim.run_robot_traj(get_q_traj_10())


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

q_traj_segment_names = [
    "0 to pre-suction",
    "pre-suction to suction",
    "suction",
    "pre-suction to suction",
    "1 to 0",
    "pre-drop to drop",
    "drop",
    "pre-drop to drop",
    "1 to 0"
]

durations_list = [durations[name] for name in q_traj_segment_names]
t_knots = np.cumsum(np.hstack([[0], durations_list]))
suction_setpoints = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]) * 8.0
suction_traj = PiecewisePolynomial.ZeroOrderHold(t_knots, suction_setpoints)
env_sim.suction_traj_source.q_traj = suction_traj

# home EE poses for bin1 and bin2.
X_WE_bin0 = env_sim.calc_ee_pose(q_iiwa_bin0)
X_WE_bin1 = env_sim.calc_ee_pose(q_iiwa_bin1)

#%%
env_sim.viz.reset_recording()
env_sim.viz.start_recording()

d = 0.105
y_positions = np.array([d / 2,
                        0.2 - d / 2 - 0.005,
                        -0.2 + d / 2 * 3 + 0.003,
                        -0.2 + d / 2 + 0.003])



# Transfer
for i_layer in range(n_layers - 1, -1, -1):
    boxes_transferred = []
    # if i_layer == 0:
    #     env_sim.unlock_boxes([3, 2, 1, 0])
    for j in range(3, -1, -1):
        i = i_layer * 4 + j
        print(f"picking up box {i}...")
        boxes_transferred.append(i)

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

        # drop ee pose.
        R_WE_drop = RollPitchYaw(np.pi, 0, np.pi).ToRotationMatrix()
        y_drop_offset = y_positions[j]
        z_drop_offset = 0.03 + 0.04 * (n_layers - 1 - i_layer) + k_suction_offset_z
        p_WB_offset = np.array([0, y_drop_offset, z_drop_offset])
        X_WE_drop = RigidTransform(R_WE_drop, X_WBin1.translation() + p_WB_offset)

        # pre-drop ee pose, which needs to be right above where the box is dropped.
        X_WE_pre_drop = RigidTransform(X_WE_bin1)
        p_WB_pre_drop = np.array(X_WE_pre_drop.translation())
        p_WB_pre_drop[1] = y_drop_offset + X_WBin1.translation()[1]  # set y.
        X_WE_pre_drop.set_translation(p_WB_pre_drop)

        # 1 to pre-drop.
        q_traj_1_to_pre_drop, q_traj_pre_drop_to_1 = calc_joint_trajectory(
            X_WE_start=X_WE_bin1, X_WE_final=X_WE_pre_drop,
            duration=durations["1 to pre-drop"],
            frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
            q_initial_guess=q_iiwa_bin1, n_knots=5)

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

        # pre_suction to pre_drop.
        q_knots = np.zeros((2, 7))
        q_knots[0] = q_traj_pre_suction_to_suction.value(0).ravel()
        q_knots[1] = q_traj_pre_drop_to_drop.value(0).ravel()
        q_traj_pre_suction_to_pre_drop = \
            PiecewisePolynomial.CubicHermite(
                [0, durations["1 to 0"]], q_knots.T, np.zeros((7, 2)))

        # pre_drop to 0
        q_knots = np.zeros((2, 7))
        q_knots[0] = q_traj_pre_drop_to_drop.value(0).squeeze()
        q_knots[1] = q_iiwa_bin0
        q_traj_pre_drop_to_0 = \
            PiecewisePolynomial.CubicHermite(
                [0, durations["1 to 0"]], q_knots.T, np.zeros((7, 2)))

        q_traj = concatenate_traj_list([
            q_traj_0_to_pre_suction, q_traj_pre_suction_to_suction,
            q_traj_suction, q_traj_suction_to_pre_suction,
            q_traj_pre_suction_to_pre_drop,
            q_traj_pre_drop_to_drop, q_traj_drop,
            q_traj_drop_to_pre_drop, q_traj_pre_drop_to_0])

        env_sim.suc_sys.idx_box_to_suck = i
        env_sim.run_robot_traj(q_traj)


env_sim.viz.stop_recording()
env_sim.viz.publish_recording()



