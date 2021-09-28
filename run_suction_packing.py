from object_pickup_suction import *


# %%
# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

n_objects = 7
env_sim = EnvSim(n_objects=n_objects, packing_mode=PackingMode.kStack)
X_WBin1 = env_sim.get_bin_pose("bin1")

'''
durations:
# 0: 1 to 0
# 1: 0 to suction
# 2: suction
# 3: suction to 0.
# 4: 0 to 1
# 5: 1 to drop
# 6: drop
# 7: drop to 1
'''
durations = np.array([3, 2, 2, 2, 3, 2, 2, 2])

t_knots = np.cumsum(np.hstack([[0], durations]))
suction_setpoints = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0]]) * 1.0
suction_traj = PiecewisePolynomial.ZeroOrderHold(t_knots, suction_setpoints)
env_sim.suction_traj_source.q_traj = suction_traj

# home EE poses for bin1 and bin2.
X_WE_bin0 = env_sim.calc_ee_pose(q_iiwa_bin0)
X_WE_bin1 = env_sim.calc_ee_pose(q_iiwa_bin1)

box_to_pick_indices = list(np.arange(n_objects))
n_packed = 0

env_sim.viz.reset_recording()
env_sim.viz.start_recording()
while len(box_to_pick_indices) > 0:
    # Find box to pick up by suction.
    X_WB_list = env_sim.get_box_poses()
    X_WE_suck, idx_into_sublist_suck = calc_suction_ee_pose(
        [X_WB_list[i] for i in box_to_pick_indices])
    idx_suck = box_to_pick_indices[idx_into_sublist_suck]
    box_to_pick_indices.remove(idx_suck)

    # 0 to suction
    q_traj_0_to_suction, q_traj_suction_to_0 = calc_joint_trajectory(
        X_WE_start=X_WE_bin0, X_WE_final=X_WE_suck, duration=durations[1],
        frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
        q_initial_guess=q_iiwa_bin0)

    # hold at suction
    q_suction = q_traj_0_to_suction.value(q_traj_0_to_suction.end_time()).ravel()
    q_traj_suction = PiecewisePolynomial.ZeroOrderHold(
        [0, durations[2]], np.vstack([q_suction, q_suction]).T)

    # 1 to drop
    R_WE_drop = RollPitchYaw(np.pi, 0, np.pi).ToRotationMatrix()
    z_W_offset = 0.03 + k_suction_offset_z + n_packed * 0.045
    X_WE_drop = RigidTransform(
        R_WE_drop,
        X_WBin1.translation() + np.array([0, 0, 1]) * z_W_offset)
    q_traj_1_to_drop, q_traj_drop_to_1 = calc_joint_trajectory(
        X_WE_start=X_WE_bin1, X_WE_final=X_WE_drop, duration=durations[5],
        frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
        q_initial_guess=q_iiwa_bin1)

    # hold at drop
    q_drop = q_traj_drop_to_1.value(0).ravel()
    q_traj_drop = PiecewisePolynomial.ZeroOrderHold(
        [0, durations[6]], np.vstack([q_drop, q_drop]).T)

    # update time in trajectory sources.
    q_traj = concatenate_traj_list([
        get_q_traj_10(), q_traj_0_to_suction, q_traj_suction, q_traj_suction_to_0,
        q_traj_01, q_traj_1_to_drop, q_traj_drop, q_traj_drop_to_1])
    t_current = env_sim.context.get_time()
    env_sim.robot_traj_source.set_t_start(t_current)
    env_sim.robot_traj_source.q_traj = q_traj

    env_sim.suction_traj_source.set_t_start(t_current)
    env_sim.suc_sys.idx_box_to_suck = idx_suck

    # simulate forward.
    env_sim.sim.AdvanceTo(t_current + q_traj.end_time())
    n_packed += 1

env_sim.viz.stop_recording()
env_sim.viz.publish_recording()

res = env_sim.viz.vis.static_html()
