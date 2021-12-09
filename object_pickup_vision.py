from typing import List, Dict
import os.path
import time

import cv2
import numpy as np
from pydrake.all import (
    PiecewisePolynomial, PidController, MultibodyPlant, RigidTransform,
    RandomGenerator, Simulator, InverseDynamicsController, ContactModel,
    ConnectContactResultsToDrakeVisualizer, DrakeVisualizer, Parser,
    DiagramBuilder, AddMultibodyPlantSceneGraph, ProcessModelDirectives,
    LoadModelDirectives, ConnectMeshcatVisualizer, RollPitchYaw)

import meshcat

from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import AddPackagePaths

from robot_utils import (SimpleTrajectorySource, concatenate_traj_list,
                   add_package_paths_local, render_system_with_graphviz)
from grasp_sampler_vision import GraspSamplerVision, zmq_url
from inverse_kinematics import calc_joint_trajectory
from build_sim_diagram import (add_controlled_iiwa_and_trj_source, add_objects,
    set_object_drop_pose)

#%%
# object SDFs
object_names = ['Mango', 'Cucumber', 'Lime']
sdf_dir = os.path.join(os.path.dirname(__file__), 'cad_files')
object_sdfs = {name: os.path.join(sdf_dir, name + '_simplified.sdf')
               for name in object_names}

bin_qs = {
    0: np.array([-np.pi / 2, 0.1, 0, -1.2, 0, 1.6, 0]),
    1: np.array([0, 0.1, 0, -1.2, 0, 1.6, 0]),
    2: np.array([np.pi / 2, 0.1, 0, -1.2, 0, 1.6, 0]),
    3: np.array([-np.pi, 0.1, 0, -1.2, 0, 1.6, 0])
}
label_to_bin = {'Cucumber': 1, 'Lime': 2, 'Mango': 3}


def make_environment_model(
        directive, rng, draw=False, n_objects=0, bin_name="bin0"):
    """
    Make one model of the environment, but the robot only gets to see the sensor
     outputs.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=2e-4)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)

    # Add objects.
    object_bodies, lime_bag_bodies = add_objects(
        n_objects=n_objects,
        obj_names=object_names,
        obj_sdfs=object_sdfs,
        plant=plant,
        parser=parser,
        rng=rng)

    # Set contact model.
    plant.set_contact_model(ContactModel.kPointContactOnly)
    plant.Finalize()
    AddRgbdSensors(builder, plant, scene_graph)

    # Robot control.
    plant_iiwa_controller, model_iiwa = add_controlled_iiwa_and_trj_source(builder, plant,
                                                                           q0=bin_qs[1])

    # Gripper Finger Control
    model_schunk = plant.GetModelInstanceByName('schunk')
    Kp_schunk = np.array([500, 500.])
    Kd_schunk = np.array([10, 10.])
    gfc = PidController(Kp_schunk, np.zeros(2), Kd_schunk)
    gfc.set_name('gripper_finger_controller')
    builder.AddSystem(gfc)
    builder.Connect(
        gfc.get_output_port_control(),
        plant.get_actuation_input_port(model_schunk))
    builder.Connect(
        plant.get_state_output_port(model_schunk),
        gfc.get_input_port_estimated_state())

    # trajectory source for gripper.
    finger_setpoints = PiecewisePolynomial.ZeroOrderHold(
        [0, 1], np.array([[-0.05, 0.05], [-0.05, 0.05]]).T)
    schunk_traj_source = SimpleTrajectorySource(finger_setpoints)
    schunk_traj_source.set_name("schunk_traj_source")

    builder.AddSystem(schunk_traj_source)
    builder.Connect(
        schunk_traj_source.x_output_port,
        gfc.get_input_port_desired_state())

    if draw:
        # Meshcat.
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment")

        # DrakeVisualizer (for hydroelasitc visualization.)
        DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)
        ConnectContactResultsToDrakeVisualizer(builder, plant)
        # # visualizer of lime bags.
        # lime_bag_vis = LimeBagVisualizer(lime_bag_bodies, viz.vis)
        # builder.AddSystem(lime_bag_vis)
        # builder.Connect(
        #     plant.get_body_poses_output_port(),
        #     lime_bag_vis.body_pose_input_port)

    diagram = builder.Build()
    render_system_with_graphviz(diagram)
    context = diagram.CreateDefaultContext()

    # Set initial conditions.
    # robot and gripper initial conditions.
    context_plant = plant.GetMyContextFromRoot(context)
    plant.SetPositions(context_plant, model_iiwa, bin_qs[1])
    plant.SetPositions(context_plant, model_schunk,
                       finger_setpoints.value(0).ravel())

    simulator = None
    if n_objects > 0:
        set_object_drop_pose(
            context_env=context,
            plant=plant,
            bin_name=bin_name,
            object_bodies=object_bodies,
            lime_bag_bodies=lime_bag_bodies,
            rng=rng)

        simulator = Simulator(diagram, context)

        simulator.AdvanceTo(5.0)
        simulator.set_target_realtime_rate(0.)

    elif draw:
        viz.load()
        diagram.Publish(context)

    return diagram, context, plant_iiwa_controller, simulator


#%%
directive_file = os.path.join(
    os.getcwd(), 'models', 'iiwa_schunk_and_two_bins.yml')

rng = np.random.default_rng()# seed=19)# seed=1215232)
# seed 12153432 looks kind of nice.

# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

# build environment and grasp sampler.
env, context_env, plant_iiwa_controller, sim = make_environment_model(
    directive=directive_file, rng=rng, draw=True, n_objects=10)
grasp_sampler = GraspSamplerVision(env)

#%% home EE poses for all bins
context_iiwa_plant = plant_iiwa_controller.CreateDefaultContext()
iiwa_model = plant_iiwa_controller.GetModelInstanceByName('iiwa')
frame_E = plant_iiwa_controller.GetBodyByName('wsg_equivalent').body_frame()

def bin_pose(bin_id):
    plant_iiwa_controller.SetPositions(context_iiwa_plant, bin_qs[bin_id])
    return plant_iiwa_controller.CalcRelativeTransform(
        context_iiwa_plant, plant_iiwa_controller.world_frame(), frame_E)

# commonly used trajectories
nq = 7

# def get_bin_traj(bin_id):
#     q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
#         [0, durations[1]], np.vstack([bin_qs[0], bin_qs[bin_id]]).T,
#         np.zeros(nq), np.zeros((nq)))
#     return q_traj


# def get_return_traj(bin_id):
#     return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
#         [0, durations[0]], np.vstack([bin_qs[bin_id], bin_qs[0]]).T,
#         np.zeros(nq), np.zeros((nq)))

#%%
robot_traj_source = env.GetSubsystemByName('robot_traj_source')
schunk_traj_source = env.GetSubsystemByName('schunk_traj_source')
viz = env.GetSubsystemByName('meshcat_visualizer')

plant_env = env.GetSubsystemByName('plant')
# context_plant = plant_env.GetMyContextFromRoot(context_env)
# model_iiwa = plant_env.GetModelInstanceByName('iiwa')

durations = np.array([3, 2, 2, 1, 2, 2, 3, 2, 1, 3])
# t_knots:
# 0: Over bin 0
# 1: Above grasp
# 2: Grasp
# 3: Grasp hold
#       close gripper
# 4: Above grasp
# 5: Above bin 0
# 6: Over bin
# 7: Lower to bin
# 8: Bin hold
#       open gripper
# 9: Over bin
t_knots = np.cumsum(np.hstack([[0], durations]))
schunk_setpoints = np.array([[-0.05, 0.05],
                             [-0.05, 0.05],
                             [-0.05, 0.05],
                             [0, 0],
                             [0, 0],
                             [0, 0],
                             [0, 0],
                             [0, 0],
                             [-0.05, 0.05],
                             [-0.05, 0.05],
                             [-0.05, 0.05]])
schunk_traj = PiecewisePolynomial.ZeroOrderHold(t_knots, schunk_setpoints.T)
schunk_traj_source.q_traj = schunk_traj

viz.reset_recording()
viz.start_recording()

cam1 = env.GetSubsystemByName('camera1')
cam1_context = cam1.GetMyMutableContextFromRoot(context_env)

last_bin = 0  # Keep track of where the arm was in the last cycle
while True:
    rgb_image = cam1.GetOutputPort('color_image').Eval(cam1_context).data
    # print(type(rgb_image))
    # print(rgb_image.shape)
    # img = rgb_image[:,:,:-1]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # file = 'test.png'
    # cv2.imwrite(file, img)
    # Sample some grasps.
    print('Sampling new grasps...')
    X_Gs_best, label = grasp_sampler.sample_grasp_candidates(
        context_env, draw_grasp_candidates=False)
    if len(X_Gs_best) == 0:
        print('No more grasp candidates, terminating...')
        break

    bin_id = label_to_bin[label[0]]

    # bin0 home to "above" pose
    X_WE_grasp = X_Gs_best[0]
    print(X_WE_grasp)
    X_WE_above = RigidTransform(X_WE_grasp)
    X_WE_above.set_translation(X_WE_grasp.translation() + np.array([0, 0, 0.3]))
    print(X_WE_above)

    # Lower bin pose
    X_WE_lower = RigidTransform(bin_pose(bin_id))
    X_WE_lower.set_translation(bin_pose(bin_id).translation() + np.array([0, 0, -0.3]))

    start_time = time.time()
    # 0: Over bin 0
    q_bin_to_0_traj, _ = calc_joint_trajectory(
        X_WE_start=bin_pose(last_bin), X_WE_final=bin_pose(0), duration=durations[0],
        frame_E=frame_E, plant=plant_iiwa_controller,
        q_initial_guess=bin_qs[last_bin])

    # 6: Over bin
    q_0_to_bin_traj, _ = calc_joint_trajectory(
        X_WE_start=bin_pose(0), X_WE_final=bin_pose(bin_id), duration=durations[6],
        frame_E=frame_E, plant=plant_iiwa_controller,
        q_initial_guess=bin_qs[0])
    print(f'calc_joint_traj time: {time.time() - start_time}')

    # 1: Above grasp   5: Above bin 0
    q_0_to_above_traj, q_above_to_0_traj = calc_joint_trajectory(
        X_WE_start=bin_pose(0), X_WE_final=X_WE_above, duration=durations[1],
        frame_E=frame_E, plant=plant_iiwa_controller,
        q_initial_guess=bin_qs[0])

    # 2: Grasp             4: Above grasp
    q_above_to_grasp_traj, q_grasp_to_above_traj = calc_joint_trajectory(
        X_WE_start=X_WE_above, X_WE_final=X_WE_grasp, duration=durations[2],
        frame_E=frame_E, plant=plant_iiwa_controller,
        q_initial_guess=q_0_to_above_traj.value(durations[1]).ravel())

    # 3: Grasp hold
    q_grasping = q_above_to_grasp_traj.value(durations[2]).ravel()  # Evaluate grasp point
    q_grasp_hold_traj = PiecewisePolynomial.ZeroOrderHold(
        [0, durations[3]], np.vstack([q_grasping, q_grasping]).T)

    # 7: Lower to bin    # 9: Over bin
    q_bin_to_lower_traj, q_lower_to_bin_traj = calc_joint_trajectory(
        X_WE_start=bin_pose(bin_id), X_WE_final=X_WE_lower, duration=durations[7],
        frame_E=frame_E, plant=plant_iiwa_controller,
        q_initial_guess=bin_qs[bin_id])

    # 8: Over bin hold
    q_lower = q_bin_to_lower_traj.value(durations[7]).ravel()
    q_lower_over_bin_hold_traj = PiecewisePolynomial.ZeroOrderHold(
        [0, durations[8]], np.vstack([q_lower, q_lower]).T)

    q_traj = concatenate_traj_list(
        [q_bin_to_0_traj,  # 0: Over bin 0
         q_0_to_above_traj,  # 1: Above grasp
         q_above_to_grasp_traj,  # 2: Grasp
         q_grasp_hold_traj,  # 3: Grasp hold
         q_grasp_to_above_traj,  # 4: Above grasp
         q_above_to_0_traj,  # 5: Above bin 0
         q_0_to_bin_traj,  # 6: Over bin
         q_bin_to_lower_traj,  # 7: Lower to bin
         q_lower_over_bin_hold_traj,  # 8: Over bin hold
         q_lower_to_bin_traj])  # 9: Over bin

    last_bin = bin_id

    # update time in trajectory sources.
    t_current = context_env.get_time()
    schunk_traj_source.set_t_start(t_current)
    robot_traj_source.set_t_start(t_current)
    robot_traj_source.q_traj = q_traj

    sim.AdvanceTo(t_current + q_traj.end_time())


viz.stop_recording()
viz.publish_recording()

