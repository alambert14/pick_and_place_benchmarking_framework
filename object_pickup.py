from typing import List

import numpy as np
import pydrake
from pydrake.all import (
    PiecewisePolynomial, PiecewiseQuaternionSlerp, Parser, PidController,
    RandomGenerator, Simulator, ProcessModelDirectives, LoadModelDirectives,
    MatrixGain, MultibodyPlant, InverseDynamicsController
)
from pydrake.math import RollPitchYaw

from manipulation.scenarios import AddRgbdSensors

from iiwa_controller.iiwa_controller.utils import (
    create_iiwa_controller_plant)

from utils import (render_system_with_graphviz, add_package_paths,
                   SimpleTrajectorySource)
from grasp_sampler import *
from lime_bag import add_bag_of_lime, initialize_bag_of_lime
from inverse_kinematics import calc_joint_trajectory

#%%
# object SDFs.
object_names = ['Lime', 'Cucumber', 'Mango']
# object_names = ['Lime']
sdf_dir = os.path.join(os.path.dirname(__file__), 'cad_files')
object_sdfs = [os.path.join(sdf_dir, name + '_simplified.sdf')
               for name in object_names]

q_iiwa_bin0 = np.array([-np.pi / 2, 0.1, 0, -1.2, 0, 1.6, 0])
q_iiwa_bin1 = np.array([0, 0.1, 0, -1.2, 0, 1.6, 0])


def concatenate_traj_list(traj_list: List[PiecewisePolynomial]):
    """
    Concatenates a list of PiecewisePolynomials into a single
        PiecewisePolynomial.
    """
    traj = traj_list[0]
    for a in traj_list[1:]:
        dt = traj.end_time()
        a.shiftRight(dt)
        traj.ConcatenateInTime(a)
        a.shiftRight(-dt)

    return traj


def make_environment_model(
        directive=None, draw=False, rng=None, num_objects=0, bin_name="bin0",
        add_robot=False):
    """
    Make one model of the environment, but the robot only gets to see the sensor
     outputs.
    """
    if not directive:
        directive = FindResource("models/two_bins_w_cameras.yaml")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths(parser)  # local.
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)

    object_bodies = []
    n_bags_of_lime = 0
    for i in range(num_objects):
        i_obj = rng.integers(len(object_sdfs))
        if object_names[i_obj] == 'Lime':
            lime_bodies = add_bag_of_lime(n_limes=5, bag_index=n_bags_of_lime,
                                          plant=plant, parser=parser)
            object_bodies.append(lime_bodies)
            n_bags_of_lime += 1
        else:
            model = parser.AddModelFromFile(object_sdfs[i_obj], f"object{i}")
            object_bodies.append(plant.GetBodyByName('base_link', model))

    plant.Finalize()
    AddRgbdSensors(builder, plant, scene_graph)
    plant_iiwa_contoller = None

    if add_robot:
        # robot control.
        plant_iiwa_controller, _ = create_iiwa_controller_plant(
            gravity=plant.gravity_field().gravity_vector(),
            add_schunk_inertia=True)

        Kp_iiwa = np.ones(7) * 100
        Kd_iiwa = 2 * np.sqrt(Kp_iiwa)
        Ki_iiwa = np.ones(7)
        idc = InverseDynamicsController(plant_iiwa_controller, Kp_iiwa,
                                        Ki_iiwa, Kd_iiwa, False)
        builder.AddSystem(idc)
        model_iiwa = plant.GetModelInstanceByName('iiwa')
        builder.Connect(plant.get_state_output_port(model_iiwa),
                        idc.get_input_port_estimated_state())
        builder.Connect(idc.get_output_port_control(),
                        plant.get_actuation_input_port(model_iiwa))

        # robot trajectory source
        q_knots = np.zeros((2, 7))
        q_knots[0] = q_iiwa_bin1
        robot_traj_source = SimpleTrajectorySource(
            PiecewisePolynomial.ZeroOrderHold(
            [0, 1], q_knots.T))
        builder.AddSystem(robot_traj_source)
        builder.Connect(robot_traj_source.x_output_port,
                        idc.get_input_port_desired_state())
        robot_traj_source.set_name('robot_traj_source')

        # Gripper Finger Control
        model_schunk = plant.GetModelInstanceByName('gripper')
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
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment")

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    if add_robot:
        # robot and gripper initial conditions.
        context_plant = plant.GetMyContextFromRoot(context)
        plant.SetPositions(context_plant, model_iiwa, q_iiwa_bin1)
        plant.SetPositions(context_plant, model_schunk,
                           finger_setpoints.value(0).ravel())

    if num_objects > 0:
        generator = RandomGenerator(rng.integers(1000))  # this is for c++
        plant_context = plant.GetMyContextFromRoot(context)
        bin_instance = plant.GetModelInstanceByName(bin_name)
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
        z = 0.3
        l_spring = 0.08  # for lime bags.
        for object_body in object_bodies:
            tf = RigidTransform(
                RotationMatrix(),
                [rng.uniform(-.15, .15), rng.uniform(-.2, .2), z])

            if isinstance(object_body, list):
                # bag of lime
                initialize_bag_of_lime(l_spring=l_spring,
                                       X_WL0=X_B.multiply(tf), plant=plant,
                                       context_plant=plant_context,
                                       lime_bodies=object_body)
            else:
                # mango, cucumber.
                plant.SetFreeBodyPose(plant_context,
                                      object_body,
                                      X_B.multiply(tf))
            z += 0.05

        simulator = Simulator(diagram, context)
        if draw:
            viz.start_recording()

        # viz.start_recording()
        simulator.AdvanceTo(3.0)
        simulator.set_target_realtime_rate(0.)
        # viz.stop_recording()
        # viz.publish_recording()

        if draw:
            viz.stop_recording()
            viz.publish_recording(play=False)
    elif draw:
        viz.load()
        diagram.Publish(context)

    return diagram, context, plant_iiwa_controller, simulator


#%%
directive_file = os.path.join(
    os.getcwd(), 'models', 'iiwa_schunk_and_two_bins.yml')

rng = np.random.default_rng(seed=10001)

# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

# build environment and grasp sampler.
env, context_env, plant_iiwa_controller, sim = make_environment_model(
    directive=directive_file, rng=rng, draw=True, num_objects=7,
    add_robot=True)
grasp_sampler = GraspSampler(env)

#%% home EE poses for bin1 and bin2.
context_iiwa_plant = plant_iiwa_controller.CreateDefaultContext()
iiwa_model = plant_iiwa_controller.GetModelInstanceByName('iiwa')
frame_E = plant_iiwa_controller.GetBodyByName('wsg_equivalent').body_frame()

plant_iiwa_controller.SetPositions(context_iiwa_plant, q_iiwa_bin0)
X_WE_bin0 = plant_iiwa_controller.CalcRelativeTransform(
    context_iiwa_plant, plant_iiwa_controller.world_frame(), frame_E)

plant_iiwa_controller.SetPositions(context_iiwa_plant, q_iiwa_bin1)
X_WE_bin1 = plant_iiwa_controller.CalcRelativeTransform(
    context_iiwa_plant, plant_iiwa_controller.world_frame(), frame_E)

# commonly used trajectories
nq = 7
q_traj_01 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    [0, 3], np.vstack([q_iiwa_bin0, q_iiwa_bin1]).T,
    np.zeros(nq), np.zeros((nq)))


def get_q_traj_10():
    q_traj_10 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        [0, 3], np.vstack([q_iiwa_bin1, q_iiwa_bin0]).T,
        np.zeros(nq), np.zeros((nq)))

    return q_traj_10


q_traj_1_hold = PiecewisePolynomial.ZeroOrderHold(
    [0, 1], np.vstack([q_iiwa_bin1, q_iiwa_bin1]).T)


#%%
robot_traj_source = env.GetSubsystemByName('robot_traj_source')
schunk_traj_source = env.GetSubsystemByName('schunk_traj_source')
viz = env.GetSubsystemByName('meshcat_visualizer')

plant_env = env.GetSubsystemByName('plant')
# context_plant = plant_env.GetMyContextFromRoot(context_env)
# model_iiwa = plant_env.GetModelInstanceByName('iiwa')

durations = np.array([3, 2, 3, 1, 3, 2, 3, 1])
# t_knots:
# 0: 1
# 1: 0
# 2: above
# 3: grasp
# 4: grasp hold
# 5: above
# 6: 0
# 7: 1
# 8: 1 hold
t_knots = np.cumsum(np.hstack([[0], durations]))
schunk_setpoints = np.array([[-0.05, 0.05],
                             [-0.05, 0.05],
                             [-0.05, 0.05],
                             [0, 0],
                             [0, 0],
                             [0, 0],
                             [0, 0],
                             [-0.05, 0.05],
                             [-0.05, 0.05]])
schunk_traj = PiecewisePolynomial.ZeroOrderHold(t_knots, schunk_setpoints.T)
schunk_traj_source.q_traj = schunk_traj

viz.reset_recording()
viz.start_recording()
while True:
    # Sample some grasps.
    print('Sampling new grasps...')
    X_Gs_best = grasp_sampler.sample_grasp_candidates(
        context_env, draw_grasp_candidates=False)
    if len(X_Gs_best) == 0:
        break

    # bin0 home to "above" pose
    X_WE_grasp = X_Gs_best[0]
    X_WE_above = RigidTransform(X_WE_grasp)
    X_WE_above.set_translation(X_WE_grasp.translation() + np.array([0, 0, 0.3]))
    q_traj_0_to_above, q_traj_above_to_0 = calc_joint_trajectory(
        X_WE_start=X_WE_bin0, X_WE_final=X_WE_above, duration=durations[1],
        frame_E=frame_E, plant=plant_iiwa_controller,
        q_initial_guess=q_iiwa_bin0)

    # above to grasp
    q_traj_above_to_grasp, q_traj_grasp_to_above = calc_joint_trajectory(
        X_WE_start=X_WE_above, X_WE_final=X_WE_grasp, duration=durations[2],
        frame_E=frame_E, plant=plant_iiwa_controller,
        q_initial_guess=q_traj_0_to_above.value(durations[2]).ravel())

    # hold
    q_grasping = q_traj_above_to_grasp.value(durations[2]).ravel()
    q_traj_grasp_hold = PiecewisePolynomial.ZeroOrderHold(
        [0, durations[3]], np.vstack([q_grasping, q_grasping]).T)

    q_traj_10 = get_q_traj_10()
    q_traj = concatenate_traj_list(
        [q_traj_10, q_traj_0_to_above, q_traj_above_to_grasp,
         q_traj_grasp_hold,
         q_traj_grasp_to_above, q_traj_above_to_0, q_traj_01, q_traj_1_hold])

    # update time in trajectory sources.
    t_current = context_env.get_time()
    schunk_traj_source.set_t_start(t_current)
    robot_traj_source.set_t_start(t_current)
    robot_traj_source.q_traj = q_traj

    sim.AdvanceTo(t_current + q_traj.end_time())

viz.stop_recording()
viz.publish_recording()

