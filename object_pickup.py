import os


from pydrake.all import (
    PiecewisePolynomial, PiecewiseQuaternionSlerp, Parser, PidController,
    RandomGenerator, Simulator, ProcessModelDirectives, LoadModelDirectives,
    MatrixGain,
)
from pydrake.math import RollPitchYaw

from manipulation.scenarios import AddRgbdSensors

from utils import render_system_with_graphviz
from grasp_sampler import *
from gripper_pose_controller import (GripperPoseController,
                                     CustomTrajectorySource)
from lime_bag import add_bag_of_lime, initialize_bag_of_lime

#%%
# object SDFs.
object_names = ['Lime', 'Cucumber', 'Mango']
# object_names = ['Lime']
object_sdfs = [os.path.join(os.getcwd(), 'cad_files', name + '_simplified.sdf')
               for name in object_names]


#%%
def make_environment_model(
        directive=None, draw=False, rng=None, num_objects=0, bin_name="bin0",
        add_gripper_control=False):
    """
    Make one model of the environment, but the robot only gets to see the sensor
     outputs.
    """
    if not directive:
        directive = FindResource("models/two_bins_w_cameras.yaml")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    AddPackagePaths(parser)
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

    if add_gripper_control:
        # Gripper Pose Control
        schunk_body = plant.GetBodyByName('body')
        gpc = GripperPoseController(gripper_body_idx=schunk_body.index())
        builder.AddSystem(gpc)
        builder.Connect(
            plant.get_body_spatial_velocities_output_port(),
            gpc.body_spatial_velocity_input_port)
        builder.Connect(
            plant.get_body_poses_output_port(),
            gpc.body_pose_input_port)
        builder.Connect(
            gpc.spatial_force_output_port,
            plant.get_applied_spatial_force_input_port())

        # Gripper Finger Control
        schunk_model = plant.GetModelInstanceByName('gripper')
        K_schunk = np.array([500, 500.])
        D_schunk = np.array([10, 10.])
        gfc = PidController(K_schunk, np.zeros(2), D_schunk)
        gfc.set_name('gripper_finger_controller')

        S = np.zeros((4, 17))
        S[0, 7] = 1
        S[1, 8] = 1
        S[2, -2] = 1
        S[3, -1] = 1
        finger_state_selector = MatrixGain(S)
        finger_state_selector.set_name('finger_state_selector')
        builder.AddSystem(gfc)
        builder.AddSystem(finger_state_selector)
        builder.Connect(
            gfc.get_output_port_control(),
            plant.get_actuation_input_port(schunk_model))
        builder.Connect(
            plant.get_state_output_port(schunk_model),
            finger_state_selector.get_input_port())
        builder.Connect(
            finger_state_selector.get_output_port(),
            gfc.get_input_port_estimated_state())

        # trajectory source for gripper.
        finger_setpoints = PiecewisePolynomial.ZeroOrderHold(
            [0, 1], np.array([[0.1], [0.1]]).T)

        p_WB_traj = PiecewisePolynomial.ZeroOrderHold(
            [0, 1], np.array([[0, 0, 0], [0, 0, 0.]]).T)

        Q_WB = RollPitchYaw(np.pi / 2, 0, 0).ToQuaternion()
        Q_WB_traj = PiecewiseQuaternionSlerp([0, 1], [Q_WB, Q_WB])

        traj_source = CustomTrajectorySource(
            p_WB_traj=p_WB_traj, Q_WB_traj=Q_WB_traj,
            finger_setpoint_traj=finger_setpoints)

        builder.AddSystem(traj_source)
        builder.Connect(
            traj_source.body_pose_output_port,
            gpc.pose_ref_input_port)

        builder.Connect(
            traj_source.finger_setpoint_output_port,
            gfc.get_input_port_desired_state())

    if draw:
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment")

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

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
        simulator.AdvanceTo(5.0)
        simulator.set_target_realtime_rate(0.)
        # viz.stop_recording()
        # viz.publish_recording()

        if draw:
            viz.stop_recording()
            viz.publish_recording(play=False)
    elif draw:
        viz.load()
        diagram.Publish(context)

    return diagram, context



#%%
directive_file = os.path.join(
    os.getcwd(), 'models', 'two_bins_and_actuated_shcunk.yml')

rng = np.random.default_rng(seed=10001)

# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

# build environment and grasp sampler.
env, context_env = make_environment_model(
    directive=directive_file, rng=rng, draw=True, num_objects=7,
    add_gripper_control=True)
grasp_sampler = GraspSampler(env)

# sample some grasps.
X_Gs_best = grasp_sampler.sample_grasp_candidates(
    context_env, draw_grasp_candidates=True)

#%%
# start gripper from 0.3m above the grasp, move in, grasp and pick up.
n_knots = 4
t_knots = [0, 2.5, 3.5, 7]
finger_setpoint_knots = np.array([[0.1, 0.01, 0.01, 0.01]])
p_WB_knots = np.zeros((4, 3))
X_WB = X_Gs_best[0]
p_WB_knots[0] = X_WB.translation() + np.array([0, 0, 0.3])
p_WB_knots[1] = X_WB.translation()
p_WB_knots[2] = X_WB.translation()
p_WB_knots[3] = p_WB_knots[0]

Q_WB = X_WB.rotation().ToQuaternion()
finger_setpoint_traj = PiecewisePolynomial.ZeroOrderHold(
    t_knots, finger_setpoint_knots)
p_WB_traj = PiecewisePolynomial.FirstOrderHold(t_knots, p_WB_knots.T)
Q_WB_traj = PiecewiseQuaternionSlerp(t_knots, [Q_WB] * n_knots)

traj_source = env.GetSubsystemByName('custom_trajectory_source')
traj_source.Q_WB_traj = Q_WB_traj
traj_source.finger_setpoint_traj = finger_setpoint_traj
traj_source.p_WB_traj = p_WB_traj

context_new = env.CreateDefaultContext()
plant_env = env.GetSubsystemByName('plant')
context_plant_old = plant_env.GetMyContextFromRoot(context_env)
context_plant_new = plant_env.GetMyContextFromRoot(context_new)
plant_env.SetPositionsAndVelocities(
    context_plant_new,
    plant_env.GetPositionsAndVelocities(context_plant_old))

# gripper
schunk_model = plant_env.GetModelInstanceByName('gripper')
schunk_body = plant_env.GetBodyByName('body')
schunk_position = np.zeros(9)
schunk_position[:4] = Q_WB.wxyz()
schunk_position[4:7] = p_WB_knots[0]
schunk_position[7:9] = [-finger_setpoint_knots[0, 0] / 2,
                        finger_setpoint_knots[0, 0] / 2]
plant_env.SetPositions(context_plant_new, schunk_model, schunk_position)

env.Publish(context_new)

# sim
sim = Simulator(env, context_new)
viz = env.GetSubsystemByName('meshcat_visualizer')
viz.reset_recording()
viz.start_recording()
sim.AdvanceTo(t_knots[-1])
viz.stop_recording()
viz.publish_recording()


