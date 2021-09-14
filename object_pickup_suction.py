import os
import numpy as np
import meshcat
from pydrake.all import (
    PiecewisePolynomial, PidController, MultibodyPlant, Context,
    RandomGenerator, Simulator, InverseDynamicsController, ContactModel,
    ConnectContactResultsToDrakeVisualizer, DrakeVisualizer, Parser,
    DiagramBuilder, AddMultibodyPlantSceneGraph, ProcessModelDirectives,
    LoadModelDirectives, ConnectMeshcatVisualizer,
    MeshcatContactVisualizer)
from pydrake.math import RollPitchYaw
from manipulation.utils import AddPackagePaths
from grasp_sampler_vision import zmq_url

from utils import (SimpleTrajectorySource, concatenate_traj_list,
                   add_package_paths_local, render_system_with_graphviz)
from build_sim_diagram import (add_controlled_iiwa_and_trj_source,
    set_object_drop_pose)
from grasp_sampler_suction import calc_suction_ee_pose, SuctionSystem
from inverse_kinematics import calc_joint_trajectory



#%%
object_sdf_path = os.path.join(os.path.dirname(__file__), 'models',
                               'blue_berry_box.sdf')

# commonly used trajectories
nq = 7
q_iiwa_bin0 = np.array([-np.pi / 2, 0.1, 0, -1.2, 0, 1.6, 0])
q_iiwa_bin1 = np.array([0, 0.1, 0, -1.2, 0, 1.6, 0])

q_traj_01 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    [0, 3], np.vstack([q_iiwa_bin0, q_iiwa_bin1]).T,
    np.zeros(nq), np.zeros((nq)))


def get_q_traj_10():
    q_traj_10 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        [0, 3], np.vstack([q_iiwa_bin1, q_iiwa_bin0]).T,
        np.zeros(nq), np.zeros((nq)))

    return q_traj_10


q_traj_1_hold = PiecewisePolynomial.ZeroOrderHold(
    [0, 2], np.vstack([q_iiwa_bin1, q_iiwa_bin1]).T)


def add_blueberry_boxes(n_objects: int, plant: MultibodyPlant,
                        parser: Parser):
    object_bodies = []
    for i in range(n_objects):
        model = parser.AddModelFromFile(object_sdf_path, f"box{i}")
        object_bodies.append(plant.GetBodyByName('base_link', model))
    return object_bodies


def make_environment_model(
        directive, rng, draw=False, n_objects=0, bin_name="bin0", draw_contact=False):

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)

    # Add objects.
    box_bodies = add_blueberry_boxes(n_objects, plant, parser)
    plant.Finalize()

    # Robot control.
    plant_iiwa_controller, model_iiwa = add_controlled_iiwa_and_trj_source(
        builder, plant, q0=q_iiwa_bin1, add_schunk_inertia=False)

    if draw:
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment")
        DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)
        if draw_contact:
            viz_c = MeshcatContactVisualizer(viz, plant=plant)
            builder.AddSystem(viz_c)
            builder.Connect(plant.get_contact_results_output_port(),
                            viz_c.GetInputPort('contact_results'))

    # Suction control.
    ss = SuctionSystem(box_body_indices=[body.index() for body in box_bodies],
                       l7_body_index=plant.GetBodyByName("iiwa_link_7").index())
    builder.AddSystem(ss)
    builder.Connect(plant.get_body_poses_output_port(),
                    ss.body_poses_input_port)
    builder.Connect(ss.easf_output_port, plant.get_applied_spatial_force_input_port())

    suction_setpoints = PiecewisePolynomial.ZeroOrderHold([0, 1], np.array([[0, 0]]))
    suction_traj_source = SimpleTrajectorySource(suction_setpoints)
    suction_traj_source.set_name("suction_traj_source")
    builder.AddSystem(suction_traj_source)
    builder.Connect(suction_traj_source.x_output_port, ss.suction_strength_input_port)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    # render_system_with_graphviz(diagram)

    # Set initial conditions.
    # robot and gripper initial conditions.
    context_plant = plant.GetMyContextFromRoot(context)
    plant.SetPositions(context_plant, model_iiwa, q_iiwa_bin1)

    simulator = None
    if n_objects > 0:
        set_object_drop_pose(
            context_environment=context,
            plant=plant,
            bin_name=bin_name,
            object_bodies=box_bodies,
            lime_bag_bodies=[],
            rng=rng,
            R_WB=RollPitchYaw(0, 0, np.pi/2),
            x_lb=-0.15, x_ub=0.05, y_lb=-0.2, y_ub=0.2)

        simulator = Simulator(diagram, context)
        viz.start_recording()
        simulator.AdvanceTo(5.0)
        simulator.set_target_realtime_rate(0.)
        viz.stop_recording()
        viz.publish_recording()

    elif draw:
        viz.load()
        diagram.Publish(context)

    return diagram, context, plant_iiwa_controller, simulator, box_bodies


class EnvSim:
    def __init__(self, n_objects: int):
        directive_file = os.path.join(
            os.getcwd(), 'models', 'iiwa_suction_and_two_bins.yml')
        rng = np.random.default_rng(seed=1215232)

        (self.env, self.context, self.plant_iiwa_c, self.sim,
         self.box_bodies) = make_environment_model(
            directive=directive_file, rng=rng, draw=True, n_objects=n_objects)

        self.iiwa_model = self.plant_iiwa_c.GetModelInstanceByName('iiwa')
        self.frame_E = self.plant_iiwa_c.GetBodyByName('iiwa_link_7').body_frame()

        self.robot_traj_source = self.env.GetSubsystemByName('robot_traj_source')
        self.suction_traj_source = self.env.GetSubsystemByName('suction_traj_source')
        self.plant_env = self.env.GetSubsystemByName('plant')
        self.viz = self.env.GetSubsystemByName('meshcat_visualizer')
        self.suc_sys = self.env.GetSubsystemByName('suction_system')

        self.context_plant_env = self.plant_env.GetMyContextFromRoot(self.context)

    def calc_ee_pose(self, q_iiwa: np.ndarray):
        context = self.plant_iiwa_c.CreateDefaultContext()
        self.plant_iiwa_c.SetPositions(context, q_iiwa)
        return self.plant_iiwa_c.CalcRelativeTransform(
            context, self.plant_iiwa_c.world_frame(), self.frame_E)

    def get_box_poses(self):
        X_WB_list = []
        for body in self.box_bodies:
            X_WB_list.append(
                self.plant_env.EvalBodyPoseInWorld(self.context_plant_env, body))

        return X_WB_list


#%%
# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

env_sim = EnvSim(n_objects=1)

durations = np.array([3, 2, 2, 2])
# durations:
# 0: 1 to 0
# 1: 0 to suction
# 2: suction
# 3: suction to 0.

t_knots = np.cumsum(np.hstack([[0], durations]))
suction_setpoints = np.array([[0, 0, 1, 1, 1]])
suction_traj = PiecewisePolynomial.ZeroOrderHold(t_knots, suction_setpoints)
env_sim.suction_traj_source.q_traj = suction_traj

# home EE poses for bin1 and bin2.
X_WE_bin0 = env_sim.calc_ee_pose(q_iiwa_bin0)
X_WE_bin1 = env_sim.calc_ee_pose(q_iiwa_bin1)

X_WB_list = env_sim.get_box_poses()
X_WE_suck, idx_suck = calc_suction_ee_pose(X_WB_list)


env_sim.viz.reset_recording()
env_sim.viz.start_recording()

# 0 to suction
q_traj_0_to_suction, q_traj_suction_to_0 = calc_joint_trajectory(
    X_WE_start=X_WE_bin0, X_WE_final=X_WE_suck, duration=durations[1],
    frame_E=env_sim.frame_E, plant=env_sim.plant_iiwa_c,
    q_initial_guess=q_iiwa_bin0)

# hold
q_suction = q_traj_0_to_suction.value(q_traj_0_to_suction.end_time()).ravel()
q_traj_suction = PiecewisePolynomial.ZeroOrderHold(
    [0, durations[2]], np.vstack([q_suction, q_suction]).T)

# update time in trajectory sources.
q_traj = concatenate_traj_list([
    get_q_traj_10(), q_traj_0_to_suction, q_traj_suction, q_traj_suction_to_0])
t_current = env_sim.context.get_time()
env_sim.robot_traj_source.set_t_start(t_current)
env_sim.robot_traj_source.q_traj = q_traj

env_sim.suction_traj_source.set_t_start(t_current)
env_sim.suc_sys.idx_box_to_suck = idx_suck

# simulate forward.
env_sim.sim.AdvanceTo(t_current + q_traj.end_time())

env_sim.viz.publish_recording()
