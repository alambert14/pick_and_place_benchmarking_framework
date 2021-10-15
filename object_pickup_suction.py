import os
from enum import Enum
from typing import List

import numpy as np
import meshcat
from pydrake.all import (
    PiecewisePolynomial, PidController, MultibodyPlant, Context,
    RandomGenerator, Simulator, InverseDynamicsController, ContactModel,
    ConnectContactResultsToDrakeVisualizer, DrakeVisualizer, Parser,
    DiagramBuilder, AddMultibodyPlantSceneGraph, ProcessModelDirectives,
    LoadModelDirectives, ConnectMeshcatVisualizer,
    MeshcatContactVisualizer)
from pydrake.math import RollPitchYaw, RigidTransform
from manipulation.utils import AddPackagePaths
from grasp_sampler_vision import zmq_url

from utils import (SimpleTrajectorySource, concatenate_traj_list,
                   add_package_paths_local, render_system_with_graphviz)
from build_sim_diagram import (add_controlled_iiwa_and_trj_source,
                               set_object_drop_pose, set_object_squeeze_pose,
                               set_object_transfer_pose)
from grasp_sampler_suction import calc_suction_ee_pose, SuctionSystem, k_suction_offset_z


# %%
blueberry_sdf_path = os.path.join(
    os.path.dirname(__file__), 'models', 'blue_berry_box.sdf')
blueberry_small_soft_sdf_path = os.path.join(
    os.path.dirname(__file__), 'models', 'blue_berry_box_small_soft.sdf')
blueberry_small_4_bottom_spheres_sdf_path = os.path.join(
    os.path.dirname(__file__), 'models', 'blue_berry_box_small_4_bottom_spheres.sdf')
blueberry_small_rigid_sdf_path = os.path.join(
    os.path.dirname(__file__), 'models', 'blue_berry_box_small_rigid.sdf')
two_bins_directive_file = os.path.join(
    os.getcwd(), 'models', 'iiwa_suction_and_two_bins.yml')
two_ocado_bins_directive_file = os.path.join(
    os.getcwd(), 'models', 'iiwa_suction_and_two_ocado_bins.yml')

# commonly used trajectories
nq = 7
q_iiwa_bin0 = np.array([-np.pi / 2, 0.1, 0, -1.2, 0, 1.6, 0])
q_iiwa_bin1 = np.array([0, 0.1, 0, -1.2, 0, 1.6, 0])

q_traj_01 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    [0, 3], np.vstack([q_iiwa_bin0, q_iiwa_bin1]).T,
    np.zeros(nq), np.zeros(nq))


def get_q_traj_10():
    q_traj_10 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        [0, 3], np.vstack([q_iiwa_bin1, q_iiwa_bin0]).T,
        np.zeros(nq), np.zeros(nq))

    return q_traj_10


q_traj_1_hold = PiecewisePolynomial.ZeroOrderHold(
    [0, 2], np.vstack([q_iiwa_bin1, q_iiwa_bin1]).T)


def add_blueberry_boxes(n_objects: int, plant: MultibodyPlant,
                        parser: Parser, box_sdf: str):
    object_bodies = []
    for i in range(n_objects):
        model = parser.AddModelFromFile(box_sdf, f"box{i}")
        object_bodies.append(plant.GetBodyByName('base_link', model))
    return object_bodies


def add_blueberry_boxes_transfer(n_objects: int, plant: MultibodyPlant, parser: Parser):
    """
    The 0, 4, 8, ... boxes do not have corner spheres. They are the boxes to be
        squeezed in.
    """
    object_bodies = []
    assert n_objects % 4 == 0
    n_layers = n_objects // 4
    for i in range(n_layers):
        model = parser.AddModelFromFile(blueberry_small_soft_sdf_path, f"box{i * 4}")
        object_bodies.append(plant.GetBodyByName('base_link', model))

        for j in range(3):
            i_box = i * 4 + j + 1
            model = parser.AddModelFromFile(
                blueberry_small_4_bottom_spheres_sdf_path, f"box{i_box}")
            object_bodies.append(plant.GetBodyByName('base_link', model))

    return object_bodies


def add_blueberry_boxes_squeeze(plant: MultibodyPlant, parser: Parser):
    """
    Add 3 hydroelastic soft boxes and 1 rigid box.
    """
    soft_boxes = []

    # Add soft boxes.
    for i in range(3):
        model = parser.AddModelFromFile(blueberry_small_soft_sdf_path,
                                        f"box{i}")
        soft_boxes.append(plant.GetBodyByName('base_link', model))

    # Add rigid box.
    model = parser.AddModelFromFile(blueberry_small_rigid_sdf_path, "rigid_box")
    rigid_box = plant.GetBodyByName('base_link', model)

    return soft_boxes, rigid_box


class PackingMode(Enum):
    # Stacks randomly dropped boxes from one wide bin to the Ocado bin.
    kStack = 0
    # Picks up one box from the wide bin and squeeze it into the Ocado bin that
    # already has three other boxes.
    kSqueeze = 1
    # Transfers densely-packed boxes between Ocado bins.
    kTransfer = 2


def make_environment_model(
        rng, draw: bool, n_objects: int, draw_contact: bool,
        packing_mode: PackingMode):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.

    # Add objects.
    if packing_mode == PackingMode.kStack:
        ProcessModelDirectives(
            LoadModelDirectives(two_bins_directive_file), plant, parser)
        box_bodies = add_blueberry_boxes(n_objects, plant, parser, blueberry_sdf_path)
    elif packing_mode == PackingMode.kSqueeze:
        ProcessModelDirectives(
            LoadModelDirectives(two_bins_directive_file), plant, parser)
        soft_boxes, rigid_box = add_blueberry_boxes_squeeze(plant, parser)
        plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
        box_bodies = [rigid_box]
    elif packing_mode == PackingMode.kTransfer:
        ProcessModelDirectives(
            LoadModelDirectives(two_ocado_bins_directive_file), plant, parser)
        box_bodies = add_blueberry_boxes_transfer(n_objects, plant, parser)
    else:
        raise RuntimeError('unknown packing mode.')
    plant.Finalize()

    # Robot control.
    plant_iiwa_controller, model_iiwa = add_controlled_iiwa_and_trj_source(
        builder, plant, q0=q_iiwa_bin1, add_schunk_inertia=False)

    if draw:
        # draw body frames of the bins.
        bin_names = ["bin0", "bin1"]
        bin_instances = [plant.GetModelInstanceByName(name) for name in bin_names]
        frames_to_draw = [
            plant.GetBodyFrameIdOrThrow(
                plant.GetBodyByName("bin_base", bin_instance).index())
            for bin_instance in bin_instances]

        # draw boxes too.
        for body in box_bodies:
            frames_to_draw.append(plant.GetBodyFrameIdOrThrow(body.index()))

        frames_to_draw.append(
            plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("iiwa_link_7").index()))

        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment",
            frames_to_draw=frames_to_draw)

        DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)
        ConnectContactResultsToDrakeVisualizer(builder, plant)
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
        if packing_mode == PackingMode.kStack:
            set_object_drop_pose(
                context_env=context,
                plant=plant,
                bin_name="bin0",
                object_bodies=box_bodies,
                lime_bag_bodies=[],
                rng=rng,
                R_WB=RollPitchYaw(0, 0, np.pi / 2),
                x_lb=-0.15, x_ub=0.05, y_lb=-0.2, y_ub=0.2)
        elif packing_mode == PackingMode.kSqueeze:
            set_object_squeeze_pose(
                context_env=context,
                plant=plant,
                soft_bodies_list=soft_boxes,
                rigid_body=rigid_box,
                rng=rng)
        elif packing_mode == PackingMode.kTransfer:
            set_object_transfer_pose(
                context_env=context,
                plant=plant,
                bodies_list=box_bodies,
                rng=rng)

        simulator = Simulator(diagram, context)
        # viz.start_recording()
        simulator.set_target_realtime_rate(0.)
        simulator.AdvanceTo(5.0)
        # viz.stop_recording()
        # viz.publish_recording()

    elif draw:
        viz.load()
        diagram.Publish(context)

    return diagram, context, plant_iiwa_controller, simulator, box_bodies


class EnvSim:
    def __init__(self, n_objects: int, packing_mode: PackingMode):
        rng = np.random.default_rng(seed=1215234)

        (self.env, self.context, self.plant_iiwa_c, self.sim,
         self.box_bodies) = make_environment_model(
            rng=rng, draw=True, n_objects=n_objects,
            draw_contact=False, packing_mode=packing_mode)

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

    def get_box_pose(self, i_box: int):
        """
        i_box: index into self.box_bodies
        """
        return self.plant_env.EvalBodyPoseInWorld(
            self.context_plant_env, self.box_bodies[i_box])

    def get_box_poses(self):
        X_WB_list = []
        for i in range(len(self.box_bodies)):
            X_WB_list.append(self.get_box_pose(i))

        return X_WB_list

    def run_robot_traj(self, q_traj_robot: PiecewisePolynomial,
                       q_traj_suction: PiecewisePolynomial):
        assert abs(q_traj_robot.end_time() - q_traj_suction.end_time()) < 1e-3
        self.robot_traj_source.q_traj = q_traj_robot
        self.suction_traj_source.q_traj = q_traj_suction

        # set start time for robot traj soource and suction traj source.
        t_current = self.context.get_time()
        self.robot_traj_source.set_t_start(t_current)
        self.suction_traj_source.set_t_start(t_current)
        # simulate forward.
        self.sim.AdvanceTo(t_current + q_traj_robot.end_time())

    def get_bin_pose(self, bin_name):
        bin_instance = self.plant_env.GetModelInstanceByName(bin_name)
        bin_body = self.plant_env.GetBodyByName("bin_base", bin_instance)
        return self.plant_env.EvalBodyPoseInWorld(self.context_plant_env, bin_body)

    def lock_boxes(self, idx_boxes_to_lock: List[int]):
        """
        locks self.box_bodies[idx_boxes_to_lock].
        """
        for i in idx_boxes_to_lock:
            self.box_bodies[i].Lock(self.context_plant_env)

    def unlock_boxes(self, idx_boxes_to_unlock: List[int]):
        """
        unlocks self.box_bodies[idx_boxes_to_lock].
        """
        for i in idx_boxes_to_unlock:
            self.box_bodies[i].Unlock(self.context_plant_env)
