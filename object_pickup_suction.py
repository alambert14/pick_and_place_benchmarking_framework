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
from grasp_sampler_suction import calc_suction_ee_pose



#%%
object_sdf_path = os.path.join(os.path.dirname(__file__), 'models',
                               'blue_berry_box.sdf')

q_iiwa_bin0 = np.array([-np.pi / 2, 0.1, 0, -1.2, 0, 1.6, 0])
q_iiwa_bin1 = np.array([0, 0.1, 0, -1.2, 0, 1.6, 0])


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
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)

    # Add objects.
    box_bodies = add_blueberry_boxes(n_objects, plant, parser)
    plant.Finalize()

    # Robot control.
    plant_iiwa_controller, model_iiwa = add_controlled_iiwa_and_trj_source(
        builder, plant, q0=q_iiwa_bin1)

    if draw:
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment")
        if draw_contact:
            viz_c = MeshcatContactVisualizer(viz, plant=plant)
            builder.AddSystem(viz_c)
            builder.Connect(plant.get_contact_results_output_port(),
                            viz_c.GetInputPort('contact_results'))

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


#%%
directive_file = os.path.join(
    os.getcwd(), 'models', 'iiwa_suction_and_two_bins.yml')

rng = np.random.default_rng(seed=1215232)

# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

env, context_env, sim, plant_iiwa_controller, box_bodies = make_environment_model(
    directive=directive_file, rng=rng, draw=True, n_objects=7)


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

robot_traj_source = env.GetSubsystemByName('robot_traj_source')
plant_env = env.GetSubsystemByName('plant')



