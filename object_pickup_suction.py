import os
import numpy as np

from pydrake.all import (
    PiecewisePolynomial, PidController, MultibodyPlant, Context,
    RandomGenerator, Simulator, InverseDynamicsController, ContactModel,
    ConnectContactResultsToDrakeVisualizer, DrakeVisualizer, Parser,
    DiagramBuilder, AddMultibodyPlantSceneGraph, ProcessModelDirectives,
    LoadModelDirectives, ConnectMeshcatVisualizer,
    MeshcatContactVisualizer)

from manipulation.utils import AddPackagePaths
from grasp_sampler_vision import zmq_url

from utils import (SimpleTrajectorySource, concatenate_traj_list,
                   add_package_paths_local)
from build_sim_diagram import (add_controlled_iiwa_and_trj_source,
    set_object_drop_pose)

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
        directive, rng, draw=False, n_objects=0, bin_name="bin0"):

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)

    # Add objects.
    object_bodies = add_blueberry_boxes(n_objects, plant, parser)

    plant.Finalize()

    if draw:
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment")
        viz_c = MeshcatContactVisualizer(viz, plant=plant)
        builder.AddSystem(viz_c)
        builder.Connect(plant.get_contact_results_output_port(),
                        viz_c.GetInputPort('contact_results'))

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    simulator = None
    if n_objects > 0:
        set_object_drop_pose(
            context_environment=context,
            plant=plant,
            bin_name=bin_name,
            object_bodies=object_bodies,
            lime_bag_bodies=[],
            rng=rng)

        simulator = Simulator(diagram, context)

        simulator.AdvanceTo(5.0)
        simulator.set_target_realtime_rate(0.)

    elif draw:
        viz.load()
        diagram.Publish(context)

    return diagram, context, simulator

#%%
directive_file = os.path.join(
    os.getcwd(), 'models', 'two_bins.yaml')

rng = np.random.default_rng(seed=1215232)

make_environment_model(
    directive=directive_file, rng=rng, draw=True, n_objects=5)
