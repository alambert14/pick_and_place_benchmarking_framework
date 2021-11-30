from typing import List, Dict
import os.path

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

from utils import (SimpleTrajectorySource, concatenate_traj_list,
                   add_package_paths_local, render_system_with_graphviz)
from grasp_sampler_vision import GraspSamplerVision, zmq_url
from inverse_kinematics import calc_joint_trajectory
from build_sim_diagram import (add_controlled_iiwa_and_trj_source, add_objects,
    set_object_drop_pose)

#%%
# object SDFs.
# object_names = ['Lime', 'Cucumber', 'Mango']
object_names = ['Mango', 'Cucumber', 'Lime']
sdf_dir = os.path.join(os.path.dirname(__file__), 'cad_files')
object_sdfs = {name: os.path.join(sdf_dir, name + '_simplified.sdf')
               for name in object_names}

training_dir = os.path.join(os.path.dirname(__file__), 'training_data')


def get_masks(point_cloud):



def make_environment_model(
        directive, rng=None, draw=False, n_objects=0, bin_name="bin0"):
    """
    Make one model of the environment, but the robot only gets to see the sensor
     outputs.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=2e-4)
    # parser = Parser(plant)
    # AddPackagePaths(parser)  # Russ's manipulation repo.
    # add_package_paths_local(parser)  # local.
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)
    inspector = scene_graph.model_inspector()
    # Add objects.
    object_bodies, lime_bag_bodies = add_objects(
        n_objects=n_objects,
        obj_names=object_names,
        obj_sdfs=object_sdfs,
        plant=plant,
        parser=parser,
        inspector=inspector,
        rng=rng)

    # Set contact model.
    plant.set_contact_model(ContactModel.kPointContactOnly)
    plant.Finalize()
    AddRgbdSensors(builder, plant, scene_graph)

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
    plant.SetPositions(context_plant, model_iiwa, q_iiwa_bin1)
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

    
    return diagram, context, simulator, inspector


#%%
directive_file = os.path.join(
    os.getcwd(), 'models', 'bin_and_cameras.yml')

rng = None# np.random.default_rng(seed=1215232)
# seed 12153432 looks kind of nice.

# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

# build environment
env, context_env, sim, inspector = make_environment_model(
    directive=directive_file, rng=rng, draw=True, n_objects=5)
grasp_sampler = GraspSamplerVision(env)

viz = env.GetSubsystemByName('meshcat_visualizer')

plant_env = env.GetSubsystemByName('plant')

viz.reset_recording()
viz.start_recording()

cameras = [env.GetSubsystemByName(f'camera{i}') for i in range(3)]
cam_contexts = [cam.GetMyMutableContextFromRoot(context_env) for cam in cameras]

while True:
    for cam, cam_context in zip(cameras, cam_contexts):
        rgb_image = cam.GetOutputPort('color_image').Eval(cam_context).data
        label_image = cam.GetOutputPort('label_image').Eval(cam_context).data

        print(label_image)

        img = rgb_image[:,:,:-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        file = 'test.png'
        cv2.imwrite(file, img)
    
    sim.AdvanceTo(t_current + q_traj.end_time())

viz.stop_recording()
viz.publish_recording()

