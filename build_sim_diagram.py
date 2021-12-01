from typing import List, Dict
import os.path

import numpy as np
from pydrake.all import (
    PiecewisePolynomial, MultibodyPlant, Context, RandomGenerator, Body,
    InverseDynamicsController)
from pydrake.math import RollPitchYaw

from robotics_utilities.iiwa_controller.utils import (
    create_iiwa_controller_plant)

from utils import SimpleTrajectorySource
from grasp_sampler_vision import *
from lime_bag import add_bag_of_lime, initialize_bag_of_lime

#%%
# object SDFs.
object_names = ['Lime', 'Cucumber', 'Mango']
#object_names = ['Cucumber']
sdf_dir = os.path.join(os.path.dirname(__file__), 'cad_files')
object_sdfs = {name: os.path.join(sdf_dir, name + '_simplified.sdf')
               for name in object_names}

q_iiwa_bin0 = np.array([-np.pi / 2, 0.1, 0, -1.2, 0, 1.6, 0])
q_iiwa_bin1 = np.array([0, 0.1, 0, -1.2, 0, 1.6, 0])


def add_objects(n_objects: int, obj_names: List[str], obj_sdfs: Dict[str, str],
                plant: MultibodyPlant, parser: Parser, rng):
    object_bodies = []
    lime_bag_bodies = []
    n_bags_of_lime = 0
    produce_amounts = rng.integers(low=0, high=n_objects, size=len(object_names))

    print('produce amts:', produce_amounts)
    for amt, obj_name in zip(produce_amounts, object_names):
        for i in range(amt):
            model = parser.AddModelFromFile(obj_sdfs[obj_name], f"object_{obj_name}_{i}")
            object_bodies.append(plant.GetBodyByName('base_link', model))
            print(f'Adding {obj_name}')

    return object_bodies, lime_bag_bodies


def add_controlled_iiwa_and_trj_source(
        builder: DiagramBuilder, plant: MultibodyPlant, q0: np.ndarray,
        add_schunk_inertia=True):
    # TODO: make inertia different for different end effectors.
    plant_iiwa_controller, _ = create_iiwa_controller_plant(
        gravity=plant.gravity_field().gravity_vector(),
        add_schunk_inertia=add_schunk_inertia)

    Kp_iiwa = np.array([800, 600, 600, 400, 400, 400, 200])
    Kd_iiwa = 2 * np.sqrt(Kp_iiwa)
    Ki_iiwa = np.zeros(7)
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
    q_knots[0] = q0
    robot_traj_source = SimpleTrajectorySource(
        PiecewisePolynomial.ZeroOrderHold(
            [0, 1], q_knots.T))
    builder.AddSystem(robot_traj_source)
    builder.Connect(robot_traj_source.x_output_port,
                    idc.get_input_port_desired_state())
    robot_traj_source.set_name('robot_traj_source')

    return plant_iiwa_controller, model_iiwa


def set_object_drop_pose(context_env: Context,
                         plant: MultibodyPlant,
                         bin_name: str,
                         object_bodies: List,
                         lime_bag_bodies: List,
                         rng,
                         R_WB=RotationMatrix(),
                         x_lb=-0.15, x_ub=0.15, y_lb=-0.2, y_ub=0.2):
    generator = RandomGenerator(rng.integers(1000))  # this is for c++
    plant_context = plant.GetMyContextFromRoot(context_env)
    bin_instance = plant.GetModelInstanceByName(bin_name)
    bin_body = plant.GetBodyByName("bin_base", bin_instance)
    X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
    z = 0.2
    l_spring = 0.08  # for lime bags.
    for object_body in (lime_bag_bodies + object_bodies):
        tf = RigidTransform(R_WB, [rng.uniform(x_lb, x_ub), rng.uniform(y_lb, y_ub), z])

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
        z += 0.08


def set_object_squeeze_pose(context_env: Context,
                            plant: MultibodyPlant,
                            soft_bodies_list: List[Body],
                            rigid_body: Body,
                            rng):
    generator = RandomGenerator(rng.integers(1000))  # this is for c++
    plant_context = plant.GetMyContextFromRoot(context_env)
    bin0 = plant.GetModelInstanceByName("bin0")
    bin1 = plant.GetModelInstanceByName("bin1")
    bin0_body = plant.GetBodyByName("bin_base", bin0)
    bin1_body = plant.GetBodyByName("bin_base", bin1)
    X_B0 = plant.EvalBodyPoseInWorld(plant_context, bin0_body)
    X_B1 = plant.EvalBodyPoseInWorld(plant_context, bin1_body)

    p_WB0 = X_B1.translation()
    d = 0.105

    # set soft box positions.
    y_positions = [-0.2 + 0.049, -0.2 + d / 2 * 3, 0.2 - d / 2]
    for i, body in enumerate(soft_bodies_list):
        X_WB = RigidTransform()
        X_WB.set_translation(
            [p_WB0[0], y_positions[i] + p_WB0[1], p_WB0[2] + 0.05])
        plant.SetFreeBodyPose(plant_context, body, X_WB)

    # set rigid box position.
    X_WB = RigidTransform()
    X_WB.set_translation(
        X_B0.translation() + np.array([0, 0, 0.05]))
    plant.SetFreeBodyPose(plant_context, rigid_body, X_WB)


def set_object_transfer_pose(context_env: Context,
                             plant: MultibodyPlant,
                             bodies_list: List[Body],
                             rng):
    generator = RandomGenerator(rng.integers(1000))  # this is for c++
    plant_context = plant.GetMyContextFromRoot(context_env)
    bin0 = plant.GetModelInstanceByName("bin0")
    bin0_body = plant.GetBodyByName("bin_base", bin0)
    X_B0 = plant.EvalBodyPoseInWorld(plant_context, bin0_body)

    p_WB0 = X_B0.translation()

    # set soft box positions.
    d = 0.1
    x_positions = [-0.2 + d / 2 + d * i for i in range(4)]
    n_boxes = len(bodies_list)
    assert n_boxes % 4 == 0
    n_layers = n_boxes // 4
    z = p_WB0[2] + 0.05
    for i in range(n_layers):
        for j in range(4):
            i_box = i * 4 + j
            body = bodies_list[i_box]
            X_WB = RigidTransform()
            X_WB.set_rotation(
                RollPitchYaw(0, 0, -np.pi/2).ToRotationMatrix())
            X_WB.set_translation([x_positions[j] + p_WB0[0], p_WB0[1], z])
            plant.SetFreeBodyPose(plant_context, body, X_WB)
        z += 0.045  # height of the blueberry boxes.
