from typing import List, Dict
import os.path

import numpy as np
from pydrake.all import (
    PiecewisePolynomial, MultibodyPlant, Context, RandomGenerator,
    InverseDynamicsController)

from robotics_utilities.iiwa_controller.utils import (
    create_iiwa_controller_plant)

from utils import SimpleTrajectorySource
from grasp_sampler_vision import *
from lime_bag import add_bag_of_lime, initialize_bag_of_lime

#%%
# object SDFs.
# object_names = ['Lime', 'Cucumber', 'Mango']
object_names = ['Cucumber']
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
    for i in range(n_objects):
        i_obj = rng.integers(len(object_sdfs))
        obj_name = obj_names[i_obj]
        if obj_name == 'Lime':
            lime_bodies = add_bag_of_lime(n_limes=5, bag_index=n_bags_of_lime,
                                          plant=plant, parser=parser)
            lime_bag_bodies.append(lime_bodies)
            n_bags_of_lime += 1
        else:
            model = parser.AddModelFromFile(obj_sdfs[obj_name], f"object{i}")
            object_bodies.append(plant.GetBodyByName('base_link', model))

    return object_bodies, lime_bag_bodies


def add_controlled_iiwa_and_trj_source(
        builder: DiagramBuilder, plant: MultibodyPlant, q0: np.ndarray):
    """

    :param builder:
    :param plant:
    :param q0:
    :return:
    """
    # TODO: make inertia different for different end effectors.
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
    q_knots[0] = q0
    robot_traj_source = SimpleTrajectorySource(
        PiecewisePolynomial.ZeroOrderHold(
            [0, 1], q_knots.T))
    builder.AddSystem(robot_traj_source)
    builder.Connect(robot_traj_source.x_output_port,
                    idc.get_input_port_desired_state())
    robot_traj_source.set_name('robot_traj_source')

    return plant_iiwa_controller, model_iiwa


def set_object_drop_pose(context_environment: Context,
                         plant: MultibodyPlant,
                         bin_name: str,
                         object_bodies: List,
                         lime_bag_bodies: List,
                         rng):
    generator = RandomGenerator(rng.integers(1000))  # this is for c++
    plant_context = plant.GetMyContextFromRoot(context_environment)
    bin_instance = plant.GetModelInstanceByName(bin_name)
    bin_body = plant.GetBodyByName("bin_base", bin_instance)
    X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
    z = 0.3
    l_spring = 0.08  # for lime bags.
    for object_body in (lime_bag_bodies + object_bodies):
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
        z += 0.08


