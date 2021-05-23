import os

import numpy as np

from pydrake.all import (DiagramBuilder, ConnectMeshcatVisualizer,
                         Simulator, Parser)
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.math import RigidTransform


#%%
station = ManipulationStation()
station.SetupClutterClearingStation()
parser = Parser(station.get_mutable_multibody_plant(),
                station.get_mutable_scene_graph())
model_names = ["Cucumber", "Mango", "Lime"]

models_object = {}
for name in model_names:
    models_object[name] = []
    for i in range(3):
        models_object[name].append(
            parser.AddModelFromFile(
                os.path.join("cad_files", name + '_simplified.sdf'),
                name + str(i)))

station.Finalize()

builder = DiagramBuilder()
builder.AddSystem(station)

ConnectMeshcatVisualizer(
    builder, scene_graph=station.get_scene_graph(),
    output_port=station.GetOutputPort("query_object"))

diagram = builder.Build()

#%%
context = diagram.CreateDefaultContext()
context_station = diagram.GetSubsystemContext(station, context)
context_plant = station.GetSubsystemContext(
    station.get_multibody_plant(), context_station)

#%%
plant = station.get_multibody_plant()
model_iiwa = plant.GetModelInstanceByName("iiwa")
model_gripper = plant.GetModelInstanceByName("gripper")
model_bin1 = plant.GetModelInstanceByName("bin1")
model_bin2 = plant.GetModelInstanceByName("bin2")


def get_default_joint_positions(model_idx):
    q_default = []
    for joint_idx in plant.GetJointIndices(model_idx):
        default_positions = plant.get_joint(joint_idx).default_positions()
        n = len(default_positions)
        if n > 0:
            assert n == 1
            q_default.append(default_positions[0])
    return q_default


q_iiwa0 = get_default_joint_positions(model_iiwa)

gripper_position_input_port = station.GetInputPort("wsg_position")
gripper_position_input_port.FixValue(context_station, [0.05])
iiwa_position_input_port = station.GetInputPort("iiwa_position")
iiwa_position_input_port.FixValue(context_station, q_iiwa0)

for name, model_list in models_object.items():
    n_objects = len(model_names)
    x = 0.4 + 0.2 * np.random.rand(n_objects)
    y = -0.2 + 0.2 * np.random.rand(n_objects)
    z = 0.4 + 0.6 * np.random.rand(n_objects)

    X_WObject_list = [
        RigidTransform(np.array([x[i], y[i], z[i]])) for i in range(n_objects)]
    for i, model in enumerate(model_list):
        plant.SetFreeBodyPose(context_plant,
                              plant.get_body(plant.GetBodyIndices(model)[0]),
                              X_WObject_list[i])

#%%
sim = Simulator(diagram, context)
sim.set_target_realtime_rate(1.0)
sim.AdvanceTo(2.0)
