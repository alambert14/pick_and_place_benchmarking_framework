import os
import numpy as np

from pydrake.all import (LinearSpringDamper, MultibodyPlant, DiagramBuilder,
                         AddMultibodyPlantSceneGraph, Parser, RigidTransform,
                         ConnectMeshcatVisualizer, Simulator,
                         ExternallyAppliedSpatialForce, SpatialForce)

lime_sdf = os.path.join(os.getcwd(), 'cad_files', 'Lime_simplified.sdf')

#%%
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
parser = Parser(plant)

n_limes = 5
lime_models = [parser.AddModelFromFile(lime_sdf, 'lime{}'.format(i))
               for i in range(n_limes)]
lime_bodies = [plant.GetBodyByName('base_link', lime_models[i])
               for i in range(n_limes)]
l_spring = 0.1
for i in range(n_limes):
    for j in range(i + 1, n_limes):
        plant.AddForceElement(
            LinearSpringDamper(
                lime_bodies[i], np.zeros(3), lime_bodies[j], np.zeros(3),
                free_length=l_spring, stiffness=50, damping=10))

plant.Finalize()

viz = ConnectMeshcatVisualizer(builder, scene_graph)
diagram = builder.Build()

# set lime initial position.
context = diagram.CreateDefaultContext()
context_plant = plant.GetMyContextFromRoot(context)
p_WB0 = np.array([[0, 0, 0],
                  [l_spring, 0, 0],
                  [0, l_spring, 0],
                  [l_spring, l_spring, 0]])
for i in range(n_limes):
    i2 = i % 4
    z_i = (i // 4) * l_spring
    X_WBi = RigidTransform()
    X_WBi.set_translation(p_WB0[i2] + np.array([0, 0, z_i]))
    plant.SetFreeBodyPose(context_plant, lime_bodies[i], X_WBi)


# apply force to lime 0
F_L0q_W = SpatialForce(np.zeros(3), np.array([0, 0, n_limes * 0.046 * 9.8]))
eaf = ExternallyAppliedSpatialForce()
eaf.F_Bq_W = F_L0q_W
eaf.body_index = lime_bodies[0].index()
plant.get_applied_spatial_force_input_port().FixValue(context_plant, [eaf])

# publish initial configuration.
viz.load()
diagram.Publish(context)

#%%
sim = Simulator(diagram, context)
viz.vis.delete()
viz.reset_recording()
viz.start_recording()
sim.AdvanceTo(5.0)
viz.stop_recording()
viz.publish_recording()
