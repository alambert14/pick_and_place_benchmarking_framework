import os
import numpy as np

from pydrake.all import (LinearSpringDamper, LinearBushingRollPitchYaw,
                         MultibodyPlant, DiagramBuilder,
                         AddMultibodyPlantSceneGraph, Parser, RigidTransform,
                         ConnectMeshcatVisualizer, Simulator, Context,
                         ExternallyAppliedSpatialForce, SpatialForce)

lime_sdf = os.path.join(os.getcwd(), 'cad_files', 'Lime_simplified.sdf')


def add_bag_of_lime(n_limes: int, l_spring: float, bag_index: int,
                    plant: MultibodyPlant, parser: Parser):
    lime_models = [parser.AddModelFromFile(
        lime_sdf, 'lime_{}_{}'.format(bag_index, i))
                   for i in range(n_limes)]
    lime_bodies = [plant.GetBodyByName('base_link', lime_models[i])
                   for i in range(n_limes)]
    for i in range(n_limes):
        for j in range(i + 1, n_limes):
            # print(i, j)
            p_Bq = np.array([0, 0, 0.0])
            plant.AddForceElement(
                LinearSpringDamper(
                    lime_bodies[i], p_Bq, lime_bodies[j], p_Bq,
                    free_length=l_spring, stiffness=100, damping=20))

            # plant.AddForceElement(
            #     LinearBushingRollPitchYaw(
            #         frameA=lime_bodies[i].body_frame(),
            #         frameC=lime_bodies[j].body_frame(),
            #         torque_stiffness_constants=np.array([5, 5, 5.]),
            #         torque_damping_constants=np.array([0.1, 0.1, 0.1]),
            #         force_stiffness_constants=np.ones(3) * 10,
            #         force_damping_constants=np.ones(3) * 1))

    return lime_bodies


def initialize_bag_of_lime(
        l_spring: float,
        X_WL0: RigidTransform, plant: MultibodyPlant, context_plant: Context,
        lime_bodies):
    """

    :param X_WL0: pose of the first lime in the bag.
    :param plant:
    :param context_plant:
    :param lime_bodies:
    :return:
    """
    p_WB0 = np.array([[0, 0, 0],
                      [l_spring, 0, 0],
                      [0, l_spring, 0],
                      [l_spring, l_spring, 0]])
    p_WB0 += X_WL0.translation()
    n_limes = len(lime_bodies)
    for i in range(n_limes):
        i2 = i % 4
        z_i = (i // 4) * l_spring
        X_WBi = RigidTransform()
        X_WBi.set_translation(p_WB0[i2] + np.array([0, 0, z_i]))
        plant.SetFreeBodyPose(context_plant, lime_bodies[i], X_WBi)

#%%
if __name__ == '__main__':
    #%%
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = Parser(plant)

    n_limes = 5
    l_spring = 0.08
    lime_bodies = add_bag_of_lime(n_limes, l_spring, 0, plant, parser)
    plant.Finalize()

    viz = ConnectMeshcatVisualizer(builder, scene_graph)
    diagram = builder.Build()

    # set lime initial position.
    context = diagram.CreateDefaultContext()
    context_plant = plant.GetMyContextFromRoot(context)
    initialize_bag_of_lime(l_spring=l_spring, X_WL0=RigidTransform(),
                           plant=plant, context_plant=context_plant,
                           lime_bodies=lime_bodies)

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
