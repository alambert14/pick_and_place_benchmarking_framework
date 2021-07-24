from typing import List
import os

import meshcat
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from pydrake.all import (LeafSystem, LinearBushingRollPitchYaw,
                         MultibodyPlant, DiagramBuilder, RigidBody,
                         AddMultibodyPlantSceneGraph, Parser, RigidTransform,
                         ConnectMeshcatVisualizer, Simulator, Context,
                         ExternallyAppliedSpatialForce, SpatialForce,
                         AbstractValue)

lime_sdf = os.path.join(os.getcwd(), 'cad_files', 'Lime_simplified.sdf')


class LimeBagVisualizer(LeafSystem):
    def __init__(self, lime_bags_bodies: List[List[RigidBody]],
                 vis: meshcat.Visualizer):
        """

        :param lime_bags_bodies:  a list of lists of lime bodies,
            each sublist consists of a bag.
        """
        super().__init__()
        lime_visual_mesh_path = os.path.join(
            os.path.dirname(__file__), 'cad_files',
            'Lime_processed_simple_vis.obj')
        lime_mesh = trimesh.load(lime_visual_mesh_path)
        self.lime_vertices = np.array(lime_mesh.vertices)
        self.lime_bags_bodies = lime_bags_bodies
        self.vis = vis
        self.body_pose_input_port = self.DeclareAbstractInputPort('body_pose',
                                      AbstractValue.Make([RigidTransform()]))

        self.DeclarePeriodicPublish(1 / 30)

    def DoPublish(self, context, event):
        super().DoPublish(context, event)
        X_WB_all = self.body_pose_input_port.Eval(context)

        nv_lime = len(self.lime_vertices)  # number of vertices per lime
        for i_bag, lime_bodies in enumerate(self.lime_bags_bodies):
            n_limes = len(lime_bodies)
            points = np.zeros((n_limes * nv_lime, 3))
            for i, lime_body in enumerate(lime_bodies):
                X_WB = X_WB_all[int(lime_body.index())]
                points[i * nv_lime: (i + 1) * nv_lime] = \
                    X_WB.multiply(self.lime_vertices.T).T

            hull = ConvexHull(points)
            self.vis['lime_bags/{}'.format(i_bag)].set_object(
                meshcat.geometry.TriangularMeshGeometry(points, hull.simplices),
                meshcat.geometry.MeshLambertMaterial(color=0xeedd22,
                                                     opacity=0.3))


def add_bag_of_lime(n_limes: int, bag_index: int, plant: MultibodyPlant,
                    parser: Parser):
    lime_models = [parser.AddModelFromFile(
        lime_sdf, 'lime_{}_{}'.format(bag_index, i))
                   for i in range(n_limes)]
    lime_bodies = [plant.GetBodyByName('base_link', lime_models[i])
                   for i in range(n_limes)]
    for i in range(n_limes):
        for j in range(i + 1, n_limes):
            plant.AddForceElement(
                LinearBushingRollPitchYaw(
                    frameA=lime_bodies[i].body_frame(),
                    frameC=lime_bodies[j].body_frame(),
                    torque_stiffness_constants=np.ones(3) * 1e-3,
                    torque_damping_constants=np.ones(3) * 2e-4,
                    force_stiffness_constants=np.ones(3) * 5,
                    force_damping_constants=np.ones(3) * 1))

    return lime_bodies


def initialize_bag_of_lime(
        l_spring: float,
        X_WL0: RigidTransform, plant: MultibodyPlant, context_plant: Context,
        lime_bodies):
    """

    :param l_spring: Initial length of the spring.
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
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=5e-4)
    parser = Parser(plant)

    n_limes = 5
    l_spring = 0.08
    lime_bodies = add_bag_of_lime(n_limes, 0, plant, parser)
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
