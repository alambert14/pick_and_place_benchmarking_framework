import os
import sys
from functools import partial

import numpy as np
import open3d as o3d
import meshcat
from tqdm import tqdm
from IPython.display import display
from ipywidgets import Textarea

from pydrake.all import (
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer,
    DiagramBuilder, RigidTransform, RotationMatrix,
    CoulombFriction, FindResourceOrThrow, Diagram,
    PiecewisePolynomial, PiecewiseQuaternionSlerp, Parser, PidController,
    RandomGenerator, Simulator, ProcessModelDirectives, LoadModelDirectives,
    MatrixGain,
)
from pydrake.math import RollPitchYaw

from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import FindResource, AddPackagePaths
from manipulation.meshcat_utils import draw_open3d_point_cloud, draw_points
from manipulation.open3d_utils import create_open3d_point_cloud

from utils import render_system_with_graphviz
from gripper_pose_controller import (GripperPoseController,
                                     CustomTrajectorySource)
from lime_bag import add_bag_of_lime, initialize_bag_of_lime

#%%
zmq_url = "tcp://127.0.0.1:6000"

# object SDFs.
object_names = ['Lime', 'Cucumber', 'Mango']
# object_names = ['Lime']
object_sdfs = [os.path.join(os.getcwd(), 'cad_files', name + '_simplified.sdf')
               for name in object_names]


#%%
# scoring grasp candidiates
def grasp_candidate_cost(plant_context, cloud, plant, scene_graph,
                         scene_graph_context, adjust_X_G=False, textbox=None,
                         meshcat=None):
    body = plant.GetBodyByName("body")
    X_G = plant.GetFreeBodyPose(plant_context, body)

    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    pts = np.asarray(cloud.points).T
    p_GC = X_GW.multiply(pts)

    # Crop to a region inside of the finger box.
    crop_min = [-.05, 0.1, -0.00625]
    crop_max = [.05, 0.1125, 0.00625]
    indices = np.all((crop_min[0] <= p_GC[0, :], p_GC[0, :] <= crop_max[0],
                      crop_min[1] <= p_GC[1, :], p_GC[1, :] <= crop_max[1],
                      crop_min[2] <= p_GC[2, :], p_GC[2, :] <= crop_max[2]),
                     axis=0)

    if meshcat:
        draw_points(meshcat["points"], pts[:, indices], [1., 0, 0], size=0.01)

    if adjust_X_G and np.sum(indices) > 0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        X_G.set_translation(
            X_G.translation() + X_G.rotation().multiply([p_Gcenter_x, 0, 0]))
        plant.SetFreeBodyPose(plant_context, body, X_G)
        X_GW = X_G.inverse()

    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    # Check collisions between the gripper and the sink
    if query_object.HasCollisions():
        cost = np.inf
        if textbox:
            textbox.value = "Gripper is colliding with the sink!\n"
            textbox.value += f"cost: {cost}"
        return cost

    # Check collisions between the gripper and the point cloud
    margin = 0.0  # must be smaller than the margin used in the point cloud preprocessing.
    for pt in cloud.points:
        distances = query_object.ComputeSignedDistanceToPoint(pt,
                                                              threshold=margin)
        if distances:
            cost = np.inf
            if textbox:
                textbox.value = "Gripper is colliding with the point cloud!\n"
                textbox.value += f"cost: {cost}"
            return cost

    n_GC = X_GW.rotation().multiply(np.asarray(cloud.normals)[indices, :].T)

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0 * X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= np.sum(n_GC[0, :] ** 2)

    if textbox:
        textbox.value = f"cost: {cost}\n"
        textbox.value += "normal terms:" + str(n_GC[0, :] ** 2)
    return cost


def process_point_cloud(diagram, context, cameras, bin_name):
    """A "no frills" version of the example above, that returns the down-sampled point cloud"""
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # Compute crop box.
    bin_instance = plant.GetModelInstanceByName(bin_name)
    bin_body = plant.GetBodyByName("bin_base", bin_instance)
    X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
    margin = 0.001  # only because simulation is perfect!
    a = X_B.multiply(
        [-.22 + 0.025 + margin, -.29 + 0.025 + margin, 0.015 + margin])
    b = X_B.multiply([.22 - 0.1 - margin, .29 - 0.025 - margin, 2.0])
    crop_min = np.minimum(a, b)
    crop_max = np.maximum(a, b)

    # Evaluate the camera output ports to get the images.
    merged_pcd = o3d.geometry.PointCloud()
    for c in cameras:
        point_cloud = diagram.GetOutputPort(f"{c}_point_cloud").Eval(context)
        pcd = create_open3d_point_cloud(point_cloud)

        # Crop to region of interest.
        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,
                                                max_bound=crop_max))

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

        camera = plant.GetModelInstanceByName(c)
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd.orient_normals_towards_camera_location(X_C.translation())

        # Merge point clouds.
        merged_pcd += pcd

    # Voxelize down-sample.  (Note that the normals still look reasonable)
    return merged_pcd.voxel_down_sample(voxel_size=0.005)


def make_environment_model(
        directive=None, draw=False, rng=None, num_objects=0, bin_name="bin0",
        add_gripper_control=False):
    """
    Make one model of the environment, but the robot only gets to see the sensor
     outputs.
    """
    if not directive:
        directive = FindResource("models/two_bins_w_cameras.yaml")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    AddPackagePaths(parser)
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)

    object_bodies = []
    n_bags_of_lime = 0
    for i in range(num_objects):
        i_obj = rng.integers(len(object_sdfs))
        if object_names[i_obj] == 'Lime':
            lime_bodies = add_bag_of_lime(n_limes=5, bag_index=n_bags_of_lime,
                                          plant=plant, parser=parser)
            object_bodies.append(lime_bodies)
            n_bags_of_lime += 1
        else:
            model = parser.AddModelFromFile(object_sdfs[i_obj], f"object{i}")
            object_bodies.append(plant.GetBodyByName('base_link', model))

    plant.Finalize()
    AddRgbdSensors(builder, plant, scene_graph)

    if add_gripper_control:
        # Gripper Pose Control
        schunk_body = plant.GetBodyByName('body')
        gpc = GripperPoseController(gripper_body_idx=schunk_body.index())
        builder.AddSystem(gpc)
        builder.Connect(
            plant.get_body_spatial_velocities_output_port(),
            gpc.body_spatial_velocity_input_port)
        builder.Connect(
            plant.get_body_poses_output_port(),
            gpc.body_pose_input_port)
        builder.Connect(
            gpc.spatial_force_output_port,
            plant.get_applied_spatial_force_input_port())

        # Gripper Finger Control
        schunk_model = plant.GetModelInstanceByName('gripper')
        K_schunk = np.array([500, 500.])
        D_schunk = np.array([10, 10.])
        gfc = PidController(K_schunk, np.zeros(2), D_schunk)
        gfc.set_name('gripper_finger_controller')

        S = np.zeros((4, 17))
        S[0, 7] = 1
        S[1, 8] = 1
        S[2, -2] = 1
        S[3, -1] = 1
        finger_state_selector = MatrixGain(S)
        finger_state_selector.set_name('finger_state_selector')
        builder.AddSystem(gfc)
        builder.AddSystem(finger_state_selector)
        builder.Connect(
            gfc.get_output_port_control(),
            plant.get_actuation_input_port(schunk_model))
        builder.Connect(
            plant.get_state_output_port(schunk_model),
            finger_state_selector.get_input_port())
        builder.Connect(
            finger_state_selector.get_output_port(),
            gfc.get_input_port_estimated_state())

        # trajectory source for gripper.
        finger_setpoints = PiecewisePolynomial.ZeroOrderHold(
            [0, 1], np.array([[0.1], [0.1]]).T)

        p_WB_traj = PiecewisePolynomial.ZeroOrderHold(
            [0, 1], np.array([[0, 0, 0], [0, 0, 0.]]).T)

        Q_WB = RollPitchYaw(np.pi / 2, 0, 0).ToQuaternion()
        Q_WB_traj = PiecewiseQuaternionSlerp([0, 1], [Q_WB, Q_WB])

        traj_source = CustomTrajectorySource(
            p_WB_traj=p_WB_traj, Q_WB_traj=Q_WB_traj,
            finger_setpoint_traj=finger_setpoints)

        builder.AddSystem(traj_source)
        builder.Connect(
            traj_source.body_pose_output_port,
            gpc.pose_ref_input_port)

        builder.Connect(
            traj_source.finger_setpoint_output_port,
            gfc.get_input_port_desired_state())

    if draw:
        viz = ConnectMeshcatVisualizer(
            builder, scene_graph, zmq_url=zmq_url, prefix="environment")

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    if num_objects > 0:
        generator = RandomGenerator(rng.integers(1000))  # this is for c++
        plant_context = plant.GetMyContextFromRoot(context)
        bin_instance = plant.GetModelInstanceByName(bin_name)
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
        z = 0.3
        l_spring = 0.08  # for lime bags.
        for object_body in object_bodies:
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
            z += 0.05

        simulator = Simulator(diagram, context)
        if draw:
            viz.start_recording()

        # viz.start_recording()
        simulator.AdvanceTo(5.0)
        simulator.set_target_realtime_rate(0.)
        # viz.stop_recording()
        # viz.publish_recording()

        if draw:
            viz.stop_recording()
            viz.publish_recording(play=False)
    elif draw:
        viz.load()
        diagram.Publish(context)

    return diagram, context


def generate_grasp_candidate_antipodal(plant_context, cloud, plant, scene_graph,
                                       scene_graph_context, rng, meshcat_vis=None):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    """
    n_tries = 0
    n_tries_ub = 100
    while n_tries < n_tries_ub:
        index = rng.integers(0, len(cloud.points) - 1)
        p_WS = np.asarray(cloud.points[index])
        n_WS = np.asarray(cloud.normals[index])
        n_WS_norm = np.linalg.norm(n_WS)
        if np.isclose(n_WS_norm, 1.0, atol=1e-2):
            n_WS /= n_WS_norm
            break
        n_tries += 1
        if n_tries == n_tries_ub:
            raise RuntimeError("cannot find a point with a good normal.")

    body = plant.GetBodyByName("body")

    if meshcat_vis:
        vertices = np.empty((3, 2))
        vertices[:, 0] = p_WS
        vertices[:, 1] = p_WS + 0.05 * n_WS
        meshcat_vis.set_object(
            meshcat.geometry.LineSegments(
                meshcat.geometry.PointsGeometry(vertices),
                meshcat.geometry.MeshBasicMaterial(color=0xff0000)))


    Gx = n_WS  # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y, Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        return None

    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0]

    # Try orientations from the center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    for theta in (min_roll + (max_roll - min_roll) * alpha):
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = - R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, body, X_G)
        cost = grasp_candidate_cost(plant_context, cloud, plant, scene_graph,
                                    scene_graph_context, adjust_X_G=True,
                                    meshcat=meshcat_vis)
        X_G = plant.GetFreeBodyPose(plant_context, body)
        if np.isfinite(cost):
            return cost, X_G

        #draw_grasp_candidate(X_G, f"collision/{theta:.1f}")

    return np.inf, None


def draw_grasp_candidate(X_G, prefix='gripper', draw_frames=True):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.package_map().Add("wsg_50_description", os.path.dirname(
        FindResourceOrThrow(
            "drake/manipulation/models/wsg_50_description/package.xml")))
    gripper = parser.AddModelFromFile(FindResource(
        "models/schunk_wsg_50_welded_fingers.sdf"), "gripper")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
    plant.Finalize()

    frames_to_draw = {"gripper": {"body"}} if draw_frames else {}
    meshcat = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url,
                                       prefix=prefix,
                                       delete_prefix_on_load=False,
                                       frames_to_draw=frames_to_draw)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    meshcat.load()
    diagram.Publish(context)


class GraspSampler:
    def __init__(self, environment: Diagram):
        self.rng = np.random.default_rng(seed=10001)
        self.env = environment

        # Another diagram for the objects the robot "knows about":
        # gripper, cameras, bins.
        # Think of this as the model in the robot's head.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.001)
        parser = Parser(plant)
        AddPackagePaths(parser)
        ProcessModelDirectives(
            LoadModelDirectives(FindResource("models/clutter_planning.yaml")),
            plant, parser)
        plant.Finalize()
        self.plant = plant
        self.sg = scene_graph

        viz = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url,
                                       prefix="planning")
        viz.load()
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.Publish(context)
        # Hide this particular gripper
        viz.vis["planning/plant/gripper"].set_property('visible', False)

        self.diagram = diagram
        self.viz = viz

    def sample_grasp_candidates(self, context_env, draw_grasp_candidates=True):
        cloud = process_point_cloud(self.env, context_env,
                                    ["camera0", "camera1", "camera2"], "bin0")

        draw_open3d_point_cloud(self.viz.vis['cloud'], cloud, size=0.003)

        context = self.diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyContextFromRoot(context)
        scene_graph_context = self.sg.GetMyContextFromRoot(context)
        costs = []
        X_Gs = []

        for i in tqdm(range(100)):
            cost, X_G = generate_grasp_candidate_antipodal(
                plant_context, cloud,
                self.plant, self.sg,
                scene_graph_context,
                self.rng)
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        indices = np.asarray(costs).argsort()[:5]
        X_Gs_best = []
        for i in indices:
            X_Gs_best.append(X_Gs[i])
            if draw_grasp_candidates:
                draw_grasp_candidate(X_Gs[i], prefix=f"{i}th best",
                                     draw_frames=False)

        return X_Gs_best


#%%
directive_file = os.path.join(
    os.getcwd(), 'models', 'two_bins_and_actuated_shcunk.yml')

rng = np.random.default_rng(seed=10001)

# clean up visualization.
v = meshcat.Visualizer(zmq_url=zmq_url)
v.delete()

# build environment and grasp sampler.
env, context_env = make_environment_model(
    directive=directive_file, rng=rng, draw=True, num_objects=7,
    add_gripper_control=True)
grasp_sampler = GraspSampler(env)

# sample some grasps.
X_Gs_best = grasp_sampler.sample_grasp_candidates(
    context_env, draw_grasp_candidates=True)

#%%
# start gripper from 0.3m above the grasp, move in, grasp and pick up.
n_knots = 4
t_knots = [0, 2.5, 3.5, 7]
finger_setpoint_knots = np.array([[0.1, 0.01, 0.01, 0.01]])
p_WB_knots = np.zeros((4, 3))
X_WB = X_Gs_best[0]
p_WB_knots[0] = X_WB.translation() + np.array([0, 0, 0.3])
p_WB_knots[1] = X_WB.translation()
p_WB_knots[2] = X_WB.translation()
p_WB_knots[3] = p_WB_knots[0]

Q_WB = X_WB.rotation().ToQuaternion()
finger_setpoint_traj = PiecewisePolynomial.ZeroOrderHold(
    t_knots, finger_setpoint_knots)
p_WB_traj = PiecewisePolynomial.FirstOrderHold(t_knots, p_WB_knots.T)
Q_WB_traj = PiecewiseQuaternionSlerp(t_knots, [Q_WB] * n_knots)

traj_source = env.GetSubsystemByName('custom_trajectory_source')
traj_source.Q_WB_traj = Q_WB_traj
traj_source.finger_setpoint_traj = finger_setpoint_traj
traj_source.p_WB_traj = p_WB_traj

context_new = env.CreateDefaultContext()
plant_env = env.GetSubsystemByName('plant')
context_plant_old = plant_env.GetMyContextFromRoot(context_env)
context_plant_new = plant_env.GetMyContextFromRoot(context_new)
plant_env.SetPositionsAndVelocities(
    context_plant_new,
    plant_env.GetPositionsAndVelocities(context_plant_old))

# gripper
schunk_model = plant_env.GetModelInstanceByName('gripper')
schunk_body = plant_env.GetBodyByName('body')
schunk_position = np.zeros(9)
schunk_position[:4] = Q_WB.wxyz()
schunk_position[4:7] = p_WB_knots[0]
schunk_position[7:9] = [-finger_setpoint_knots[0, 0] / 2,
                        finger_setpoint_knots[0, 0] / 2]
plant_env.SetPositions(context_plant_new, schunk_model, schunk_position)

env.Publish(context_new)

# sim
sim = Simulator(env, context_new)
viz = env.GetSubsystemByName('meshcat_visualizer')
viz.reset_recording()
viz.start_recording()
sim.AdvanceTo(t_knots[-1])
viz.stop_recording()
viz.publish_recording()


