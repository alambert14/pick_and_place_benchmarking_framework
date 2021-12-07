import os

import meshcat
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from manipulation.meshcat_utils import draw_open3d_point_cloud, draw_points
from manipulation.open3d_utils import create_open3d_point_cloud
from manipulation.utils import FindResource, AddPackagePaths
from pydrake.all import (
    AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer,
    DiagramBuilder, RigidTransform, RotationMatrix,
    FindResourceOrThrow, Diagram,
    Parser, ProcessModelDirectives, LoadModelDirectives,
    PointCloud, Fields, BaseField
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from robot_utils import add_package_paths_local

import train_model
import torch
from torchvision.transforms import functional as F

zmq_url = "tcp://127.0.0.1:6000"


# scoring grasp candidiates
def grasp_candidate_cost(plant_context, cloud, plant, scene_graph,
                         scene_graph_context, grasp_points, adjust_X_G=False, textbox=None,
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
    """
    A "no frills" version of the example above, that returns the down-sampled
    point cloud.
    """
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
        print(c)
        point_cloud = diagram.GetOutputPort(f"{c}_point_cloud").Eval(context)
        print(f'point cloud shape: {point_cloud.size()}')
        pcd = create_open3d_point_cloud(point_cloud)
        print(f'open3d shape: {np.asarray(pcd.points)}')

        # Crop to region of interest.
        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,
                                                max_bound=crop_max))
        if pcd.is_empty():
            continue

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

        camera = plant.GetModelInstanceByName(c)
        body = plant.GetBodyByName("base", camera)
        X_WC = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd.orient_normals_towards_camera_location(X_WC.translation())

        # Merge point clouds.
        merged_pcd += pcd

    # Voxelize down-sample.  (Note that the normals still look reasonable)'
    print(f'open3d shape: {np.asarray(merged_pcd.points).shape}')
    return merged_pcd.voxel_down_sample(voxel_size=0.005)

# From pose_estimation_icp.ipynb
def pcl_np2drake(np_cloud):
    assert(np_cloud.shape[1] == 3)
    pcl_drake = PointCloud(new_size = np_cloud.shape[0],
                           fields=Fields(BaseField.kXYZs | BaseField.kRGBs))

    xyzs = pcl_drake.mutable_xyzs()
    xyzs[:,:] = np.array(np_cloud).transpose()

    return pcl_drake

def pcl_to_camera1(diagram, context, cloud):
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    camera = plant.GetModelInstanceByName('camera1')
    body = plant.GetBodyByName("base", camera)
    X_WC = plant.EvalBodyPoseInWorld(plant_context, body)
    X_WP = np.asarray(cloud.points)
    # p means points :)
    X_CP = X_WC.inverse() @ X_WP.T

    return X_CP.T

def get_masked_pcl(diagram, cloud, mask):
    cam = diagram.GetSubsystemByName('camera1')
    intrinsics = cam.depth_camera_info()
    cx = intrinsics.center_x()
    cy = intrinsics.center_y()
    fx = intrinsics.focal_x()
    fy = intrinsics.focal_y()

    masked_pcl = []
    for_display = np.zeros([480, 640])
    print(cloud)
    for p in cloud:
        u = (p[1] * fy) / p[2] + cy
        v = (p[0] * fx) / p[2] + cx

        print(u,v)
        print('got here!!!')
        try:
            if mask[round(u), round(v)]:
                masked_pcl.append(p)
                for_display[round(u), round(v)] = p[2]
        except IndexError:
            pass

    print('got to plot')
    print(np.unique(for_display))
    print(np.unique((for_display * 255).astype(np.uint8)))
    cv2.imwrite('masked_pcl.png', (for_display * 255).astype(np.uint8))

def generate_grasp_candidate_antipodal(plant_context, cloud, plant, scene_graph,
                                       scene_graph_context, rng,
                                       meshcat_vis=None):
    """
    Picks a random point in the cloud, and aligns the robot finger with the
     normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution
     over [min_roll, max_roll].
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

def prediction_to_masks(prediction):
    masks = []
    for i in range(prediction[0]['masks'].size()[0]):
        print(f'label is: {prediction[0]["labels"][i]}')
        img_array = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
        _, thresh = cv2.threshold(img_array,90,255,cv2.THRESH_BINARY)
        dilation = cv2.dilate(thresh,np.ones((5,5)).astype(np.uint8), iterations = 1)
        thresh = cv2.erode(dilation,np.ones((5,5)).astype(np.uint8), iterations = 1)
        masks.append((thresh == 255).astype(np.uint8))

    return masks

class GraspSamplerVision:
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
        add_package_paths_local(parser)
        ProcessModelDirectives(
            LoadModelDirectives(
                os.path.join(os.path.dirname(__file__),
                             "models/clutter_planning.yaml")),
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

        self.model = train_model.get_model_instance_segmentation(4)
        self.model.load_state_dict(torch.load('../veggie_master_20000.pth', map_location=torch.device('cpu')))


    def sample_grasp_candidates(self, context_env, draw_grasp_candidates=True):
        cloud = process_point_cloud(self.env, context_env,
                                    ["camera0", "camera1", "camera2"], "bin0")

        camera = self.env.GetSubsystemByName('camera1')
        cam_context = camera.GetMyMutableContextFromRoot(context_env)
        rgb_image_np = camera.GetOutputPort('color_image').Eval(cam_context).data
        rgb_image_pil = Image.fromarray(rgb_image_np).convert("RGB")
        rgb_image_pil.save('rgb_for_figure.png')
        rgb_image = F.to_tensor(rgb_image_pil)


        print(rgb_image.size())
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([rgb_image.to(torch.device('cpu'))])

        mask = prediction_to_masks(prediction)[0]
        cv2.imwrite('nice_mask.png', mask * 255)
        label = prediction[0]['labels'][0]

        # Create pretty drawings for paper[0][0]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # center of contour
        M = cv2.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        label_to_veg = {1: 'Cucumber', 2: 'Lime', 3: 'Mango'}

        drawing = cv2.cvtColor(np.array(rgb_image_pil), cv2.COLOR_BGR2RGB)
        cv2.drawContours(drawing, contours, -1, (0, 255, 0), 3, cv2.LINE_8, hierarchy, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        drawing = cv2.putText(drawing, f'label: {label_to_veg[int(label)]}', (cX, cY), font,
               fontScale, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imwrite('labeled_image.png', drawing)

        # TRY OUR FUNCTION:
        X_CP = pcl_to_camera1(self.env, context_env, cloud)

        get_masked_pcl(self.env, X_CP, mask)

        if cloud.is_empty():
            return []

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

        # TODO: get label of grasp
        label = 'Cucumber'

        return X_Gs_best, label
