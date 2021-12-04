from typing import List, Dict
import os.path
from datetime import datetime
import json
import time
import sys
import cv2
import numpy as np
from pydrake.all import (
    PiecewisePolynomial, PidController, MultibodyPlant, RigidTransform,
    RandomGenerator, Simulator, InverseDynamicsController, ContactModel,
    ConnectContactResultsToDrakeVisualizer, DrakeVisualizer, Parser,
    DiagramBuilder, AddMultibodyPlantSceneGraph, ProcessModelDirectives,
    LoadModelDirectives, ConnectMeshcatVisualizer, RollPitchYaw, PerceptionProperties, RenderLabel)

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


def make_environment_model(
        directive, rng=None, draw=False, n_objects=0, bin_name="bin"):
    """
    Make one model of the environment, but the robot only gets to see the sensor
     outputs.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.
    start_time = time.time()
    ProcessModelDirectives(LoadModelDirectives(directive), plant, parser)
    inspector = scene_graph.model_inspector()
    # Add objects.
    start_time = time.time()
    object_bodies, lime_bag_bodies = add_objects(
        n_objects=n_objects,
        obj_names=object_names,
        obj_sdfs=object_sdfs,
        plant=plant,
        parser=parser,
        #inspector=inspector,
        rng=rng)

    # Set contact model.
    start_time = time.time()
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
    # render_system_with_graphviz(diagram)
    context = diagram.CreateDefaultContext()

    # Set initial conditions.
    # robot and gripper initial conditions.
    context_plant = plant.GetMyContextFromRoot(context)
    #plant.SetPositions(context_plant, model_iiwa, q_iiwa_bin1)
    #plant.SetPositions(context_plant, model_schunk,
    #                   finger_setpoints.value(0).ravel())

    start_time = time.time()
    simulator = None
    if n_objects > 0:
        set_object_drop_pose(
            context_env=context,
            plant=plant,
            bin_name=bin_name,
            object_bodies=object_bodies,
            lime_bag_bodies=lime_bag_bodies,
            rng=rng)


    elif draw:
        viz.load()
        diagram.Publish(context)

    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(2.0)
    simulator.set_target_realtime_rate(0.)
    
    return plant, diagram, context, simulator, scene_graph, inspector


categories_dict = {
    'Cucumber': 1,
    'Lime': 2,
    'Mango': 3,
}
annotation_id = 0

def get_annotations(label_image, object_data, image_id):
    global annotation_id
    global categories_dict

    annotations = []
    for i, info in object_data.items():
        annotation = {}
        mask = (label_image == i).astype(np.uint8)
        if np.all(mask == 0):
            continue
        # cv2.imshow('mask', mask * 255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Get contours
        start_time = time.time()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # drawing = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2, cv2.LINE_8, hierarchy, 0)
        
        good_contours = []
        threshold = 1000
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                good_contours.append(contour)
                areas.append(area)
            # org = (25, 50)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # # fontScale
            # fontScale = 0.75
            # # Blue color in BGR
            # color = (0, 255, 0)
            # drawing = cv2.putText(drawing, f'area (px): {area}', org, font,
            #        fontScale, color, 3, cv2.LINE_AA)

            # drawing = cv2.putText(drawing, f'label: {info[1]}', (350, 50), font,
            #        fontScale, color, 3, cv2.LINE_AA)


        if not good_contours:
            continue

        all_xys = np.concatenate(good_contours)
        x, y, w, h = cv2.boundingRect(all_xys)

        # Construct the annotation
        annotation_id += 1
        annotation['segmentation'] = all_xys.flatten().tolist()
        annotation['area'] = sum(areas)
        annotation['image_id'] = image_id
        annotation['category_id'] = categories_dict[info[1]]
        annotation['bounding_box'] = [x, y, w, h]
        annotation['id'] = annotation_id
        annotations.append(annotation)

        # cv2.imshow(f'contours {image_id}', drawing)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return annotations


if __name__ == '__main__':
    #%%
    sys.stdout.flush()
    directive_file = os.path.join(
        os.getcwd(), 'models', 'bin_and_cameras.yaml')

    rng = np.random.default_rng(seed=1215232)
    # seed 12153432 looks kind of nice.

    # clean up visualization.
    v = meshcat.Visualizer(zmq_url=zmq_url)
    v.delete()
    
    num_iters = 250

    images = []
    annotations = []

    categories = [
        {'id': 1, 'name': 'cucumber'},
        {'id': 2, 'name': 'lime'},
        {'id': 3, 'name': 'mango'},
    ]

    info = {
        'year': '2021',
        'version': 1,
        'description': 'Synthetic training data for fruits',
        'date_created': datetime.now().isoformat().replace('T', ' '),
    }

    num_skipped = 0

    for n in range(num_iters):
        print('iter', n)
        # delete me to use the default nice seed
        rng = np.random.default_rng()

        # build environment
        try:
            plant, env, context_env, sim, scene_graph, inspector = make_environment_model(
                directive=directive_file, rng=rng, draw=False, n_objects=10)
        except RuntimeError:
            num_skipped += 1
            continue

        # dictionary mapping id number (i) to tuple geometry id and type of produce
        produce_geo_ids = {} 

        start_time = time.time()
        for i, geo_id in enumerate(inspector.GetAllGeometryIds()):
            name = inspector.GetName(geo_id)
            if 'visual' not in name or 'bin' in name or 'camera' in name:
                if 'visual' in name:
                    not_id = inspector.GetPerceptionProperties(geo_id).GetProperty('label', 'id')
                continue

            realname = name.split('::')[1].split('_')[0]
            not_id = inspector.GetPerceptionProperties(geo_id).GetProperty('label', 'id')
            realid = None
            for j in range(100):
                if j == not_id:
                    realid = j
                    break
            if realid is None:
                print('uh ohh')
            perception = inspector.GetPerceptionProperties(geo_id)
            perception.UpdateProperty('label', 'id', RenderLabel(realid))
            # perception.UpdateProperty('label', 'type', realname)
            produce_geo_ids[realid] = (geo_id, realname)
        
            #scene_graph.AssignRole(plant.get_source_id(), geo_id, perception)
            geometry_label = inspector.GetPerceptionProperties(geo_id).GetProperty('label', 'id')


        #viz = env.GetSubsystemByName('meshcat_visualizer')

        plant_env = env.GetSubsystemByName('plant')

        #viz.reset_recording()
        #viz.start_recording()

        cameras = [env.GetSubsystemByName(f'camera{i}') for i in range(3)]
        cam_contexts = [cam.GetMyMutableContextFromRoot(context_env) for cam in cameras]

        for idx, (cam, cam_context) in enumerate(zip(cameras, cam_contexts)):
            rgb_image = cam.GetOutputPort('color_image').Eval(cam_context).data
            label_image = cam.GetOutputPort('label_image').Eval(cam_context).data

            filename = os.path.join(training_dir, f'cam{idx}_{n}.png')

            img = rgb_image[:,:,:-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imwrite(filename, img)


            image_info = {}
            image_info['file_name'] = filename
            image_info['height'] = img.shape[0]
            image_info['width'] = img.shape[1]
            image_info['id'] = 10000 * idx + n  # 1e4 * cam_id + iteration
            image_info['date_captured'] = datetime.now().isoformat().replace('T', ' ')

            images.append(image_info)
            
            annotations += get_annotations(label_image, produce_geo_ids, image_info['id'])

        #viz.stop_recording()
        #viz.publish_recording()
    print()
    print('##########################')
    print('skipped:',num_skipped)
    print('##########################')
    print()
    coco = {
        'info': info,
        'images': images,
        'categories': categories,
        'annotations': annotations,
    }

    coco_file = os.path.join(training_dir, 'coco.json')

    with open(coco_file, 'w') as outfile:
        json.dump(coco, outfile)
