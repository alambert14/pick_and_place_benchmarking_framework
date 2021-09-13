from typing import List
import numpy as np

from pydrake.all import RigidTransform


def calc_suction_ee_pose(X_WB_list: List[RigidTransform]):
    n_boxes = len(X_WB_list)
    angles_with_world_z = np.zeros(n_boxes)
    world_z = np.zeros(n_boxes)
    for i, X_WB in enumerate(X_WB_list):
        angles_with_world_z[i] = X_WB.rotation().matrix()[2, 2]
        world_z[i] = X_WB.translation()[2]

    '''
    best box to grasp: 
        1. cos(angle) close enough to 1.
        2. highest among the first.
    '''

    small_angle = angles_with_world_z > np.cos(np.pi / 6)
    idx_best = np.arange(n_boxes)[small_angle][np.argmax(world_z[small_angle])]

    X_WB_best = X_WB_list[idx_best]
    R_WB_best = X_WB_best.rotation()
    X_WE = RigidTransform(R_WB_best,
                          X_WB_best.translation() + R_WB_best.matrix()[:, 2] * 0.2)

    return X_WE


