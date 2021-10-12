from typing import List
import numpy as np

from pydrake.all import (LeafSystem, AbstractValue, BasicVector,
                         ExternallyAppliedSpatialForce, BodyIndex,
                         SpatialForce)
from pydrake.math import RigidTransform, RollPitchYaw


k_suction_offset_z = 0.29


class SuctionSystem(LeafSystem):
    def __init__(self, box_body_indices: List[BodyIndex],
                 l7_body_index: BodyIndex):
        super().__init__()
        self.set_name("suction_system")

        # input ports
        self.body_poses_input_port = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))
        self.suction_strength_input_port = self.DeclareVectorInputPort(
            "suction_strength", BasicVector(2))

        # output ports
        self.easf_output_port = self.DeclareAbstractOutputPort(
            "suction_spatial_force",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.calc_suction_force)

        self.box_body_indices = box_body_indices
        self.l7_body_index = l7_body_index
        # index into self.box_body_indices.
        self.idx_box_to_suck = None

    def calc_suction_force(self, context, suction_force_abstract_value):
        if self.idx_box_to_suck is None:
            suction_force_abstract_value.set_value([])
            return

        body_poses = self.body_poses_input_port.Eval(context)
        suction_strength = self.suction_strength_input_port.Eval(context)[0]
        if suction_strength < 1e-3:
            suction_force_abstract_value.set_value([])
            return

        X_WL7 = body_poses[int(self.l7_body_index)]
        box_idx = self.box_body_indices[self.idx_box_to_suck]
        X_WB = body_poses[int(box_idx)]

        # Spatial force application point in box frame B.
        # Hard-coded based on the suction_cups.sdf file.
        p_Bc_list = np.array([[0.03, 0.02, 0],
                              [-0.03, 0.02, 0],
                              [-0.03, -0.02, 0],
                              [0.03, -0.02, 0]])

        p_Wc_W_list = X_WB.multiply(p_Bc_list.T).T
        p_WL7o_W = X_WL7.translation()

        easf_list = []
        for p_Wc_W, p_Bc in zip(p_Wc_W_list, p_Bc_list):
            f = suction_strength * 1.0 * (p_WL7o_W - p_Wc_W)
            F_Bq_W = SpatialForce(np.zeros(3), f)

            easf = ExternallyAppliedSpatialForce()
            easf.body_index = box_idx
            easf.p_BoBq_B = p_Bc
            easf.F_Bq_W = F_Bq_W
            easf_list.append(easf)

        suction_force_abstract_value.set_value(easf_list)


def calc_suction_ee_pose(X_WB_list: List[RigidTransform]):
    """
    X_WB_list: list of poses of all boxes.
    return idx_best is an index into X_WB_list.
    """
    n_boxes = len(X_WB_list)
    angles_with_world_z = np.zeros(n_boxes)
    world_z = np.zeros(n_boxes)
    X_WB_adjusted_list = []  # if z is pointing downwards, flip it back up.
    for i, X_WB in enumerate(X_WB_list):
        angle = X_WB.rotation().matrix()[2, 2]
        X_WB_adjusted = RigidTransform(X_WB)
        if angle < 0:
            X_WB_adjusted = RigidTransform(
                X_WB.rotation().multiply(RollPitchYaw(0, np.pi, 0).ToRotationMatrix()),
                X_WB.translation())
            angle *= -1

        angles_with_world_z[i] = angle
        world_z[i] = X_WB_adjusted.translation()[2]
        X_WB_adjusted_list.append(X_WB_adjusted)

    '''
    best box to grasp: 
        1. cos(angle) close enough to 1.
        2. highest among the first.
    '''

    small_angle = angles_with_world_z > np.cos(np.pi / 4)
    idx_best = np.arange(n_boxes)[small_angle][np.argmax(world_z[small_angle])]

    X_WB_best = X_WB_adjusted_list[idx_best]
    R_WB_best = X_WB_best.rotation()
    X_WE = RigidTransform(
        R_WB_best.multiply(RollPitchYaw(0, np.pi, 0).ToRotationMatrix()),
        X_WB_best.translation() + R_WB_best.matrix()[:, 2] * k_suction_offset_z)

    return X_WE, idx_best






