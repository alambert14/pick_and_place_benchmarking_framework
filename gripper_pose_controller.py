import numpy as np

from pydrake.all import (ExternallyAppliedSpatialForce, SpatialForce,
                         LeafSystem, AbstractValue, RigidTransform,
                         SpatialVelocity, MultibodyPlant, PortDataType,
                         BodyIndex, Quaternion, RotationMatrix)


#%%
class GripperPoseController(LeafSystem):
    def __init__(self, gripper_body_idx: BodyIndex):
        super().__init__()
        self.set_name('gripper_pose_controller')
        self.body_idx = gripper_body_idx

        self.spatial_force_output_port = self.DeclareAbstractOutputPort(
            'spatial_force',
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.calc_spatial_force)

        self.body_pose_input_port = self.DeclareAbstractInputPort('body_pose',
                                      AbstractValue.Make([RigidTransform()]))

        self.body_spatial_velocity_input_port = self.DeclareAbstractInputPort(
            'spatial_velocity', AbstractValue.Make([SpatialVelocity()]))

        # 4 quaternion + 3 position
        self.pose_ref_input_port = self.DeclareInputPort(
            'pose_ref', PortDataType.kVectorValued, 7)

        # feedback gain
        self.Kp = np.array([200, 200, 200.], dtype=float)
        self.Dp = 2 * 1.0 * np.sqrt(self.Kp)
        self.Kr = np.array([50, 50, 50.], dtype=float)
        self.Dr = 2 * 1.0 * np.sqrt(self.Kr)

    def calc_spatial_force(self, context, output):
        X_WB = self.body_pose_input_port.Eval(context)[int(self.body_idx)]
        V_WB = self.body_spatial_velocity_input_port.Eval(
            context)[int(self.body_idx)]
        q_and_p_ref = self.pose_ref_input_port.Eval(context)
        R_WB_ref = RotationMatrix(Quaternion(q_and_p_ref[:4]))
        p_WB_ref = q_and_p_ref[4:]

        # add 10N in +z direction to compensate for gravity.
        f_WB = self.Kp * (p_WB_ref - X_WB.translation()) + np.array([0, 0, 10])
        f_WB += -self.Dp * V_WB.translational()

        R_WB = X_WB.rotation()
        R_BBref = R_WB.inverse().multiply(R_WB_ref)
        angle_axis = R_BBref.ToAngleAxis()
        m_WB = self.Kr * R_WB.multiply(angle_axis.axis() * angle_axis.angle())
        m_WB += -self.Dr * V_WB.rotational()

        F_Bq_W = SpatialForce(m_WB, f_WB)
        eaf = ExternallyAppliedSpatialForce()
        eaf.F_Bq_W = F_Bq_W
        eaf.body_index = self.body_idx
        output.set_value([eaf])
