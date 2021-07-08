import numpy as np

from pydrake.all import (ExternallyAppliedSpatialForce, SpatialForce,
                         LeafSystem, AbstractValue, RigidTransform,
                         SpatialVelocity, MultibodyPlant, PortDataType,
                         BodyIndex, Quaternion, RotationMatrix, BasicVector,
                         PiecewisePolynomial, PiecewiseQuaternionSlerp)


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


class CustomTrajectorySource(LeafSystem):
    def __init__(self, Q_WB_traj: PiecewiseQuaternionSlerp,
                 p_WB_traj: PiecewisePolynomial,
                 finger_setpoint_traj: PiecewisePolynomial):
        """
        For gripper pose and finger position trajectory references.
        :param Q_WB_traj:
        :param p_WB_traj:
        :param finger_setpoint_traj:
        """
        super().__init__()
        self.set_name('custom_trajectory_source')

        # self.Q_WB_traj_input_port = self.DeclareAbstractInputPort(
        #     'Q_WB_traj', AbstractValue.Make(PiecewiseQuaternionSlerp()))
        #
        # self.p_WB_traj_input_port = self.DeclareAbstractInputPort(
        #     'p_WB_traj', AbstractValue.Make(PiecewisePolynomial()))

        self.Q_WB_traj = Q_WB_traj
        self.p_WB_traj = p_WB_traj
        self.finger_setpoint_traj = finger_setpoint_traj

        self.body_pose_output_port = self.DeclareVectorOutputPort(
            'q_and_p', BasicVector(7), self.calc_q_and_p)

        self.finger_setpoint_output_port = self.DeclareVectorOutputPort(
            'finger', BasicVector(4), self.calc_finger_setpoint)

    def calc_q_and_p(self, context, output):
        t = context.get_time()
        q = RotationMatrix(self.Q_WB_traj.value(t)).ToQuaternion().wxyz()
        p = self.p_WB_traj.value(t).ravel()
        output.SetFromVector(np.hstack([q, p]))

    def calc_finger_setpoint(self, context, output):
        t = context.get_time()
        d_open = self.finger_setpoint_traj.value(t).ravel()[0] / 2
        v_open = self.finger_setpoint_traj.derivative(1).value(t).ravel()[0]
        setpoints = np.array([-d_open, d_open, -v_open, v_open])
        output.SetFromVector(setpoints)

