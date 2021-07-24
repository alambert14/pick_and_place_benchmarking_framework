import os

import numpy as np
import pydrake
from pydrake.all import (Parser, MultibodyPlant, ProcessModelDirectives,
                         LoadModelDirectives, LeafSystem, PiecewisePolynomial,
                         BasicVector, Joint, SpatialInertia, RigidTransform)
from robotics_utilities.iiwa_controller.utils import models_dir as \
    iiwa_controller_models_dir

models_dir = os.path.join(os.path.dirname(__file__), 'models')


def render_system_with_graphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


def add_package_paths_local(parser: Parser):
    parser.package_map().Add(
        "drake_manipulation_models",
        os.path.join(pydrake.common.GetDrakePath(),
                     "manipulation/models"))

    parser.package_map().Add("local", models_dir)

    parser.package_map().Add('iiwa_controller',
                             iiwa_controller_models_dir)

    parser.package_map().PopulateFromFolder(models_dir)


class SimpleTrajectorySource(LeafSystem):
    def __init__(self, q_traj: PiecewisePolynomial):
        super().__init__()
        self.q_traj = q_traj

        self.x_output_port = self.DeclareVectorOutputPort(
            'x', BasicVector(q_traj.rows() * 2), self.calc_x)

        self.t_start = 0.

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(np.hstack([q, v]))

    def set_t_start(self, t_start_new: float):
        self.t_start = t_start_new
