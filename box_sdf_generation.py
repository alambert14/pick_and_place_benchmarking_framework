import os

import numpy as np
from lxml import etree as et
import trimesh

#%% file names
stl_folder = "models"

# inputs to function.
name = "blue_berry_box_pretty_visual"
x = 0.18
y = 0.10
z = 0.045
mass = 0.25  # kg
rgb_berry = np.array([70, 65, 150]) / 255

# in the function.
root = et.Element("sdf", version="1.7")
node_model = et.SubElement(root, "model", name=name)
node_link = et.SubElement(node_model, "link", name="base_link")

# inertial
node_inertial = et.SubElement(node_link, "inertial")
node_mass = et.SubElement(node_inertial, "mass")
node_mass.text = str(mass)
inertia_dict = {
    "ixx": mass / 12 * (y**2 + z**2),
    "ixy": 0.,
    "ixz": 0.,
    "iyy": mass / 12 * (x**2 + z**2),
    "iyz": 0.,
    "izz": mass / 12 * (x**2 + y**2)
}
node_inertia = et.SubElement(node_inertial, "inertia")

for inertia_name, inertia_value in inertia_dict.items():
    i = et.SubElement(node_inertia, inertia_name)
    i.text = '{:.3e}'.format(inertia_value)


def add_box_geometry(node, x, y, z):
    node_g = et.SubElement(node, "geometry")
    node_g_box = et.SubElement(node_g, "box")
    node_g_box_size = et.SubElement(node_g_box, 'size')
    node_g_box_size.text = "{} {} {}".format(x, y, z)


def add_sphere_geometry(node, x_c, y_c, z_c, r):
    node_pose = et.SubElement(node, "pose")
    node_pose.text = "{} {} {} {} {} {}".format(x_c, y_c, z_c, 0, 0, 0)

    node_g = et.SubElement(node, "geometry")
    node_g_sphere = et.SubElement(node_g, "sphere")
    node_g_sphere_radius = et.SubElement(node_g_sphere, 'radius')
    node_g_sphere_radius.text = "{}".format(r)


# visual
node_visual = et.SubElement(node_link, "visual", name=name + "_visual_box")
add_box_geometry(node_visual, x, y, z)
node_visual_material = et.SubElement(node_visual, "material")
node_visual_material_diffuse = et.SubElement(node_visual_material, "diffuse")
node_visual_material_diffuse.text = "1 1 1 0.5"

box_dimensions = np.array([x, y, z]) * np.array([0.9, 0.8, 0.7])
for i in range(100):
    xyz_berry = np.random.rand(3) * box_dimensions - box_dimensions / 2
    node_visual = et.SubElement(node_link, "visual", name=name + f"_visual_berry_{i}")
    add_sphere_geometry(node_visual, xyz_berry[0], xyz_berry[1], xyz_berry[2], 0.007)
    node_visual_material = et.SubElement(node_visual, "material")
    node_visual_material_diffuse = et.SubElement(node_visual_material, "diffuse")
    node_visual_material_diffuse.text = "{:.2f} {:.2f} {:.2f} 1.0".format(
        rgb_berry[0], rgb_berry[1], rgb_berry[2])

#  collision
node_collision = et.SubElement(node_link, "collision", name="body")
add_box_geometry(node_collision, x - 2e-3, y - 2e-3, z - 2e-3)

idx = 0
for i in [-1, 1]:
    for j in [-1, 1]:
        for k in [-1, 1]:
            node_collision = et.SubElement(
                node_link, "collision", name="corner_{}".format(idx))
            add_sphere_geometry(node_collision,
                                x_c=x / 2 * i,
                                y_c=y / 2 * j,
                                z_c=z / 2 * k,
                                r=1e-3)
            idx += 1


sdf_binary = et.tostring(root, pretty_print=True)
print(sdf_binary.decode('utf-8'))

file_name_save = os.path.join(stl_folder, name + ".sdf")
with open(file_name_save, "w") as file:
    file.write(sdf_binary.decode('utf-8'))




