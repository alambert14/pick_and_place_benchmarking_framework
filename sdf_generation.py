import os

import numpy as np
from scipy.stats import moment
from lxml import etree as et
import trimesh


#%% file names
stl_folder = "cad_files"
stl_paths = []
object_names = []
object_mass = {"Mango": 0.2, "Cucumber": 0.18, "Lime": 0.045}
model_colors = {
    "Lime": np.array([191, 255, 0.]) / 255,
    "Cucumber": np.array([93, 121, 25.]) / 255,
    "Mango": np.array([246, 189, 22.]) / 255
}

for file in os.listdir(stl_folder):
    if file.endswith('.stl'):
        name, _ = file.rsplit('.', 1)
        stl_paths.append(os.path.join(stl_folder, file))
        object_names.append(name)

#%% process mesh
object_meshes = {}
for idx in range(len(object_names)):
    mesh = trimesh.load(stl_paths[idx])
    scale = 0.001  # mesh vertices are probably in mm.
    mesh.apply_scale(scale)
    mesh.apply_transform(mesh.principal_inertia_transform)

    name = object_names[idx]
    file_name_save = os.path.join(stl_folder, name + "_processed.obj")
    with open(file_name_save, "w") as file:
        mesh.export(file, "obj")

    object_meshes[name] = mesh


#%%
name = object_names[2]

root = et.Element("sdf", version="1.7")
model = et.SubElement(root, "model", name=name)
link = et.SubElement(model, "link", name="base_link")

# inertial
inertial = et.SubElement(link, "inertial")
mass = et.SubElement(inertial, "mass")
mass.text = str(object_mass[name])

inertia = et.SubElement(inertial, "inertia")
m = object_mass[name]
x, y, z = object_meshes[name].bounding_box.extents
inertia_dict = {
    "ixx": m / 12 * (y**2 + z**2),
    "ixy": 0.,
    "ixz": 0.,
    "iyy": m / 12 * (x**2 + z**2),
    "iyz": 0.,
    "izz": m / 12 * (x**2 + y**2)
}

for inertia_name, inertia_value in inertia_dict.items():
    i = et.SubElement(inertia, inertia_name)
    i.text = '{:.3e}'.format(inertia_value)

# visual
visual = et.SubElement(link, "visual", name=name + "_visual")
visual_g = et.SubElement(visual, "geometry")
visual_g_mesh = et.SubElement(visual_g, "mesh")
visual_g_mesh_scale = et.SubElement(visual_g_mesh, "scale")
visual_g_mesh_scale.text = "1 1 1"
visual_g_mesh_uri = et.SubElement(visual_g_mesh, "uri")
visual_g_mesh_uri.text = name + "_processed.obj"

# visual RGBA.
visual_material = et.SubElement(visual, "material")
visual_material_diffuse = et.SubElement(visual_material, "diffuse")
visual_material_diffuse.text = ''
for c in model_colors[name]:
    visual_material_diffuse.text += "{:.2} ".format(c)
visual_material_diffuse.text += '1.0'

# collision
collision = et.SubElement(link, "collision", name=name + "_collision")
collision_g = et.SubElement(collision, "geometry")
collision_g_mesh = et.SubElement(collision_g, "mesh")
collision_g_mesh_scale = et.SubElement(collision_g_mesh, "scale")
collision_g_mesh_scale.text = "1 1 1"
collision_g_mesh_uri = et.SubElement(collision_g_mesh, "uri")
collision_g_mesh_uri.text = name + "_processed.obj"

sdf_binary = et.tostring(root, pretty_print=True)
# print(sdf_binary.decode('utf-8'))

file_name_save = os.path.join(stl_folder, name + ".sdf")
with open(file_name_save, "w") as file:
    file.write(sdf_binary.decode('utf-8'))
