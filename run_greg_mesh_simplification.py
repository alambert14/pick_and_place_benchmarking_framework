from spatial_scene_grammars.models.do_model_updates import *


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

#%%
do_collision_mesh_simplification(stl_paths[2], True)

#%%
sdf_file_name = os.path.join('cad_files', 'Mango.sdf')
update_sdf_with_convex_decomp(sdf_file_name)
