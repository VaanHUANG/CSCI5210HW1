import numpy as np
import os
from tqdm import tqdm

def read_off(file_path):
    file = open(file_path, 'r')
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF file')
    n_verts, n_faces, n_edges = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def compute_triangle_area(verts, faces):

    n_faces = len(faces)
    tri_areas = np.zeros(shape=(n_faces,))
    tri_areas_sum = 0.0
    for i_face in range(n_faces):
        x1 = verts[faces[i_face][0]][0]
        y1 = verts[faces[i_face][0]][1]
        z1 = verts[faces[i_face][0]][2]
        x2 = verts[faces[i_face][1]][0]
        y2 = verts[faces[i_face][1]][1]
        z2 = verts[faces[i_face][1]][2]
        x3 = verts[faces[i_face][2]][0]
        y3 = verts[faces[i_face][2]][1]
        z3 = verts[faces[i_face][2]][2]

        term1 = np.square(x2*y1-x3*y1-x1*y2+x3*y2+x1*y3-x2*y3)
        term2 = np.square(x2*z1-x3*z1-x1*z2+x3*z2+x1*z3-x2*z3)
        term3 = np.square(y2*z1-y3*z1-y1*z2+y3*z2+y1*z3-y2*z3)
        tri_area = 0.5*(np.sqrt(term1+term2+term3))

        tri_areas_sum = tri_areas_sum + tri_area

        tri_areas[i_face] = tri_area

    tri_areas = tri_areas / tri_areas_sum  ## normalize

    return tri_areas

def weighted_random_sampling(verts, faces, npoints):
    ## npoints: how many points to sample from mesh

    tri_areas = compute_triangle_area(verts, faces)

    points = np.zeros(shape=(npoints, 3))
    for i in range(npoints):
        tri_index = np.random.choice(range(len(faces)), size=1, p=tri_areas)
        # tri_index = np.random.choice(range(len(faces)), size=1)
        tri_index = tri_index[0]
        x1 = verts[faces[tri_index][0]][0]
        y1 = verts[faces[tri_index][0]][1]
        z1 = verts[faces[tri_index][0]][2]
        x2 = verts[faces[tri_index][1]][0]
        y2 = verts[faces[tri_index][1]][1]
        z2 = verts[faces[tri_index][1]][2]
        x3 = verts[faces[tri_index][2]][0]
        y3 = verts[faces[tri_index][2]][1]
        z3 = verts[faces[tri_index][2]][2]

        # sample point on the selected triangle
        u = np.random.rand()
        v = np.random.rand()
        if u+v > 1.0:
            u = 1 - u
            v = 1 - v
        w = 1 - (u + v)

        new_point_x = x1 * u + x2 * v + x3 * w
        new_point_y = y1 * u + y2 * v + y3 * w
        new_point_z = z1 * u + z2 * v + z3 * w

        points[i, 0] = new_point_x
        points[i, 1] = new_point_y
        points[i, 2] = new_point_z

    return points


if __name__ == '__main__':

    input_folder = "/data/xzli/Data/downsampling/mesh_off"   # mesh input folder
    output_folder = "/data/xzli/Data/downsampling/output"    # point cloud output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    items = os.listdir(input_folder)
    npoints = 2048     # sample point number

    for item in tqdm(items):
        model_path = os.path.join(input_folder, items[0])
        model_name = item.split('.')[0]
        # read off mesh
        verts, faces = read_off(model_path)

        # sample points from mesh
        points = weighted_random_sampling(verts, faces, npoints)

        # save to disk
        new_model_name = model_name + ".xyz"
        output_path = os.path.join(output_folder, new_model_name)
        np.savetxt(output_path, points, fmt='%.3f')