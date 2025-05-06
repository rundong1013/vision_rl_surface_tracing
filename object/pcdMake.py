import numpy as np
import os
# Generate the 6 faces of a cube, sampled uniformly
def generate_cube_surface(edge_length=0.2, resolution=10):
    half = edge_length / 2
    lin = np.linspace(-half, half, resolution)
    faces = []
    for axis in range(3):
        for sign in [-half, half]:
            grid = np.meshgrid(lin, lin)
            coords = np.zeros((resolution * resolution, 3))
            for i in range(3):
                if i == axis:
                    coords[:, i] = sign
                else:
                    coords[:, i] = grid.pop(0).flatten()
            faces.append(coords)
    return np.vstack(faces) 

# Generate a cylinder surface including side + top and bottom circles
def generate_cylinder_surface(radius=0.1, height=0.2, radial_res=30, height_res=10):
    angles = np.linspace(0, 2 * np.pi, radial_res)
    heights = np.linspace(-height / 2, height / 2, height_res)
    angle_grid, height_grid = np.meshgrid(angles, heights)
    x = radius * np.cos(angle_grid)
    y = radius * np.sin(angle_grid)
    z = height_grid
    side = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    top_bottom = []
    for z_val in [-height / 2, height / 2]:
        r = np.linspace(0, radius, 10)
        theta = np.linspace(0, 2 * np.pi, 20)
        r_grid, t_grid = np.meshgrid(r, theta)
        x = r_grid * np.cos(t_grid)
        y = r_grid * np.sin(t_grid)
        z = np.full_like(x, z_val)
        top_bottom.append(np.vstack((x.flatten(), y.flatten(), z.flatten())).T)

    return np.vstack([side] + top_bottom)

# Apply random scaling and rotation to point cloud
def apply_random_transform(points, scale_range=(0.5, 1.5), rot_deg_range=(-45, 45)):
    scale = np.random.uniform(*scale_range)
    points = points * scale

    angles = np.radians(np.random.uniform(*rot_deg_range, size=3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return points @ R.T

# Save point cloud as pcd file
def save_pcd(filename, points):
    with open(filename, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

os.makedirs("generated_pcds", exist_ok=True)

for shape in ["cube", "cylinder"]:
    for i in range(3):
        if shape == "cube":
            base = generate_cube_surface()
        else:
            base = generate_cylinder_surface()
        transformed = apply_random_transform(base)
        filename = f"generated_pcds/{shape}_{i}.pcd"
        save_pcd(filename, transformed)
        print(f"Saved {filename} with {len(transformed)} points.")
