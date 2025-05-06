import numpy as np
import open3d as o3d

# Global set to keep track of visited surface point indices
visited_surface = set()

# Compute a sparse reward based on how well EE traces the surface
def compute_surface_reward(ee_pos, surface_points, threshold=0.015):
    global visited_surface
    surface_points = np.asarray(surface_points)

    dists = np.linalg.norm(surface_points - ee_pos, axis=1)
    close_indices = np.where(dists < threshold)[0]

    new_visits = sum(idx not in visited_surface for idx in close_indices)
    for idx in close_indices:
        visited_surface.add(idx)

    bonus_reward = new_visits * 1.0
    coverage_ratio = len(visited_surface) / len(surface_points)
    coverage_reward = coverage_ratio * 2.0

    if new_visits == 0:
        return -0.1 

    return bonus_reward + coverage_reward

# Generate surface trajectory points based on PCA axes of the object
def generate_surface_points_from_pcd(pcd_path, n_lines=10, points_per_line=50):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    mean = np.mean(points, axis=0)
    points_centered = points - mean

    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    main_axis = eigvecs[:, order[0]]
    ortho_axis = eigvecs[:, order[1]]


    bbox = np.max(points, axis=0) - np.min(points, axis=0)
    scale_main = np.dot(bbox, np.abs(main_axis)) * 0.8 
    scale_ortho = np.dot(bbox, np.abs(ortho_axis)) * 0.8 

    surface_points = []
    for offset in np.linspace(-scale_ortho / 2, scale_ortho / 2, n_lines):
        for t in np.linspace(-scale_main / 2, scale_main / 2, points_per_line):
            point = mean + offset * ortho_axis + t * main_axis
            surface_points.append(point)

    return np.array(surface_points)
