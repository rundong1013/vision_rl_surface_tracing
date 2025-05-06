import os
import torch
import argparse
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from env.vision_surface_env import VisionSurfaceTracingEnv

# Compute how much of the surface has been covered by the end-effector
def compute_coverage(trajectory, surface_points, threshold=0.02):
    covered = np.zeros(len(surface_points), dtype=bool)
    for ee in trajectory:
        dists = np.linalg.norm(surface_points - ee, axis=1)
        covered |= (dists < threshold)
    return covered.sum() / len(surface_points)

# Load .pcd file and return point coordinates
def load_surface_points(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

# Visualize coverage and save figure
def draw_coverage(surface_points, ee_trajectory, save_path=None, threshold=0.03):
    surface_points = np.array(surface_points)
    ee_trajectory = np.array(ee_trajectory)

    dists = np.linalg.norm(surface_points[:, None, :] - ee_trajectory[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    covered_mask = min_dists < threshold

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(surface_points[~covered_mask, 0], surface_points[~covered_mask, 1], surface_points[~covered_mask, 2], c='gray', s=3, label='Unvisited')
    ax.scatter(surface_points[covered_mask, 0], surface_points[covered_mask, 1], surface_points[covered_mask, 2], c='red', s=5, label='Visited')
    ax.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2], c='blue', linewidth=1, label='EE Trajectory')

    ax.legend()
    ax.set_title("Surface Coverage Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

# Parse arguments: PPO model, PCD directory, output file path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="models/ppo_surface.zip")
parser.add_argument("--variant_dir", type=str, default="object/generated_pcds")
parser.add_argument("--output_csv", type=str, default="outputs/eval_summary.csv")
args = parser.parse_args()

model = PPO.load(args.model_path)

# Load all .pcd files from directory for evaluation
pcd_files = sorted([f for f in os.listdir(args.variant_dir) if f.endswith(".pcd")])
os.makedirs("outputs/eval_variants", exist_ok=True)

results = []

# Evaluate model on each test object
for pcd_file in pcd_files:
    print(f"Evaluating: {pcd_file}")
    pcd_path = os.path.join(args.variant_dir, pcd_file)
    surface_points = load_surface_points(pcd_path)

    env = VisionSurfaceTracingEnv(pcd_path=pcd_path, auto_switch_pcd=False)
    obs = env.reset()

    total_reward = 0.0
    ee_trajectory = []
    step = 0
    done = False

    while not done and step < env.max_episode_steps:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        ee_pos = env.robot.get_ee_position()
        ee_trajectory.append(np.array(ee_pos))
        step += 1

    ee_traj_np = np.unique(np.array(ee_trajectory), axis=0)


    surface_center = surface_points.mean(axis=0)
    ee_center = ee_traj_np.mean(axis=0)
    surface_points_aligned = surface_points + (ee_center - surface_center)

    coverage = compute_coverage(ee_traj_np, surface_points_aligned)
    visited_points = int(coverage * len(surface_points))

    draw_coverage(
        surface_points_aligned, ee_traj_np,
        save_path=f"outputs/eval_variants/{pcd_file.replace('.pcd', '_coverage.png')}"
    )

    print(f"{pcd_file}: Reward={total_reward:.2f}, Coverage={coverage*100:.2f}%")

    results.append({
        "pcd_file": pcd_file,
        "reward": total_reward,
        "coverage": coverage,
        "visited_points": visited_points
    })

# Save all results to a CSV file
df = pd.DataFrame(results)
df.to_csv(args.output_csv, index=False)
print(f"\n Summary saved to: {args.output_csv}")
