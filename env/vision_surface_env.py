import os
import random
import numpy as np
import open3d as o3d
import pybullet as p
from env import utils
from gym.spaces import Box, Dict
from panda_gym.envs.panda_tasks.panda_reach import PandaReachEnv
from env.utils import generate_surface_points_from_pcd, compute_surface_reward, visited_surface


class VisionSurfaceTracingEnv(PandaReachEnv):
    def __init__(self, pcd_path=None, train_pcd_dir="object/train_pcds", auto_switch_pcd=False):
        if p.isConnected() == 0:
            p.connect(p.DIRECT) # Ensure PyBullet is connected
        
        self.episode_id = 0
        self.auto_switch_pcd = auto_switch_pcd
        self.train_pcd_dir = train_pcd_dir

        if pcd_path is None:
            candidates = [f for f in os.listdir(train_pcd_dir) if f.endswith(".pcd")]
            if not candidates:
                raise FileNotFoundError(f"No .pcd files found in {train_pcd_dir}")
            selected_pcd = random.choice(candidates)
            self.pcd_path = os.path.join(train_pcd_dir, selected_pcd)
        else:
            self.pcd_path = pcd_path

        self.surface_shape = os.path.basename(self.pcd_path).split(".")[0]
        self.visualize_trajectory = True
        self.render_high_res = True

        self.n_lines = 10
        self.points_per_line = 50
        self.target_points = self._load_target_points()
        self.current_point_idx = 0

        self.episode_step = 0

        super().__init__()

        # Extend observation space to include image and current target point
        base_obs = super()._get_obs()
        self.observation_space = Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=base_obs["observation"].shape, dtype=np.float32),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=base_obs["achieved_goal"].shape, dtype=np.float32),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=base_obs["desired_goal"].shape, dtype=np.float32),
            "image": Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "target_point": Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        })
    
    # Load target points from the given .pcd and sample surface paths
    def _load_target_points(self):
        pcd = o3d.io.read_point_cloud(self.pcd_path)
        points = np.asarray(pcd.points)
        bbox = np.max(points, axis=0) - np.min(points, axis=0)

        points_centered = points - np.mean(points, axis=0)
        cov = np.cov(points_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        main_axis = eigvecs[:, order[0]]
        ortho_axis = eigvecs[:, order[1]]

        scale_main = np.dot(bbox, np.abs(main_axis))
        scale_ortho = np.dot(bbox, np.abs(ortho_axis))

        self.points_per_line = max(10, int(scale_main * 200))
        self.n_lines = max(5, int(scale_ortho * 100))

        return generate_surface_points_from_pcd(
            self.pcd_path,
            n_lines=self.n_lines,
            points_per_line=self.points_per_line
        )
    
    # Observation includes base robot obs + rendered image + current surface point
    def _get_obs(self):
        base_obs = super()._get_obs()
        width, height = 224, 224
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=p.computeViewMatrix(
                cameraEyePosition=[1.1, 0, 0.6],
                cameraTargetPosition=[0.5, 0, 0.3],
                cameraUpVector=[0, 0, 1],
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0),
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_img = np.array(rgb_img).reshape((height, width, 4))[:, :, :3]

        return {
            "observation": base_obs["observation"],
            "achieved_goal": base_obs["achieved_goal"],
            "desired_goal": base_obs["desired_goal"],
            "image": rgb_img.astype(np.uint8),
            "target_point": np.array(self.target_points[self.current_point_idx], dtype=np.float32)
        }   
    
    # Custom step with surface-based reward
    def step(self, action):
        obs, _, _, info = super().step(action)
        ee_pos = self.robot.get_ee_position()
        reward = compute_surface_reward(ee_pos, self.target_points)

        self.episode_step += 1
        done = self.episode_step >= self.max_episode_steps

        return obs, reward, done, info   
    
    # Reset environment, optionally switching to a new PCD shape
    def reset(self):
        if self.auto_switch_pcd:
            candidates = [f for f in os.listdir(self.train_pcd_dir) if f.endswith(".pcd")]
            selected_pcd = random.choice(candidates)
            self.pcd_path = os.path.join(self.train_pcd_dir, selected_pcd)
            self.surface_shape = os.path.basename(self.pcd_path).split(".")[0]
            print(f"[AutoSwitch] Episode {self.episode_id}: selected {self.pcd_path}")

        self.target_points = self._load_target_points()
        self.max_episode_steps = int(len(self.target_points) * 2) 
        self.current_point_idx = 0
        self.episode_step = 0

        utils.visited_surface.clear()
        print(f"[Reset] Cleared visited_surface, total target points: {len(self.target_points)}")

        self.episode_id += 1 
        return super().reset()
