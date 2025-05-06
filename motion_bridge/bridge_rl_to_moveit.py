import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from stable_baselines3 import PPO
from env.vision_surface_env import VisionSurfaceTracingEnv
import argparse
import torch

# Convert RL action (dx, dy, dz) into an absolute end-effector Pose
def action_to_pose(base_pose, action, scale=0.05):
    dx, dy, dz = action * scale
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.pose.position.x = base_pose[0] + dx
    pose.pose.position.y = base_pose[1] + dy
    pose.pose.position.z = base_pose[2] + dz
    pose.pose.orientation.w = 1.0
    return pose

def main(pcd_path):
    rospy.init_node("ppo_to_moveit_bridge")
    robot = RobotCommander()
    group = MoveGroupCommander("panda_arm")
    scene = PlanningSceneInterface()
    rospy.sleep(2.0)

 
    model = PPO.load("models/ppo_surface_1M.zip")

  
    env = VisionSurfaceTracingEnv(pcd_path=pcd_path, auto_switch_pcd=False)
    obs = env.reset()

 
    current_pose = group.get_current_pose().pose
    base_position = np.array([
        current_pose.position.x,
        current_pose.position.y,
        current_pose.position.z,
    ])

    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        target_pose = action_to_pose(base_position, action)

        group.set_pose_target(target_pose)
        group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

        obs, reward, done, info = env.step(action)
        base_position = np.array([
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
        ])

# Parse input PCD file path and run bridge
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_path", type=str, required=True, help="Path to the input .pcd file")
    args = parser.parse_args()

    main(args.pcd_path)
