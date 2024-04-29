from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    params = PathJoinSubstitution(
        [
            FindPackageShare("ros2_foundation_pose"),
            "config",
            "foundation_pose_config.yaml",
        ]
    )
    
    return LaunchDescription(
        [
            Node(
                package="ros2_foundation_pose",
                executable="foundation_pose_node.py",
                name="foundation_pose_node",
                parameters=[params],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },
            )
        ]
    )
