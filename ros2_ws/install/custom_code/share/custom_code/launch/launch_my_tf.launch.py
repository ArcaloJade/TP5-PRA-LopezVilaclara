#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('custom_code')
    # RViz configuration (can reuse the one from slam_toolbox setup)
    rviz_config_file = LaunchConfiguration('rviz_config', 
                                           default=os.path.join(pkg_share, 'rviz', 'tf.rviz'))

    # Our Python SLAM Node
    python_custom_code = Node(
        package='custom_code',
        executable='my_tf',
        name='my_tf',
        output='screen',
        parameters=[]
    )
    
    # RViz2
    start_rviz2_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file])

    ld = LaunchDescription()
    ld.add_action(start_rviz2_cmd)
    
    # Add actions to LaunchDescription
    ld.add_action(python_custom_code)

    return ld 