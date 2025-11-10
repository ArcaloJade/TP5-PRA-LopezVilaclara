#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration



def generate_launch_description():
    pkg_share = get_package_share_directory('custom_code')
    # RViz configuration (can reuse the one from slam_toolbox setup)
    rviz_config_file = LaunchConfiguration('rviz_config', 
                                           default=os.path.join(pkg_share, 'rviz', 'ekf.rviz'))

    # --- Define nodes ---
    feature_finder = Node(
        package='custom_code',
        executable='feature_finder',
        name='feature_finder',
        output='screen'
    )

    features = Node(
        package='custom_code',
        executable='features',
        name='features',
        output='screen'
    )

    ekf = Node(
        package='custom_code',
        executable='ekf',
        name='ekf',
        output='screen'
    )

    # --- Event handlers to enforce order ---
    # Start 'features' only after 'feature_finder' has started
    start_features_after_finder = RegisterEventHandler(
        OnProcessStart(
            target_action=feature_finder,
            on_start=[features]
        )
    )

    # Start 'ekf' only after 'features' has started
    start_ekf_after_features = RegisterEventHandler(
        OnProcessStart(
            target_action=features,
            on_start=[ekf]
        )
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
    
    # Add feature_finder first
    ld.add_action(feature_finder)

    # Add ordered starts
    ld.add_action(start_features_after_finder)
    ld.add_action(start_ekf_after_features)

    return ld 