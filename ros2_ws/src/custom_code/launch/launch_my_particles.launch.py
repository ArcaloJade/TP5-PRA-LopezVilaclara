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
                                           default=os.path.join(pkg_share, 'rviz', 'particulas.rviz'))
    
    # Declare argument
    num_particles_arg = DeclareLaunchArgument(
        "num_particles",
        default_value="1000",
        description="Number of particles for the filter"
    )

    likelihood = Node(
        package='custom_code',
        executable='likelihood',
        name='likelihood',
        output='screen',
        parameters=[]
    )

    localization = Node(
        package='custom_code',
        executable='localization',
        name='localization',
        output='screen',
        parameters=[{"num_particles": LaunchConfiguration("num_particles")}]
    )
    
    # RViz2
    start_rviz2_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file])

    ld = LaunchDescription()
    ld.add_action(num_particles_arg)
    ld.add_action(start_rviz2_cmd)
    
    # Add actions to LaunchDescription
    ld.add_action(likelihood)
        
    ld.add_action(localization)

    return ld 