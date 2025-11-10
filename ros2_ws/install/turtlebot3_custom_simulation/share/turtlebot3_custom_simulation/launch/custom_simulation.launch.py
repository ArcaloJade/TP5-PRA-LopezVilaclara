from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory 


def generate_launch_description():
    # Paths
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    world = os.path.join(
        get_package_share_directory('turtlebot3_custom_simulation'),
        'worlds',
        'slippery.world'
    )

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    return LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        spawn_turtlebot_cmd,
        robot_state_publisher_cmd,
        # Static transform between map and odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_map_to_odom',
            arguments=[x_pose, y_pose, '0', '0', '0', '0', 'map', 'calc_odom'],
            output='screen'
        ),
        Node(
            package='turtlebot3_custom_simulation',
            executable='turtlebot3_custom_simulation',
            name='turtlebot3_custom_simulation',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'odom_frame': 'calc_odom',
                'base_frame': 'calc_base_footprint',
                "joint_states_frame": "base_footprint",
                'wheels.separation': 0.160,
                'wheels.radius': 0.033,
                'initial_pose.x': 0.0,
                'initial_pose.y': 0.0,
                'initial_pose.yaw': 0.0,
            }]
        ),

        # Optionally start RViz with a config file
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', '/path/to/your/rviz/config.rviz'],
        #     output='screen'
        # ),
    ])