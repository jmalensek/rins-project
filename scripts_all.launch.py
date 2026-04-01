from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _script_node(executable: str, arg_name: str, namespace: LaunchConfiguration) -> Node:
    return Node(
        package='dis_tutorial3',
        executable=executable,
        name=executable.replace('.py', ''),
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration(arg_name)),
    )

# DeclareLaunchArgument('run_robot_explorer', default_value='true', description='Run robot_explorer.py'),
def generate_launch_description() -> LaunchDescription:
    namespace = LaunchConfiguration('namespace')

    args = [
        DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
        DeclareLaunchArgument('run_detect_people2', default_value='true', description='Run detect_people2.py'),
        DeclareLaunchArgument('run_detect_rings27', default_value='true', description='Run detect_rings27.py'),
        DeclareLaunchArgument('run_greet_people', default_value='true', description='Run greet_people_faster.py'),
    ]

    nodes = [
        _script_node('detect_people2.py', 'run_detect_people2', namespace),
        _script_node('detect_rings27.py', 'run_detect_rings27', namespace),
        _script_node('greet_people_faster.py', 'run_greet_people', namespace),
    ]

    return LaunchDescription(args + nodes)
