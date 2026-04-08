#! /usr/bin/env python3

import rclpy
import time
import math
import tf2_ros

from tf_transformations import euler_from_quaternion
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

map_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

class RobotExplorer(Node):

    def __init__(self, node_name='robot_explorer', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.scan_data = None
        self.map_data = None
        self.finished_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.finished_pub = self.create_publisher(Bool, "/finished", 10)

        # Subscribers
        self.create_subscription(LaserScan, 'scan_filtered', self._scanCallback, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, 'map', self._mapCallback, map_qos)
        self.create_subscription(Bool, '/finished', self._finishedCallback, 10)

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Initialisation successful
        self.get_logger().info(f"Robot explorer has been initialized!")

    # BASIC MOTORIC METHODS
    def move_straight(self, distance: float, speed: float = 0.2) -> None:

        """
        Moves the robot in a straight line for a specified distance at a specified speed.
        Args:
            distance(float): meters to move forward
            speed(float): linear speed in m/s (default: 0.2)

        Returns:
            None
        """

        if distance <= 0:
            self.get_logger().error("Distance must be positive.")
            return
        
        speed = abs(speed)
        duration = distance / speed

        twist_msg = TwistStamped()
        twist_msg.twist.linear.x = speed

        stop_msg = TwistStamped()

        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(twist_msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)

        for _ in range(3):
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(stop_msg)
            time.sleep(0.05)

    def turn(self, angle: float, angular_speed: float = 0.5):

        """
        Turns the robot by a specified angle at a specified angular speed.
        Args:
            angle(float): angle to turn in radians (positive for counterclockwise, negative for clockwise)
            angular_speed(float): angular speed in rad/s (default: 0.5)
        
        Returns:
            None
        """

        if angle is None:
            self.get_logger().warn("Turn angle is None; skipping turn command")
            return

        if angular_speed <= 0:
            self.get_logger().error("Angular speed must be positive")
            return
            
        angular_speed = abs(angular_speed)
        duration = abs(angle) / angular_speed

        twist_msg = TwistStamped()
        if angle > 0:
            twist_msg.twist.angular.z = angular_speed
        else:
            twist_msg.twist.angular.z = -angular_speed
            
        stop_msg = TwistStamped()

        start_time = self.get_clock().now()
        while(self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(twist_msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)

        for _ in range(3):
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(stop_msg)
            time.sleep(0.05)

    def go_to_pose(self, x: float, y: float, yaw: float = 0.0) -> bool:
        # Wait for Nav2 action server
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for navigate_to_pose action server...")

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.orientation = self._yaw_to_quaternion(yaw)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if self.goal_handle is None or not self.goal_handle.accepted:
            self.get_logger().warn(f"Goal rejected at ({x:.2f}, {y:.2f})")
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def cover_environment(self, waypoints: list[tuple[float, float]]):
        for x, y in waypoints:
            self.get_logger(). info(f"Navigating to waypoint ({x:.2f}, {y:.2f})")

            if not self.go_to_pose(x, y):
                self.get_logger().error(f"Failed to send goal to ({x:.2f}, {y:.2f})")
                continue
            
            if not self.wait_task_done(timeout_sec=360.0):
                self.get_logger().error(f"Failed to reach ({x:.2f}, {y:.2f}) within timeout")
            else:
                self.get_logger().info(f"Successfully reached ({x:.2f}, {y:.2f})")

            # Check if 2 finished signal received, terminate exploration, and publish signal
            if self.finished_count >= 2:
                self.get_logger().info("Received /finished=True trigger twice, stopping exploration, publishing /finished=True trigger.")
                self.finished_pub.publish(Bool(data=True))
                return
        
    # NAVIGATION HELPER METHODS
    def get_waypoints_from_map(self, step: float = 1.0) -> list[tuple[float, float]]:
        if self.map_data is None:
            self.get_logger().error("Map data not available")
            return []

        waypoints = []
        info = self.map_data.info
        stride = max(1, int(step / info.resolution))

        for my in range(0, info.height, stride):
            x_iter = range(0, info.width, stride)
            if (my // stride) % 2 == 1:
                x_iter = reversed(x_iter)
            
            for mx in x_iter:
                if self._cell(mx, my) == 0:
                    wx, wy = self._map_to_world(mx, my)
                    waypoints.append((wx, wy))

        self.get_logger().info(f"Generated {len(waypoints)} waypoints from map with step {step}m")
        return waypoints

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    def wait_for_scan_data(self, timeout_sec: float = 5.0) -> bool:
        
        """
        Waits until laser scan data is received or a timeout occurs.
        Args:
            timeout_sec(float): maximum time to wait for scan data in seconds (default: 5.0)
        Returns:
            bool: True if scan data was received, False if timeout occurred
        """
        start_time = self.get_clock().now()

        while rclpy.ok():
            if self.scan_data is not None:
                return True

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= timeout_sec:
                self.get_logger().warn(f"No scan data received within {timeout_sec:.1f}s")
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for_map_data(self, timeout_sec: float = 5.0) -> bool:
        start_time = self.get_clock().now()

        while rclpy.ok():
            if self.map_data is not None:
                return True

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= timeout_sec:
                self.get_logger().warn(f"No map data received within {timeout_sec:.1f}s")
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

    def _cell(self, mx: int, my: int) -> int:
        w = self.map_data.info.width
        return self.map_data.data[my * w + mx]
    
    def _world_to_map(self, x: float, y: float) -> tuple[int, int]:
        info = self.map_data.info
        ox = info.origin.position.x
        oy = info.origin.position.y
        res = info.resolution
        mx = int((x - ox) / res)
        my = int((y - oy) / res)
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    # UTILITY METHODS
    def wait_task_done(self, timeout_sec: float = 120.0) -> bool:
        start = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.2)
            if self.result_future.done():
                result = self.result_future.result()
                if result is None:
                    return False
                status = result.status
                return status == GoalStatus.STATUS_SUCCEEDED

            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn("Navigation timeout, canceling goal.")
                if self.goal_handle is not None:
                    cancel_future = self.goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future)
                return False
        return False

    def _map_to_world(self, mx: int, my: int) -> tuple[float, float]:
        info = self.map_data.info
        wx = info.origin.position.x + (mx + 0.5) * info.resolution
        wy = info.origin.position.y + (my + 0.5) * info.resolution
        return wx, wy

    # CALLBACKS
    def _scanCallback(self, msg: LaserScan) -> None:
        self.scan_data = msg

    def _mapCallback(self, msg: OccupancyGrid) -> None:
        self.map_data = msg

    def _finishedCallback(self, msg: Bool) -> None:
        if msg.data:
            self.finished_count += 1
            #self.get_logger().info(f"Received /finished=True trigger ({self.finished_true_count}/2)")

def main(args = None):

    rclpy.init(args=args)
    re = RobotExplorer()

    time.sleep(3.0)

    if not re.wait_for_map_data(timeout_sec=10.0):
        re.get_logger().error('Map was not received, aborting.')
        re.destroy_node()
        rclpy.shutdown()
        return

    # hardcoded waypoints for the specific map
    waypoints = [(2.212346248653394, 0.009680876809087835),
                (2.1833755802533554, -1.9678722468643208),
                (1.417301595011825, -2.666432722817975),
                (1.289321485429959, -3.4529791198569413),
                (0.12696714192880537, -3.6631788241089063),
                (0.03727206723370793, -1.5493085954305073),
                (-0.011229038107887649,-2.3238756805248095),
                (-1.0622132922149174, -2.411171268639826),
                (-1.2901576432001445, -1.089920196088445),
                (-1.7972970814236089, -1.0256533410565287),
                (-1.9365451987254536, 0.4943115005568502),
                (-1.1507406707967371, 1.1061146346431132),
                (0.3933188510497119, 1.2343684129661787),
                (1.261126302529096, 1.6370923481849204),
                (0.6003921547337148, 2.6967011040456237),
                (-2.4022780168470765, 2.4666737072366414),
                (-2.6826873711016357, 0.7170108058401835)]

    #waypoints = re.get_waypoints_from_map(step=1.0)
    #print(waypoints)
    re.cover_environment(waypoints)
    re.get_logger().info("Exploration complete, shutting down.")
    re.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
