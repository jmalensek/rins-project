#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import time
import math
import subprocess

from robot_commander import RobotCommander


class greet_people(Node):

    def __init__(self, rc):
        super().__init__('greet_people')

        self.rc = rc

        # Subscribe to face detections
        self.create_subscription(PoseStamped, '/face_detections', self.detections_callback, 10)

        # Start greeting only after two /finished triggers
        self.create_subscription(Bool, '/finished', self.finished_callback, 10)

        self.queue = []
        self.greet_started = False
        self.processing = False
        self.finished = False

        self.finished_true_count = 0

        self.n_persons = 3
        self.stop_distance = 0.65

        self.get_logger().info("Node started. Waiting for people detections...")

        self.text = ["Smile", "Hair", "Eyes"]
        self.person_ix = 0



    # make sure rings and faces are both finished
    def finished_callback(self, msg: Bool):
        self.finished_true_count += 1
        self.get_logger().info(f"Received /finished=True trigger ({self.finished_true_count}/2)")
        self.maybe_start_greeting()


    # save people detections coords
    def detections_callback(self, msg):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        self.get_logger().info(f"Received face detection at ({x:.2f}, {y:.2f})")
        self.queue.append((x,y))
        self.maybe_start_greeting()


    # check if everything is ok to start
    def maybe_start_greeting(self):
        if self.greet_started:
            return
        if len(self.queue) < self.n_persons or self.finished_true_count < 2:
            return
        if not self.rc.initial_pose_received:
            self.get_logger().warn(
                f"Have {len(self.queue)} detections and {self.finished_true_count} /finished triggers; waiting for first amcl_pose before greeting")
            return

        self.greet_started = True


    def process_pending_greetings(self):
        if self.finished or not self.greet_started or self.processing:
            return

        self.processing = True
        try:
            self.greet()
            self.finished = True
            self.get_logger().info(f"Visited all {self.n_persons} faces. Shutting down.")
            rclpy.shutdown()
        finally:
            self.processing = False


    def greet(self):
        for x,y in self.queue[:self.n_persons]:
            self.greet_person(x, y)


    def greet_person(self, x, y):
        stop_distance = self.stop_distance

        if not self.rc.initial_pose_received or self.rc.current_pose is None:
            self.get_logger().warn("Robot pose not available yet (waiting for amcl_pose)")
            return
        
        # Get robot current position
        x_robot = self.rc.current_pose.pose.position.x
        y_robot = self.rc.current_pose.pose.position.y
        
        # Calculate direction
        dx = x - x_robot
        dy = y - y_robot
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:
            ux, uy = 0, 0
        else:
            ux = dx / dist
            uy = dy / dist
        
        # Goal: stop_distance away from person
        goal_x = x - ux * stop_distance
        goal_y = y - uy * stop_distance
        yaw = math.atan2(dy, dx)
        
        # Create and send goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation = self.rc.YawToQuaternion(yaw)
        
        self.get_logger().info(f"Greeting face at ({x:.2f}, {y:.2f})")
        self.rc.goToPose(goal_pose)
        
        # Wait for completion
        while not self.rc.isTaskComplete():
            time.sleep(0.5)

        # say something
        self.say(f"Hello person, I like your {self.text[self.person_ix]}!")
        time.sleep(4)
        self.person_ix += 1


    def say(self, text):
        """Use system spd-say for text-to-speech"""
        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")
        


def main():
    rclpy.init(args=None)
    rc = RobotCommander()

    rc.waitUntilNav2Active()

    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    if rc.is_docked:
        rc.undock()

    node = greet_people(rc)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        rclpy.spin_once(rc, timeout_sec=0.1)
        node.process_pending_greetings()

    node.destroy_node()
    rc.destroyNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
