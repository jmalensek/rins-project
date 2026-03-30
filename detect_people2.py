#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool



class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		faces_topic = "/face_detections"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		# TF2 listener for transform base_link -> map
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# df for my detections
		#self.detections_df = pd.DataFrame(columns=['x', 'y', 'z', 'person_id'])
		self.detections = []    # [person_id][x,y,z]  2D array
		self.coords_published = False
		self.face_t = 0.7
		

		# subscribe to image + pointcloud
		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)


		self.model = YOLO("yolov8n.pt") # for people detection
		self.faces = []

		self.pose_pub = self.create_publisher(PoseStamped, faces_topic, 10)
		self.marker_pub = self.create_publisher(Marker, "/face_markers", 10)
		self.get_logger().info(f"Will publish face coordinates to {faces_topic}.")
		self.get_logger().info("Will publish person markers to /face_markers.")
		self.finished_pub = self.create_publisher(Bool, "/finished", 10)
		self.get_logger().info("Will publish finished trigger to /finished.")


	# detects faces
	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for result in res:
				if result.boxes is None or len(result.boxes) == 0:
					continue

				for box in result.boxes:
					bbox = box.xyxy[0]

					x = float(box.conf)
					if x < self.face_t:
						continue

					self.get_logger().info("Person has been detected!")

					# draw rectangle
					cv_image = cv2.rectangle(cv_image,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),self.detection_color,3)

					cx = int((bbox[0] + bbox[2]) / 2)
					cy = int((bbox[1] + bbox[3]) / 2)

					# draw the center of bounding box
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

					self.faces.append((cx, cy))
				
			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)
			

	def match_person(self, x, y, z):
		if z > 0.5:
			return  # not on a fence!

		n = len(self.detections)
		person_id = n  # default: assume new person

		for p_id in range(n):  # try to match with existing person
			person_detections = np.array(self.detections[p_id], dtype=float)
			xp, yp, zp = np.mean(person_detections, axis=0)

			d = np.linalg.norm([x-xp, y-yp, z-zp])
			if d < 0.5:
				person_id = p_id  # found a match
				self.get_logger().info(f"Matched person {person_id}")
				break

		# save
		is_new_person = person_id >= len(self.detections)
		if person_id >= len(self.detections):
			self.detections.append([])
			self.get_logger().info(f"New person found! (ID {person_id})")
		self.detections[person_id].append([x, y, z])

		if is_new_person:
			self.publish_person_marker(person_id, x, y, z)

		if len(self.detections) >= 3 and not self.coords_published:
			self.publish_people()


	def publish_person_marker(self, person_id, x, y, z):
		msg = Marker()
		msg.header.frame_id = "map"
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.ns = "detected_people"
		msg.id = int(person_id)
		msg.type = Marker.SPHERE
		msg.action = Marker.ADD

		msg.pose.position.x = float(x)
		msg.pose.position.y = float(y)
		msg.pose.position.z = float(z) + 0.25
		msg.pose.orientation.w = 1.0

		msg.scale.x = 0.25
		msg.scale.y = 0.25
		msg.scale.z = 0.25

		msg.color.r = 1.0
		msg.color.g = 0.2
		msg.color.b = 0.1
		msg.color.a = 0.9
		self.marker_pub.publish(msg)
		self.get_logger().info(f"Published marker for person {person_id}")



	def publish_people(self):
		if self.coords_published: return
		if len(self.detections) < 3:
			return
		if any(len(self.detections[person_id]) < 5 for person_id in range(3)):
			return
		self.coords_published = True

		for person_id in range(3):
			person_detections = np.array(self.detections[person_id], dtype=float)
			x, y, z = np.mean(person_detections, axis=0)

			msg = PoseStamped()
			msg.header.frame_id = "map"
			msg.header.stamp = self.get_clock().now().to_msg()

			msg.pose.position.x = x
			msg.pose.position.y = y
			msg.pose.position.z = z
			msg.pose.orientation.w = 1.0

			self.pose_pub.publish(msg)
			self.get_logger().info(f"Published person {person_id}")


		self.finished_pub.publish(Bool(data=True))
		self.get_logger().info("Done. Published /finished=True. Shutting down node.")
		rclpy.shutdown()


	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step		

		# get 3-channel representation of the point cloud in numpy format (once)
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))

		# lookup TF once per callback: base_link -> map
		try:
			transform = self.tf_buffer.lookup_transform("map", "base_link", data.header.stamp)
		except TransformException:
			self.get_logger().warn("map frame not available yet, skipping detections")
			return


		# iterate over face coordinates
		for x, y in self.faces:
			if x < 0 or x >= width or y < 0 or y >= height:
				continue

			# read center coordinates in base_link frame
			d = a[y, x, :]

			point_stamped = PointStamped()
			point_stamped.header.frame_id = "base_link"
			point_stamped.header.stamp = data.header.stamp
			point_stamped.point.x = float(d[0])
			point_stamped.point.y = float(d[1])
			point_stamped.point.z = float(d[2])

			# transform to map
			point_map = do_transform_point(point_stamped, transform)

			# store map coords in df
			x = float(point_map.point.x)
			y = float(point_map.point.y)
			z = float(point_map.point.z)

			self.match_person(x, y, z)




def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
