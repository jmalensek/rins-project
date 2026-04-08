#!/usr/bin/env python3


# HAVE TO DETECT TWO RINGS AND THEIR COLOR
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import subprocess

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point

from std_msgs.msg import Bool


class detect_rings(Node):

    def __init__(self):
        super().__init__('detect_rings')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
        ])

        marker_topic = "/rings_marker"

        self.detection_color = (0,0,255)  # question about that
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.scan = None

        self.detected_colors = set()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.RELIABLE)
        self.finished_pub = self.create_publisher(Bool, "/finished", 10)

        self.rings = []
        # clustering
        self.ring_clusters = []
        self.cluster_threshold = 0.20  # 20 centimers

        self.min_detections = 5  # min hits before publishing a marker

        self.ema_alpha = 0.1  # EMA smoothing factor for centroid

        self.create_timer(1.0, self.publish_clusters)
       

        self.get_logger().info(f"Node has been initialized! Will publish ring markers to {marker_topic}.")

    def say(self, text):
        """Use system spd-say for text-to-speech"""
        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")

    def rgb_callback(self, data):
        self.rings = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            #self.get_logger().info(f"Running Hough Circle detection...")

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 1)

            # detectinh ONLY on upper third :)
            upper_h = gray.shape[0] // 2
            gray_upper = gray[:upper_h, :]

            cv2.imshow("gray", gray_upper)
            key = cv2.waitKey(1)
            if key == 27:
                print("exiting")
                exit()

            circles = cv2.HoughCircles(gray_upper, cv2.HOUGH_GRADIENT, dp=1.0, minDist=50,
                                     param1=30, param2=30, minRadius=3, maxRadius=200)  # params could be changed
# maybe change param2 si it would depend also on radius? smaller radius -> smaller threshold for detection?
            if circles is not None:
                circles_draw = np.uint16(np.around(circles[0, :]))
                circles_proc = np.around(circles[0, :]).astype(np.int32)

                accepted = []

                for c_proc, c_draw in zip(circles_proc, circles_draw):
                    cx, cy, r = int(c_proc[0]), int(c_proc[1]), int(c_proc[2])
                    draw_cx, draw_cy, draw_r = int(c_draw[0]), int(c_draw[1]), int(c_draw[2])
                    
                    '''
                    duplicate = False
                    for ax, ay, ar in accepted:
                        if np.hypot(cx - ax, cy - ay) < max(8, 0.5 * min(r, ar)) and abs(int(r) - int(ar)) < max(5, 0.3 * min(r, ar)):
                            duplicate = True
                            break
                    if duplicate:
                        continue
                        '''

                    accepted.append((cx, cy, r))

                    #self.get_logger().info("Ring detected!")

                    # color detection
                    color = self.detect_ring_color(cv_image, cx, cy, r)



                    # Draw circle
                    cv2.circle(cv_image, (draw_cx, draw_cy), draw_r, self.detection_color, 2)

                    # draw center
                    cv2.circle(cv_image, (draw_cx, draw_cy), 3, (0, 0, 255), -1)

                    if color:
                        #self.get_logger().info(f"Ring colour: {color['name']} | LAB: {[round(v,1) for v in color['lab']]}")
                        # Label on image
                        cv2.putText(cv_image, color['name'], (cx - r, cy - r - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    self.rings.append((cx, cy, r, color))

            cv2.imshow("rings", cv_image)
            key = cv2.waitKey(1)
            if key == 27:
                print("exiting")
                exit()

        except CvBridgeError as e:
            print(e)



    
    def pointcloud_callback(self, data):

        # get point cloud attributes
        height = data.height
        width = data.width

        # get 3-channel representation of the point cloud in numpy format once
        a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        a = a.reshape((height, width, 3))

        source_frame = data.header.frame_id if data.header.frame_id else "base_link"

        # iterate over ring coordinates
        for i, (x, y, r, color) in enumerate(self.rings):
            # maybe needed a bounds check

            if not (0 <= x < width and 0 <= y < height):
                continue

            radius_px = max(4, int(0.9 * r))

            ys_off, xs_off = np.mgrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
            within_circle = (xs_off**2 + ys_off**2) <= radius_px**2

            abs_xs = x + xs_off[within_circle]
            abs_ys = y + ys_off[within_circle]
            dist_from_center = np.sqrt(xs_off[within_circle]**2 + ys_off[within_circle]**2)

            valid = (abs_xs >= 0) & (abs_xs < width) & (abs_ys >= 0) & (abs_ys < height)
            abs_xs = abs_xs[valid]
            abs_ys = abs_ys[valid]
            dist_from_center = dist_from_center[valid]


            if len(abs_xs) < 10:
                continue

            disc_points = a[abs_ys, abs_xs, :]

            # Remove only NaN — keep inf for now, it carries meaning
            nan_mask = ~np.isnan(disc_points).any(axis=1)
            disc_points_with_inf = disc_points[nan_mask]
            dist_from_center = dist_from_center[nan_mask]

            if len(disc_points_with_inf) < 10:
                continue

            # Count how many center pixels are inf — these are looking through the hole
            center_mask = dist_from_center <= (radius_px * 0.5)
            center_points = disc_points_with_inf[center_mask]
            center_inf_ratio = np.sum(~np.isfinite(center_points[:, 2])) / max(len(center_points), 1)

            # Count how many edge pixels are finite — these should be the ring material
            edge_mask = dist_from_center > (radius_px * 0.5)
            edge_points_with_inf = disc_points_with_inf[edge_mask]
            edge_finite_ratio = np.sum(np.isfinite(edge_points_with_inf[:, 2])) / max(len(edge_points_with_inf), 1)

            # A real hollow ring: center is mostly inf (hole), edge is mostly finite (ring material)
            # A flat surface: center is finite, edge is finite → both ratios fail
            # A missing/far object: everything is inf → edge_finite_ratio fails
            if center_inf_ratio < 0.3 or edge_finite_ratio < 0.1:
                #self.get_logger().info(f"c_inf_rati: {center_inf_ratio}      tra drugi: {edge_finite_ratio}")

                # center_inf_ratio < 0.3 → center is not hollow enough → flat surface
                # edge_finite_ratio < 0.5 → ring material not reliably detected
                self.get_logger().debug(
                    f"Ring {i}: rejected — center_inf={center_inf_ratio:.2f}, edge_finite={edge_finite_ratio:.2f}"
                )
                continue

            # Now filter to only finite points for position estimate
            finite_mask = np.isfinite(disc_points_with_inf).all(axis=1)
            edge_points = disc_points_with_inf[edge_mask & finite_mask]

            #self.get_logger().info(f"Edge points")


            if len(edge_points) < 4:
                continue

            ring_point = np.median(edge_points, axis=0)

            # Transform point from point cloud frame to map
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = data.header.stamp
            point_stamped.point.x = float(ring_point[0])
            point_stamped.point.y = float(ring_point[1])
            point_stamped.point.z = float(ring_point[2])

            try:
                # Transform to map frame
                transform = self.tf_buffer.lookup_transform("map", source_frame, data.header.stamp)
                point_map = do_transform_point(point_stamped, transform)

                map_x = point_map.point.x
                map_y = point_map.point.y
                map_z = point_map.point.z

                self.add_to_clusters(map_x, map_y, map_z, color)
            except TransformException:
                self.get_logger().warn("map frame not available yet, skipping ring")


            
    def add_to_clusters(self, x, y, z, color):
        new_point = np.array([x, y, z])
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for cluster in self.ring_clusters:
            centroid = np.array(cluster["centroid"])
            # Match in XY to reduce depth-noise splitting of the same physical ring
            distance = np.linalg.norm(new_point[:2] - centroid[:2])

            time_dur = cluster["last_seen"] - now_sec
            
            self.get_logger().info(f"Distance {distance}.")


            if distance < self.cluster_threshold:
                cluster["count"] += 1
                cluster["last_seen"] = now_sec

                if cluster["count"] == 3 and not cluster.get("spoken", False):
                    name = (cluster.get("color") or {}).get("name", "unknown")
                    
                    self.say(f"{name} ring")
                    cluster["spoken"] = True

                old = np.array(cluster["centroid"])
                # each new detection nudges the centroid 10% toward it.
                # as cluster grows, the centroid slowly corrects toward the true position
                cluster["centroid"] = ((1 - self.ema_alpha) * old + self.ema_alpha * new_point).tolist()
                
                # update the color similarly to centroid
                if color and cluster.get("color"):
                    old_lab = np.array(cluster["color"]["lab"])
                    new_lab = np.array(color["lab"])
                    smoothed_lab = (1 - self.ema_alpha) * old_lab + self.ema_alpha * new_lab
                    L, A, B = smoothed_lab
                    cluster["color"] = {"lab": smoothed_lab.tolist(), "name": self.classify_lab(L, A, B)}

                return


        # no existing cluster close enough -> new one
        self.ring_clusters.append({
            "centroid": new_point.tolist(),
            "count": 1,
            "spoken": False,
            "color": color,
            "last_seen" : now_sec
        })
        self.get_logger().info(f"New cluster")


    def publish_clusters(self):
        self.get_logger().info(f"Checking {len(self.ring_clusters)} clusters for publishing.")
        for i, cluster in enumerate(self.ring_clusters):
            if cluster["count"] < self.min_detections:
                continue

            self.get_logger().info(f"Publishing marker for cluster {i} at {cluster['centroid']} with count {cluster['count']}")

            marker = Marker()

            marker.header.frame_id = "map"

            marker.type = 2  # could change
            marker.id = i  # unique id per ring

            cx, cy, cz = cluster["centroid"]

            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz

            marker.pose.orientation.w = 1.0

            marker.header.stamp = self.get_clock().now().to_msg()

            # Set the scale of the marker
            scale = 0.25  # 15 cm
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Set the color
            color = cluster.get("color")
            if color:
                r_val, g_val, b_val = self.lab_to_marker_rgb(color["lab"])
            else:
                r_val, g_val, b_val = 0.0, 1.0, 1.0 

            marker.color.r = r_val
            marker.color.g = g_val
            marker.color.b = b_val
            marker.color.a = 1.0

            self.marker_pub.publish(marker)
            
            name = color["name"] if color else "unknown"
            if name not in self.detected_colors:
                self.detected_colors.add(name)

            if len(self.detected_colors) >= 2:
                self.get_logger().info(f"Detected 2 rings with colors: {', '.join(self.detected_colors)}. Publishing finished signal.")
                self.finished_pub.publish(Bool(data=True))
                self.get_logger().info("Done. Published /finished=True. Shutting down node.")

                # Optionally, stop further processing or exit
                rclpy.shutdown()
    def detect_ring_color(self, cv_image, cx, cy, r):
        h_img, w_img = cv_image.shape[:2]

        x0 = max(0, int(cx - r))
        x1 = min(w_img, int(cx + r) + 1)
        y0 = max(0, int(cy - r))
        y1 = min(h_img, int(cy + r) + 1)

        roi = cv_image[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        img_float = roi.astype(np.float32) / 255.0
        lab_roi = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
        pixels = lab_roi.reshape(-1, 3)

        if pixels.shape[0] < 20:
            return None

        labels = [self.classify_lab(L, A, B) for L, A, B in pixels]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1


        # so we have a bounding box and we count the colored pixels by color
        # we suspect,. that over 50% of pixel are background
        # we take the second most dominant color, thats not too similar to the background
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        dominant_name, dominant_count = sorted_counts[0]
        dominant_ratio = dominant_count / len(labels)

        class_medians = {}
        for class_name, _ in sorted_counts:
            class_pixels = np.array([pix for pix, label in zip(pixels, labels) if label == class_name])
            if class_pixels.size == 0:
                continue
            class_medians[class_name] = np.median(class_pixels, axis=0)

        selected_name = dominant_name

        # If dominant class is likely background, pick the strongest alternative
        # that is not too close to background color in LAB space.
        if dominant_ratio > 0.5:
            dominant_lab = class_medians.get(dominant_name)
            min_lab_distance = 20.0
            min_secondary_ratio = 0.05

            for class_name, class_count in sorted_counts[1:]:
                candidate_ratio = class_count / len(labels)
                if candidate_ratio < min_secondary_ratio:
                    continue

                candidate_lab = class_medians.get(class_name)
                if dominant_lab is None or candidate_lab is None:
                    continue

                lab_distance = np.linalg.norm(candidate_lab - dominant_lab)
                if lab_distance >= min_lab_distance:
                    selected_name = class_name
                    break

        selected_lab = class_medians.get(selected_name)
        if selected_lab is None:
            return None

        return {
            "lab": selected_lab.tolist(),
            "name": selected_name
        }


    def classify_lab(self, L, A, B):
        # Chroma = distance from the grey axis in the AB plane
        # Low chroma = achromatic (black / grey / white)
        chroma = np.sqrt(A ** 2 + B ** 2)

        if chroma < 15:
            # Achromatic — classify purely by lightness
            if L < 25:      return "black"
            elif L < 70:    return "grey"
            else:           return "white"

        # Chromatic — hue angle in LAB (atan2 of B over A)
        hue = np.degrees(np.arctan2(B, A)) % 360

        if hue < 15 or hue >= 345:     return "red"
        elif hue < 45:                  return "orange"
        elif hue < 75:                  return "yellow"
        elif hue < 150:                 return "green"
        elif hue < 195:                 return "cyan"
        elif hue < 255:                 return "blue"
        elif hue < 285:                 return "purple"
        else:                           return "magenta"


    def lab_to_marker_rgb(self, lab):
        # Convert LAB centroid back to BGR, then return normalised RGB for RViz marker
        lab_pixel = np.array(lab, dtype=np.float32).reshape(1, 1, 3)
        bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)
        bgr = np.clip(bgr, 0, 1)
        b, g, r = bgr[0, 0]
        return float(r), float(g), float(b)
            

def main():
    print('Ring detection node starting.')

    rclpy.init(args=None)
    node = detect_rings()
    rclpy.spin(node)
    node.destroy_node()
        
    try:    
        rclpy.shutdown()
    except Exception as e:
        print(f"Error during shutdown: {e}")

if __name__ == '__main__':
    main()
