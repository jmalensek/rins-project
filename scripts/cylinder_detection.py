#!/usr/bin/env python3

'''
Has to detect barrels -> count the number of barrels and the colors of barrels.

Finish the node when leaving the first room - at the same time send the barrel report 
to the node, that will make a final report

So this node should run passively

workflow:
1. detect a barrel and its color
    check this barrel was already detected
2. detect, whether the barrel is horizontal or vertical
    if it is horizontal, check whether it is leaking (alert)
    (leaking can be checked with the comparison of floor colors)
3. keep a count of the barrels and their color and orientation
4. for each barrel put a marker of the right color, so if we detect it again, we dont count it again




'''
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import subprocess

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point

from std_msgs.msg import Bool, String

try:
    import pcl                          # python-pcl 
    HAS_PCL = True
except ImportError:
    HAS_PCL = False
 
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

# Helper dataclass-like dict keys
#   cluster = {
#       "centroid"  : [x, y, z],        # map frame
#       "axis"      : [ax, ay, az],      # cylinder axis unit vector
#       "radius"    : float,             # metres
#       "count"     : int,
#       "color"     : {"lab": [...], "name": str} | None,
#       "orientation": "vertical" | "horizontal",
#       "leaking"   : bool,
#       "last_seen" : float,             # nanoseconds / 1e9
#       "published" : bool,
#   }


class detect_barrels(Node):

    # FLOOR_LAB_REF = np.array([75.0, 0.0, 5.0]) # reference floor color - have to calibrate

    def __init__(self):
        super().__init__('detect_barrels')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
                # camera intrinsics (used for 3-D → pixel projection)
                # override via ros param or camera_info topic
                ('fx', 277.0),
                ('fy', 277.0),
                ('cx', 160.0),
                ('cy', 120.0),
                # RANSAC cylinder segmentation
                ('ransac_distance_threshold', 0.025),   # metres
                ('ransac_max_iterations', 500),
                ('cylinder_min_radius', 0.10),          # metres
                ('cylinder_max_radius', 0.45),          # metres
                # clustering
                ('cluster_threshold', 0.20),            # metres
                ('min_detections', 5),
                # leak detection
                ('leak_lab_distance_threshold', 18.0),  # ΔE in LAB space
                ('leak_sample_radius_px', 30),          # pixels around barrel
        ])


        marker_topic = "/barrels_marker"

        # self.detection_color = (0,0,255)  # question about that
        self.device = self.get_parameter('device').get_parameter_value().string_value

        # camera intrinsics (may be updated from /camera_info)
        self.fx = self.get_parameter('fx').get_parameter_value().double_value
        self.fy = self.get_parameter('fy').get_parameter_value().double_value
        self.cx = self.get_parameter('cx').get_parameter_value().double_value
        self.cy = self.get_parameter('cy').get_parameter_value().double_value
 
        # RANSAC params
        self.ransac_dist   = self.get_parameter('ransac_distance_threshold').get_parameter_value().double_value
        self.ransac_iters  = self.get_parameter('ransac_max_iterations').get_parameter_value().integer_value
        self.cyl_r_min     = self.get_parameter('cylinder_min_radius').get_parameter_value().double_value
        self.cyl_r_max     = self.get_parameter('cylinder_max_radius').get_parameter_value().double_value
 
        # clustering
        self.cluster_threshold = self.get_parameter('cluster_threshold').get_parameter_value().double_value
        self.min_detections    = self.get_parameter('min_detections').get_parameter_value().integer_value
 
        # leak detection
        self.leak_lab_thresh   = self.get_parameter('leak_lab_distance_threshold').get_parameter_value().double_value
        self.leak_sample_r     = self.get_parameter('leak_sample_radius_px').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.latest_rgb = None
        # self.scan = None

        self.detected_colors: dict = {} # name -> count

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
        self.cam_info_sub = self.create_subscription(CameraInfo, "/oakd/rgb/preview/camera_info", self.camera_info_callback, 10)

        # listen for "left room 1" signal so we can shut down gracefully
        #self.room_exit_sub = self.create_subscription(
        #    Bool, '/room1_exit', self.room_exit_callback, 10)
        
        self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.RELIABLE)

        self.finished_pub = self.create_publisher(Bool, "/finished", 10)

        self.report_pub = self.create_publisher(String, "/barrel_report", 10)

        # clustering
        self.barrels_clusters: list = []

        self.ema_alpha = 0.1  # EMA smoothing factor for centroid

        self.create_timer(1.0, self.publish_clusters)
       

        self.get_logger().info(f"Node has been initialized! Will publish barrel markers to {marker_topic}.")

        if not HAS_PCL:
            self.get_logger().warn(
                'python-pcl not found — cylinder segmentation disabled. '
                'Install with: pip install python-pcl')
        if not HAS_O3D:
            self.get_logger().warn(
                'open3d not found — using raw numpy fallback for downsampling.')






    def say(self, text):
        """Use system spd-say for text-to-speech"""
        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")


    
    def camera_info_callback(self, msg: CameraInfo):
        # update intrinsics from camera_info topic
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        # only need this once, destroy
        self.destroy_subscription(self.cam_info_sub)

        self.get_logger().info(f"Camera info received. Updated intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}.")



    def rgb_callback(self, data):
        # we keep the latest RGB frame for color sampling when we get a pointcloud detection - we dont want to do heavy processing on every frame, only when we have a candidate pointcloud cluster that might be a barrel
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")

        # TO DO: barrel detection: here we also call for orientation and color detection

        # tukaj imamo tudi poslušalca za izhod iz prve sobe, ko pride do tega, shuttamo down ta node 
        # in publishamo /finished=True, da lahko drugi node naredi končno poročilo o barvah in številu sodov

    
    def pointcloud_callback(self, data):

        """
        Main perception pipeline:
          raw points → floor removal → downsample → normal estimation
          → RANSAC cylinder fit → 3-D centroid → colour → cluster
        """

        if self.latest_rgb is None:
            self.get_logger().warn("No RGB image received yet, skipping pointcloud processing.")
            return
        
        # 1. read raw points
        gen = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.array(list(gen), dtype=np.float32)
        if pts.shape[0] < 50:
            return  # not enough points to process
        
        # 2. transform to map frame
        pts_map = self._transform_points_to_map(pts, data.header)
        if pts_map is None:
            return  # transform failed, skip this frame
        
        # 3. floor removal (fast RANSAC plane fit)
        pts_no_floor = self._remove_floor(pts_map)
        if pts_no_floor is None or len(pts_no_floor) < 30:
            return # not enough points left after floor removal
        
        # 4. downsample (voxel grid)
        pts_down = self._downsample(pts_no_floor, voxel=0.02)

        # 5. cylinder segmentation (RANSAC)
        cylinders = self._ransac_cylinder(pts_down)


        # 6. per-cylinder processing: color, orientation, leak, clustering
        for cyl in cylinders:
            cx3, cy3, cz3 = cyl["centroid"]
            axis = cyl["axis"]
            radius = cyl["radius"]

            color = self.detect_barrel_color(self.latest_rgb, cx3, cy3, cz3)
            orientation = self.detect_orientation(axis)
            leaking = False
            if orientation == "horizontal":
                leaking = self.check_leak(self.latest_rgb, cx3, cy3, cz3)
                if leaking:
                    color_name = (color or {}).get("name", "unknown")
                    self.get_logger().warn(f"Leak detected on {color_name} barrel at ({cx3:.2f}, {cy3:.2f}, {cz3:.2f})!")
                    self.say(f"Warning! Warning! Iuiuiuiuiuiuiu! Leak detected on {color_name} barrel!")

            self.add_to_clusters(cx3, cy3, cz3, color=color, axis=axis, radius=radius, orientation=orientation, leaking=leaking)

        # TO DO: pointcloud conversion to map coordinates and clustering

    # pointcloud helpers

    def _transform_points_to_map(self, pts:np.ndarray, header) -> np.ndarray | None:
        """
        Transform an (N,3) array from the sensor frame to the map frame.
        Falls back to returning the original array when TF is unavailable.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', header.frame_id, rclpy.time.Time())
        except TransformException:
            # TF not ready yet — use sensor-frame coordinates for now
            return pts
 
        # Build rotation matrix from quaternion
        q = transform.transform.rotation
        t = transform.transform.translation
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z])
        return (R @ pts.T).T + translation
    
    @staticmethod
    def _quat_to_rot(x, y, z, w) -> np.ndarray:
        """Quaternion → 3×3 rotation matrix."""
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
 
    def _remove_floor(self, pts: np.ndarray) -> np.ndarray:
        """
        Remove the dominant horizontal plane (floor) with a quick
        height-threshold approach.  Returns points above the floor.
        If open3d is available, uses its RANSAC plane segmentation for
        robustness on slopes.
        """
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            _, inliers = pcd.segment_plane(
                distance_threshold=0.03, ransac_n=3, num_iterations=200)
            mask = np.ones(len(pts), dtype=bool)
            mask[inliers] = False
            return pts[mask]
 
        # Fallback: remove points within 5 cm of the lowest Z
        floor_z = np.percentile(pts[:, 2], 5)
        return pts[pts[:, 2] > floor_z + 0.05]
 
    def _voxel_downsample(self, pts: np.ndarray, voxel: float = 0.02) -> np.ndarray:
        """Voxel-grid downsampling.  Uses open3d when available."""
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            return np.asarray(pcd.points, dtype=np.float32)
 
        # Fallback: random subsample to at most 1000 points
        if len(pts) > 1000:
            idx = np.random.choice(len(pts), 1000, replace=False)
            return pts[idx]
        return pts
 
    def _estimate_normals(self, pts: np.ndarray) -> np.ndarray:
        """
        Estimate per-point normals.  Returns (N,3) array of unit normals.
        Uses open3d when available, else a simple PCA neighbourhood approach.
        """
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.05, max_nn=30))
            pcd.orient_normals_towards_camera_location(np.array([0., 0., 5.]))
            return np.asarray(pcd.normals, dtype=np.float32)
 
        # Fallback: PCA over a fixed neighbourhood (slow but dependency-free)
        from scipy.spatial import KDTree   # lazy import
        tree  = KDTree(pts)
        normals = np.zeros_like(pts)
        for i, p in enumerate(pts):
            _, idx = tree.query(p, k=min(15, len(pts)))
            neighbours = pts[idx]
            cov = np.cov((neighbours - neighbours.mean(0)).T)
            _, vecs = np.linalg.eigh(cov)
            normals[i] = vecs[:, 0]        # eigenvector with smallest eigenvalue
        # Flip toward +Z
        flip = normals[:, 2] < 0
        normals[flip] *= -1
        return normals
 
    def _segment_cylinders(self, pts: np.ndarray) -> list:
        """
        Iteratively fit cylinder models to the point cloud using RANSAC.
        Returns a list of dicts with keys: centroid, axis, radius, inlier_pts.
 
        Uses python-pcl when available; falls back to a pure-numpy RANSAC
        implementation.
        """
        remaining = pts.copy()
        results   = []
        max_cylinders = 6   # don't look for more than this per frame
 
        for _ in range(max_cylinders):
            if len(remaining) < 30:
                break
 
            if HAS_PCL:
                cyl = self._ransac_cylinder_pcl(remaining)
            else:
                cyl = self._ransac_cylinder_numpy(remaining)
 
            if cyl is None:
                break
 
            results.append(cyl)
 
            # Remove inliers so next iteration finds a different cylinder
            mask = np.ones(len(remaining), dtype=bool)
            mask[cyl['inlier_idx']] = False
            remaining = remaining[mask]
 
        return results
 
    # ── python-pcl path ──────────────────────────────────────────────────────
 
    def _ransac_cylinder_pcl(self, pts: np.ndarray) -> dict | None:
        """Fit one cylinder via python-pcl SACSegmentationFromNormals."""
        cloud = pcl.PointCloud(pts[:, :3])
 
        # Normal estimation
        ne    = cloud.make_NormalEstimation()
        tree  = cloud.make_kdtree()
        ne.set_SearchMethod(tree)
        ne.set_KSearch(30)
        cloud_normals = ne.compute()
 
        # Segmentation
        seg = cloud.make_segmenter_normals(ksearch=30)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_CYLINDER)
        seg.set_normal_distance_weight(0.1)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(self.ransac_iters)
        seg.set_distance_threshold(self.ransac_dist)
        seg.set_radius_limits(self.cyl_r_min, self.cyl_r_max)
 
        indices, coefficients = seg.segment()
        if len(indices) < 20:
            return None
 
        # coefficients = [point_on_axis x,y,z | axis x,y,z | radius]
        axis   = np.array(coefficients[3:6])
        axis  /= np.linalg.norm(axis) + 1e-9
        radius = coefficients[6]
 
        inlier_pts = pts[indices]
        centroid   = inlier_pts.mean(axis=0)
 
        return {
            'centroid'   : centroid.tolist(),
            'axis'       : axis.tolist(),
            'radius'     : float(radius),
            'inlier_idx' : indices,
            'inlier_pts' : inlier_pts,
        }
 
    # ── pure-numpy fallback path ─────────────────────────────────────────────
 
    def _ransac_cylinder_numpy(self, pts: np.ndarray) -> dict | None:
        """
        Minimal RANSAC cylinder segmentation without external PCL.
 
        A cylinder in 3D is parameterised by:
          - a point P on the axis
          - a unit direction vector D
          - a radius r
 
        In each RANSAC iteration we:
          1. Sample 2 points and use their normals to estimate D.
          2. Project all points onto the plane ⊥ D and fit a circle (another
             2-point sample + radius from centroid distance).
          3. Count inliers (points whose distance to the cylinder surface ≤ threshold).
        """
        normals = self._estimate_normals(pts)
        N       = len(pts)
        best_inliers = []
        best_model   = None
 
        rng = np.random.default_rng(42)
 
        for _ in range(self.ransac_iters):
            # Sample 2 points
            i, j = rng.choice(N, 2, replace=False)
 
            # Estimate axis direction from cross-product of normals
            n1, n2 = normals[i], normals[j]
            axis   = np.cross(n1, n2)
            norm   = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis /= norm
 
            # Project all points onto the plane perpendicular to axis
            dot    = (pts - pts[i]) @ axis            # (N,)
            proj   = pts - np.outer(dot, axis)         # (N,3) projected points
 
            # Estimate centre and radius from the two sampled projected points
            p1p = proj[i]
            p2p = proj[j]
            centre_2d = (p1p + p2p) / 2.0
            r_est     = np.linalg.norm(p1p - p2p) / 2.0
 
            if r_est < self.cyl_r_min or r_est > self.cyl_r_max:
                continue
 
            # Distances from each projected point to the estimated axis centre
            dists  = np.linalg.norm(proj - centre_2d, axis=1)
            inlier_mask = np.abs(dists - r_est) < self.ransac_dist
 
            if inlier_mask.sum() > len(best_inliers):
                best_inliers = np.where(inlier_mask)[0]
                best_model   = (centre_2d, axis, r_est)
 
        if len(best_inliers) < 20 or best_model is None:
            return None
 
        _, axis, radius = best_model
        inlier_pts = pts[best_inliers]
        centroid   = inlier_pts.mean(axis=0)
 
        return {
            'centroid'   : centroid.tolist(),
            'axis'       : axis.tolist(),
            'radius'     : float(radius),
            'inlier_idx' : best_inliers.tolist(),
            'inlier_pts' : inlier_pts,
        }
    

    def detect_orientation(self, axis: list | np.ndarray) -> str:
        """
        Determine whether a barrel is standing upright (vertical) or
        lying on its side (horizontal).
 
        The cylinder axis returned by RANSAC can point in either direction
        along the axis (the sign is arbitrary), so we compare against the
        absolute dot-product with the world Z vector.
 
        Returns:
            'vertical'   — barrel standing upright  (tilt ≤ 45°)
            'horizontal' — barrel on its side        (tilt > 45°)
        """
        axis = np.array(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return 'vertical'   # degenerate — assume upright
        axis /= norm
 
        world_z = np.array([0.0, 0.0, 1.0])
 
        # Use |dot| because RANSAC axis direction is arbitrary
        cos_angle = float(np.abs(np.dot(axis, world_z)))
        cos_angle = np.clip(cos_angle, 0.0, 1.0)
        tilt_deg  = np.degrees(np.arccos(cos_angle))
 
        self.get_logger().debug(f'Cylinder tilt from vertical: {tilt_deg:.1f}°')
 
        return 'vertical' if tilt_deg <= 45.0 else 'horizontal'
 
    # ─────────────────────────────────────────────────────────────────────────
    # Leak detection
    # ─────────────────────────────────────────────────────────────────────────
 
    def check_leak(
        self,
        cv_image: np.ndarray,
        x: float,
        y: float,
        z: float,
    ) -> bool:
        """
        Detect liquid leaking from a horizontal barrel by comparing the colour
        of the floor patches around the barrel to a reference floor colour.
 
        Strategy:
          1. Project barrel 3D position to pixel (u, v).
          2. Sample four small patches just outside the barrel footprint
             (above, below, left, right) in the image.
          3. Convert each patch to LAB and compute ΔE (Euclidean distance in
             LAB space) against self.FLOOR_LAB_REF.
          4. If any patch deviates by more than self.leak_lab_thresh, a liquid
             is assumed to be present → return True.
 
        Returns:
            True  — leak likely
            False — floor looks normal
        """
        if cv_image is None or z <= 0:
            return False
 
        h, w = cv_image.shape[:2]
 
        # Project barrel centre to 2D
        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)
 
        r = self.leak_sample_r    # pixel radius for sampling around the barrel
        p = 12                    # patch half-size
 
        # Sample locations: below barrel in image (gravity pulls liquid down)
        sample_offsets = [
            (0,   r),    # below
            (0,  -r),    # above
            (-r,  0),    # left
            ( r,  0),    # right
        ]
 
        leak_detected = False
        for du, dv in sample_offsets:
            su, sv = u + du, v + dv
            u0 = max(0, su - p);  u1 = min(w, su + p)
            v0 = max(0, sv - p);  v1 = min(h, sv + p)
            if u1 <= u0 or v1 <= v0:
                continue
 
            patch_bgr = cv_image[v0:v1, u0:u1]
            if patch_bgr.size == 0:
                continue
 
            patch_f32  = patch_bgr.astype(np.float32) / 255.0
            patch_lab  = cv2.cvtColor(patch_f32, cv2.COLOR_BGR2Lab)
            mean_lab   = patch_lab.mean(axis=(0, 1))
 
            delta_e = float(np.linalg.norm(mean_lab - self.FLOOR_LAB_REF))
            self.get_logger().debug(
                f'Leak check patch ({du},{dv}): ΔE = {delta_e:.1f}')
 
            if delta_e > self.leak_lab_thresh:
                leak_detected = True
                break
 
        return leak_detected
 
 

            
    def add_to_clusters(self, x:float, y:float, z:float, color:dict | None, axis:list | None=None, radius:float = 0.0, orientation:str ='vertical', leaking:bool=False):
        """
        Merge a new detection into the nearest existing cluster or start a new
        one.  Centroid and colour are smoothed with EMA.
        """

        new_point = np.array([x, y, z])
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for cluster in self.barrels_clusters:
            centroid = np.array(cluster["centroid"])

            distance = np.linalg.norm(new_point[:2] - centroid[:2])
            
            self.get_logger().info(f"Distance {distance}.")


            if distance < self.cluster_threshold:
                cluster["count"] += 1
                cluster["last_seen"] = now_sec

                if cluster["count"] == self.min_detections and not cluster["published"]:
                    name = (cluster.get("color") or {}).get("name", "unknown")

                    self.get_logger().info(f'Confirmed barrel: {name} {cluster["orientation"]}'
                                           f'{"(LEAKING)" if cluster["leaking"] else ""} at {cluster["centroid"]}')

            
                # EMA centroid update - 10%
                cluster['centroid'] = (
                    (1 - self.ema_alpha) * centroid
                    + self.ema_alpha * new_point
                ).tolist()
                
                # update the color similarly to centroid
                if color and cluster.get("color"):
                    old_lab = np.array(cluster["color"]["lab"])
                    new_lab = np.array(color["lab"])
                    smoothed_lab = (1 - self.ema_alpha) * old_lab + self.ema_alpha * new_lab
                    L, A, B = smoothed_lab
                    cluster["color"] = {"lab": smoothed_lab.tolist(), "name": self.classify_lab(L, A, B)}
                elif color:
                    cluster["color"] = color

                cluster['orientation'] = orientation
                if leaking:
                    cluster['leaking'] = True

                return


        # no existing cluster close enough -> new one
        self.ring_clusters.append({
            "centroid": new_point.tolist(),
            "axis": axis or [0.0, 0.0, 1.0],
            "radius": radius,
            "count": 1,
            "color": color,
            "orientation": orientation,
            "leaking": leaking,
            "last_seen" : now_sec,
            "published": False
        })
        self.get_logger().info(f"New cluster")


    def publish_clusters(self):
        self.get_logger().info(f"Checking {len(self.barrels_clusters)} clusters for publishing.")
        for i, cluster in enumerate(self.barrels_clusters):
            if cluster["count"] < self.min_detections:
                continue
            
            color = cluster.get("color")
            name = (color or {}).get("name", "unknown")
            self.get_logger().info(f"Publishing marker for cluster {i} at {cluster['centroid']} with count {cluster['count']} and color {name}.")

            self.detected_colors[name] = self.detected_colors.get(name, 0) + 1

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "barrels"
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.id = i

            cx, cy, cz = cluster["centroid"]

            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz
            marker.pose.orientation.w = 1.0

            r = max(float(cluster.get('radius', 0.2)), 0.1)
            
            marker.scale.x = r * 2
            marker.scale.y = r * 2
            marker.scale.z = 0.85

            # Set the color
            if color:
                r_val, g_val, b_val = self.lab_to_marker_rgb(color["lab"])
            else:
                r_val, g_val, b_val = 0.5, 0.5, 0.5

            marker.color.r = r_val
            marker.color.g = g_val
            marker.color.b = b_val
            marker.color.a = 0.85

            # Flash red if leaking
            if cluster.get('leaking'):
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
 
            marker.lifetime.sec = 5   # refresh every 5 s or disappear

            self.marker_pub.publish(marker)
            cluster["published"] = True

            self.get_logger().info(f"Published marker for cluster {i} at {cluster['centroid']} with color {name}.")



    def detect_barrel_color(self, cv_image: np.ndarray, x: float, y: float, z: float, patch_size:int=20) -> dict | None:
        # extract a small patch inside the rectangle and convert to LAB to get average color
        """
        Project the 3D centroid (x, y, z) in the *camera* frame to pixel
        coordinates, extract a square patch, convert to LAB, and classify.
 
        NOTE: x, y, z here are expected in the *camera optical frame*
        (x-right, y-down, z-forward).  If you have map-frame coordinates,
        you need to inverse-transform them first.
        """
        if cv_image is None or z <= 0:
            return None
 
        h, w = cv_image.shape[:2]
 
        # Project 3D → 2D pixel
        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)
 
        # Clamp patch to image bounds
        half  = patch_size // 2
        u0    = max(0, u - half)
        u1    = min(w, u + half)
        v0    = max(0, v - half)
        v1    = min(h, v + half)
 
        if u1 <= u0 or v1 <= v0:
            return None
 
        patch_bgr = cv_image[v0:v1, u0:u1]
        if patch_bgr.size == 0:
            return None
 
        # Convert to float32 [0,1] LAB
        patch_f32 = patch_bgr.astype(np.float32) / 255.0
        patch_lab = cv2.cvtColor(patch_f32, cv2.COLOR_BGR2Lab)
        mean_lab  = patch_lab.mean(axis=(0, 1))
 
        L, A, B = mean_lab
        name    = self.classify_lab(L, A, B)
 
        return {'lab': mean_lab.tolist(), 'name': name}
    




    def classify_lab(self, L, A, B):
        """
        Classify a LAB color as one of: red, green, blue, black, white, yellow.
        """
        chroma = np.sqrt(A ** 2 + B ** 2)
    
        # --- Achromatic colors (low chroma) ---
        if chroma < 15:
            if L < 20:
                return 'black'
            if L > 90:
                return 'white'
            return 'unknown'
 
        # Chromatic — hue angle in degrees [0, 360)
        hue = float(np.degrees(np.arctan2(B, A))) % 360.0
 
        if hue < 20 or hue >= 340:
            return 'red'
        if 20 <= hue < 50:
            return 'orange'
        if 50 <= hue < 80:
            return 'yellow'
        if 80 <= hue < 155:
            # distinguish green (positive B) from yellow-green / cyan
            if A < 10 and B > 0:
                return 'green'
            return 'green'
        if 155 <= hue < 265:
            return 'blue'
        if 265 <= hue < 340:
            return 'red'   # magenta / purple mapped to red for typical barrel sets
 
        return 'unknown'


    def lab_to_marker_rgb(self, lab):
        # Convert LAB centroid back to BGR, then return normalised RGB for RViz marker
        lab_pixel = np.array(lab, dtype=np.float32).reshape(1, 1, 3)
        bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)
        bgr = np.clip(bgr, 0, 1)
        b, g, r = bgr[0, 0]
        return float(r), float(g), float(b)
            



    def room_exit_callback(self, msg: Bool):
        """Called when the robot leaves room 1.  Publish report, then shutdown."""
        if not msg.data:
            return
 
        self.get_logger().info('Room 1 exit detected — publishing barrel report.')
        self.publish_final_report()
 
        # Signal other nodes
        fin = Bool()
        fin.data = True
        self.finished_pub.publish(fin)
 
        # Give other nodes a moment to receive the message
        self.create_timer(1.0, lambda: rclpy.shutdown())
 
    def publish_final_report(self):
        """
        Build and publish a human-readable summary of all detected barrels.
        """
        confirmed = [
            c for c in self.barrels_clusters
            if c['count'] >= self.min_detections
        ]
 
        total     = len(confirmed)
        vertical  = sum(1 for c in confirmed if c['orientation'] == 'vertical')
        horiz     = total - vertical
        leaking   = sum(1 for c in confirmed if c.get('leaking'))
 
        color_summary = ', '.join(
            f'{cnt} {col}' for col, cnt in self.detected_colors.items()
        ) or 'none'
 
        report_lines = [
            '=== Barrel Report ===',
            f'Total barrels : {total}',
            f'Vertical      : {vertical}',
            f'Horizontal    : {horiz}',
            f'Leaking       : {leaking}',
            f'Colors        : {color_summary}',
        ]
 
        for i, c in enumerate(confirmed):
            cx, cy, _ = c['centroid']
            cname     = (c.get('color') or {}).get('name', 'unknown')
            leak_flag = ' [LEAKING]' if c.get('leaking') else ''
            report_lines.append(
                f'  Barrel {i+1}: {cname} {c["orientation"]}{leak_flag} '
                f'@ ({cx:.2f}, {cy:.2f})')
 
        report_str = '\n'.join(report_lines)
        self.get_logger().info('\n' + report_str)
 
        msg      = String()
        msg.data = report_str
        self.report_pub.publish(msg)
 
        '''
        speech = (
            f'Room 1 complete. Found {total} barrel{"s" if total != 1 else ""}. '
            f'{color_summary}. '
            + (f'{leaking} leaking.' if leaking else 'No leaks detected.')
        )
        self.say(speech)
        '''







def main():
    print('Barrel detection node starting.')
    rclpy.init(args=None)
    node = detect_barrels()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_final_report()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f'Shutdown error: {e}')

if __name__ == '__main__':
    main()
