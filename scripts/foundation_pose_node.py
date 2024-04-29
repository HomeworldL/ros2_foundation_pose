#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterEvent, ParameterDescriptor, ParameterValue
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, TimeSynchronizer
# from vision_msgs
from ros2_foundation_pose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from ros2_foundation_pose.Utils import dr, draw_posed_3d_box, draw_xyz_axis
from ros2_foundation_pose.detector import Detector
import trimesh
import numpy as np
import time

class FoundationPoseNode(Node):
    def __init__(self):
        super().__init__('foundation_pose_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', '', ParameterDescriptor(description="Device")),
                ('obj_file', '', ParameterDescriptor(description="Object .obj file")),
                ('image_topic', '', ParameterDescriptor(description="Image topic")),
                ('depth_topic', '', ParameterDescriptor(description="Depth topic")),
                ('camera_info_topic', '', ParameterDescriptor(description="Camera info topic")),
                ('est_refine_iter', 5, ParameterDescriptor(description="Estimate refinement iterations")),
                ('track_refine_iter', 5, ParameterDescriptor(description="Track refinement iterations")),
            ]
        )
        
        self.device = self.get_parameter("device").value
        self.obj_file = self.get_parameter("obj_file").value
        self.est_refine_iter = self.get_parameter("est_refine_iter").value
        self.track_refine_iter = self.get_parameter("track_refine_iter").value
        self.mesh = trimesh.load(self.obj_file)
        self.mesh.apply_scale(1/1000)
        # self.mesh.show()
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2,3)
        
        # set_logging_format()
        # set_seed(0)
        debug_dir = './debug'
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=0, glctx=glctx)        
               
        self.detector = Detector(self.device)
        
               
        self.bridge = CvBridge()
        self.color_sub = Subscriber(self, Image, self.get_parameter("image_topic").value)
        self.depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value)
        self.ts = TimeSynchronizer([self.depth_sub, self.color_sub], 10)
        self.ts.registerCallback(self.image_callback)
        
        self.camera_info_received = False
        self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.get_parameter("camera_info_topic").value,
                self.camera_info_callback,
                1)        
        self.i = 0

    def image_callback(self, depth_msg, color_msg):
        if self.camera_info_received is False:
            return
        
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
        color_image = self.bridge.imgmsg_to_cv2(color_msg) # RGB
        
        # overlay_image = cv2.addWeighted(color_image, 0.7, depth_image, 0.3, 0)
        # cv2.imshow("Overlay Image", overlay_image)
        # cv2.waitKey(1)
        # max_depth = depth_image.max()
        # min_depth = depth_image.min()
        # mean_depth = depth_image.mean()
        # # Print max, min, and mean depth values
        # self.get_logger().info(f"Max Depth: {max_depth}")
        # self.get_logger().info(f"Min Depth: {min_depth}")
        # self.get_logger().info(f"Mean Depth: {mean_depth}")
        
        # # Convert depth image to pseudo-color for visualization
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_JET)

        # # Resize depth image to match the size of the color image
        # depth_colormap_resized = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))

        # # Overlay color image and depth pseudo-color image
        # overlay_image = cv2.addWeighted(color_image, 0.7, depth_colormap_resized, 0.3, 0)
        
        # concat_image = cv2.hconcat([color_image, depth_colormap_resized, overlay_image])
        # concat_image = cv2.resize(concat_image, (0, 0), fx=0.5, fy=0.5)

        # cv2.imshow("Overlay Image", concat_image)
        # # cv2.imwrite(f"results/align/{self.i:05d}.png", overlay_image)
        # cv2.waitKey(1)
        
        tstart = time.time()
        if self.i == 0:
            classes=["mustard bottle"]
            detections = self.detector.run_grounding_classes(
                image=cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR).copy(),
                classes=classes,
            )
            # self.get_logger().info(f"detections: {detections}")
            # self.get_logger().info(f"detections: {detections.xyxy}")
            
            # annotated_image = self.detector.get_detection_image(color_image.copy(), detections, classes)
            # cv2.imshow("annotated_image", annotated_image)
            # # self.image_tmp = color_image
            
            mask = np.zeros_like(depth_image)
            for det in detections.xyxy:
                xmin, ymin, xmax, ymax = det[:4].astype(int)
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255), -1)
            # cv2.imshow("Mask", mask)
            # cv2.imshow("color_image", color_image)
            pose = self.est.register(K=self.K, rgb=color_image.copy(), depth=depth_image.copy(), ob_mask=mask, iteration=self.est_refine_iter)
            
        else:
            pose = self.est.track_one(rgb=color_image.copy(), depth=depth_image.copy(), K=self.K, iteration=self.track_refine_iter)
            # pose = self.est.register(K=self.K, rgb=color_image.copy(), depth=depth_image.copy(), ob_mask=mask, iteration=self.est_refine_iter)
            # cv2.imshow("annotated_image", self.image_tmp)
        self.get_logger().info(f"time : {time.time()-tstart}")
        # exit(1)
        # cv2.imshow("Overlay Image", color_image)
        # cv2.waitKey(1)
        
        # self.get_logger().info(f"pose: {pose}")
        
        center_pose = pose@np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(self.K, img=color_image, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(color_image, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)
        
        self.i += 1
        
    def camera_info_callback(self, info: CameraInfo): 
        if self.camera_info_received is False:
            self.camera_d = info.d
            self.camera_height = info.height
            self.camera_width = info.width
            self.camera_k = info.k
            self.K = np.array(self.camera_k).reshape((3, 3))
            self.camera_r = info.r
            self.binning_x = info.binning_x
            self.get_logger().info(f'camera_height: {self.camera_height}\n \
                    camera_width: {self.camera_width}\n \
                    camera_d: {self.camera_d}\n \
                    camera_k: {self.camera_k}\n \
                    camera_r: {self.camera_r}\n \
                    binning_x: {self.binning_x}\n')
            self.camera_info_received = True
        else:
            return
        
def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
