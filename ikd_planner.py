#!/usr/bin/env python3

import os
import sys

import pickle
import time
import numpy as np
import utils as Utils
import cv2
from typing import List, Optional, Tuple

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path , Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray

from Grid import MapProcessor

import torch
import torch.nn as nn

from collections import deque
from ikd_utils import total_path_planner, edge_weight_calculation

import torchvision.transforms.functional as F
torch.set_float32_matmul_precision('high')


MAX_VEL = 1.0
MIN_VEL = -1.0
publish_itrs = 80
ackermann = False

class IKD():
    def __init__(self, ):
        #Robot Limits
        self.max_vel = MAX_VEL
        self.min_vel = MIN_VEL
        self.robot_length = 0.54

        self.mp = MapProcessor() 

        #General
        self.robot_pose: Optional[List] = None 
        self.prev_robot_pose_a: Optional[List] = None
        self.prev_robot_pose_m: Optional[List] = None
    
        #loop Specific
        self.gt_path: List = []
        self.pred_path_a: List = []
        self.pred_path_m: List = []
        self.no_iter: int = 0 
        self.init: int = 0

        self.gt_cmds: List = []
        
        # Circumvent GridMap being updated mid path plan algo
        self.is_planning = False
        self.map_hold = []

        # Used to build graphs out of current pose for path planning
        self.change_vals = []
        self.K = 10
        for i in range(-10, 11, self.K):
            self.change_vals.append(Utils.ackermann(0.1, i/100))

        rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, self.gridMap_callback, queue_size=5, buff_size=6*sys.getsizeof(np.zeros((360,360))))
        rospy.Subscriber("/dlio/odom_node/odom", Odometry, self.odom_cb, queue_size = 100, buff_size=100*sys.getsizeof(Odometry()))
        rospy.Subscriber("/cmd_vel1", Float32MultiArray, self.cmd_vel_cb, queue_size = 100, buff_size=100*8*sys.getsizeof(float()))

        self.path_pub_pred_a = rospy.Publisher("/paths_pred_ack", Path, queue_size = 100 )
        self.path_pub_gt = rospy.Publisher("/paths_gt", Path, queue_size = 100)
        self.path_pub_pred_m = rospy.Publisher("/paths_pred_ssl", Path, queue_size = 100)

    def gridMap_callback(self, gridmap_msg):

        # Flag is set to true whenever building graph for path planning to 
        # prevent the map from updating and the edge weights between
        # poses to not be based on the same gridmap
        if not self.is_planning:
            self.mp.update_map(gridmap_msg)
        else:
            self.map_hold.append(gridmap_msg)
            
    def odom_cb(self, odom_msg):

        self.robot_pose = Utils.odometry_to_particle(odom_msg)
    
    def cmd_vel_cb(self, cmd_vel_msg):

        self.gt_cmds.append([cmd_vel_msg.data[1], cmd_vel_msg.data[0]])
    
    def build_graph(self, steps = 2, final_x = 3.414258, final_y = 0.039283, patch_sum = False):

        """
        Builds the graph used in the path planning algorithm
        Uses values from Ackerman model to get updated poses
        Nodes are poses and edges are sum of traversibility map
        (another option for edges are difference between front and back wheels
        in the traversibility map)

        steps: number of poses to calculate graph for planning
        final_x : value of goal x coordinate 
        final_y: value of goal y coordinate
        patch_sum: boolean, whether to use sum of traversability footprint or 
                   wheel difference for edge weight

        Returns: dictionary as a weighted graph
        {1x6 tuple-pose : {1x6 tuple-pose : float-weight}}
        2 special keys: 'start', 'end'
        """
        # x, y, z, roll, pitch, yaw = curr_pose

        # flag to prevent gridMap_callback from updating the traversibility map
        # mid algo. When the flag is true the gridMap_callback method stores
        # the message in a list. At the end of the build_graph method if the 
        # list has length greater than 0 the update_map method is called and
        # the traversibility map is updated to the most recent map message data
        # the list is then cleared
        self.is_planning = True
        
        curr_pose = self.robot_pose.copy()

        weight_dict = {}
        poses = deque([tuple(curr_pose)])
        closest_pose, closest_dist = curr_pose.copy(), ((curr_pose[0]-final_x)**2 + (curr_pose[1]-final_y)**2)**(1/2)
        
        add_one = 0
        if steps > 1:
            add_one = 1


        # Graph is built to have the following dictionary structure
        # key : current pose - tuple
        # value : dictionary - key : estimated pose, tuple
        #                      value : edge weight, float 
        # two special keys
        # 'start' <- beginning pose
        # 'end' <- pose closest to final_x, final_y in Euclidian distance 

        cnt = 0
        while cnt < 3 ** (steps - 1) + add_one and len(poses) > 0:
            
            cnt+=1
            len_pose = len(poses)
            for _ in range(len_pose):
                build_pose = poses.popleft()

                weight_dict.update({build_pose : {}})

                for move in self.change_vals:
                    new_pose = Utils.to_world_torch(build_pose, move).squeeze().tolist()

                    # BOUND CHECKING FOR NEW POSES
                    if not self.mp.is_on_map(new_pose):
                        print("OFF MAP")
                        continue

                    poses.append(tuple(new_pose))
                    
                    trav_map = torch.tensor(self.mp.get_trav_footprint(np.array(new_pose), (40,40)),\
                                                dtype=torch.float32).cuda().unsqueeze(0)

                    edge_weight = edge_weight_calculation(trav_map, patch_sum)

                    weight_dict[tuple(build_pose)].update({tuple(new_pose) : edge_weight})

                    if ((new_pose[0]-final_x)**2 + (new_pose[1]-final_y)**2)**(1/2) < closest_dist:
                        closest_pose = new_pose.copy()

        weight_dict['start'] = tuple(curr_pose)
        weight_dict['end'] = tuple(closest_pose)

        if len(self.map_hold) > 0:
            self.mp.update_map(self.map_hold[-1])
            self.map_hold.clear()
        self.is_planning = False

        return weight_dict
    
    def poses(self, planner = 1, steps = 2, final_x=3.1415, final_y=3.1415, patch_sum = False):
        """
        Method to return path

        planner : 1 - Dijkstra
                  2 - A* Normalized edge weight, Euclidean distance heuristic
                  3 - A* Non-normalized edge weight, product of distance, edge weight heuristic

        steps, final_x, final_y, patch_sum: same definition as those given in build_graph function above

        Returns numpy array of path
        """

        weight_dict = self.build_graph(steps , final_x, final_y, patch_sum)

        return total_path_planner(weight_dict, planner)
                         

# USED FOR TESTING
#     def loop(self, _)-> None:

#         if self.robot_pose is None:
#             print("waiting for robot pose")
#             self.gt_cmds = []
#             return
#         if self.mp.map_elevation is None:
#             print("waiting for map data")
#             self.gt_cmds = []
#             return
#         if len(self.gt_cmds) == 0:            
#             self.gt_cmds = []
#             return 
#         if len(self.gt_path) == 0:
#             self.pred_path_a = [self.robot_pose]
#             self.pred_path_m = [self.robot_pose]
#             self.gt_path = [self.robot_pose]
#             self.gt_cmds = []
#             return
#         if self.gt_path[-1] == self.robot_pose and len(self.gt_path) != 1:
#             self.gt_cmds = []
#             return 
        
#         print(f"Start pose: {self.robot_pose}")
#         pred_x_t1 = self.poses()

#         print(pred_x_t1)

#         print(f"Current pose: {self.robot_pose}")
#         if self.no_iter == publish_itrs:
#             Utils.visualize(self.path_pub_gt, self.gt_path)
#             Utils.visualize(self.path_pub_pred_a, self.pred_path_a)
#             Utils.visualize(self.path_pub_pred_m, self.pred_path_m)
#             self.no_iter = 0
#             self.prev_robot_pose_a = self.robot_pose
#             self.prev_robot_pose_m = self.robot_pose
#             self.pred_path_a = [self.robot_pose]
#             self.pred_path_m = [self.robot_pose]
#             self.gt_path = [self.robot_pose]
#         self.gt_cmds = []


# if __name__ == '__main__':

#     rospy.init_node("bc_TVERTI", anonymous=True)
#     deploy = IKD()
#     rospy.Timer(rospy.Duration(0.1), deploy.loop)
    
#     try:
#         while not rospy.is_shutdown():
#             rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down")
#         cv2.destroyAllWindows()
#         pass
    