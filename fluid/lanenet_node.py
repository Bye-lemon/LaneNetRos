#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# @Author  : Luo Yao
# @Modified  : AdamShan
# @Original site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_node.py


import time
import math
import tensorflow as tf
import numpy as np
import cv2

import paddle.fluid as fluid

from reader import LaneNetDataset
from models.model_builder import build_model
from models.model_builder import ModelPhase
from utils import lanenet_postprocess

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from lane_detector.msg import Lane_Image

CFG = global_config.cfg


class lanenet_detector():
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic')
        self.output_image = rospy.get_param('~output_image')
        self.output_lane = rospy.get_param('~output_lane')
        self.weight_path = rospy.get_param('~weight_path')
        self.use_gpu = rospy.get_param('~use_gpu')
        self.lane_image_topic = rospy.get_param('~lane_image_topic')

        self.init_lanenet()
        self.bridge = CvBridge()
        sub_image = rospy.Subscriber(
            self.image_topic, Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher(
            self.output_image, Image, queue_size=1)
        self.pub_laneimage = rospy.Publisher(
            self.lane_image_topic, Lane_Image, queue_size=1)

    def init_lanenet(self):
        '''
        initlize the paddlepaddle model
        '''

        startup_prog = fluid.Program()
        test_prog = fluid.Program()
        self.pred, self.logit = build_model(test_prog, startup_prog, phase=ModelPhase.VISUAL)
        # Clone forward graph
        test_prog = test_prog.clone(for_test=True)

        # Get device environment
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(startup_prog)

        ckpt_dir = self.weight_path
        if ckpt_dir is not None:
            print('load test model:', ckpt_dir)
            try:
                fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), self.exe)
            except:
                fluid.io.load_params(self.exe, ckpt_dir, main_program=test_prog)
                
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", cv_image)
        # cv2.waitKey(0)
        original_img = cv_image.copy()
        resized_image = self.preprocessing(cv_image)
        mask_image = self.inference_net(resized_image, original_img)
        cv2.imwrite("/home/li/img/"+str(time.time())+".png", mask_image)
        out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, "bgr8")
        self.pub_image.publish(out_img_msg)

    def preprocessing(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", image)
        # cv2.waitKey(1)
        return image

    def inference_net(self, img, original_img):
        fetch_list = [self.pred.name, self.logit.name]
        segLogits, emLogits = self.exe.run(
            program=test_prog,
            feed={'image': img},
            fetch_list=fetch_list,
            return_numpy=True)
            
        binary_seg_image, instance_seg_image = segLogits[0].squeeze(
            -1), emLogits[0].transpose((1, 2, 0))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image,
            instance_seg_result=instance_seg_image,
            source_image=original_img)
       
        mask_image = postprocess_result['mask_image']
        
        mask_image = cv2.resize(mask_image, (original_img.shape[1],
                                             original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.addWeighted(original_img, 0.6, mask_image, 5.0, 0)
        return mask_image

    def minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node')
    lanenet_detector()
    rospy.spin()
