# --------------------------------------------------------
# FPN - FPN
# Copyright (c) 2017
# Licensed under The MIT License [see LICENSE for details]
# Written by xmyqsh
# --------------------------------------------------------

import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg


class FPN_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):

        n_classes = cfg.NCLASSES
        num_anchor_ratio = 3 # 1:2, 1:1, 2:1
        anchor_size = [None, None, 32, 64, 128, 256, 512] # P6 should be in RPN, but not Fast-RCNN, according to the paper
        _feat_stride = [None, 2, 4, 8, 16, 32, 64]

        with tf.variable_scope('res1_2'):

            (self.feed('data')
                 .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
                 .batch_normalization(relu=True, name='bn_conv1', is_training=False)
                 .max_pool(3, 3, 2, 2, padding='VALID',name='pool1')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
                 .batch_normalization(name='bn2a_branch1',is_training=False,relu=False))

            (self.feed('pool1')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
                 .batch_normalization(relu=True, name='bn2a_branch2a',is_training=False)
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
                 .batch_normalization(relu=True, name='bn2a_branch2b',is_training=False)
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
                 .batch_normalization(name='bn2a_branch2c',is_training=False,relu=False))

            (self.feed('bn2a_branch1',
                       'bn2a_branch2c')
                 .add(name='res2a')
                 .relu(name='res2a_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
                 .batch_normalization(relu=True, name='bn2b_branch2a',is_training=False)
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
                 .batch_normalization(relu=True, name='bn2b_branch2b',is_training=False)
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
                 .batch_normalization(name='bn2b_branch2c',is_training=False,relu=False))

            (self.feed('res2a_relu',
                       'bn2b_branch2c')
                 .add(name='res2b')
                 .relu(name='res2b_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
                 .batch_normalization(relu=True, name='bn2c_branch2a',is_training=False)
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
                 .batch_normalization(relu=True, name='bn2c_branch2b',is_training=False)
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
                 .batch_normalization(name='bn2c_branch2c',is_training=False,relu=False))

        with tf.variable_scope('res3_5'):

            (self.feed('res2b_relu',
                       'bn2c_branch2c')
                 .add(name='res2c')
                 .relu(name='res2c_relu')
                 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1', padding='VALID')
                 .batch_normalization(name='bn3a_branch1',is_training=False,relu=False))

            (self.feed('res2c_relu')
                 .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a', padding='VALID')
                 .batch_normalization(relu=True, name='bn3a_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
                 .batch_normalization(relu=True, name='bn3a_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
                 .batch_normalization(name='bn3a_branch2c',is_training=False,relu=False))

            (self.feed('bn3a_branch1',
                       'bn3a_branch2c')
                 .add(name='res3a')
                 .relu(name='res3a_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
                 .batch_normalization(relu=True, name='bn3b_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
                 .batch_normalization(relu=True, name='bn3b_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
                 .batch_normalization(name='bn3b_branch2c',is_training=False,relu=False))

            (self.feed('res3a_relu',
                       'bn3b_branch2c')
                 .add(name='res3b')
                 .relu(name='res3b_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
                 .batch_normalization(relu=True, name='bn3c_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
                 .batch_normalization(relu=True, name='bn3c_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
                 .batch_normalization(name='bn3c_branch2c',is_training=False,relu=False))

            (self.feed('res3b_relu',
                       'bn3c_branch2c')
                 .add(name='res3c')
                 .relu(name='res3c_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
                 .batch_normalization(relu=True, name='bn3d_branch2a',is_training=False)
                 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
                 .batch_normalization(relu=True, name='bn3d_branch2b',is_training=False)
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
                 .batch_normalization(name='bn3d_branch2c',is_training=False,relu=False))

            (self.feed('res3c_relu',
                       'bn3d_branch2c')
                 .add(name='res3d')
                 .relu(name='res3d_relu')
                 .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1', padding='VALID')
                 .batch_normalization(name='bn4a_branch1',is_training=False,relu=False))

            (self.feed('res3d_relu')
                 .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a', padding='VALID')
                 .batch_normalization(relu=True, name='bn4a_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
                 .batch_normalization(relu=True, name='bn4a_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
                 .batch_normalization(name='bn4a_branch2c',is_training=False,relu=False))

            (self.feed('bn4a_branch1',
                       'bn4a_branch2c')
                 .add(name='res4a')
                 .relu(name='res4a_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
                 .batch_normalization(relu=True, name='bn4b_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
                 .batch_normalization(relu=True, name='bn4b_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
                 .batch_normalization(name='bn4b_branch2c',is_training=False,relu=False))

            (self.feed('res4a_relu',
                       'bn4b_branch2c')
                 .add(name='res4b')
                 .relu(name='res4b_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
                 .batch_normalization(relu=True, name='bn4c_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
                 .batch_normalization(relu=True, name='bn4c_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
                 .batch_normalization(name='bn4c_branch2c',is_training=False,relu=False))

            (self.feed('res4b_relu',
                       'bn4c_branch2c')
                 .add(name='res4c')
                 .relu(name='res4c_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
                 .batch_normalization(relu=True, name='bn4d_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
                 .batch_normalization(relu=True, name='bn4d_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
                 .batch_normalization(name='bn4d_branch2c',is_training=False,relu=False))

            (self.feed('res4c_relu',
                       'bn4d_branch2c')
                 .add(name='res4d')
                 .relu(name='res4d_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
                 .batch_normalization(relu=True, name='bn4e_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
                 .batch_normalization(relu=True, name='bn4e_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
                 .batch_normalization(name='bn4e_branch2c',is_training=False,relu=False))

            (self.feed('res4d_relu',
                       'bn4e_branch2c')
                 .add(name='res4e')
                 .relu(name='res4e_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
                 .batch_normalization(relu=True, name='bn4f_branch2a',is_training=False)
                 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
                 .batch_normalization(relu=True, name='bn4f_branch2b',is_training=False)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
                 .batch_normalization(name='bn4f_branch2c',is_training=False,relu=False))

            (self.feed('res4e_relu',
                       'bn4f_branch2c')
                 .add(name='res4f')
                 .relu(name='res4f_relu'))

            # conv5
            (self.feed('res4f_relu')
                 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a', padding='VALID')
                 .batch_normalization(relu=True, name='bn5a_branch2a',is_training=False)
                 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
                 .batch_normalization(relu=True, name='bn5a_branch2b',is_training=False)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
                 .batch_normalization(name='bn5a_branch2c',is_training=False,relu=False))

            (self.feed('res4f_relu')
                 .conv(1,1,2048,2,2,biased=False, relu=False, name='res5a_branch1', padding='VALID')
                 .batch_normalization(name='bn5a_branch1',is_training=False,relu=False))

            (self.feed('bn5a_branch2c','bn5a_branch1')
                 .add(name='res5a')
                 .relu(name='res5a_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
                 .batch_normalization(relu=True, name='bn5b_branch2a',is_training=False)
                 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
                 .batch_normalization(relu=True, name='bn5b_branch2b',is_training=False)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
                 .batch_normalization(name='bn5b_branch2c',is_training=False,relu=False))

            (self.feed('res5a_relu',
                       'bn5b_branch2c')
                 .add(name='res5b')
                 .relu(name='res5b_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
                 .batch_normalization(relu=True, name='bn5c_branch2a',is_training=False)
                 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
                 .batch_normalization(relu=True, name='bn5c_branch2b',is_training=False)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
                 .batch_normalization(name='bn5c_branch2c',is_training=False,relu=False))

            (self.feed('res5b_relu',
                       'bn5c_branch2c')
                 .add(name='res5c')
                 .relu(name='res5c_relu'))

        with tf.variable_scope('Top-Down'):

            # Top-Down
            (self.feed('res5c_relu') # C5
                 .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='P5'))

            (self.feed('P5')
                 .max_pool(2, 2, 2, 2, padding='VALID',name='P6'))

            (self.feed('res4f_relu') # C4
                 .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='C4_lateral'))

            (self.feed('P5',
                       'C4_lateral')
                 .upbilinear(name='C5_topdown'))

            (self.feed('C5_topdown',
                       'C4_lateral')
                 .add(name='P4_pre')
                 .conv(3, 3, 256, 1, 1, biased=True, relu=False, name='P4'))

            (self.feed('res3d_relu') #C3
                 .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='C3_lateral'))

            (self.feed('P4',
                       'C3_lateral')
                 .upbilinear(name='C4_topdown'))

            (self.feed('C4_topdown',
                       'C3_lateral')
                 .add(name='P3_pre')
                 .conv(3, 3, 256, 1, 1, biased=True, relu= False, name='P3'))

            (self.feed('res2c_relu') #C2
                 .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='C2_lateral'))

            (self.feed('P3',
                       'C2_lateral')
                 .upbilinear(name='C3_topdown'))

            (self.feed('C3_topdown',
                       'C2_lateral')
                 .add(name='P2_pre')
                 .conv(3, 3, 256, 1, 1, biased=True, relu= False, name='P2'))


        with tf.variable_scope('RPN') as scope:
            #========= RPN ============
            # TODO: the head of 3*3 conv and two sibling 1*1 convs in each pyramid could share weights,
            #       add a shared weights version for comparison
            # P2
            (self.feed('P2')
                 .conv(3,3,512,1,1,name='rpn_conv/3x3/P2',reuse=True)
                 .conv(1,1,num_anchor_ratio*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P2', reuse=True))

            (self.feed('rpn_conv/3x3/P2')
                 .conv(1,1,num_anchor_ratio*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P2', reuse=True))

            (self.feed('rpn_cls_score/P2')
                 .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P2')
                 .spatial_softmax(name='rpn_cls_prob/P2'))

            (self.feed('rpn_cls_prob/P2')
                 .spatial_reshape_layer(num_anchor_ratio*2, name = 'rpn_cls_prob_reshape/P2'))

            scope.reuse_variables()

            # P3
            (self.feed('P3')
                 .conv(3,3,512,1,1,name='rpn_conv/3x3/P3', reuse=True)
                 .conv(1,1,num_anchor_ratio*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P3', reuse=True))

            (self.feed('rpn_conv/3x3/P3')
                 .conv(1,1,num_anchor_ratio*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P3', reuse=True))

            (self.feed('rpn_cls_score/P3')
                 .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P3')
                 .spatial_softmax(name='rpn_cls_prob/P3'))

            (self.feed('rpn_cls_prob/P3')
                 .spatial_reshape_layer(num_anchor_ratio*2, name = 'rpn_cls_prob_reshape/P3'))

            # P4
            (self.feed('P4')
                 .conv(3,3,512,1,1,name='rpn_conv/3x3/P4',reuse=True)
                 .conv(1,1,num_anchor_ratio*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P4', reuse=True))

            (self.feed('rpn_conv/3x3/P4')
                 .conv(1,1,num_anchor_ratio*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P4', reuse=True))

            (self.feed('rpn_cls_score/P4')
                 .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P4')
                 .spatial_softmax(name='rpn_cls_prob/P4'))

            (self.feed('rpn_cls_prob/P4')
                 .spatial_reshape_layer(num_anchor_ratio*2, name = 'rpn_cls_prob_reshape/P4'))

            # P5
            (self.feed('P5')
                 .conv(3,3,512,1,1,name='rpn_conv/3x3/P5', reuse=True)
                 .conv(1,1,num_anchor_ratio*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P5', reuse=True))

            (self.feed('rpn_conv/3x3/P5')
                 .conv(1,1,num_anchor_ratio*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P5', reuse=True))

            (self.feed('rpn_cls_score/P5')
                 .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P5')
                 .spatial_softmax(name='rpn_cls_prob/P5'))

            (self.feed('rpn_cls_prob/P5')
                 .spatial_reshape_layer(num_anchor_ratio*2, name = 'rpn_cls_prob_reshape/P5'))

            # P6
            (self.feed('P6')
                 .conv(3,3,512,1,1,name='rpn_conv/3x3/P6', reuse=True)
                 .conv(1,1,num_anchor_ratio*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P6', reuse=True))

            (self.feed('rpn_conv/3x3/P6')
                 .conv(1,1,num_anchor_ratio*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P6', reuse=True))

            (self.feed('rpn_cls_score/P6')
                 .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P6')
                 .spatial_softmax(name='rpn_cls_prob/P6'))

            (self.feed('rpn_cls_prob/P6')
                 .spatial_reshape_layer(num_anchor_ratio*2, name = 'rpn_cls_prob_reshape/P6'))



            (self.feed('rpn_cls_score_reshape/P2')
                 .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P2'))

            (self.feed('rpn_cls_score_reshape/P3')
                 .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P3'))

            (self.feed('rpn_cls_score_reshape/P4')
                 .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P4'))

            (self.feed('rpn_cls_score_reshape/P5')
                 .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P5'))

            (self.feed('rpn_cls_score_reshape/P6')
                 .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P6'))

            (self.feed('rpn_cls_score_reshape_reshape/P2',
                       'rpn_cls_score_reshape_reshape/P3',
                       'rpn_cls_score_reshape_reshape/P4',
                       'rpn_cls_score_reshape_reshape/P5',
                       'rpn_cls_score_reshape_reshape/P6')
                 .concat(0, name = 'rpn_cls_score_reshape_reshape_concat'))


            (self.feed('rpn_bbox_pred/P2')
                 .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P2'))

            (self.feed('rpn_bbox_pred/P3')
                 .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P3'))

            (self.feed('rpn_bbox_pred/P4')
                 .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P4'))

            (self.feed('rpn_bbox_pred/P5')
                 .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P5'))

            (self.feed('rpn_bbox_pred/P6')
                 .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P6'))

            (self.feed('rpn_bbox_pred_reshape/P2',
                       'rpn_bbox_pred_reshape/P3',
                       'rpn_bbox_pred_reshape/P4',
                       'rpn_bbox_pred_reshape/P5',
                       'rpn_bbox_pred_reshape/P6')
                 .concat(0, name = 'rpn_bbox_pred_reshape_concat'))


            (self.feed('rpn_cls_score/P2',
                       'rpn_cls_score/P3',
                       'rpn_cls_score/P4',
                       'rpn_cls_score/P5',
                       'rpn_cls_score/P6',
                       'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
                 .anchor_target_layer(_feat_stride[2:], anchor_size[2:], name = 'rpn-data'))

            #========= RoI Proposal ============
            (self.feed('rpn_cls_prob_reshape/P2', 'rpn_bbox_pred/P2',
                       'rpn_cls_prob_reshape/P3', 'rpn_bbox_pred/P3',
                       'rpn_cls_prob_reshape/P4', 'rpn_bbox_pred/P4',
                       'rpn_cls_prob_reshape/P5', 'rpn_bbox_pred/P5',
                       'rpn_cls_prob_reshape/P6', 'rpn_bbox_pred/P6',
                       'im_info')
                 .proposal_layer(_feat_stride[2:], anchor_size[2:], 'TRAIN',name = 'rpn_rois'))

            (self.feed('rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas')
                 .proposal_target_layer(n_classes,name = 'roi-data'))

        with tf.variable_scope('Fast-RCNN'):
            #========= RCNN ============
            (self.feed('P2', 'P3', 'P4', 'P5', 'P6', 'roi-data')
                 .fpn_roi_pool(7, 7, name='fpn_roi_pooling')
                 .fc(1024, name='fc6')
                 .fc(1024, name='fc7')
                 .fc(n_classes, relu=False, name='cls_score')
                 .softmax(name='cls_prob'))

            (self.feed('fc7')
                 .fc(n_classes*4, relu=False, name='bbox_pred'))


    def build_loss(self):
        ############# RPN
        rpn_cls_score = self.get_output('rpn_cls_score_reshape_reshape_concat') # shape(sum(HxWxA), 2)
        rpn_bbox_pred = self.get_output('rpn_bbox_pred_reshape_concat') # shape (sum(HxWxA), 4)

        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        rpn_bbox_targets = self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]

        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep), [-1, 2]) # shape (N, 2)
        rpn_bbox_pred = tf.reshape(tf.gather(rpn_bbox_pred, rpn_keep), [-1, 4]) # shape (N, 4)

        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep), [-1])
        rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_inside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_outside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep), [-1, 4])

        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        rpn_loss_box_n = tf.reduce_sum(self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), axis=[1])
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1.0)
        #rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (cfg.TRAIN.RPN_BATCHSIZE)

        ############# R-CNN
        # classification loss
        cls_score = self.get_output('cls_score') # (R, C+1)
        label = tf.reshape(self.get_output('roi-data')[5], [-1]) # (R)
        cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)

        # bounding box regression L1 loss
        bbox_pred = self.get_output('bbox_pred') # (R, (C+1)x4)
        bbox_targets = self.get_output('roi-data')[6] # (R, (C+1)x4)
        # each element is {0, 1}, represents background (0), objects (1)
        bbox_inside_weights = self.get_output('roi-data')[7] # (R, (C+1)x4)
        bbox_outside_weights = self.get_output('roi-data')[8] # (R, (C+1)x4)

        loss_box_n = tf.reduce_sum( \
            bbox_outside_weights * self.smooth_l1_dist(bbox_inside_weights * (bbox_pred - bbox_targets)), \
            axis=[1])

        loss_n = loss_box_n + cross_entropy_n
        loss_n = tf.reshape(loss_n, [-1])

        #loss_box = tf.reduce_mean(loss_box_n)
        loss_box = tf.reduce_sum(loss_box_n)
        cross_entropy = tf.reduce_mean(cross_entropy_n)

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        vs_name = ['res3_5', 'RPN', 'Top-Down', 'Fast-RCNN']

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            print 'regularization_losses: '
            print regularization_losses
            regularization_losses = [v for v in regularization_losses for vs in vs_name if v.op.name.startswith(vs + '/')]
            print 'filtered regularization_losses: '
            print regularization_losses
            loss = tf.add_n(regularization_losses) + loss

        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box
