# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""
import numpy as np
import os
import pprint
import cPickle
import tensorflow as tf
from tensorflow.python.client import timeline
import cv2

from .nms_wrapper import nms_wrapper
from ..roi_data_layer.layer import RoIDataLayer
from ..utils.timer import Timer
from ..gt_data_layer import roidb as gdl_roidb
from ..roi_data_layer import roidb as rdl_roidb
from ..fast_rcnn.config import get_log_dir, get_output_dir
from ..datasets.factory import get_imdb
from ..utils.blob import im_list_to_blob

# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
# <<<< obsolete

_DEBUG = False

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb_name, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.pretrained_model = pretrained_model
        self.imdb_name = imdb_name

        roidb, imdb = get_roidb(self.imdb_name)
        #imdb = get_imdb(self.imdb_name)
        self.logdir = get_log_dir(imdb)
        self.writer = tf.summary.FileWriter(logdir=self.logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)


        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        self.global_step = tf.Variable(0, trainable=False)

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)

        # get train_op
        self.lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        momentum = cfg.TRAIN.MOMENTUM
        self.opt = tf.train.MomentumOptimizer(self.lr, momentum)
        #self.opt = tf.train.GradientDescentOptimizer(self.lr)

    def snapshot(self, sess, iter, output_dir):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred') and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('Fast-RCNN', reuse=True):
                with tf.variable_scope('bbox_pred'):
                    weights = tf.get_variable("weights")
                    biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def train_rpn(self, sess, vs_names, max_iter, init_model=None):
        """Train a Region Proposal Network in a separate training process.
        """

        # Not using any proposals, just ground-truth boxes
        cfg.TRAIN.HAS_RPN = True
        cfg.TRAIN.BBOX_REG = False  # applies only to Fast R-CNN bbox regression
        cfg.TRAIN.PROPOSAL_METHOD = 'gt'
        cfg.TRAIN.IMS_PER_BATCH = 1
        cfg.TRAIN.STEPSIZE = 60000
        print 'Init model: {}'.format(init_model)
        print('Using config:')
        pprint.pprint(cfg)

        roidb, imdb = get_roidb(self.imdb_name)
        print 'roidb len: {}'.format(len(roidb))
        output_dir = get_output_dir(imdb)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        # get data_layer
        data_layer = RoIDataLayer(roidb, imdb.num_classes)

        # get loss
        rpn_loss, rpn_cross_entropy, rpn_loss_box = self.net.build_RPN_loss(vs_names)

        # get summery_op
        tf.summary.scalar('rpn_cls_loss', rpn_cross_entropy, collections=['RPN'])
        tf.summary.scalar('rpn_rgs_loss', rpn_loss_box, collections=['RPN'])
        tf.summary.scalar('rpn_loss', rpn_loss, collections=['RPN'])
        summary_op = tf.summary.merge_all('RPN')

        tf.assign(self.lr, cfg.TRAIN.LEARNING_RATE)

        tf.assign(self.global_step, 0)

        trainable_vars = []
        for vs_name in vs_names:
            trainable_vars.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, vs_name))
        train_op = self.opt.minimize(rpn_loss, global_step=self.global_step, var_list=trainable_vars)

        if init_model:
            # intialize variables
            sess.run(tf.global_variables_initializer())
            try:
                print ('Loading pretrained model '
                   'weights from {:s}').format(init_model)
                self.net.load(init_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(init_model)

        # initialized uninitialized_vars
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        print 'uninitialized_vars in RPN:'
        print uninitialized_vars

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)


        last_snapshot_iter = -1
        timer = Timer()
        for iter in xrange(max_iter):
            timer.tic()

            # learning rate
            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(self.lr, self.lr.eval() * cfg.TRAIN.GAMMA))

            # get one batch
            blobs = data_layer.forward()

            feed_dict={
                self.net.data: blobs['data'],
                self.net.im_info: blobs['im_info'],
                self.net.gt_boxes: blobs['gt_boxes'],
                self.net.gt_ishard: blobs['gt_ishard'],
                self.net.dontcare_areas: blobs['dontcare_areas']
            }

            fetch_list = [rpn_cross_entropy,
                          rpn_loss_box,
                          summary_op,
                          train_op]

            rpn_loss_cls_value, rpn_loss_box_value, \
            summary_str, _ = sess.run(fetches=fetch_list, feed_dict=feed_dict)

            self.writer.add_summary(summary=summary_str, global_step=self.global_step.eval())

            _diff_time = timer.toc(average=False)

            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %.10f'%\
                        (iter, max_iter, rpn_loss_cls_value + rpn_loss_box_value,\
                         rpn_loss_cls_value, rpn_loss_box_value, self.lr.eval())
                print 'speed: {:.3f}s / iter'.format(_diff_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter, output_dir)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter, output_dir)

    def rpn_generate(self, sess, rpn_net_name):
        """Use a trained RPN to generate proposals.
        """

        cfg.TRAIN.RPN_PRE_NMS_TOP_N = -1     # no pre NMS filtering
        cfg.TRAIN.RPN_POST_NMS_TOP_N = 2000  # limit top boxes after NMS
        print('Using config:')
        pprint.pprint(cfg)

        imdb = get_imdb(self.imdb_name)
        print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

        output_dir = get_output_dir(imdb)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        # Generate proposals on the imdb
        def imdb_proposals(sess, imdb):
            """Generate RPN proposals on all images in an imdb."""

            def im_proposals(sess, im):
                """Generate RPN proposals on a single image."""
                blobs = {}
                blobs['data'], blobs['im_info'] = _get_image_blob(im)

                feed_dict={
                    self.net.data: blobs['data'],
                    self.net.im_info: blobs['im_info'],
                }

                rois = sess.run(fetches=[self.net.get_output('rpn_rois')], feed_dict=feed_dict)

                scale = blobs['im_info'][0, 2]
                boxes = rois[0][:, 1:].copy() / scale
                return boxes

            _t = Timer()
            imdb_boxes = [[] for _ in xrange(imdb.num_images)]
            for i in xrange(imdb.num_images):
                im = cv2.imread(imdb.image_path_at(i))
                _t.tic()
                imdb_boxes[i] = im_proposals(sess, im)
                _t.toc()
                print 'im_proposals: {:d}/{:d} {:.3f}s' \
                      .format(i + 1, imdb.num_images, _t.average_time)

            return imdb_boxes

        rpn_proposals = imdb_proposals(sess, imdb)
        # Write proposals to disk and send the proposal file path through the
        # multiprocessing queue
        rpn_proposals_path = os.path.join(
            output_dir, rpn_net_name + '_proposals.pkl')
        with open(rpn_proposals_path, 'wb') as f:
            cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
        print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)

        return rpn_proposals_path

    def train_fast_rcnn(self, sess, vs_names, max_iter, rpn_file, init_model=None):
        """Train a Fast R-CNN using proposals generated by an RPN.
        """

        cfg.TRAIN.HAS_RPN = False           # not generating prosals on-the-fly
        cfg.TRAIN.BBOX_REG = True
        cfg.TRAIN.PROPOSAL_METHOD = 'rpn'   # use pre-computed RPN proposals instead
        cfg.TRAIN.IMS_PER_BATCH = 2
        cfg.TRAIN.STEPSIZE = 30000
        print 'Init model: {}'.format(init_model)
        print 'RPN proposals: {}'.format(rpn_file)
        print('Using config:')
        pprint.pprint(cfg)

        roidb, imdb = get_roidb(self.imdb_name, rpn_file)
        print 'roidb len: {}'.format(len(roidb))

        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        output_dir = get_output_dir(imdb)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        # get data_layer
        data_layer = RoIDataLayer(roidb, imdb.num_classes)

        # get loss
        loss, cross_entropy, loss_box = self.net.build_Fast_RCNN_loss(vs_names)

        # get summery_op
        tf.summary.scalar('cls_loss', cross_entropy, collections=['Fast-RCNN'])
        tf.summary.scalar('rgs_loss', loss_box, collections=['Fast-RCNN'])
        tf.summary.scalar('loss', loss, collections=['Fast-RCNN'])
        summary_op = tf.summary.merge_all('Fast-RCNN')

        tf.assign(self.lr, cfg.TRAIN.LEARNING_RATE)

        tf.assign(self.global_step, 0)

        trainable_vars = []
        for vs_name in vs_names:
            trainable_vars.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, vs_name))
        train_op = self.opt.minimize(loss, global_step=self.global_step, var_list=trainable_vars)

        if init_model:
            # intialize variables
            sess.run(tf.global_variables_initializer())
            try:
                print ('Loading pretrained model '
                   'weights from {:s}').format(init_model)
                self.net.load(init_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(init_model)

        # initialized uninitialized_vars
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        print 'uninitialized_vars in Fast-RCNN:'
        print uninitialized_vars

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)


        last_snapshot_iter = -1
        timer = Timer()
        for iter in xrange(max_iter):
            timer.tic()

            # learning rate
            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(self.lr, self.lr.eval() * cfg.TRAIN.GAMMA))

            # get one batch
            blobs = data_layer.forward()

            feed_dict={
                self.net.data: blobs['data'],
                self.net.leveled_rois_0: blobs['leveled_rois_0'],
                self.net.leveled_rois_1: blobs['leveled_rois_1'],
                self.net.leveled_rois_2: blobs['leveled_rois_2'],
                self.net.leveled_rois_3: blobs['leveled_rois_3'],
                self.net.rois: blobs['rois'],
                self.net.labels: blobs['labels'],
                self.net.bbox_targets: blobs['bbox_targets'],
                self.net.bbox_inside_weights: blobs['bbox_inside_weights'],
                self.net.bbox_outside_weights: blobs['bbox_outside_weights'],
                #self.net.roi_data: blobs['roi-data'],
            }

            fetch_list = [cross_entropy,
                          loss_box,
                          summary_op,
                          train_op]

            loss_cls_value, loss_box_value, \
            summary_str, _ = sess.run(fetches=fetch_list, feed_dict=feed_dict)

            self.writer.add_summary(summary=summary_str, global_step=self.global_step.eval())

            _diff_time = timer.toc(average=False)

            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %.10f'%\
                        (iter, max_iter, loss_cls_value + loss_box_value,\
                         loss_cls_value, loss_box_value, self.lr.eval())
                print 'speed: {:.3f}s / iter'.format(_diff_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter, output_dir)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter, output_dir)

    def train_model(self, sess, vs_names, max_iters):
        """Network training loop."""

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Stage 1 RPN, init from ImageNet model'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        cfg.TRAIN.SNAPSHOT_INFIX = 'stage1_RPN'

        self.train_rpn(sess, vs_names[0], max_iters[0], init_model=self.pretrained_model)

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Stage 1 RPN, generate proposals'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        rpn_file_stage1 = self.rpn_generate(sess, cfg.TRAIN.SNAPSHOT_INFIX)
        #rpn_file_stage1 = 'output/FPN_alt_opt/voc_0712_trainval/stage1_RPN_proposals.pkl'

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        cfg.TRAIN.SNAPSHOT_INFIX = 'stage1_Fast_RCNN'

        self.train_fast_rcnn(sess, vs_names[1], max_iters[1], rpn_file = rpn_file_stage1, init_model=self.pretrained_model)

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Stage 2 RPN, init from stage 1 Fast R-CNN model'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        cfg.TRAIN.SNAPSHOT_INFIX = 'stage2_RPN'

        self.train_rpn(sess, vs_names[2], max_iters[2])

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Stage 2 RPN, generate proposals'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        rpn_file_stage2 = self.rpn_generate(sess, cfg.TRAIN.SNAPSHOT_INFIX)

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        cfg.TRAIN.SNAPSHOT_INFIX = 'stage2_Fast_RCNN'

        self.train_fast_rcnn(sess, vs_names[3], max_iters[3], rpn_file=rpn_file_stage2)

def get_roidb(imdb_name, rpn_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []

    assert len(cfg.TRAIN.SCALES) == 1
    target_size = cfg.TRAIN.SCALES[0]

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TRAIN.MAX_SIZE:
        im_scale = float(cfg.TRAIN.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_info = np.hstack((im.shape[:2], im_scale))[np.newaxis, :]
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_info

def train_net(network, imdb_name, vs_names, pretrained_model=None, max_iters=[80000, 40000, 80000, 40000]):
    """Train a Fast R-CNN network."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.90
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb_name, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, vs_names, max_iters)
        print 'done solving'
