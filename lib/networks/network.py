import numpy as np
import tensorflow as tf

from ..fast_rcnn.config import cfg
from ..roi_pooling_layer import roi_pooling_op as roi_pool_op
from ..rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from ..rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from ..rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py


DEFAULT_PADDING = 'SAME'

def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        # scope: res1_2
        for key in data_dict:
            with tf.variable_scope('res1_2', reuse=True):
                with tf.variable_scope(key):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key + " " + subkey
                            if not ignore_missing:
                                raise

        # scope: res3_5
        for key in data_dict:
            with tf.variable_scope('res3_5', reuse=True):
                with tf.variable_scope(key):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key + " " + subkey
                            if not ignore_missing:
                                raise

        # scope: Top-Down
        for key in data_dict:
            with tf.variable_scope('Top-Down', reuse=True):
                with tf.variable_scope(key):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key + " " + subkey
                            if not ignore_missing:
                                raise

        # scope: RPN
        for key in data_dict:
            with tf.variable_scope('RPN', reuse=True):
                with tf.variable_scope(key):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key + " " + subkey
                            if not ignore_missing:
                                raise

        # scope: Fast-RCNN
        for key in data_dict:
            with tf.variable_scope('Fast-RCNN', reuse=True):
                with tf.variable_scope(key):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key + " " + subkey
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, reuse=False, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name if reuse == False else '/'.join(name.split('/')[:-1])) as scope:

            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            #init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0 if relu else 1.0, mode='FAN_IN', uniform=False if relu else True)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def upbilinear(self, input, name):
        up_h = tf.shape(input[1])[1]
        up_w = tf.shape(input[1])[2]
        return tf.image.resize_bilinear(input[0], [up_h, up_w], name=name)

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride = 2, name = 'upconv', biased=False, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = ((in_shape[1] ) * stride)
            w = ((in_shape[2] ) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            #init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0 if relu else 1.0, mode='FAN_IN', uniform=False if relu else True)
            filters = self.make_var('weights', filter_shape, init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def fpn_roi_pool(self, input, pooled_height, pooled_width, name):
        # fake op, just parallel roi_pool and concat them
        # only use the first input
        if isinstance(input[0], tuple): # P2
            input[0] = input[0][0]

        if isinstance(input[1], tuple): # P3
            input[1] = input[1][0]

        if isinstance(input[2], tuple): # P4
            input[2] = input[2][0]

        if isinstance(input[3], tuple): # P5
            input[3] = input[3][0]

        if isinstance(input[4], tuple): # P6
            input[4] = input[4][0]

        '''
        if isinstance(input[4], tuple): # roi-data
            input[4] = input[4][0]
        '''


        print input
        with tf.variable_scope(name) as scope:
            roi_pool_P2 = roi_pool_op.roi_pool(input[0], input[5][0],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 4.0,
                                    name='roi_pool_P2')[0]
            roi_pool_P3 = roi_pool_op.roi_pool(input[1], input[5][1],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 8.0,
                                    name='roi_pool_P3')[0]
            roi_pool_P4 = roi_pool_op.roi_pool(input[2], input[5][2],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 16.0,
                                    name='roi_pool_P4')[0]
            roi_pool_P5 = roi_pool_op.roi_pool(input[3], input[5][3],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 32.0,
                                    name='roi_pool_P5')[0]
            roi_pool_P6 = roi_pool_op.roi_pool(input[4], input[5][4],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 64.0,
                                    name='roi_pool_P6')[0]

            return tf.concat(axis=0, values=[roi_pool_P2, roi_pool_P3, roi_pool_P4, roi_pool_P5, roi_pool_P6], name='roi_pool_concat')

    @layer
    def proposal_layer(self, input, _feat_strides, anchor_sizes, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        if cfg_key == 'TRAIN':
            # 'rpn_cls_prob_reshape/P2', 'rpn_bbox_pred/P2',
            # 'rpn_cls_prob_reshape/P3', 'rpn_bbox_pred/P3',
            # 'rpn_cls_prob_reshape/P4', 'rpn_bbox_pred/P4',
            # 'rpn_cls_prob_reshape/P5', 'rpn_bbox_pred/P5',
            # 'im_info'
            with tf.variable_scope(name) as scope:
                return tf.reshape(tf.py_func(proposal_layer_py,\
                                     [input[0], input[1],\
                                      input[2], input[3],\
                                      input[4], input[5],\
                                      input[6], input[7],\
                                      input[8], input[9],\
                                      input[10], cfg_key, _feat_strides, anchor_sizes],\
                                     [tf.float32]),\
                                     [-1,5], name = 'rpn_rois')

        with tf.variable_scope(name) as scope:
            rpn_rois_P2, rpn_rois_P3, rpn_rois_P4, rpn_rois_P5, rpn_rois_P6, rpn_rois = tf.py_func(proposal_layer_py,\
                                                        [input[0], input[1],\
                                                         input[2], input[3],\
                                                         input[4], input[5],\
                                                         input[6], input[7],\
                                                         input[8], input[9],\
                                                         input[10], cfg_key, _feat_strides, anchor_sizes],\
                                                         [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]);

            rpn_rois_P2 = tf.reshape(rpn_rois_P2, [-1, 5], name = 'rpn_rois_P2') # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P3 = tf.reshape(rpn_rois_P3, [-1, 5], name = 'rpn_rois_P3') # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P4 = tf.reshape(rpn_rois_P4, [-1, 5], name = 'rpn_rois_P4') # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P5 = tf.reshape(rpn_rois_P5, [-1, 5], name = 'rpn_rois_P5') # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P6 = tf.reshape(rpn_rois_P6, [-1, 5], name = 'rpn_rois_P6') # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois    = tf.reshape(rpn_rois, [-1, 5], name = 'rpn_rois') # shape is (1 x H(P) x W(P) x A(P), 5)

            self.layers['rois'] = rpn_rois

            return rpn_rois_P2, rpn_rois_P3, rpn_rois_P4, rpn_rois_P5, rpn_rois_P6

    @layer
    def anchor_target_layer(self, input, _feat_strides, anchor_sizes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0],input[1],input[2],input[3],input[4],input[5],input[6],input[7],input[8], _feat_strides, anchor_sizes],
                           [tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels') # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets') # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights') # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights') # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    '''
    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_size, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0],input[1],input[2],input[3],input[4], _feat_stride, anchor_size],
                           [tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels') # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets') # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights') # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights') # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    '''

    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            #inputs: 'rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas'
            rois_P2,rois_P3,rois_P4,rois_P5,rois_P6,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights,rois \
                = tf.py_func(proposal_target_layer_py,
                             [input[0],input[1],input[2],input[3],classes],
                             [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            # rois_Px <- (1 x H x W x A(x), 5) e.g. [0, x1, y1, x2, y2]
            # rois = tf.convert_to_tensor(rois, name='rois')
            rois = tf.reshape(rois, [-1, 5], name='rois') # goes to roi_pooling
            rois_P2 = tf.reshape(rois_P2, [-1, 5], name='rois_P2') # goes to roi_pooling
            rois_P3 = tf.reshape(rois_P3, [-1, 5], name='rois_P3') # goes to roi_pooling
            rois_P4 = tf.reshape(rois_P4, [-1, 5], name='rois_P4') # goes to roi_pooling
            rois_P5 = tf.reshape(rois_P5, [-1, 5], name='rois_P5') # goes to roi_pooling
            rois_P6 = tf.reshape(rois_P6, [-1, 5], name='rois_P6') # goes to roi_pooling
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels') # goes to FRCNN loss
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets') # goes to FRCNN loss
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

            self.layers['rois'] = rois

            return rois_P2, rois_P3, rois_P4, rois_P5, rois_P6, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois


    '''
    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)
    '''

    @layer
    def reshape_layer(self, input, output_shape, name):
        return tf.reshape(input, output_shape, name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            '''
            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)
            '''

            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2.0 if relu else 1.0, mode='FAN_IN', uniform=False if relu else True)
            init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    @layer
    def batch_normalization(self,input,name,relu=True, is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def scale(self, input, c_in, name):
        with tf.variable_scope(name) as scope:

            alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True,
                                    regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=True,
                                   regularizer=self.l2_regularizer(0.00001))
            return tf.add(tf.multiply(input, alpha), beta)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=3.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)
