#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_from_host, host_from_gpu
from theano.tensor.nnet import conv
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs
from theano_ops.conv_bc01_fft import (ConvBC01, ConvBC01ImgsGrad,
                                      ConvBC01FiltersGrad)
import theano_ops.conv_bc01_fft


def avg_running_time(fun):
    n_iter = 20
    start_time = time.time()
    for _ in range(n_iter):
        fun()
    duration = time.time() - start_time
    return duration / float(n_iter)


def allclose(a, b):
    atol = 1e-3
    rtol = 1e-4
    return np.allclose(a, b, atol=atol, rtol=rtol)


def benchmark(n_imgs, n_channels, img_shape, n_filters, filter_shape, pad):
    print('Benchmarking n_imgs: %i, n_channels: %i, ' % (n_imgs, n_channels)
          + 'img_shape: (%i, %i), ' % img_shape
          + 'filter_shape: (%i, %i), pad: %i' % (filter_shape + (pad,)))

    # Setup arrays
    img_h, img_w = img_shape
    filter_h, filter_w = filter_shape
    convout_h = img_h + 2*pad - filter_h + 1
    convout_w = img_w + 2*pad - filter_w + 1

    imgs_bc01_shape = (n_imgs, n_channels, img_h, img_w)
    filters_bc01_shape = (n_filters, n_channels, filter_h, filter_w)

    imgs_bc01 = np.random.randn(n_imgs, n_channels, img_h, img_w)
    imgs_c01b = np.transpose(imgs_bc01, (1, 2, 3, 0))
    filters_fc01 = np.random.randn(n_filters, n_channels, filter_h, filter_w)
    filters_c01f = np.transpose(filters_fc01, (1, 2, 3, 0))
    convout_bc01 = np.random.randn(n_imgs, n_filters, convout_h, convout_w)
    convout_c01b = np.transpose(convout_bc01, (1, 2, 3, 0))

    imgs_bc01_t = theano.shared(imgs_bc01.astype(theano.config.floatX))
    imgs_c01b_t = theano.shared(imgs_c01b.astype(theano.config.floatX))
    filters_fc01_t = theano.shared(filters_fc01.astype(theano.config.floatX))
    filters_c01f_t = theano.shared(filters_c01f.astype(theano.config.floatX))
    convout_bc01_t = theano.shared(convout_bc01.astype(theano.config.floatX))
    convout_c01b_t = theano.shared(convout_c01b.astype(theano.config.floatX))

    # Forward propagation
    convout_cc_op = FilterActs(stride=1, partial_sum=4, pad=pad)
    convout_cc_expr = convout_cc_op(imgs_c01b_t, filters_c01f_t)
    convout_cc_fun = theano.function([], convout_cc_expr)
    convout_cc = convout_cc_fun()
    convout_cc = np.transpose(convout_cc, (3, 0, 1, 2))

    convout_fft_op = ConvBC01(n_imgs, n_channels, n_filters, img_shape,
                              filter_shape, (pad, pad))
    convout_fft_expr = convout_fft_op(imgs_bc01_t, filters_fc01_t)
    convout_fft_fun = theano.function([], host_from_gpu(convout_fft_expr))
    convout_fft = convout_fft_fun()

    print('fprop')
    print('allclose: ' + str(allclose(convout_fft, convout_cc)))
    duration_cc = avg_running_time(convout_cc_fun)
    convout_fft_fun = theano.function([], convout_fft_expr)
    duration_fft = avg_running_time(convout_fft_fun)
    print('cuda_convnet: %f  fft: %f' % (duration_cc, duration_fft))

    # Back propagation, imgs
    dimgs_cc_op = ImageActs(stride=1, partial_sum=1, pad=pad)
    dimgs_cc_expr = dimgs_cc_op(convout_c01b_t, filters_c01f_t)
    dimgs_cc_fun = theano.function([], dimgs_cc_expr)
    dimgs_cc = dimgs_cc_fun()
    dimgs_cc = np.transpose(dimgs_cc, (3, 0, 1, 2))

    dimgs_fft_op = ConvBC01ImgsGrad(n_imgs, n_channels, n_filters, img_shape,
                                    filter_shape, (pad, pad))
    dimgs_fft_expr = dimgs_fft_op(filters_fc01_t, convout_bc01_t)
    dimgs_fft_fun = theano.function([], host_from_gpu(dimgs_fft_expr))
    dimgs_fft = dimgs_fft_fun()
    print('bprop imgs')
    print('allclose: ' + str(allclose(dimgs_fft, dimgs_cc)))
    duration_cc = avg_running_time(dimgs_cc_fun)
    dimgs_fft_fun = theano.function([], dimgs_fft_expr)
    duration_fft = avg_running_time(dimgs_fft_fun)
    print('cuda_convnet: %f  fft: %f' % (duration_cc, duration_fft))

    # Back propagation, filters
    dfilters_cc_op = WeightActs(stride=1, partial_sum=1, pad=pad)
    dfilters_cc_expr = dfilters_cc_op(imgs_c01b_t, convout_c01b_t,
                                      T.as_tensor_variable(filter_shape))
    dfilters_cc_fun = theano.function([], dfilters_cc_expr)
    dfilters_cc = dfilters_cc_fun()[0]
    dfilters_cc = np.transpose(dfilters_cc, (3, 0, 1, 2))

    dfilters_fft_op = ConvBC01FiltersGrad(n_imgs, n_channels, n_filters,
                                          img_shape, filter_shape, (pad, pad))
    dfilters_fft_expr = dfilters_fft_op(imgs_bc01_t, convout_bc01_t)
    dfilters_fft_fun = theano.function([], host_from_gpu(dfilters_fft_expr))
    dfilters_fft = dfilters_fft_fun()
    print('bprop filters')
    print('allclose: ' + str(allclose(dfilters_fft, dfilters_cc)))
    duration_cc = avg_running_time(dfilters_cc_fun)
    dfilters_fft_fun = theano.function([], dfilters_fft_expr)
    duration_fft = avg_running_time(dfilters_fft_fun)
    print('cuda_convnet: %f  fft: %f' % (duration_cc, duration_fft))


def run():
    np.random.seed(1)
    # NB: In the current implementation of conv_bc01_fft, memory does not get
    #     free'd correctly. Therefore, we can only run one configuration for
    #     each program run.
    configurations = [
        (128, 3, (32, 32), 96, (11, 11), 0),
#        (128, 96, (32, 32), 256, (7, 7), 0),
#        (128, 256, (16, 16), 384, (5, 5), 0),
#        (128, 384, (16, 16), 384, (5, 5), 0),
#        (128, 384, (16, 16), 384, (3, 3), 0),
    ]

    for conf in configurations:
        benchmark(*conf)


if __name__ == '__main__':
    run()
