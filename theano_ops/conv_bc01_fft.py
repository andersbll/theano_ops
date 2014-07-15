from theano.sandbox.cuda import CudaNdarrayType
from theano.gof import Apply
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda import gpu_from_host

from .abll_base import ABLLOp


def conv_bc01(imgs, filters, n_imgs, n_channels, n_filters, img_shape,
              filter_shape, pad_shape):
    op = ConvBC01(n_imgs, n_channels, n_filters, img_shape, filter_shape,
                  pad_shape)
    if 'Cuda' not in str(type(imgs)):
        imgs = gpu_from_host(imgs)
        imgs = gpu_contiguous(imgs)
    if 'Cuda' not in str(type(filters)):
        filters = gpu_from_host(filters)
        filters = gpu_contiguous(filters)
    return op(imgs, filters)


class ConvBC01Base(ABLLOp):
    def __init__(self, n_imgs, n_channels, n_filters, img_shape, filter_shape,
                 pad_shape):
        # TODO: infer shapes from matrices
        self.n_imgs = n_imgs
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.img_shape = img_shape
        self.filter_shape = filter_shape
        self.pad_shape = pad_shape

    def c_headers(self):
        return [
            'abll/conv_bc01_fft.hpp',
        ]

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.n_imgs == other.n_imgs
            and self.n_channels == other.n_channels
            and self.n_filters == other.n_filters
            and self.img_shape == other.img_shape
            and self.filter_shape == other.filter_shape
            and self.pad_shape == other.pad_shape
        )

    def __hash__(self):
        msg = map(str, [self.n_imgs, self.n_channels, self.n_filters,
                        self.img_shape, self.filter_shape, self.pad_shape])
        msg.append(self.__class__.__name__)
        return hash(tuple(msg))

    def c_support_code_apply(self, node, name):
        # XXX: How do you destroy convBC01FFT properly? Implementing
        #      c_code_cleanup() doesn't work.
        # XXX: Is it possible to reuse convBC01FFT for the gradient back
        #      propagations? Currently, a convBC01FFT is created per function.
        n_imgs = self.n_imgs
        n_channels = self.n_channels
        n_filters = self.n_filters
        img_height, img_width = self.img_shape
        filter_height, filter_width = self.filter_shape
        pad_y, pad_x = self.pad_shape
        code = """
ConvBC01FFT convBC01FFT(
    %(n_imgs)d, %(n_channels)d, %(img_height)d, %(img_width)d, %(n_filters)d,
    %(filter_height)d, %(filter_width)d, %(pad_y)d, %(pad_x)d
);
        """ % locals()
        return code

    def c_code_cache_version(self):
        return (0, 1)


class ConvBC01(ConvBC01Base):
    def make_node(self, imgs, filters):
        # XXX: The following works but is probably not ideal wrt. theano.
        if not isinstance(imgs.type, CudaNdarrayType):
            raise TypeError('imgs.type should be CudaNdarrayType, '
                            'got '+str(imgs.type))
        if not isinstance(filters.type, CudaNdarrayType):
            raise TypeError('filters.type should be CudaNdarrayType, '
                            'got '+str(filters.type))
        if not imgs.ndim == 4:
            raise ValueError('imgs.ndim should be 4, got %d' % imgs.ndim)
        if not filters.ndim == 4:
            raise ValueError('filters.ndim should be 4, got %d' % imgs.ndim)

        batch_broadcastable = imgs.type.broadcastable[0]
        channel_broadcastable = imgs.type.broadcastable[1]
        rows_broadcastable = False
        cols_broadcastable = False
        convout_broadcastable = (batch_broadcastable, channel_broadcastable,
                                 rows_broadcastable, cols_broadcastable)
        convout_type = CudaNdarrayType(broadcastable=convout_broadcastable)
        convout = convout_type()
        return Apply(self, [imgs, filters], [convout])

    def c_code(self, node, name, inputs, outputs, sub):
        imgs, filters = inputs
        convout, = outputs
        fail = sub['fail']
        n_imgs = self.n_imgs
        n_filters = self.n_filters
        img_height, img_width = self.img_shape
        filter_height, filter_width = self.filter_shape
        pad_y, pad_x = self.pad_shape
        convout_height = img_height + 2*pad_y - filter_height + 1
        convout_width = img_width + 2*pad_x - filter_width + 1

        code = """
{
const int convout_dims [] = {
    %(n_imgs)d,
    %(n_filters)d,
    %(convout_height)d,
    %(convout_width)d
};
if (CudaNdarray_prep_output(&%(convout)s, 4, convout_dims)) {
    %(fail)s;
}

float *imgs = %(imgs)s->devdata;
float *filters = %(filters)s->devdata;
float *convout = %(convout)s->devdata;

convBC01FFT.conv(imgs, filters, convout);
}
        """ % locals()
        return code

    def grad(self, inputs, d_outputs):
        imgs, filters = inputs
        d_convout, = d_outputs
        if 'Cuda' not in str(type(imgs)):
            raise TypeError('imgs must be CUDA')
        if 'Cuda' not in str(type(filters)):
            raise TypeError('filters must be CUDA')
        if 'Cuda' not in str(type(d_convout)):
            raise TypeError('output gradients must be CUDA')

        d_imgs = ConvBC01ImgsGrad(
            self.n_imgs, self.n_channels, self.n_filters, self.img_shape,
            self.filter_shape, self.pad_shape)(filters, d_convout)
        d_filters = ConvBC01FiltersGrad(
            self.n_imgs, self.n_channels, self.n_filters, self.img_shape,
            self.filter_shape, self.pad_shape)(imgs, d_convout)
        return d_imgs, d_filters


class ConvBC01ImgsGrad(ConvBC01Base):
    def make_node(self, filters, d_convout):
        # XXX: The following works but is probably not ideal wrt. theano.
        if not isinstance(d_convout.type, CudaNdarrayType):
            raise TypeError('d_convout.type should be CudaNdarrayType, '
                            'got '+str(d_convout.type))
        if not isinstance(filters.type, CudaNdarrayType):
            raise TypeError('filters.type should be CudaNdarrayType, '
                            'got '+str(filters.type))
        channel_broadcastable = filters.type.broadcastable[3]
        batch_broadcastable = d_convout.type.broadcastable[3]
        rows_broadcastable = False
        cols_broadcastable = False
        d_imgs_broadcastable = (channel_broadcastable, rows_broadcastable,
                                cols_broadcastable, batch_broadcastable)
        d_imgs_type = CudaNdarrayType(broadcastable=d_imgs_broadcastable)
        d_imgs = d_imgs_type()
        return Apply(self, [filters, d_convout], [d_imgs])

    def c_code(self, node, name, inputs, outputs, sub):
        filters, d_convout = inputs
        d_imgs, = outputs
        fail = sub['fail']
        n_imgs = self.n_imgs
        n_channels = self.n_channels
        img_height, img_width = self.img_shape
        code = """
{
const int imgs_dims [] = {
    %(n_imgs)d,
    %(n_channels)d,
    %(img_height)d,
    %(img_width)d
};
if (CudaNdarray_prep_output(&%(d_imgs)s, 4, imgs_dims)) {
    %(fail)s;
}

float *filters = %(filters)s->devdata;
float *d_convout = %(d_convout)s->devdata;
float *d_imgs = %(d_imgs)s->devdata;

convBC01FFT.bprop_imgs(filters, d_convout, d_imgs);
}
        """ % locals()
        return code


class ConvBC01FiltersGrad(ConvBC01Base):
    def make_node(self, imgs, d_convout):
        # XXX: The following works but is probably not ideal wrt. theano.
        input_channel_broadcastable = imgs.type.broadcastable[0]
        filter_rows_broadcastable = False
        filter_cols_broadcastable = False
        output_channel_broadcastable = d_convout.type.broadcastable[0]
        d_filters_type = CudaNdarrayType(
            (input_channel_broadcastable, filter_rows_broadcastable,
             filter_cols_broadcastable, output_channel_broadcastable)
        )
        d_filters = d_filters_type()
        return Apply(self, [imgs, d_convout], [d_filters])

    def c_code(self, node, name, inputs, outputs, sub):
        imgs, d_convout = inputs
        d_filters, = outputs
        fail = sub['fail']
        n_channels = self.n_channels
        n_filters = self.n_filters
        filter_height, filter_width = self.filter_shape
        code = """
{
const int filters_dims [] = {
    %(n_filters)d,
    %(n_channels)d,
    %(filter_height)d,
    %(filter_width)d
};
if (CudaNdarray_prep_output(&%(d_filters)s, 4, filters_dims)) {
    %(fail)s;
}

float *imgs = %(imgs)s->devdata;
float *d_convout = %(d_convout)s->devdata;
float *d_filters = %(d_filters)s->devdata;

convBC01FFT.bprop_filters(imgs, d_convout, d_filters);
}
        """ % locals()
        return code
