import sys
import numpy as np
import theano
import theano.tensor as T
from theano.gof import Apply, local_optimizer
from theano.sandbox.cuda import GpuOp, CudaNdarrayType
from theano.sandbox.cuda.opt import register_opt
from theano.sandbox.cuda.basic_ops import gpu_contiguous

from .abll_base import ABLLOp

_cache_version = (0, 1)


def conv_bc01(imgs, filters, n_imgs, n_channels, n_filters, img_shape,
              filter_shape, pad_shape):
    op = ConvBC01(n_imgs, n_channels, n_filters, img_shape, filter_shape,
                  pad_shape)
    return op(imgs, filters)


def round_up_fft_size(size):
    ''' Round up signal dimension to the nearest power of 2, 3, 5, or 7 for
    faster cuFFT operations.'''
    power = sys.maxsize
    for b in [2, 3, 5, 7]:
        for e in range(2, 13):
            candidate = b**e
            if candidate < power and candidate >= size:
                power = candidate
            elif candidate > power:
                break
    return power


def half_fft(size):
    ''' Calculate the size of the 'half' dimension in Fourier space. For the
    real-valued transforms CUFFT_R2C and CUFFT_C2R, cuFFT takes advantage of
    the Hermitian symmetry and use half the signal only.'''
    if size % 2 == 0:
        return size/2 + 2
    else:
        return size/2 + 1


def convout_shape(img_shape, filter_shape, pad_shape):
    return tuple(np.array(img_shape) + 2*np.array(pad_shape)
                 - np.array(filter_shape) + 1)


def fft_shape(img_shape, pad_shape):
    return (round_up_fft_size(img_shape[0] + 2*pad_shape[0]),
            round_up_fft_size(img_shape[1] + 2*pad_shape[1]))


def fft_to_01bc(imgs, n_imgs, n_channels, img_shape):
    op = FFT_TO_01BC(n_imgs, n_channels, img_shape)
    if 'Cuda' not in str(type(imgs)):
        imgs = gpu_contiguous(imgs)
    return op(imgs)


def ifft_to_bc01(ffts, n_imgs, n_channels, img_shape):
    op = IFFT_TO_BC01(n_imgs, n_channels, img_shape)
    if 'Cuda' not in str(type(ffts)):
        ffts = gpu_contiguous(ffts)
    return op(ffts)


def cgemm_batched(a, b, trans_a, trans_b, m, n, k, alpha, batch_size):
    if 'Cuda' not in str(type(a)):
        ffts = gpu_contiguous(a)
    if 'Cuda' not in str(type(b)):
        ffts = gpu_contiguous(b)
    op = CGEMMBatched(trans_a, trans_b, m, n, k, alpha, batch_size)
    return op(a, b)


def pad_bc01(imgs, n_imgs, n_channels, img_shape, padded_shape,
             offsets=(0, 0)):
    if img_shape == padded_shape:
        return imgs
    padded = T.zeros((n_imgs, n_channels) + padded_shape)
    start_y = offsets[0]
    end_y = img_shape[0]+offsets[0]
    start_x = offsets[1]
    end_x = img_shape[1]+offsets[1]
    padded = T.set_subtensor(padded[:, :, start_y:end_y, start_x:end_x], imgs)
    return padded


def crop_bc01(imgs, n_imgs, n_channels, img_shape, cropped_shape,
              offsets=(0, 0)):
    if img_shape == cropped_shape:
        return imgs
    start_y = offsets[0]
    end_y = cropped_shape[0]+offsets[0]
    start_x = offsets[1]
    end_x = cropped_shape[1]+offsets[1]
    return imgs[:, :, start_y:end_y, start_x:end_x]


def conv_bc01_fprop(imgs, filters, n_imgs, n_channels, n_filters, img_shape,
                    filter_shape, pad_shape):
    fft_shp = fft_shape(img_shape, pad_shape)
    convout_shp = convout_shape(img_shape, filter_shape, pad_shape)

    # Pad inputs
    imgs_padded = pad_bc01(imgs, n_imgs, n_channels, img_shape, fft_shp,
                           pad_shape)
    filters_padded = pad_bc01(filters, n_filters, n_channels, filter_shape,
                              fft_shp)

    # FFT
    imgs_fft = fft_to_01bc(imgs_padded, n_imgs, n_channels, fft_shp)
    filters_fft = fft_to_01bc(filters_padded, n_filters, n_channels, fft_shp)

    # Element-wise convolution
    m = n_filters
    n = n_imgs
    k = n_channels
    fft_size = fft_shp[0] * half_fft(fft_shp[1])
    alpha = 1.0/(np.prod(fft_shp))
    convout_fft = cgemm_batched(filters_fft, imgs_fft, True, False, m, n, k,
                                alpha, fft_size)

    # IFFT
    convout_padded = ifft_to_bc01(convout_fft, n_imgs, n_filters, fft_shp)

    # Crop output
    convout = crop_bc01(convout_padded, n_imgs, n_filters, fft_shp,
                        convout_shp)
    convout = gpu_contiguous(convout)
    return convout


def conv_bc01_bprop_imgs(filters, d_convout, n_imgs, n_channels, n_filters,
                         img_shape, filter_shape, pad_shape):
    fft_shp = fft_shape(img_shape, pad_shape)
    convout_shp = convout_shape(img_shape, filter_shape, pad_shape)

    # Pad inputs
    filters_padded = pad_bc01(filters, n_filters, n_channels, filter_shape,
                              fft_shp)
    d_convout_padded = pad_bc01(d_convout, n_imgs, n_filters, convout_shp,
                                fft_shp)

    # FFT
    filters_fft = fft_to_01bc(filters_padded, n_filters, n_channels, fft_shp)
    d_convout_fft = fft_to_01bc(d_convout_padded, n_imgs, n_filters, fft_shp)

    # Element-wise convolution
    m = n_channels
    n = n_imgs
    k = n_filters
    fft_size = fft_shp[0] * half_fft(fft_shp[1])
    alpha = 1.0/(np.prod(fft_shp))
    d_imgs_fft = cgemm_batched(filters_fft, d_convout_fft, False, False, m, n,
                               k, alpha, fft_size)

    # IFFT
    d_imgs_padded = ifft_to_bc01(d_imgs_fft, n_imgs, n_channels, fft_shp)

    # Crop output
    d_imgs = crop_bc01(d_imgs_padded, n_imgs, n_channels, fft_shp, img_shape,
                       pad_shape)
    d_imgs = gpu_contiguous(d_imgs)
    return d_imgs


def conv_bc01_bprop_filters(imgs, d_convout, n_imgs, n_channels, n_filters,
                            img_shape, filter_shape, pad_shape):
    fft_shp = fft_shape(img_shape, pad_shape)
    convout_shp = convout_shape(img_shape, filter_shape, pad_shape)

    # Pad inputs
    imgs_padded = pad_bc01(imgs, n_imgs, n_channels, img_shape, fft_shp,
                           pad_shape)
    d_convout_padded = pad_bc01(d_convout, n_imgs, n_filters, convout_shp,
                                fft_shp)

    # FFT
    imgs_fft = fft_to_01bc(imgs_padded, n_imgs, n_channels, fft_shp)
    d_convout_fft = fft_to_01bc(d_convout_padded, n_imgs, n_filters, fft_shp)

    # Element-wise convolution
    m = n_channels
    n = n_filters
    k = n_imgs
    fft_size = fft_shp[0] * half_fft(fft_shp[1])
    alpha = 1.0/(np.prod(fft_shp))
    d_filters_fft = cgemm_batched(imgs_fft, d_convout_fft, False, True, m, n,
                                  k, alpha, fft_size)

    # IFFT
    d_filters_padded = ifft_to_bc01(d_filters_fft, n_filters, n_channels,
                                    fft_shp)

    # Crop
    d_filters = crop_bc01(d_filters_padded, n_filters, n_channels, fft_shp,
                          filter_shape)
    d_filters = gpu_contiguous(d_filters)
    return d_filters


class CGEMMBatched(ABLLOp):
    def __init__(self, trans_a, trans_b, m, n, k, alpha, batch_size):
        # XXX: how do you get the Theano cublas handle?
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.m = m
        self.n = n
        self.k = k
        self.alpha = alpha
        self.batch_size = batch_size

    def c_headers(self):
        return [
            '<cublas_v2.h>',
            'abll/cuda.hpp',
        ]

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.trans_a == other.trans_a
            and self.trans_b == other.trans_b
            and self.m == other.m
            and self.n == other.n
            and self.k == other.k
            and self.alpha == other.alpha
            and self.batch_size == other.batch_size
        )

    def __hash__(self):
        msg = map(str, [self.trans_a, self.trans_b, self.m, self.n, self.k,
                        self.alpha, self.batch_size])
        msg.append(self.__class__.__name__)
        return hash(tuple(msg))

    def make_node(self, a, b):
        c_type = CudaNdarrayType((False,))
        c = c_type()
        return Apply(self, [a, b], [c])

    def c_support_code_apply(self, node, name):
        batch_size = self.batch_size
        code = """
void create_ptr_list(float2 ***ptrs_dev, float2 *base, int stride) {
  float2 *ptrs_host[%(batch_size)d];
  for(int i = 0; i < %(batch_size)d; i++){
    ptrs_host[i] = base + i * stride;
  }
  CUDA_CHECK(cudaMemcpy(*ptrs_dev, ptrs_host, %(batch_size)d*sizeof(float2 *),
                        cudaMemcpyHostToDevice));
}
float2 **a_ptrs = NULL;
float2 *a_ptrs_base = NULL;
float2 **b_ptrs = NULL;
float2 *b_ptrs_base = NULL;
float2 **c_ptrs = NULL;
float2 *c_ptrs_base = NULL;
        """ % locals()
        return code

    def c_init_code_apply(self, node, name):
        # TODO: deallocated ptrs
        batch_size = self.batch_size
        code = """
CUDA_CHECK(cudaMalloc(&a_ptrs, %(batch_size)d * sizeof(float2 **)));
CUDA_CHECK(cudaMalloc(&b_ptrs, %(batch_size)d * sizeof(float2 **)));
CUDA_CHECK(cudaMalloc(&c_ptrs, %(batch_size)d * sizeof(float2 **)));
        """ % locals()
        return code

    def c_code(self, node, name, inputs, outputs, sub):
        a, b = inputs
        c, = outputs
        fail = sub['fail']
        m = self.m
        n = self.n
        k = self.k
        if self.trans_a:
            trans_a = 'CUBLAS_OP_C'
            lda = k
            ldc = m
        else:
            trans_a = 'CUBLAS_OP_N'
            lda = m
            ldc = m
        if self.trans_b:
            trans_b = 'CUBLAS_OP_C'
            ldb = n
        else:
            trans_b = 'CUBLAS_OP_N'
            ldb = k
        alpha = float(self.alpha)
        batch_size = self.batch_size
        code = """
{
const int c_dims [] = {
    %(n)d * %(m)d * %(batch_size)d * 2,
};
if (CudaNdarray_prep_output(&%(c)s, 1, c_dims)) {
    %(fail)s;
}

float2 *a = (float2 *) %(a)s->devdata;
float2 *b = (float2 *) %(b)s->devdata;
float2 *c = (float2 *) %(c)s->devdata;

if (a_ptrs_base != a) {
    create_ptr_list(&a_ptrs, a, %(m)d * %(k)d);
    a_ptrs_base = a;
}
if (b_ptrs_base != b) {
    create_ptr_list(&b_ptrs, b, %(n)d * %(k)d);
    b_ptrs_base = b;
}
if (c_ptrs_base != c) {
    create_ptr_list(&c_ptrs, c, %(n)d * %(m)d);
    c_ptrs_base = c;
}

float2 alpha = {%(alpha).20f, 0.0};
float2 beta = {0.0, 0.0};
CUBLAS_CHECK(cublasCgemmBatched(
    CUDA::cublas_handle(), %(trans_a)s, %(trans_b)s, %(m)d, %(n)d, %(k)d,
    &alpha, (const float2**) a_ptrs, %(lda)d, (const float2**) b_ptrs, %(ldb)d,
    &beta, c_ptrs, %(ldc)d, %(batch_size)d
));
}
        """ % locals()
        return code

    def c_code_cache_version(self):
        return _cache_version


class FFTBase(ABLLOp):
    def __init__(self, n_imgs, n_channels, img_shape):
        self.n_imgs = n_imgs
        self.n_channels = n_channels
        self.img_shape = img_shape

    def c_headers(self):
        return [
            '<cufft.h>',
            'abll/cuda.hpp',
        ]

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.n_imgs == other.n_imgs
            and self.n_channels == other.n_channels
            and self.img_shape == other.img_shape
        )

    def __hash__(self):
        msg = map(str, [self.n_imgs, self.n_channels, self.n_channels])
        msg.append(self.__class__.__name__)
        return hash(tuple(msg))

    def c_code_cache_version(self):
        return _cache_version


class FFT_TO_01BC(FFTBase):
    def make_node(self, imgs):
        ffts_type = CudaNdarrayType((False,))
        ffts = ffts_type()
        return Apply(self, [imgs], [ffts])

    def c_code(self, node, name, inputs, outputs, sub):
        imgs, = inputs
        ffts, = outputs
        fail = sub['fail']
        batch_size = self.n_imgs * self.n_channels
        img_h, img_w = self.img_shape
        img_w_half = half_fft(img_w)
        # Double size because we are working with float2
        fft_size = batch_size * img_h * half_fft(img_w) * 2
        code = """
{
const int ffts_dims [] = {
    %(fft_size)d
};
if (CudaNdarray_prep_output(&%(ffts)s, 1, ffts_dims)) {
    %(fail)s;
}
float *imgs = %(imgs)s->devdata;
float2 *ffts = (float2 *) %(ffts)s->devdata;

int rank = 2;
int input_dims[2] = {%(img_h)d, %(img_w)d};
int inembed[2] = {%(img_h)d, %(img_w)d};
int onembed[2] = {%(img_h)d, %(img_w_half)d};
int idist = %(img_w)d * %(img_h)d;
int istride = 1;
int odist = 1;
int ostride = %(batch_size)d;

cufftHandle plan;
CUFFT_CHECK(cufftCreate(&plan));
CUFFT_CHECK(cufftPlanMany(&plan, rank, input_dims, inembed, istride, idist,
                          onembed, ostride, odist, CUFFT_R2C, %(batch_size)d));
CUFFT_CHECK(cufftExecR2C(plan, imgs, ffts));
CUFFT_CHECK(cufftDestroy(plan));
}
        """ % locals()
        return code


class IFFT_TO_BC01(FFTBase):
    def make_node(self, imgs):
        ffts_type = CudaNdarrayType((False, False, False, False))
        ffts = ffts_type()
        return Apply(self, [imgs], [ffts])

    def c_code(self, node, name, inputs, outputs, sub):
        ffts, = inputs
        imgs, = outputs
        fail = sub['fail']
        n_imgs = self.n_imgs
        n_channels = self.n_channels
        batch_size = self.n_imgs * self.n_channels
        img_h, img_w = self.img_shape
        img_w_half = half_fft(img_w)
        code = """
{
const int imgs_dims [] = {
    %(n_imgs)d,
    %(n_channels)d,
    %(img_h)d,
    %(img_w)d
};
if (CudaNdarray_prep_output(&%(imgs)s, 4, imgs_dims)) {
    %(fail)s;
}
float2 *ffts = (float2 *) %(ffts)s->devdata;
float *imgs = %(imgs)s->devdata;

int rank = 2;
int input_dims[2] = {%(img_h)d, %(img_w)d};
int inembed[2] = {%(img_h)d, %(img_w_half)d};
int onembed[2] = {%(img_h)d, %(img_w)d};
int idist = 1;
int istride = %(batch_size)d;
int odist = %(img_w)d * %(img_h)d;
int ostride = 1;
cufftHandle plan;
CUFFT_CHECK(cufftCreate(&plan));
CUFFT_CHECK(cufftPlanMany(&plan, rank, input_dims, inembed, istride, idist,
                          onembed, ostride, odist, CUFFT_C2R, %(batch_size)d));
CUFFT_CHECK(cufftExecC2R(plan, ffts, imgs));
CUFFT_CHECK(cufftDestroy(plan));
}
        """ % locals()
        return code


class ConvBC01Base(GpuOp):
    def __init__(self, n_imgs, n_channels, n_filters, img_shape, filter_shape,
                 pad_shape):
        # XXX: can theano be used to infer shapes (using infer_shape())
        self.n_imgs = n_imgs
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.img_shape = img_shape
        self.filter_shape = filter_shape
        self.pad_shape = pad_shape

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

    def make_node(self, in1, in2):
        out = CudaNdarrayType((False, False, False, False))
        return Apply(self, [in1, in2], [out()])

    def c_code_cache_version(self):
        return _cache_version


class ConvBC01(ConvBC01Base):
    def grad(self, inputs, d_outputs):
        imgs, filters = inputs
        d_convout, = d_outputs
        d_imgs = ConvBC01ImgsGrad(
            self.n_imgs, self.n_channels, self.n_filters, self.img_shape,
            self.filter_shape, self.pad_shape)(filters, d_convout)
        d_filters = ConvBC01FiltersGrad(
            self.n_imgs, self.n_channels, self.n_filters, self.img_shape,
            self.filter_shape, self.pad_shape)(imgs, d_convout)
        return d_imgs, d_filters


class ConvBC01ImgsGrad(ConvBC01Base):
    pass


class ConvBC01FiltersGrad(ConvBC01Base):
    pass


@register_opt()
@local_optimizer([ConvBC01])
def conv_bco1_fprop_wrap(node):
    op = node.op
    if isinstance(op, ConvBC01):
        imgs = node.inputs[0]
        filters = node.inputs[1]
        convout = conv_bc01_fprop(
            imgs, filters, op.n_imgs, op.n_channels, op.n_filters,
            op.img_shape, op.filter_shape, op.pad_shape
        )
        return [convout]


@register_opt()
@local_optimizer([ConvBC01ImgsGrad])
def conv_bco1_bprop_imgs_wrap(node):
    op = node.op
    if isinstance(op, ConvBC01ImgsGrad):
        filters = node.inputs[0]
        d_convout = node.inputs[1]
        d_imgs = conv_bc01_bprop_imgs(
            filters, d_convout, op.n_imgs, op.n_channels, op.n_filters,
            op.img_shape, op.filter_shape, op.pad_shape
        )
        return [d_imgs]


@register_opt()
@local_optimizer([ConvBC01FiltersGrad])
def conv_bco1_bprop_filters_wrap(node):
    op = node.op
    if isinstance(op, ConvBC01FiltersGrad):
        imgs = node.inputs[0]
        d_convout = node.inputs[1]
        d_filters = conv_bc01_bprop_filters(
            imgs, d_convout, op.n_imgs, op.n_channels, op.n_filters,
            op.img_shape, op.filter_shape, op.pad_shape
        )
        return [d_filters]
