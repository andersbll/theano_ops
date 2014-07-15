#include <climits>
#include <cmath>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

#include "abll/cuda.hpp"
#include "abll/conv_bc01_fft.hpp"


/*
  Round up signal dimension to the nearest power of 2, 3, 5, or 7 for faster
  cuFFT operations.
*/
unsigned int round_up_fft_size(int size) {
  unsigned int power = UINT_MAX;
  int bases[] = {2, 3, 5, 7};
  for (int b = 0; b < 4; ++b) {
    for (int e = 0; e < 12; ++e) {
      int candidate = std::pow(bases[b], e);
      if (candidate < power && candidate >= size) {
        power = candidate;
      } else if (candidate > power) {
        break;
      }
    }
  }
  return power;
}


/*
  Calculate the size of the 'half' dimension in Fourier space.
  For the real-valued transforms CUFFT_R2C and CUFFT_C2R, cuFFT takes advantage
  of the Hermitian symmetry and use half the signal only.
*/
inline int half_fft(int size) {
  if (size % 2 == 0) {
    return size/2 + 2;
  } else {
    return size/2 + 1;
  }
}


__global__ void pad_b01_kernel(const float *imgs, int n_threads, int n_imgs,
    int img_h, int img_w, int y_offset, int x_offset, int padded_h,
    int padded_w, float *imgs_padded) {
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    int x = idx % img_w;
    int y = (idx / img_w) % img_h;
    int b = idx / img_w / img_h;
    y = (padded_h + y + y_offset) % padded_h;
    x = (padded_w + x + x_offset) % padded_w;
    int padded_idx = (b * padded_h + y) * padded_w + x;
    imgs_padded[padded_idx] = imgs[idx];
  }
}

void pad_b01(const float *imgs, int n_imgs, int img_h, int img_w, int y_offset,
             int x_offset, int padded_h, int padded_w, float *imgs_padded) {
  int imgs_size = n_imgs * img_h * img_w;
  int imgs_padded_size = n_imgs * padded_h * padded_w;
  cudaMemset(imgs_padded, 0, sizeof(float)*imgs_padded_size);
  CUDA_DEBUG_SYNC("memset(filters) failed.");
  pad_b01_kernel<<<CUDA_BLOCKS(imgs_size), CUDA_NUM_THREADS>>>(
      imgs, imgs_size, n_imgs, img_h, img_w, y_offset, x_offset, padded_h,
      padded_w, imgs_padded
  );
  CUDA_DEBUG_SYNC("pad_b01 failed.");
}


__global__ void crop_b01_kernel(const float *imgs, int n_threads, int n_imgs,
    int img_h, int img_w, int y_offset, int x_offset,
    int cropped_h, int cropped_w, float *imgs_cropped) {
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    int x = idx % cropped_w;
    int y = (idx / cropped_w) % cropped_h;
    const int b = idx / cropped_w / cropped_h;
    y = (y_offset + y + img_h) % img_h;
    x = (x_offset + x + img_w) % img_w;
    imgs_cropped[idx] = imgs[(b * img_h + y) * img_w + x];
  }
}


void crop_b01(const float *imgs, int n_imgs, int img_h, int img_w,
              int y_offset, int x_offset, int cropped_h, int cropped_w,
              float *imgs_cropped) {
  int imgs_cropped_size = n_imgs * cropped_h * cropped_w;
  crop_b01_kernel<<<CUDA_BLOCKS(imgs_cropped_size), CUDA_NUM_THREADS>>>(
      imgs, imgs_cropped_size, n_imgs, img_h, img_w, y_offset,
      x_offset, cropped_h, cropped_w, imgs_cropped);
  CUDA_DEBUG_SYNC("crop_b01 failed.");
}


/*
  img: (n_imgs X w X h)
  img_f: (w X h X n_imgs)
*/
void plan_fft_b01(int n_imgs, int w, int h, cufftHandle *plan) {
    cufftType_t type = CUFFT_R2C;
    int rank = 2;
    int input_dims[2] = {w, h};
    int inembed[2] = {w, h};
    int onembed[2] = {w, half_fft(h)};
    int idist = w * h;
    int istride = 1;
    int odist = 1;
    int ostride = n_imgs;
    CUFFT_CHECK(cufftCreate(plan));
    CUFFT_CHECK(cufftPlanMany(plan, rank, input_dims, inembed, istride, idist,
                              onembed, ostride, odist, type, n_imgs));
}


/*
  img_f: (w X h X n_imgs)
  img: (n_imgs X w X h)
*/
void plan_ifft_b01(int n_imgs, int w, int h, cufftHandle *plan) {
    cufftType_t type = CUFFT_C2R;
    int rank = 2;
    int input_dims[2] = {w, h};
    int inembed[2] = {w, half_fft(h)};
    int onembed[2] = {w, h};
    int idist = 1;
    int istride = n_imgs;
    int odist = w * h;
    int ostride = 1;
    CUFFT_CHECK(cufftCreate(plan));
    CUFFT_CHECK(cufftPlanMany(plan, rank, input_dims, inembed, istride, idist,
                              onembed, ostride, odist, type, n_imgs));
}


/*
  Allocate a pointer array on the GPU and fill it up with references to
  base + i*stride for i = 0...stride-1.
*/
float2 **create_ptr_list(float2 *base, int size, int stride) {
  float2 *list_host[size];
  for(int i = 0; i < size; i++){
    list_host[i] = base + i * stride;
  }
  float2 **list_dev;
  CUDA_CHECK(cudaMalloc(&list_dev, size * sizeof(float2 **)));
  CUDA_CHECK(cudaMemcpy(list_dev, list_host, size * sizeof(float2 *),
                        cudaMemcpyHostToDevice));
  return list_dev;
}


ConvBC01FFT::ConvBC01FFT(int n_imgs, int n_channels, int img_h, int img_w,
    int n_filters, int filter_h, int filter_w, int pad_y, int pad_x)
    : n_imgs(n_imgs), n_channels(n_channels), img_h(img_h), img_w(img_w),
      n_filters(n_filters), filter_h(filter_h), filter_w(filter_w),
      pad_y(pad_y), pad_x(pad_x) {

  assert(filter_h <= img_h && filter_w <= img_w);

  if (pad_y > 0 || pad_x > 0) { 
    // Round up FFT sizes to speed up cuFFT.
    fft_h = round_up_fft_size(img_h + 2*pad_y);
    fft_w = round_up_fft_size(img_w + 2*pad_x);
  } else {
    // Don't round up FFTs; this is cheaper as we can avoid padding imgs.
    // XXX: is this really faster?
    fft_h = img_h;
    fft_w = img_w;
  }
  convout_h = img_h + 2*pad_y - filter_h + 1;
  convout_w = img_w + 2*pad_x - filter_w + 1;

  fft_size = fft_h * half_fft(fft_w);
  fft_scale = 1.0f / float(fft_h*fft_w);

  // Setup cuFFT plans
  plan_fft_b01(n_imgs * n_channels, fft_h, fft_w, &plan_imgs_fft);
  plan_fft_b01(n_filters * n_channels, fft_h, fft_w, &plan_filters_fft);
  plan_fft_b01(n_imgs * n_filters, fft_h, fft_w, &plan_convout_fft);
  plan_ifft_b01(n_imgs * n_channels, fft_h, fft_w, &plan_imgs_ifft);
  plan_ifft_b01(n_filters * n_channels, fft_h, fft_w, &plan_filters_ifft);
  plan_ifft_b01(n_imgs * n_filters, fft_h, fft_w, &plan_convout_ifft);

  // Request a buffer size to contain all buffers
  int filters_fft_size = n_filters * n_channels * fft_size;
  int imgs_fft_size = n_imgs * n_channels * fft_size;
  int convout_fft_size = n_imgs * n_filters * fft_size;
  CUDA::require_buffer_size(
      sizeof(float2) * filters_fft_size
      + sizeof(float2) * imgs_fft_size
      + sizeof(float2) * convout_fft_size
  );
  // XXX: this is not correct; if the buffer changes the pointers are invalid.
  // TODO: figure out how the buffer should work (must work with theano).
  filters_fft = (float2 *) CUDA::buffer();
  imgs_fft = filters_fft + filters_fft_size;
  convout_fft = imgs_fft + imgs_fft_size;
  // Perform FFT operations in-place by letting padded arrays point to the fft
  // arrays
  filters_padded = (float *) filters_fft;
  imgs_padded = (float *) imgs_fft;
  convout_padded = (float *) convout_fft;
  
  // Pointer lists for cuBLAS operations
  imgs_ptrs = create_ptr_list(imgs_fft, fft_size, n_imgs*n_channels);
  filters_ptrs = create_ptr_list(filters_fft, fft_size, n_filters*n_channels);
  convout_ptrs = create_ptr_list(convout_fft, fft_size, n_imgs*n_filters);
}


ConvBC01FFT::~ConvBC01FFT() {
  cudaFree(imgs_ptrs);
  cudaFree(filters_ptrs);
  cudaFree(convout_ptrs);
  CUFFT_CHECK(cufftDestroy(plan_imgs_fft));
  CUFFT_CHECK(cufftDestroy(plan_filters_fft));
  CUFFT_CHECK(cufftDestroy(plan_convout_fft));
  CUFFT_CHECK(cufftDestroy(plan_imgs_ifft));
  CUFFT_CHECK(cufftDestroy(plan_filters_ifft));
  CUFFT_CHECK(cufftDestroy(plan_convout_ifft));
}


void ConvBC01FFT::conv(float* imgs, float* filters, float* convout) {
  float *imgs_ptr = imgs;
  if (img_h != fft_h || img_w != fft_w) {
    pad_b01(imgs, n_imgs * n_channels, img_h, img_w, pad_y, pad_x, fft_h,
            fft_w, imgs_padded);
    imgs_ptr = imgs_padded;
  }
  pad_b01(filters, n_filters * n_channels, filter_h, filter_w, 0, 0, fft_h,
          fft_w, filters_padded);

  CUFFT_CHECK(cufftExecR2C(plan_imgs_fft, imgs_ptr, imgs_fft));
  CUFFT_CHECK(cufftExecR2C(plan_filters_fft, filters_padded, filters_fft));

  int m = n_filters;
  int n = n_imgs;
  int k = n_channels;
  int batch = fft_size;
  float2 alpha = {fft_scale, 0.0};
  float2 beta = {0.0, 0.0};
  int lda = k;
  int ldb = k;
  int ldc = m;
  CUBLAS_CHECK(cublasCgemmBatched(
      CUDA::cublas_handle(), CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha,
      (const float2**) filters_ptrs, lda, (const float2**) imgs_ptrs, ldb,
      &beta, convout_ptrs, ldc, batch
  ));

  CUFFT_CHECK(cufftExecC2R(plan_convout_ifft, convout_fft, convout_padded));
  crop_b01(convout_padded, n_imgs*n_filters, fft_h, fft_w,
           0, 0, convout_h, convout_w, convout);
}


void ConvBC01FFT::bprop_imgs(float *filters, float *convout_grad,
                             float *imgs_grad) {
  pad_b01(filters, n_filters * n_channels, filter_h, filter_w, 0, 0, fft_h,
          fft_w, filters_padded);
  pad_b01(convout_grad, n_imgs * n_filters, convout_h, convout_w, -pad_y,
          -pad_x, fft_h, fft_w, convout_padded);

  CUFFT_CHECK(cufftExecR2C(plan_filters_fft, filters_padded, filters_fft));
  CUFFT_CHECK(cufftExecR2C(plan_convout_fft, convout_padded, convout_fft));

  int m = n_channels;
  int n = n_imgs;
  int k = n_filters;
  int batch = fft_size;
  float2 alpha = {fft_scale, 0.0};
  float2 beta = {0.0, 0.0};
  int lda = m;
  int ldb = k;
  int ldc = m;
  CUBLAS_CHECK(cublasCgemmBatched(
      CUDA::cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      (const float2**) filters_ptrs, lda, (const float2**) convout_ptrs, ldb,
      &beta, imgs_ptrs, ldc, batch
  ));

  if (img_h != fft_h || img_w != fft_w) {
    CUFFT_CHECK(cufftExecC2R(plan_imgs_ifft, imgs_fft, imgs_padded));
    crop_b01(imgs_padded, n_imgs*n_channels, fft_h, fft_w, 0, 0, img_h, img_w,
             imgs_grad);
  } else {
    CUFFT_CHECK(cufftExecC2R(plan_imgs_ifft, imgs_fft, imgs_grad));  
  }
}


void ConvBC01FFT::bprop_filters(float *imgs, float *convout_grad,
                                float *filters_grad) {
  float *imgs_ptr = imgs;
  if (img_h != fft_h || img_w != fft_w) {
    pad_b01(imgs, n_imgs * n_channels, img_h, img_w, 0, 0, fft_h, fft_w,
            imgs_padded);
    imgs_ptr = imgs_padded;
  }
  pad_b01(convout_grad, n_imgs * n_filters, convout_h, convout_w,
          -pad_y, -pad_x, fft_h, fft_w, convout_padded);

  CUFFT_CHECK(cufftExecR2C(plan_imgs_fft, imgs_ptr, imgs_fft));
  CUFFT_CHECK(cufftExecR2C(plan_convout_fft, convout_padded, convout_fft));

  int m = n_channels;
  int n = n_filters;
  int k = n_imgs;
  int batch = fft_size;
  float2 alpha = {fft_scale, 0.0};
  float2 beta = {0.0, 0.0};
  int lda = m;
  int ldb = n;
  int ldc = m;
  CUBLAS_CHECK(cublasCgemmBatched(
      CUDA::cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_C, m, n, k, &alpha,
      (const float2**) imgs_ptrs, lda, (const float2**) convout_ptrs, ldb,
      &beta, filters_ptrs, ldc, batch
  ));

  CUFFT_CHECK(cufftExecC2R(plan_filters_ifft, filters_fft, filters_padded));
  crop_b01(filters_padded, n_filters*n_channels, fft_h, fft_w, 0, 0, filter_h,
           filter_w, filters_grad);
}
