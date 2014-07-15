#ifndef CONV_BC01_FFT_HPP_
#define CONV_BC01_FFT_HPP_

#include <cufft.h>


class ConvBC01FFT {
public:
  ConvBC01FFT(int n_imgs, int n_channels, int img_h, int img_w, int n_filters,
              int filter_h, int filter_w, int pad_y, int pad_x);
  ~ConvBC01FFT();
  void conv(float *imgs, float *filters, float *convout);
  void bprop_imgs(float *filters, float *convout_grad, float *imgs_grad);
  void bprop_filters(float *imgs, float *convout_grad, float *filters_grad);

private:
  int n_imgs;
  int n_channels;
  int img_h;
  int img_w;
  int n_filters;
  int filter_h;
  int filter_w;
  int pad_y;
  int pad_x;
  int convout_h;
  int convout_w;
  int fft_h;
  int fft_w;
  int fft_size;
  float fft_scale;
  float2 *filters_fft;
  float2 *imgs_fft;
  float2 *convout_fft;
  float *filters_padded;
  float *imgs_padded;
  float *convout_padded;

  cufftHandle plan_imgs_fft;
  cufftHandle plan_filters_fft;
  cufftHandle plan_convout_fft;
  cufftHandle plan_imgs_ifft;
  cufftHandle plan_filters_ifft;
  cufftHandle plan_convout_ifft;
  float2 **imgs_ptrs;
  float2 **filters_ptrs;
  float2 **convout_ptrs;
};
 
#endif  // CONV_BC01_HPP_
