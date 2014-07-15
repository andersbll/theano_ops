#ifndef CUDA_HPP_
#define CUDA_HPP_

#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>


#define CUDA_CHECK(condition) { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
        std::cout << " " << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)


char *cublasErrorString(cublasStatus_t err){
	switch(err) {
		case CUBLAS_STATUS_SUCCESS :
			return "operation completed successfully";
		case CUBLAS_STATUS_NOT_INITIALIZED :
			return "CUBLAS library not initialized";
		case CUBLAS_STATUS_ALLOC_FAILED :
			return "resource allocation failed";
		case CUBLAS_STATUS_INVALID_VALUE :
			return "unsupported numerical value was passed to function";
		case CUBLAS_STATUS_ARCH_MISMATCH :
			return "function requires an architectural feature absent from \
			the architecture of the device";
		case CUBLAS_STATUS_MAPPING_ERROR :
			return "access to GPU memory space failed";
		case CUBLAS_STATUS_EXECUTION_FAILED :
			return "GPU program failed to execute";
		case CUBLAS_STATUS_INTERNAL_ERROR :
			return "an internal CUBLAS operation failed";
		default :
			return "unknown error type";
	}
}
#define CUBLAS_CHECK(condition) { \
    cublasStatus_t status = condition; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cout << " " << cublasErrorString(status) << std::endl; \
    } \
  }


static const char *cufftErrorEnum(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }
    return "<unknown>";
}
#define CUFFT_CHECK(condition) { \
    cufftResult status = condition; \
    if (status != CUFFT_SUCCESS) { \
        std::cout << " " << cufftErrorEnum(status) << std::endl; \
    } \
  }


inline void cudaSyncCheck(const char *msg, const char *file, const int line) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i):\nCUDA error : (%d) %s\n", file, line, (int)err,
            cudaGetErrorString(err));
    fprintf(stderr, "Message : %s\n", msg);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#ifdef DEBUG
#define CUDA_DEBUG_SYNC(msg) cudaSyncCheck(msg, __FILE__, __LINE__)
#else
#define CUDA_DEBUG_SYNC(msg)
#endif

#define CUDA_GRID_STRIDE_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; \
       i += blockDim.x * gridDim.x)

#define CUDA_NUM_THREADS 1024

#define CUDA_BLOCKS(n_threads) \
  ((n_threads + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)


/*
  Singleton class to handle CUDA resources.
*/
class CUDA {
public:
  inline static CUDA &instance() {
    static CUDA instance_;
    return instance_;
  }

  static void require_buffer_size(int size) {
    if (instance().buffer_size < size) {
      instance().buffer_size = size;
      if (instance().buffer_) {
        cudaFree(instance().buffer_);
        CUDA_DEBUG_SYNC("Could not free buffer.");
        instance().buffer_ = NULL;
      }
    }
  }

  inline static void *buffer() {
    if (!instance().buffer_) {
      if (instance().buffer_size <= 0) {
        throw std::runtime_error("No buffer size has been specified.");
      }
      CUDA_CHECK(cudaMalloc(&instance().buffer_, instance().buffer_size));
    }
    return instance().buffer_;
  }

  inline static cublasHandle_t &cublas_handle() { 
    return instance().cublas_handle_;
  }

private:
  cublasHandle_t cublas_handle_;
  void *buffer_;
  int buffer_size;

  CUDA() : buffer_(NULL), buffer_size(-1) {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  }

  ~CUDA() {
    if (buffer_) {
      // This segfaults with Theano (I the CUDA runtime is already shut down
      // at this point) 
//      CUDA_CHECK(cudaFree(buffer_));
    }
    cudaDeviceReset();
  }

  CUDA(CUDA const&);

  void operator=(CUDA const&);
};


#endif  // CUDA_HPP_
