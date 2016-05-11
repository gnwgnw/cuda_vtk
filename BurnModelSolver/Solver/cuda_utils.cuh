#define cuda_check_error(ans) { cuda_assert((ans), __FILE__, __LINE__); }

inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "GPU assert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    throw Cuda_exception(cudaGetErrorString(code));
  }
}
