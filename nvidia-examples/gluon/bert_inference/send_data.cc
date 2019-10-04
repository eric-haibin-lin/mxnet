#include <cuda_runtime.h>
#include <cstdlib>

extern "C"
void send_data(void * cpu_ptr, void * gpu_ptr, size_t nbytes) {
    cudaMemcpyAsync(gpu_ptr, cpu_ptr, nbytes, cudaMemcpyHostToDevice, 0);
}

extern "C"
void recv_data_async(void * cpu_ptr, void * gpu_ptr, size_t nbytes) {
    cudaMemcpyAsync(cpu_ptr, gpu_ptr, nbytes, cudaMemcpyDeviceToHost, 0);
}

extern "C"
void recv_data_sync(void * cpu_ptr, void * gpu_ptr, size_t nbytes) {
    cudaMemcpy(cpu_ptr, gpu_ptr, nbytes, cudaMemcpyDeviceToHost);
}
