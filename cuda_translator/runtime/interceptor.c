/**
 * CUDA Driver API Interceptor Library (nvcuda.dll / libcuda.so drop-in replacement prototype).
 * Intercepts application calls to CUDA Driver API and redirects them to the CUDA_Translator engine.
 */

#include "include/cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

EXPORT_API CUresult cuInit(unsigned int Flags) {
    printf("[CUDA_Translator] Intercepted cuInit(Flags=%u)\n", Flags);
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuDeviceGetCount(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = 1;
    printf("[CUDA_Translator] Intercepted cuDeviceGetCount -> 1\n");
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = ordinal;
    printf("[CUDA_Translator] Intercepted cuDeviceGet(ordinal=%d)\n", ordinal);
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    const char *dev_name = "CUDA_Translator Virtualized Accelerator (Vulkan/ROCm)";
    strncpy(name, dev_name, len - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = (CUcontext)0x1;
    printf("[CUDA_Translator] Intercepted cuCtxCreate -> Context Initialized\n");
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuCtxDestroy(CUcontext ctx) {
    printf("[CUDA_Translator] Intercepted cuCtxDestroy\n");
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    if (!dptr || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    void *ptr = malloc(bytesize);
    if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
    *dptr = (CUdeviceptr)ptr;
    printf("[CUDA_Translator] Intercepted cuMemAlloc(bytes=%zu) -> 0x%zx\n", bytesize, (size_t)*dptr);
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuMemFree(CUdeviceptr dptr) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    free((void*)dptr);
    printf("[CUDA_Translator] Intercepted cuMemFree(0x%zx)\n", (size_t)dptr);
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    if (!dstDevice || !srcHost) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void*)dstDevice, srcHost, ByteCount);
    printf("[CUDA_Translator] Intercepted cuMemcpyHtoD(bytes=%zu)\n", ByteCount);
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    if (!dstHost || !srcDevice) return CUDA_ERROR_INVALID_VALUE;
    memcpy(dstHost, (void*)srcDevice, ByteCount);
    printf("[CUDA_Translator] Intercepted cuMemcpyDtoH(bytes=%zu)\n", ByteCount);
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    *module = (CUmodule)0x2;
    printf("[CUDA_Translator] Intercepted cuModuleLoadData -> Passing PTX image to Translation Pipeline\n");
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    if (!hfunc || !name) return CUDA_ERROR_INVALID_VALUE;
    *hfunc = (CUfunction)0x3;
    printf("[CUDA_Translator] Intercepted cuModuleGetFunction(name='%s')\n", name);
    return CUDA_SUCCESS;
}

EXPORT_API CUresult cuLaunchKernel(CUfunction f,
                                   unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, void *hStream,
                                   void **kernelParams, void **extra) {
    printf("[CUDA_Translator] Intercepted cuLaunchKernel: Grid=(%u,%u,%u), Block=(%u,%u,%u), SharedMem=%u bytes\n",
           gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes);
    return CUDA_SUCCESS;
}
