#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>
#include <math.h>
#include <assert.h>
#include "fp16_conversion.h"

extern "C" {
#include "HMem.h"
}

#define CEIL(x,y) (((x)+(y)-1) / (y))

#define CHECKCUDASTATUS(status, invoker) {                                                    \
    if (status != cudaSuccess) printf("ERROR %s: %s", invoker, cudaGetErrorString(status)); \
}

#define CHECKCUBLASSTATUS(status, invoker, message) {                                \
    if (status != CUBLAS_STATUS_SUCCESS) printf("ERROR %s: %s", invoker, message); \
}

#define CHECKCUDNNSTATUS(status, invoker) {                                                        \
    if (status != CUDNN_STATUS_SUCCESS) printf("ERROR %s: %s", invoker, cudnnGetErrorString(status)); \
}

#define CHECKCURANDSTATUS(status, invoker, message) {                                \
    if (status != CURAND_STATUS_SUCCESS) printf("ERROR %s: %s", invoker, message); \
}

#define THREADPERBLOCK 256  

static int GPUDevId = -1; 
static Boolean GPUInit = FALSE;
cublasHandle_t cublasHandle;
#ifdef CUDNN
cudnnHandle_t cudnnHandle;
cudnnDataType_t dataTypeCUDNN;
cudnnTensorFormat_t tensorFormatCUDNN;
cudnnTensorDescriptor_t srcTensorDescCUDNN;
cudnnTensorDescriptor_t dstTensorDescCUDNN;
cudnnTensorDescriptor_t biasTensorDescCUDNN;
cudnnActivationDescriptor_t actfunDescCUDNN;
#endif


#define GPU_DEV_ID 0
#define FP16MM

extern "C" {
    static void ShowAllGPUs(void) {
        int nGPU, i;
        cudaDeviceProp prop;

        /*  */
        CHECKCUDASTATUS(cudaGetDeviceCount(&nGPU), "ShowAllGPUs")
        if (nGPU == 0) {
            printf("ERROR ShowAllGPUs: No GPU device");
            exit(1);
        }
        /*  */
        for (i = 0; i < nGPU; ++i) {
            CHECKCUDASTATUS(cudaGetDeviceProperties(&prop, i), "ShowAllGPUs")
            printf("GPU %d: %s, %luMB, SM = %d.%d", i, prop.name, prop.totalGlobalMem / 1048576, prop.major, prop.minor);
            if (GPUDevId == i)
                printf(" [Selected]");
            printf("\n");
        }
    }
}

extern "C" {
    void StartCUDA(void) {
        cudaDeviceProp prop;

        /* initialize the library and device */
        if (!GPUInit) {
            /* select GPU device 0 */
            GPUDevId = GPU_DEV_ID;
            CHECKCUDASTATUS(cudaSetDevice(GPUDevId), "InitCUDA")
            CHECKCUDASTATUS(cudaGetDeviceProperties(&prop, GPUDevId), "InitCUDA")
            /* initiate CUBLAS */
            CHECKCUBLASSTATUS(cublasCreate(&cublasHandle), "InitCUDA", "Fail to initialise CUBLAS")
            /* set GPUInit flag */
            GPUInit = TRUE;
            /* show devices */
            ShowAllGPUs();
    #ifdef CUDNN
    #ifdef DOUBLEANN
            dataTypeCUDNN = CUDNN_DATA_DOUBLE;
    #else
            dataTypeCUDNN = CUDNN_DATA_FLOAT;
    #endif
            tensorFormatCUDNN = CUDNN_TENSOR_NCHW;
            CHECKCUDNNSTATUS(cudnnCreate(&cudnnHandle), "StartCUDA")
            CHECKCUDNNSTATUS(cudnnCreateTensorDescriptor(&srcTensorDescCUDNN), "StartCUDA")
            CHECKCUDNNSTATUS(cudnnCreateTensorDescriptor(&dstTensorDescCUDNN), "StartCUDA")
            CHECKCUDNNSTATUS(cudnnCreateTensorDescriptor(&biasTensorDescCUDNN), "StartCUDA")
            CHECKCUDNNSTATUS(cudnnCreateActivationDescriptor(&actfunDescCUDNN), "StartCUDA")
    #endif

        }
        else
            printf("InitCUDA: GPU device %d already initialised", GPUDevId);

        printf("\n");
    }
}

extern "C" {
    void StopCUDA(void) {

        if (GPUInit) {
    #ifdef CUDNN
            CHECKCUDNNSTATUS(cudnnDestroyTensorDescriptor(srcTensorDescCUDNN), "StopCUDA")
            CHECKCUDNNSTATUS(cudnnDestroyTensorDescriptor(dstTensorDescCUDNN), "StopCUDA")
            CHECKCUDNNSTATUS(cudnnDestroyTensorDescriptor(biasTensorDescCUDNN), "StopCUDA")
            CHECKCUDNNSTATUS(cudnnDestroyActivationDescriptor(actfunDescCUDNN), "StopCUDA")
            CHECKCUDNNSTATUS(cudnnDestroy(cudnnHandle), "StopCUDA")
    #endif
            /* destroy the context on the GPU */
            CHECKCUBLASSTATUS(cublasDestroy(cublasHandle), "StopCUDA", "Fail to destroy CUBLAS")
            /* shutdown CUBLAS */
            CHECKCUDASTATUS(cudaDeviceReset(), "StopCUDA");
            /* reset GPU IDs and the flag */
            GPUDevId = -1;
            GPUInit = FALSE;
        }
        else
            printf("StopCUDA: GPU device has already stopped");
    }
}


extern "C" {
    void SyncDev2Host(void *devPtr, void *hostPtr, size_t size) {
        cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
    }
}

extern "C" {
    void SyncHost2Dev(void *hostPtr, void *devPtr, size_t size) {
        cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
    }
}

extern "C" {
    void DevDispose(void *devPtr, size_t size) {
        cudaFree(devPtr);
    }
}

extern "C" {
    Boolean DevNew(void **devAddr, size_t size) {
        if (cudaMalloc(devAddr, size) != cudaSuccess)
            return FALSE;
        return TRUE;
    }
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

/*----------Kernel Functions----------*/

__global__
void HKern_SelfAddNSegment(NFloat *rhPtr, int segLen, NFloat *lhPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        lhPtr[pos] = lhPtr[pos] + rhPtr[pos];
        
        /*if (pos == 0) {
            printf("Size of half: %lu, Size of half2: %lu", sizeof(half), sizeof(half2));
        }*/
    }
}

__global__
void HKern_SelfAddNSegmentHalf(NFloat *rhPtr, int segLen, NFloat *lhPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
        lhPtr[pos] = __half2float(__hadd(__float2half(lhPtr[pos]), __float2half(rhPtr[pos])));
    }
}

__global__
void HKern_MulMatrices(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N) {
    /* Assuming both matrices are N x N */

    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    float tmpSum = 0;

    if (col < N && row < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += lhPtr[row * N + i] * rhPtr[i * N + col];
        }
	resPtr[row * N + col] = tmpSum; 
    }
}

__global__
void HKern_MulMatrices_Half(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    //float tmpSum = 0;
    __half tmpSum = __int2half_rn(0);

    if (col < N && row < N) {
        for (int i = 0; i < N; i++) {
            //tmpSum += lhPtr[row * N + i] * rhPtr[i * N + col];
            tmpSum = __hadd(tmpSum, __hmul(__float2half(lhPtr[row * N + i]), __float2half(rhPtr[i * N + col])));
        }
	    resPtr[row * N + col] = __half2float(tmpSum);
    }
}



__global__
void HKern_MulMatrices_Half2(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N, cublasHandle_t cublasHandle) {
    /*
    cublasStatus_t stat;
    
    stat = cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, , lda, d_B, ldb, beta, d_C, ldc);


    cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const __half *alpha,
                           const __half *A, int lda,
                           const __half *B, int ldb,
                           const __half *beta,
                           __half *C, int ldc)
    */
        
}


/*
__global__
void HKern_SelfMulNSegment(NFloat *rhPtr, int segLen, NFloat *lhPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) {
	lhPtr[pos] = lhPtr[pos] * rhPtr[pos];
    }
}
*/

/*
__global__
void HKern_MulNSegment(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
    int pos;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < segLen) 
        resPtr[pos] = lhPtr[pos] * rhPtr[pos];
}
*/

/*----------C Wrappers----------*/

extern "C" {
    void SelfAddNSegmentCUDA(NFloat *rhPtr, int segLen, NFloat *lhPtr) {
        int nBlocks;

        nBlocks = CEIL(segLen, THREADPERBLOCK);
        /*if (nBlocks > MAXBLOCKNUM)
            HError(8890, (char *)"MulNSegmentCUDA: Block number exceeds the maximum");*/
        HKern_SelfAddNSegment<<<nBlocks, THREADPERBLOCK>>>(rhPtr, segLen, lhPtr);
    }
}

extern "C" {
    void SelfAddNSegmentCUDAHalf(NFloat *rhPtr, int segLen, NFloat *lhPtr) {
        int nBlocks;

        nBlocks = CEIL(segLen, THREADPERBLOCK);
        //if (nBlocks > MAXBLOCKNUM)
        //  HError(8890, (char *)"MulNSegmentCUDA: Block number exceeds the maximum");
        HKern_SelfAddNSegmentHalf<<<nBlocks, THREADPERBLOCK>>>(rhPtr, segLen, lhPtr);
    }
}

extern "C" {
    void MulMatricesCUDA(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N, int segLen) {
	    int nBlocks;

	    dim3 threadsPerBlock(16, 16);  // hard coded as 16 x 16 for now
	    nBlocks = CEIL(segLen, THREADPERBLOCK);
        /*if (nBlocks > MAXBLOCKNUM)
            HError(8890, (char *)"MulNSegmentCUDA: Block number exceeds the maximum");*/
	    dim3 blocksPerGrid(nBlocks, nBlocks);
	    HKern_MulMatrices<<<blocksPerGrid, threadsPerBlock>>>(lhPtr, rhPtr, resPtr, N);
    }
}

extern "C" {
     void MulMatricesCUDAHalf(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N, int segLen) {
	    int nBlocks;
	    dim3 threadsPerBlock(16, 16);  // NOTE: hard coded as 16 x 16 for now
	    nBlocks = CEIL(segLen, THREADPERBLOCK);
        //if (nBlocks > MAXBLOCKNUM)
        //    HError(8890, (char *)"MulNSegmentCUDA: Block number exceeds the maximum");
	    dim3 blocksPerGrid(nBlocks, nBlocks);

        //printf("nBlocks: %d, segLen: %d", nBlocks, segLen);
	    HKern_MulMatrices_Half<<<blocksPerGrid, threadsPerBlock>>>(lhPtr, rhPtr, resPtr, N);
    }
}

extern "C" {
     void MulMatricesCUDAHalf2(NFloat *lhPtr, NFloat *rhPtr, NFloat *resPtr, int N, int segLen) {
        /*
	    int nBlocks;
	    dim3 threadsPerBlock(16, 16);  // NOTE: hard coded as 16 x 16 for now
	    nBlocks = CEIL(segLen, THREADPERBLOCK);
        //if (nBlocks > MAXBLOCKNUM)
        //    HError(8890, (char *)"MulNSegmentCUDA: Block number exceeds the maximum");
	    dim3 blocksPerGrid(nBlocks, nBlocks);
        
        //printf("nBlocks: %d, segLen: %d", nBlocks, segLen);
	    HKern_MulMatrices_Half2<<<blocksPerGrid, threadsPerBlock>>>(lhPtr, rhPtr, resPtr, N);
        */

		cublasStatus_t stat;
        cublasHandle_t handle;
        
        checkCublas(cublasCreate(&handle));

        __half *d_A, *d_B, *d_C;
        checkCuda(cudaMallocManaged(&d_A, N * N * sizeof(__half)));
        checkCuda(cudaMallocManaged(&d_B, N * N * sizeof(__half)));
        checkCuda(cudaMallocManaged(&d_C, N * N * sizeof(__half)));
        
        for (int i = 0; i < N * N; i++) {
              d_A[i] = approx_float_to_half(lhPtr[i]);
          	  d_B[i] = approx_float_to_half(rhPtr[i]);
          	  d_C[i] = approx_float_to_half(resPtr[i]);
        }

        int lda, ldb, ldc, m, n, k;
        const __half alf = approx_float_to_half(1.0);
        const __half bet = approx_float_to_half(0.0);
        const __half *alpha = &alf;
        const __half *beta = &bet;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        double sum = 0.0;

        cudaEventRecord(start, 0);
        m=n=k=N;
        lda = m;
        ldb = k;
        ldc = m;
        
        stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        if(stat != CUBLAS_STATUS_SUCCESS){
            printf("cublasHgemm failed");
            exit(1);
        }
        assert(!cudaGetLastError());
      
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        sum += elapsed;

        printf("Time elapsed: %e", sum);

        for (int i = 0; i < N * N; i++) {
              d_A[i] = approx_float_to_half(lhPtr[i]);
          	  d_B[i] = approx_float_to_half(rhPtr[i]);
          	  d_C[i] = approx_float_to_half(resPtr[i]);
        }
        
        
        //Free GPU memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

    }
}

/*
extern "C" {
    void SelfMulNSegmentCUDA(NFloat *rhPtr, int segLen, NFloat *lhPtr) {
        int nBlocks;

        nBlocks = CEIL(segLen, THREADPERBLOCK);
        //if (nBlocks > MAXBLOCKNUM)
        //  HError(8890, (char *)"MulNSegmentCUDA: Block number exceeds the maximum");
        HKern_SelfMulNSegment<<<nBlocks, THREADPERBLOCK>>>(rhPtr, segLen, lhPtr);
    }
}

extern "C" {
	void MulNSegmentCUDA(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr) {
        int nBlocks;

        nBlocks = CEIL(segLen, THREADPERBLOCK);
        //if (nBlocks > MAXBLOCKNUM)
        //  HError(8890, (char *)"MulNSegmentCUDA: Block number exceeds the maximum");
        HKern_MulNSegment<<<nBlocks, THREADPERBLOCK>>>(lhPtr, rhPtr, segLen, resPtr);
    }
}
*/
/*---------------------------END OF HCUDA_ext.cu---------------------------*/
