#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<stdio.h>
int main(){
    unsigned int N=1024;
    unsigned int size=N*N*sizeof(float);
    //initialize data
    float *d_A = (float*)malloc(size);
    float *d_B = (float*)malloc(size);
    float *d_C = (float*)malloc(size);
    // Initialize matrices A and B
    for(unsigned int i=0;i<N*N;i++){
        d_A[i]=1.0f;
        d_B[i]=2.0f;
    }
    float *A,*B,*C;
    //allocate memory on the device
    cudaMalloc((void**)&A, size);
    cudaMalloc((void**)&B, size);
    cudaMalloc((void**)&C, size);
    //copy data to device
    cudaMemcpy(A, d_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, d_B, size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(start);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);
    
    cublasDestroy(handle);
    // copy data back to host
    cudaMemcpy(d_C, C, size, cudaMemcpyDeviceToHost);
    // Free memory
    free(d_A);
    free(d_B);
    free(d_C);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}