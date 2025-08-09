/*improve naive by tiling*/
#include<stdio.h>
#define TILE_DIM 32
__global__ void mm_kernel(float* A,float* B,float*C,unsigned int N){
    unsigned int row=blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum=0.0f;
    for(unsigned int i=0;i<N/TILE_DIM;i++){
        A_s[threadIdx.y][threadIdx.x]=A[row*N+i*TILE_DIM+threadIdx.x];
        B_s[threadIdx.y][threadIdx.x]=B[(i*TILE_DIM+threadIdx.y)*N+col];
        __syncthreads();
        for(unsigned int j=0;j<TILE_DIM;j++){
            sum+=A_s[threadIdx.y][j]*B_s[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(row<N && col<N){
        C[row*N+col]=sum;
    }
}
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
    //cuda events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Free host memory
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    cudaEventRecord(start);
    mm_kernel<<<gridDim, blockDim>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    /*
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        return -1; 
    }
    // copy data back to host
    cudaMemcpy(d_C, C, size, cudaMemcpyDeviceToHost);
    // Print a few results
    printf("Result of matrix multiplication (first 10 elements):\n");
    for(unsigned int i=0;i<10;i++){
        printf("%f ", d_C[i]);
    }
    printf("\n");
    
    */
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