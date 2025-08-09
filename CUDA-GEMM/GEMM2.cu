/*improve naive by tiling*/
#include<stdio.h>
#define TILE_DIM 32
#define COARSE_FACTOR 2

__global__ void mm_tiled_coarse_kernel(float* A, float* B, float* C, unsigned int M,
                                                     unsigned int N, unsigned int K) {
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x*blockDim.x*COARSE_FACTOR + threadIdx.x;
    float sum[COARSE_FACTOR];
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        sum[c] = 0.0f;
    }
    for(unsigned int tile = 0; tile < N/TILE_DIM; ++tile) {
        // Load A tile
        A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_DIM + threadIdx.x];
        for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            unsigned int col = colStart + c*TILE_DIM;
            // Load B tile
            B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];
            __syncthreads();
            // Compute with tile
            for(unsigned int i = 0; i < TILE_DIM; ++i) {
                sum[c] += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int col = colStart + c*TILE_DIM;
        C[row*N + col] = sum[c];
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
    dim3 numBlocks((N + TILE_DIM - 1)/TILE_DIM/COARSE_FACTOR, (N + TILE_DIM - 1)/TILE_DIM);
    cudaEventRecord(start);
    mm_tiled_coarse_kernel<<<numBlocks, blockDim>>>(A, B, C, N, N, N);
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
