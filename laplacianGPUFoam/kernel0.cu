#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 

__global__ void cellKernel( double *vcs,
                        double* tot,
                        double rDelgaG,
                        double *diag, 
                        double *source,
                        int sizeDiag)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id>= sizeDiag)
        return;

    diag[id] = rDelgaG*vcs[id];
    source[id] = rDelgaG*vcs[id]*tot[id];
}


__global__ void faceKernel(double *delta,
                        double *gamma,
                        int *upperAddr,
                        int *lowerAddr,
                        double *upper, 
                        double *lower,
                        double *diag,
                        int sizeFace)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id>= sizeFace)
        return;

    double temp=delta[id]*gamma[id];
    lower[id] = temp;
    upper[id] = temp;
    atomicAdd(&diag[lowerAddr[id]], -temp);
    atomicAdd(&diag[upperAddr[id]], -temp);
}

__global__ void boundaryKernel(int *patchCounter,
                                int *pC_Array,  
                                double *pDiag,
                                double *pSource,
                                double *diag,
                                double *source,
                                int maxPatches,
                                int numberOfPatches)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id>= maxPatches)
        return;

    atomicAdd(&diag[pC_Array[id]],pDiag[id]);
    atomicAdd(&source[pC_Array[id]],pSource[id]);
}


void laplasKernel( int sizeDiag,
                int sizeFace,
                double *vcs, 
                double *tot,
                double *delta,
                double *gamma,
                int *upperAddr,
                int *lowerAddr,
                int *patchCounter,
                int *pC_Array,
                double *pDiag,
                double *pSource,
                int maxPatches,
                int numberOfPatches,
                double rDelgaG,
                double *h_diag,
                double *h_source,
                double *h_upper,
                double *h_lower
                )
{
    // Device output vectors
    double *d_diag;
    double *d_source;
    double *d_lower;
    double *d_upper;

 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_diag, sizeDiag*sizeof(double));
    cudaMalloc(&d_source, sizeDiag*sizeof(double));
    cudaMalloc(&d_lower, sizeFace*sizeof(double));
    cudaMalloc(&d_upper, sizeFace*sizeof(double));

    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 512;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)(sizeDiag)/blockSize);
 
    // Execute the kernel
    cellKernel<<<gridSize, blockSize>>>(vcs,
                                        tot,
                                        rDelgaG,   
                                        d_diag, 
                                        d_source, 
                                        sizeDiag
                                        );

    // Update grid size for new kernel
    gridSize = (int)ceil((float)(sizeFace)/blockSize);
    faceKernel<<<gridSize, blockSize>>>(delta,
                                        gamma,  
                                        upperAddr,
                                        lowerAddr,
                                        d_upper,
                                        d_lower,
                                        d_diag,
                                        sizeFace
                                        );

    gridSize = (int)ceil((float)(maxPatches)/blockSize);
    boundaryKernel<<<gridSize, blockSize>>>(patchCounter,
                                        pC_Array,  
                                        pDiag,
                                        pSource,
                                        d_diag,
                                        d_source,
                                        maxPatches,
                                        numberOfPatches
                                        );

    // Copy array back to host
    cudaMemcpy( h_diag, d_diag, sizeDiag*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( h_source, d_source, sizeDiag*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( h_lower, d_lower, sizeFace*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( h_upper, d_upper, sizeFace*sizeof(double), cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(d_diag);
    cudaFree(d_source);
    cudaFree(d_lower);
    cudaFree(d_upper);
}
