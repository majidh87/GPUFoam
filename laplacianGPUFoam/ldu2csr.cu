
#include <iostream>
using namespace std;

__global__ void kernelRowSize(int *upperAddr, int *lowerAddr, int* rIdx, int nFaces)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= nFaces)
		return;

	atomicAdd(&rIdx[lowerAddr[id]+1], 1);
	atomicAdd(&rIdx[upperAddr[id]+1], 1);	
}

__global__ void kernelRowPtr(int *rIdx, int* rPtr, int nCells)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx > nCells)
		return;

	for (int i=0; i<= idx; i++)
		rPtr[idx]+=rIdx[i];
}
__global__ void kernelLower(int *upperAddr, int *lowerAddr, double *lower, int* rIdx, int* colIdx, double* val, int nFaces)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= nFaces)
		return;
	int row    = upperAddr[id] ;
	int column = lowerAddr[id] ;

	int idx = atomicAdd(&rIdx[row],1);
	val[idx] = lower[id] ;
	colIdx[idx] = column ;
}

__global__ void kernelDiag(double *diag, int* rIdx, int* colIdx, double* val, int nCells)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= nCells)
		return;

	int idx =  atomicAdd(&rIdx[id],1);
	val[idx] = diag[id] ;
	colIdx[idx] = id ;
}

__global__ void kernelUpper(int *upperAddr, int *lowerAddr, double *upper, int* rIdx, int* colIdx, double* val, int nFaces)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= nFaces)
		return;
		
	int row    = lowerAddr[id] ;
	int column = upperAddr[id] ;

	int idx =  atomicAdd(&rIdx[row],1);
	val[idx] = upper[id] ;
	colIdx[idx] = column ;
}


void ldu2csrWrapper (	double *d_diag,
				double *d_lower,
				double *d_upper,
				int *d_lowerAddr,
				int *d_upperAddr,
				int *d_rowPtr,
				int *d_colIdx,
				double *d_value,
				int nCells,
				int nFaces
) 
{


	/////////////////////////////////////////////////////GPU/////////////////////////////////////////////////////
	
	int *d_rIdx;
	
	cudaMalloc(&d_rIdx, (nCells+1)*sizeof(int));
	cudaMemset(d_rIdx, 0, (nCells+1)*sizeof(int));
	

	int blockSize = 512;
    int gridSizeF = (int)ceil((double)nFaces/blockSize);
	int gridSizeC = (int)ceil((double)nCells/blockSize);
	int gridSizeC1 = (int)ceil((double)(nCells+1)/blockSize);

	printf("here in device 1 \n");
    kernelRowSize<<<gridSizeF, blockSize>>>(d_upperAddr, d_lowerAddr, d_rIdx, nFaces);


	kernelRowPtr<<<gridSizeC1, blockSize>>>(d_rIdx, d_rowPtr, nCells);

	cudaMemcpy( d_rIdx, d_rowPtr, (nCells+1)*sizeof(int), cudaMemcpyDeviceToDevice ); 

	kernelLower<<<gridSizeF, blockSize>>>(d_upperAddr, d_lowerAddr, d_lower , d_rIdx, d_colIdx, d_value, nFaces);
	
	kernelDiag<<<gridSizeC, blockSize>>>(d_diag , d_rIdx, d_colIdx, d_value, nCells);

	kernelUpper<<<gridSizeF, blockSize>>>(d_upperAddr, d_lowerAddr, d_upper , d_rIdx, d_colIdx, d_value, nFaces);
 
   
	
}