#include "discretizationKernel.h"
#define BSIZE 512
#define BSIZEX 32
#define BSIZEY 32

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

__global__ void faceKernel( double *delta,
                            double *gamma,
                            int *upperAddr,
                            int *lowerAddr,
                            double *upper, 
                            double *lower,
                            double *diag,
                            int sizeFace
                            )
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

__global__ void boundaryKernel( int *pSize,
                                int **pAdrr,
                                double **pf_BC,
                                double **pf_IC,
                                double **pf_GammaSf,
                                double *diag,
                                double *source,
                                int maxPatches,
                                int numberOfPatches
                                )
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx >= pSize[idy] || idy >= numberOfPatches)
		return;
    atomicAdd(&diag[pAdrr[idy][idx]],pf_GammaSf[idy][idx]*pf_IC[idy][idx]);
    atomicAdd(&source[pAdrr[idy][idx]],-pf_GammaSf[idy][idx]*pf_BC[idy][idx]);
    //printf("***** %f %f\n",source[pAdrr[idy][idx]],-pf_GammaSf[idy][idx]*pf_BC[idy][idx] );
}

__global__ void updateSourceKernel( double *vcs,
                            double* tot,
                            double rDelgaG,
                            double *source,
                            int sizeDiag)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id>= sizeDiag)
        return;

    source[id] = rDelgaG*vcs[id]*tot[id];
     //printf("***** %f %f\n",source[pAdrr[idy][idx]],-pf_GammaSf[idy][idx]*pf_BC[idy][idx] );
}

__global__ void updateBoundaryKernel( int *pSize,
                                int **pAdrr,
                                double **pf_BC,
                                double **pf_IC,
                                double **pf_GammaSf,
                                double *source,
                                int maxPatches,
                                int numberOfPatches
                                )
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
    if (idx >= pSize[idy] || idy >= numberOfPatches)
		return;
    atomicAdd(&source[pAdrr[idy][idx]],-pf_GammaSf[idy][idx]*pf_BC[idy][idx]);
    //printf("***** %f %f\n",source[pAdrr[idy][idx]],-pf_GammaSf[idy][idx]*pf_BC[idy][idx] );
}

void discKernelWrapper( int sizeDiag,
                        int sizeFace,
                        double *vcs, 
                        double *tot,
                        double *delta,
                        double *gamma,
                        int *upperAddr,
                        int *lowerAddr,
                        int numOfPatches,
                        int maxPatches,
                        int *d_pSize,
                        int **d_pAdrr,
                        double **d_pf_BC,
                        double **d_pf_IC,
                        double **d_pf_GammaSf,
                        double rDelgaG,
                        double *d_diag,
                        double *d_source,
                        double *d_upper,
                        double *d_lower
                        )
{
    int blockSize = BSIZE;
    int gridCell = (int)ceil((float)(sizeDiag)/blockSize);
 
    cellKernel<<<gridCell, blockSize>>>(vcs,
                                        tot,
                                        rDelgaG,   
                                        d_diag, 
                                        d_source, 
                                        sizeDiag
                                        );

    
    int gridFace = (int)ceil((float)(sizeFace)/blockSize);
    faceKernel<<<gridFace, blockSize>>>(delta,
                                        gamma,  
                                        upperAddr,
                                        lowerAddr,
                                        d_upper,
                                        d_lower,
                                        d_diag,
                                        sizeFace
                                        );

    dim3 blockBoundary(BSIZEX, BSIZEY, 1);
	dim3 gridBoundary((int)ceil((float)(maxPatches)/blockBoundary.x), (int)ceil((float)(numOfPatches)/blockBoundary.y), 1);
    boundaryKernel<<<gridBoundary, blockBoundary>>>(d_pSize,
                                            d_pAdrr,
                                            d_pf_BC,
                                            d_pf_IC,
                                            d_pf_GammaSf,
                                            d_diag,
                                            d_source,
                                            maxPatches,
                                            numOfPatches
                                            );
   
    
}

void updateDiscWrapper(int sizeDiag,
                        double *vcs, 
                        double *tot,
                        int numOfPatches,
                        int maxPatches,
                        int *d_pSize,
                        int **d_pAdrr,
                        double **d_pf_BC,
                        double **d_pf_IC,
                        double **d_pf_GammaSf,
                        double rDelgaG,
                        double *d_source
                        )
{
    int blockSize = BSIZE;
    int gridCell = (int)ceil((float)(sizeDiag)/blockSize);
 
    updateSourceKernel<<<gridCell, blockSize>>>(vcs,
                                        tot,
                                        rDelgaG,    
                                        d_source, 
                                        sizeDiag
                                        );

    dim3 blockBoundary(BSIZEX, BSIZEY, 1);
	dim3 gridBoundary((int)ceil((float)(maxPatches)/blockBoundary.x), (int)ceil((float)(numOfPatches)/blockBoundary.y), 1);
    updateBoundaryKernel<<<gridBoundary, blockBoundary>>>(d_pSize,
                                            d_pAdrr,
                                            d_pf_BC,
                                            d_pf_IC,
                                            d_pf_GammaSf,
                                            d_source,
                                            maxPatches,
                                            numOfPatches
                                            );
   
    
}

