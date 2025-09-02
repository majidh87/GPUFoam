#include <stdio.h>
// Define block sizes for CUDA kernels
#define BSIZE 512       // Block size for 1D kernels (cellKernel and faceKernel)
#define BSIZEX 32       // Block size in the x-direction for 2D kernel (boundaryKernel)
#define BSIZEY 32       // Block size in the y-direction for 2D kernel (boundaryKernel)

// Kernel to compute diagonal and source terms for each cell
__global__ void cellKernel(double *vcs,          // Volume of the cells
                           double *tot,          // Total value (e.g., temperature) at each cell
                           double rDelgaG,       // Scaling factor
                           double *diag,         // Diagonal terms of the matrix
                           double *source,       // Source terms of the linear system
                           int sizeDiag)         // Number of cells
{
    // Calculate the thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread ID is within bounds
    if (id >= sizeDiag)
        return;

    //printf("cellKernel: id: %d, vcs[id]: %e, tot[id]: %e\n", id, vcs[id], tot[id]);
    // Compute diagonal term: diag[id] = rDelgaG * vcs[id]
    diag[id] = 0; //rDelgaG * vcs[id];

    // Compute source term: source[id] = rDelgaG * vcs[id] * tot[id]
    source[id] = 0; //rDelgaG * vcs[id] * tot[id];
    
}

// Kernel to compute off-diagonal terms (upper and lower) for each face
__global__ void faceKernel(double *delta,        // Delta coefficient for faces
                           double *gamma,        // Gamma coefficient for faces
                            double *DT_surf,      // Total value (e.g., temperature) at each cell
                           int *upperAddr,       // Indices of the upper cells for each face
                           int *lowerAddr,       // Indices of the lower cells for each face
                           double *upper,        // Upper off-diagonal terms
                           double *lower,        // Lower off-diagonal terms
                           double *diag,         // Diagonal terms of the matrix
                           int sizeFace)         // Number of faces
{
    // Calculate the thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread ID is within bounds
    if (id >= sizeFace)
        return;
   // Compute the off-diagonal term: temp = delta[id] * gamma[id]
    double temp = delta[id] * gamma[id] * DT_surf[id];
    //printf("upper[%d]: %f , delta: %f,  gammaMagSf: : %f\n", id, temp, delta[id], gamma[id]* DT_surf[id]);
  
    // Assign the off-diagonal terms
    lower[id] = temp;  // Lower term
    upper[id] = temp;  // Upper term

    // Update the diagonal terms for the lower and upper cells using atomicAdd
    atomicAdd(&diag[lowerAddr[id]], -temp);  // Subtract temp from the diagonal of the lower cell
    atomicAdd(&diag[upperAddr[id]], -temp);  // Subtract temp from the diagonal of the upper cell

}

// Kernel to handle boundary conditions for the patches
__global__ void boundaryKernel(int *pSize,        // Number of faces in each patch
                               int **pAdrr,       // Addresses of the cells for each patch
                               double **pf_BC,    // Boundary coefficients for each patch
                               double **pf_IC,    // Internal coefficients for each patch
                               double **pf_GammaSf, // Gamma coefficient for each patch
                               double **pf_DT_surf, // DT patch field
                               double *diag,      // Diagonal terms of the matrix
                               double *source,    // Source terms of the linear system
                               int maxPatches,    // Maximum number of faces in any patch
                               int numberOfPatches) // Total number of patches
{
    // Calculate the thread IDs for 2D grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Thread ID in x-direction
    int idy = blockIdx.y * blockDim.y + threadIdx.y;  // Thread ID in y-direction
 
    // Ensure the thread IDs are within bounds
    if (idx >= pSize[idy] || idy >= numberOfPatches)
        return;

    // Update the diagonal and source terms for boundary faces
    atomicAdd(&diag[pAdrr[idy][idx]], pf_GammaSf[idy][idx] * pf_DT_surf[idy][idx]* pf_IC[idy][idx]);  // Add to diagonal
    atomicAdd(&source[pAdrr[idy][idx]], -pf_GammaSf[idy][idx]* pf_DT_surf[idy][idx] * pf_BC[idy][idx]); // Subtract from source
   
}
__global__ void gradTKernel(
    const double* T,
    const int *upperAddr,
    const int *lowerAddr,
    const double *weight,
    const double *faceSfX,
    const double *faceSfY,
    const double *faceSfZ,
    double *gradTX,
    double *gradTY,
    double *gradTZ,
    int sizeFace
) 
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id>= sizeFace)
        return;

    // Face face = faces[idx];
    int i = lowerAddr[id];
    int j = upperAddr[id];

    //if (j == -1) return;  // skip boundary faces (can be extended)

    // Interpolate T at face (simple average)
    double interpT = (1.0 - weight[id]) * T[i] + weight[id] * T[j];
    //if (id<10)
     //    printf("x = %f y = %f z = %f interp = %f\n", faceSfX[id], faceSfY[id], faceSfZ[id],interpT);
   

    // Accumulate to cells
    atomicAdd(&gradTX[i], faceSfX[id] * interpT);
    atomicAdd(&gradTY[i], faceSfY[id] * interpT);
    atomicAdd(&gradTZ[i], faceSfZ[id] * interpT);

    atomicAdd(&gradTX[j], -faceSfX[id] * interpT);
    atomicAdd(&gradTY[j], -faceSfY[id] * interpT);
    atomicAdd(&gradTZ[j], -faceSfZ[id] * interpT);
}
__global__ void normalizeGradT(
    const double *cellVolume,
    double *gradTX,
    double *gradTY,
    double *gradTZ,
    int sizeDiag
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sizeDiag) return;

    double V = cellVolume[i];
    if (V > 1e-12) {
        gradTX[i]/= V;
        gradTY[i]/= V;
        gradTZ[i]/= V;
    }
}
__global__ void interpololateKernel(double* TX,                     // [nCells]
                                    double* TY,                     // [nCells]
                                    double* TZ,                     // [nCells]
                            const int *upperAddr,
                            const int *lowerAddr,
                            const double * w,
                            double* interpTX,                     // [nCells]
                            double* interpTY,                     // [nCells]
                            double* interpTZ,                     // [nCells]
                            int sizeFace
                            )
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id>= sizeFace)
        return;

    int own = lowerAddr[id];
    int neighbour = upperAddr[id];

    interpTX[id] = (1.0 - w[id]) * TX[own] + w[id] * TX[neighbour];
    interpTY[id] = (1.0 - w[id]) * TY[own] + w[id] * TY[neighbour];
    interpTZ[id] = (1.0 - w[id]) * TZ[own] + w[id] * TZ[neighbour];

}

__global__ void applyBoundaryGradientKernel(
    const int* __restrict__ faceCells, // [N]
    const double* __restrict__ SfX,    // [N]
    const double* __restrict__ SfY,    // [N]
    const double* __restrict__ SfZ,    // [N]
    const double* __restrict__ ssf,    // [N]
    double* gradX,                     // [nCells]
    double* gradY,                     // [nCells]
    double* gradZ,                     // [nCells]
    int N                              // number of boundary faces
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cell = faceCells[i];
    double scale = ssf[i];

    // Compute contribution
    double cx = SfX[i] * scale;
    double cy = SfY[i] * scale;
    double cz = SfZ[i] * scale;

    //printf("face %d: cell: %d  SfX = %f, SfY = %f, SfZ = %f, ssf = %f\n", i, cell, SfX[i], SfY[i], SfZ[i], ssf[i]);

    // Accumulate to gradient (atomic in case of race conditions)
    atomicAdd(&gradX[cell], cx);
    atomicAdd(&gradY[cell], cy);
    atomicAdd(&gradZ[cell], cz);
}

__global__ void computeCorrectionDiv(
    const double* __restrict__ correctionF,
    const int* __restrict__ upperAddr,
    const int* __restrict__ lowerAddr,
    const double* __restrict__ cellVolume,
    double* correctionDiv,
    int sizeFace
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sizeFace)
    {
        int own = lowerAddr[i];
        int neighbour = upperAddr[i];

        double flux = correctionF[i];

        atomicAdd(&correctionDiv[own], -flux / cellVolume[own]);
        atomicAdd(&correctionDiv[neighbour],  +flux / cellVolume[neighbour]);
    }
}

__global__ void updateSourceKernel(
    const double* __restrict__ cellVolume,
    const double* __restrict__ correctionDiv,
    double* source,
    int sizeDiag
)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sizeDiag) return;

    source[i] += correctionDiv[i]*cellVolume[i];
}
// Computes: correctionF = (sf_DT * magSf) * (corrVec2 · gradT_f)
__global__ void fusedCorrectionFluxKernel(
    const double* __restrict__ corrX,
    const double* __restrict__ corrY,
    const double* __restrict__ corrZ,
    const double* __restrict__ gradX,
    const double* __restrict__ gradY,
    const double* __restrict__ gradZ,
    const double* __restrict__ sf_DT,
    const double* __restrict__ magSf,
    double* __restrict__ correctionF,
    int sizeFace)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= sizeFace) return;

    double dot = corrX[f]*gradX[f] + corrY[f]*gradY[f] + corrZ[f]*gradZ[f];
    double gammaSf = sf_DT[f] * magSf[f];
    correctionF[f] = gammaSf * dot;
}




// Wrapper function to call the CUDA kernels and manage the discretization process
void discKernelWrapper(int sizeDiag,              // Number of cells
                       int sizeFace,              // Number of faces
                       double *vcs,               // Volume of the cells
                       double *tot,               // Total value (e.g., temperature) at each cell
                       double *DT_surf,               // Total value (e.g., temperature) at each cell
                       double *delta,             // Delta coefficient for faces
                       double *gamma,             // Gamma coefficient for faces
                       int *upperAddr,            // Indices of the upper cells for each face
                       int *lowerAddr,            // Indices of the lower cells for each face
                       int numOfPatches,          // Number of patches
                       int maxPatches,            // Maximum number of faces in any patch
                       int *d_pSize,              // Number of faces in each patch (on device)
                       int **d_pAdrr,             // Addresses of the cells for each patch (on device)
                       double **d_pf_BC,          // Boundary coefficients for each patch (on device)
                       double **d_pf_IC,          // Internal coefficients for each patch (on device)
                       double **d_pf_GammaSf,     // Gamma coefficient for each patch (on device)
                       double **d_pf_DT_surf,     // DT patch field(on device)
                       double rDelgaG,            // Scaling factor
                       double *d_diag,            // Diagonal terms of the matrix (on device)
                       double *d_source,          // Source terms of the linear system (on device)
                       double *d_upper,           // Upper off-diagonal terms (on device)
                       double *d_lower           //// Lower off-diagonal terms (on device)
                    )           
{
    // Set block size for 1D kernels
    int blockSize = BSIZE;

    // Compute grid size for cellKernel
    int gridCell = (int)ceil((float)(sizeDiag) / blockSize);

    //printf("pf_BC[0]:%p , pf_IC[0]:%p \n",d_pf_BC[0],d_pf_IC[0]);
    // Launch cellKernel to compute diagonal and source terms
    cellKernel<<<gridCell, blockSize>>>(vcs, tot, rDelgaG, d_diag, d_source, sizeDiag);

    // Compute grid size for faceKernel
    int gridFace = (int)ceil((float)(sizeFace) / blockSize);

    // Launch faceKernel to compute off-diagonal terms
    faceKernel<<<gridFace, blockSize>>>(delta, gamma,DT_surf, upperAddr, lowerAddr, d_upper, d_lower, d_diag, sizeFace);

    // // // Set block and grid sizes for boundaryKernel (2D kernel)
    dim3 blockBoundary(BSIZEX, BSIZEY, 1);  // 2D block size
    dim3 gridBoundary((int)ceil((float)(maxPatches) / blockBoundary.x),  // Grid size in x-direction
                     (int)ceil((float)(numOfPatches) / blockBoundary.y), // Grid size in y-direction
                     1);

       // Launch boundaryKernel to handle boundary conditions
    boundaryKernel<<<gridBoundary, blockBoundary>>>(d_pSize, d_pAdrr, d_pf_BC, d_pf_IC, d_pf_GammaSf,d_pf_DT_surf, d_diag, d_source, maxPatches, numOfPatches);

    // Optional: Debugging code to check the source terms (commented out)
    
    // double* hg_source = (double*)malloc((9) * sizeof(double));
    // cudaMemcpy(hg_source, d_source, (9) * sizeof(double), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 9; i++)
    //     printf("*******%f\n", hg_source[i]);
    
}
void cellKernelWrapper(int sizeDiag,              // Number of cells
    double *vcs,               // Volume of the cells
    double *tot,               // Total value (e.g., temperature) at each cell
    double rDelgaG,            // Scaling factor
    double *d_diag,            // Diagonal terms of the matrix (on device)
    double *d_source          // Source terms of the linear system (on device)
)
{
    // Set block size for 1D kernels
    int blockSize = BSIZE;

    // Compute grid size for cellKernel
    int gridCell = (int)ceil((float)(sizeDiag) / blockSize);

    //printf("pf_BC[0]:%p , pf_IC[0]:%p \n",d_pf_BC[0],d_pf_IC[0]);
    // Launch cellKernel to compute diagonal and source terms
    cellKernel<<<gridCell, blockSize>>>(vcs, tot, rDelgaG, d_diag, d_source, sizeDiag);

}
void faceKernelWrapper(
    int sizeFace,              // Number of faces
    double *delta,             // Delta coefficient for faces
    double *gamma,             // Gamma coefficient for faces
    double *DT_surf,               // Total value (e.g., temperature) at each cell
    int *upperAddr,            // Indices of the upper cells for each face
    int *lowerAddr,            // Indices of the lower cells for each face
    double *d_upper,           // Upper off-diagonal terms (on device)
    double *d_lower,            // Lower off-diagonal terms (on device)
    double *d_diag            // Diagonal terms of the matrix (on device)
)
{
    // Set block size for 1D kernels
    int blockSize = BSIZE;
    // Compute grid size for faceKernel
    int gridFace = (int)ceil((float)(sizeFace) / blockSize);

    // Launch faceKernel to compute off-diagonal terms
    faceKernel<<<gridFace, blockSize>>>(delta, gamma,DT_surf, upperAddr, lowerAddr, d_upper, d_lower, d_diag, sizeFace);

}
void boundaryKernelWrapper(
    int numOfPatches,          // Number of patches
    int maxPatches,            // Maximum number of faces in any patch
    int *d_pSize,              // Number of faces in each patch (on device)
    int **d_pAdrr,             // Addresses of the cells for each patch (on device)
    double **d_pf_BC,          // Boundary coefficients for each patch (on device)
    double **d_pf_IC,          // Internal coefficients for each patch (on device)
    double **d_pf_GammaSf,     // Gamma coefficient for each patch (on device)
    double **d_pf_DT_surf,     // DT patch field(on device)
    double *d_diag,            // Diagonal terms of the matrix (on device)
    double *d_source          // Source terms of the linear system (on device)
)
{
    //printf(" Hi here in boundaryKernelWrapper\n");
    // // // Set block and grid sizes for boundaryKernel (2D kernel)
    dim3 blockBoundary(BSIZEX, BSIZEY, 1);  // 2D block size
    dim3 gridBoundary((int)ceil((float)(maxPatches) / blockBoundary.x),  // Grid size in x-direction
                     (int)ceil((float)(numOfPatches) / blockBoundary.y), // Grid size in y-direction
                     1);

       // Launch boundaryKernel to handle boundary conditions
    boundaryKernel<<<gridBoundary, blockBoundary>>>(d_pSize, d_pAdrr, d_pf_BC, d_pf_IC, d_pf_GammaSf,d_pf_DT_surf, d_diag, d_source, maxPatches, numOfPatches);

    // Optional: Debugging code to check the source terms (commented out)
    
    // double* hg_source = (double*)malloc((9) * sizeof(double));
    // cudaMemcpy(hg_source, d_source, (9) * sizeof(double), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 9; i++)
    //     printf("*******%f\n", hg_source[i]);
}

void gradTKernelWrapper(
    const double* d_T,
    const int *d_upperAddr,
    const int *d_lowerAddr,
    const double *d_weight,
    const double *d_faceSfX,
    const double *d_faceSfY,
    const double *d_faceSfZ,
    double *d_gradTX,
    double *d_gradTY,
    double *d_gradTZ,
    int sizeFace
)
{
    // Set block size for 1D kernels
    int blockSize = BSIZE;
    // Compute grid size for faceKernel
    int gridFace = (int)ceil((float)(sizeFace) / blockSize);
    
    gradTKernel<<<gridFace, blockSize>>> ( 
    d_T,
    d_upperAddr,
    d_lowerAddr,
    d_weight,
    d_faceSfX,
    d_faceSfY,
    d_faceSfZ,
    d_gradTX,
    d_gradTY,
    d_gradTZ,
    sizeFace);

    
}
void applyBoundaryGradientWrapper (
    int* d_faceCells, // [N]
    double* d_SfX,    // [N]
    double* d_SfY,    // [N]
    double* d_SfZ,    // [N]
    double* d_ssf,    // [N]
    const double *d_cellVolumes,
    double* d_gradX,                     // [nCells]
    double* d_gradY,                     // [nCells]
    double* d_gradZ,                     // [nCells]
    int N  ,                            // number of boundary faces,
    int sizeDiag

)
{
int blockSize = BSIZE;
int gridSize = (N + blockSize - 1) / blockSize;

applyBoundaryGradientKernel<<<gridSize, blockSize>>>(
    d_faceCells,
    d_SfX, d_SfY, d_SfZ,
    d_ssf,
    d_gradX, d_gradY, d_gradZ,
    N
);

int gridDiag = (int)ceil((float)(sizeDiag) / blockSize);
    normalizeGradT <<<gridDiag, blockSize>>> ( 
    d_cellVolumes,
    d_gradX,
    d_gradY,
    d_gradZ,
    sizeDiag);

}

void computeCorrectionDivWrapper(
    const double* d_correctionF,
    const int* d_upperAddr,
    const int* d_lowerAddr,
    const double* d_cellVolumes,
    double* d_correctionDiv,
    int sizeFace
)
{
    // Set block size for 1D kernels
    int blockSize = BSIZE;
    // Compute grid size for faceKernel
    int gridFace = (int)ceil((float)(sizeFace) / blockSize);

    computeCorrectionDiv<<<gridFace, blockSize>>>(
        d_correctionF,
        d_upperAddr,
        d_lowerAddr,
        d_cellVolumes,
        d_correctionDiv,
        sizeFace
    );

    
}


void interpololateKernelWrapper( double* d_TX,
                                 double* d_TY,
                                 double* d_TZ,
                                 const int* d_upperAddr,
                                 const int* d_lowerAddr,
                                 double* w,
                                 double* d_interpTX,
                                 double* d_interpTY,
                                 double* d_interpTZ,
                                 int sizeFace
                                )
{
    int blockSize = BSIZE;
    int gridFace = (int)ceil((float)(sizeFace) / blockSize);

    //printf(" launching interpolateKernel\n");
    interpololateKernel<<<gridFace, blockSize>>>(
        d_TX,
        d_TY,
        d_TZ,
        d_upperAddr,
        d_lowerAddr,
        w,
        d_interpTX,
        d_interpTY,
        d_interpTZ,
        sizeFace
    );
}

void fusedCorrectionFluxWrapper(
     double*  d_corrX,
     double*  d_corrY,
     double*  d_corrZ,
     double*  d_gradX,
     double*  d_gradY,
     double*  d_gradZ,
     double*  d_sf_DT,
     double*  d_magSf,
    double*  d_correctionF,
    int sizeFace)
{
    int blockSize = BSIZE;
    int gridFace = (int)ceil((float)(sizeFace) / blockSize);

    fusedCorrectionFluxKernel<<<gridFace, blockSize>>>(d_corrX, d_corrY, d_corrZ,
                                            d_gradX, d_gradY, d_gradZ,
                                            d_sf_DT, d_magSf,
                                            d_correctionF, sizeFace);
}

void updateSourceWrapper(
    double*  d_cellVolume,
    double*  d_correctionDiv,
    double* d_source,
    int sizeDiag
)
{
    int blockSize = BSIZE;
    int gridDiag = (int)ceil((float)(sizeDiag) / blockSize);

    updateSourceKernel<<<gridDiag, blockSize>>>(
    d_cellVolume,
    d_correctionDiv,
    d_source,
    sizeDiag
    );
}