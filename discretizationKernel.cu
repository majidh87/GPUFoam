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

    // Compute diagonal term: diag[id] = rDelgaG * vcs[id]
    diag[id] = rDelgaG * vcs[id];

    // Compute source term: source[id] = rDelgaG * vcs[id] * tot[id]
    source[id] = rDelgaG * vcs[id] * tot[id];
    
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
    atomicAdd(&diag[pAdrr[idy][idx]], pf_GammaSf[idy][idx] * pf_IC[idy][idx]);  // Add to diagonal
    atomicAdd(&source[pAdrr[idy][idx]], -pf_GammaSf[idy][idx] * pf_BC[idy][idx]); // Subtract from source
   
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
                       double rDelgaG,            // Scaling factor
                       double *d_diag,            // Diagonal terms of the matrix (on device)
                       double *d_source,          // Source terms of the linear system (on device)
                       double *d_upper,           // Upper off-diagonal terms (on device)
                       double *d_lower)           // Lower off-diagonal terms (on device)
{
    // Set block size for 1D kernels
    int blockSize = BSIZE;

    // Compute grid size for cellKernel
    int gridCell = (int)ceil((float)(sizeDiag) / blockSize);

    // Launch cellKernel to compute diagonal and source terms
    cellKernel<<<gridCell, blockSize>>>(vcs, tot, rDelgaG, d_diag, d_source, sizeDiag);

    // Compute grid size for faceKernel
    int gridFace = (int)ceil((float)(sizeFace) / blockSize);

    // Launch faceKernel to compute off-diagonal terms
    faceKernel<<<gridFace, blockSize>>>(delta, gamma,DT_surf, upperAddr, lowerAddr, d_upper, d_lower, d_diag, sizeFace);

    // // // Set block and grid sizes for boundaryKernel (2D kernel)
    // dim3 blockBoundary(BSIZEX, BSIZEY, 1);  // 2D block size
    // dim3 gridBoundary((int)ceil((float)(maxPatches) / blockBoundary.x),  // Grid size in x-direction
    //                  (int)ceil((float)(numOfPatches) / blockBoundary.y), // Grid size in y-direction
    //                  1);

    // // Launch boundaryKernel to handle boundary conditions
    // boundaryKernel<<<gridBoundary, blockBoundary>>>(d_pSize, d_pAdrr, d_pf_BC, d_pf_IC, d_pf_GammaSf, d_diag, d_source, maxPatches, numOfPatches);

    // Optional: Debugging code to check the source terms (commented out)
    /*
    double* hg_source = (double*)malloc((9) * sizeof(double));
    cudaMemcpy(hg_source, d_source, (9) * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 9; i++)
        printf("*******%f\n", hg_source[i]);
    */
}