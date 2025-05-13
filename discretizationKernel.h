#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function declaration for the discretization kernel wrapper
void discKernelWrapper(
    int sizeDiag,            // Number of cells (size of diagonal terms)
    int sizeFace,            // Number of faces (size of off-diagonal terms)
    double *vcs,             // Volume of the cells (on host)
    double *tot,             // Total value (e.g., temperature) at each cell (on host)
    double *surfDT,          // Surface diffusion term (on host)
    double *delta,           // Delta coefficient for faces (on host)
    double *gamma,           // Gamma coefficient for faces (on host)
    int *upperAddr,          // Indices of the upper cells for each face (on host)
    int *lowerAddr,          // Indices of the lower cells for each face (on host)
    int numOfPatches,        // Number of patches
    int maxPatches,          // Maximum number of faces in any patch
    int *d_pSize,            // Number of faces in each patch (on device)
    int **d_pAdrr,           // Addresses of the cells for each patch (on device)
    double **d_pf_BC,        // Boundary coefficients for each patch (on device)
    double **d_pf_IC,        // Internal coefficients for each patch (on device)
    double **d_pf_GammaSf,   // Gamma coefficient for each patch (on device)
    double **d_pf_SfDT,      // Surface diffusion term for each patch (on device)
    double rDelgaG,          // Scaling factor for diagonal and source terms
    double *h_diag,          // Diagonal terms of the matrix (on host)
    double *h_source,        // Source terms of the linear system (on host)
    double *h_upper,         // Upper off-diagonal terms (on host)
    double *h_lower          // Lower off-diagonal terms (on host)
);


