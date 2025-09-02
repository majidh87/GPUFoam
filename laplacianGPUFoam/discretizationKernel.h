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



void cellKernelWrapper(int sizeDiag,              // Number of cells
    double *vcs,               // Volume of the cells
    double *tot,               // Total value (e.g., temperature) at each cell
    double rDelgaG,            // Scaling factor
    double *d_diag,            // Diagonal terms of the matrix (on device)
    double *d_source          // Source terms of the linear system (on device)
);

void faceKernelWrapper(
    int sizeFace,              // Number of faces
    double *delta,             // Delta coefficient for faces
    double *gamma,             // Gamma coefficient for faces
    double *DT_surf,               // Total value (e.g., temperature) at each cell
    int *upperAddr,            // Indices of the upper cells for each face
    int *lowerAddr,            // Indices of the lower cells for each face
    double *d_upper,           // Upper off-diagonal terms (on device)
    double *d_lower,
    double *d_diag            // Diagonal terms of the matrix (on device)
);


/*void boundaryKernelWrapper(
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
);*/

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
);

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
);

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
    int N,
    int sizeDiag                           

);
void computeCorrectionDivWrapper(
    const double* d_correctionF,
    const int* d_upperAddr,
    const int* d_lowerAddr,
    const double* d_cellVolumes,
    double* d_correctionDiv,
    int sizeFace
);

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
                                );
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
    int sizeFace
);

void updateSourceWrapper(
    double*  d_cellVolume,
    double*  d_correctionDiv,
    double* d_source,
    int sizeDiag
);

