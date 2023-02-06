#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
                double *h_diag,
                double *h_source,
                double *h_upper,
                double *h_lower
                );

