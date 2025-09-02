/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::LduMatrixFields

Description
    A template class for dynamic array allocation of output LDU matrix fields. 

SourceFiles
    LduMatrixFieldsI.H
\*---------------------------------------------------------------------------*/
#ifndef LduMatrixFieldsI_H
#define LduMatrixFieldsI_H

#include "UIndirectList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

// * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * //

LduMatrixFields::LduMatrixFields()
:
    diagonal(nullptr),
    source(nullptr),
    lower(nullptr),
    upper(nullptr),
    numCells_(0),
    numInternalFaces_(0),
    onGPU_(false)
{}

LduMatrixFields::LduMatrixFields(const label numCells, const label numInternalFaces , const Switch onGPU)
:
    diagonal(nullptr),
    source(nullptr),
    lower(nullptr),
    upper(nullptr),
    numCells_(numCells),
    numInternalFaces_(numInternalFaces),
    onGPU_(onGPU)
{

}

LduMatrixFields::~LduMatrixFields()
{
    deallocate();
    finalizeSolver();
}

void LduMatrixFields::allocate()
{
    deallocate();

    if(onGPU_)
    {
#ifdef CUDA_USE
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&diagonal)    ,numCells_*sizeof(Foam::scalar)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&source)  ,numCells_*sizeof(Foam::scalar)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&lower)   ,numInternalFaces_*sizeof(Foam::scalar)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&upper)   ,numInternalFaces_*sizeof(Foam::scalar)));
#endif
    }
    else
    {
        diagonal = static_cast<Foam::scalar*>(std::malloc(numCells_ * sizeof(Foam::scalar)));
        source = static_cast<Foam::scalar*>(std::malloc(numCells_ * sizeof(Foam::scalar)));
        lower = static_cast<Foam::scalar*>(std::malloc(numInternalFaces_ * sizeof(Foam::scalar)));
        upper = static_cast<Foam::scalar*>(std::malloc(numInternalFaces_ * sizeof(Foam::scalar)));
    }
}

void LduMatrixFields::deallocate()
{
    if (diagonal)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(diagonal)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(diagonal);
        }
        diagonal = nullptr;
    }
    if (source)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(source)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(source);
        }
        source = nullptr;
    }
    if (lower)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(lower)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(lower);
        }
        lower = nullptr;
    }
    if (upper)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(upper)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(upper);
        }
        upper = nullptr;
    }
}

void LduMatrixFields::init(const label numCells, const label numInternalFaces , const Switch onGPU)

{
    numCells_ = numCells;
    numInternalFaces_ = numInternalFaces;
    onGPU_ = onGPU;
	allocate();
    
    initSolver();
}


void LduMatrixFields::init(fvMesh& mesh,const Switch onGPU)
{
    onGPU_ = onGPU;


    
    numCells_          = static_cast<Foam::label>(mesh.cells().size());
    numInternalFaces_  = static_cast<Foam::label>(mesh.faceNeighbour().size());
    
	allocate();

    initSolver();
}

void LduMatrixFields::cellKernelWrapper(int sizeDiag,              // Number of cells
    double *vcs,               // Volume of the cells
    double *tot,               // Total value (e.g., temperature) at each cell
    double rDelgaG            // Scaling factor
    )
{
    ::cellKernelWrapper(sizeDiag,vcs,tot,rDelgaG,diagonal,source);

}

void LduMatrixFields::faceKernelWrapper(
    int sizeFace,              // Number of faces
    double *delta,             // Delta coefficient for faces
    double *gamma,             // Gamma coefficient for faces
    double *DT_surf,               // Total value (e.g., temperature) at each cell
    int *upperAddr,            // Indices of the upper cells for each face
    int *lowerAddr            // Indices of the lower cells for each face
    )
{
    ::faceKernelWrapper(sizeFace,delta,gamma,DT_surf,upperAddr,lowerAddr,upper,lower,diagonal);

}

void LduMatrixFields::boundaryKernelWrapper(
    int numOfPatches,          // Number of patches
    int maxPatches,            // Maximum number of faces in any patch
    int *d_pSize,              // Number of faces in each patch (on device)
    int **d_pAdrr,             // Addresses of the cells for each patch (on device)
    double **d_pf_BC,          // Boundary coefficients for each patch (on device)
    double **d_pf_IC,          // Internal coefficients for each patch (on device)
    double **d_pf_GammaSf,     // Gamma coefficient for each patch (on device)
    double **d_pf_DT_surf     // DT patch field(on device)
    )
{
    ::boundaryKernelWrapper(numOfPatches,maxPatches,d_pSize,d_pAdrr,d_pf_BC,d_pf_IC,d_pf_GammaSf,d_pf_DT_surf,diagonal,source);
}


void LduMatrixFields:: gradTKernelWrapper(
    double *d_T,
    int *d_upperAddr,
    int *d_lowerAddr,
    double *d_weight,
    double *d_faceSfX,
    double *d_faceSfY,
    double *d_faceSfZ,
    double *d_gradTX,
    double *d_gradTY,
    double *d_gradTZ,
    int sizeFace
)
{
    ::gradTKernelWrapper(d_T,
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

void LduMatrixFields:: applyBoundaryGradientWrapper (
                int* d_faceCells, // [N]
                double* d_SfX,    // [N]
                double* d_SfY,    // [N]
                double* d_SfZ,    // [N]
                double* d_ssf,    // [N]
                double *d_cellVolumes,
                double* d_gradX,                     // [nCells]
                double* d_gradY,                     // [nCells]
                double* d_gradZ,                     // [nCells]
                int N,
                int sizeDiag                             
                )
{
                ::applyBoundaryGradientWrapper (
                d_faceCells, // [N]
                d_SfX,    // [N]
                d_SfY,    // [N]
                d_SfZ,    // [N]
                d_ssf,    // [N]
                d_cellVolumes,
                d_gradX,                     // [nCells]
                d_gradY,                     // [nCells]
                d_gradZ,                     // [nCells]
                N ,
                sizeDiag                            
                );
}
void LduMatrixFields:: computeCorrectionDivWrapper(
    const double* d_correctionF,
    const int* d_upperAddr,
    const int* d_lowerAddr,
    const double* d_cellVolumes,
    double* d_correctionDiv,
    int sizeFace
)
{
    ::computeCorrectionDivWrapper(
        d_correctionF,
        d_upperAddr,
        d_lowerAddr,
        d_cellVolumes,
        d_correctionDiv,
        sizeFace
    );
}

void LduMatrixFields:: interpololateKernelWrapper( double* d_TX,
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
    ::interpololateKernelWrapper(d_TX,
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
void LduMatrixFields:: fusedCorrectionFluxWrapper(
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
                                )
{
    ::fusedCorrectionFluxWrapper(
                                   d_corrX,
                                   d_corrY,
                                   d_corrZ,
                                   d_gradX,
                                   d_gradY,
                                   d_gradZ,
                                   d_sf_DT,
                                   d_magSf,
                                   d_correctionF,
                                   sizeFace
                                );
}
void LduMatrixFields:: updateSourceWrapper(
                                double*  d_cellVolume,
                                double*  d_correctionDiv,
                                int sizeDiag
                                )
{
    ::updateSourceWrapper(d_cellVolume,
                           d_correctionDiv,
                           source,
                           sizeDiag
                                );
}

void LduMatrixFields::set(MeshFields& mesh)
{
            csrMatrix_.setValuesLDU
            (
                mesh.numCells,
                mesh.numInternalFaces,
                0,
                0,
                0,
                mesh.upperAddress.Data(),
                mesh.lowerAddress.Data(),
                0,
                NULL,
                NULL,
                diagonal,
                upper,
                lower,
                NULL
            );
            amgxSolver_.setOperator
            (
                mesh.numCells,
                mesh.numCells,
                mesh.numCells+2*mesh.numInternalFaces,
                csrMatrix_
            );

}

void LduMatrixFields::solve(MeshFields& mesh , Foam::scalar *T)
{
            amgxSolver_.solve
            (
                mesh.numCells,
                &T[0],
                source,  
                csrMatrix_
            );

}

void LduMatrixFields::initSolver()
{
    // Initialize MPI for parallel computation
    MPI_Init(NULL, NULL);
    const MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD; // MPI communicator for AMGX
    const std::string modeStr = "dDDI";            // AMGX mode (e.g., double precision)
    const std::string cfgFile = "./system/config";  // AMGX configuration file

    // Initialize AMGX matrix communication
    csrMatrix_.initialiseComms(amgx_mpi_comm, 0);

    // Initialize the AMGX solver
    amgxSolver_.initialize(amgx_mpi_comm, modeStr, cfgFile);

}


void LduMatrixFields::finalizeSolver()
{
    // Finalize MPI
    MPI_Finalize();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif
