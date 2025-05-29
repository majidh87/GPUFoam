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
}


void LduMatrixFields::init(fvMesh& mesh,const Switch onGPU)
{
    onGPU_ = onGPU;


    
    numCells_          = static_cast<Foam::label>(mesh.cells().size());
    numInternalFaces_  = static_cast<Foam::label>(mesh.faceNeighbour().size());
    
	allocate();

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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif
