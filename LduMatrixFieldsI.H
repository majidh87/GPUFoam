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
    numCells_(0),
    numInternalFaces_(0),
    onGPU_(false),
    diagonal_(nullptr),
    source_(nullptr),
    lower_(nullptr),
    upper_(nullptr)
{}

LduMatrixFields::LduMatrixFields(const label numCells, const label numInternalFaces , const Switch onGPU)
:
    numCells_(numCells),
    numInternalFaces_(numInternalFaces),
    onGPU_(onGPU),
    diagonal_(nullptr),
    source_(nullptr),
    lower_(nullptr),
    upper_(nullptr)
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
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&diagonal_)    ,numCells_*sizeof(Foam::scalar)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&source_)  ,numCells_*sizeof(Foam::scalar)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&lower_)   ,numInternalFaces_*sizeof(Foam::scalar)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&upper_)   ,numInternalFaces_*sizeof(Foam::scalar)));
#endif
    }
    else
    {
        diagonal_ = static_cast<Foam::scalar*>(std::malloc(numCells_ * sizeof(Foam::scalar)));
        source_ = static_cast<Foam::scalar*>(std::malloc(numCells_ * sizeof(Foam::scalar)));
        lower_ = static_cast<Foam::scalar*>(std::malloc(numInternalFaces_ * sizeof(Foam::scalar)));
        upper_ = static_cast<Foam::scalar*>(std::malloc(numInternalFaces_ * sizeof(Foam::scalar)));
    }
}

void LduMatrixFields::deallocate()
{
    if (diagonal_)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(diagonal_)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(diagonal_);
        }
        diagonal_ = nullptr;
    }
    if (source_)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(source_)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(source_);
        }
        source_ = nullptr;
    }
    if (lower_)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(lower_)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(lower_);
        }
        lower_ = nullptr;
    }
    if (upper_)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(upper_)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(upper_);
        }
        upper_ = nullptr;
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif
