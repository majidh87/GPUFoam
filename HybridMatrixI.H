/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
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
    Foam::HybridMatrixI

Description
    A template class for dynamic matrix allocation supporting both host and GPU 
    memory using CUDA. 

SourceFiles
    HybridMatrixI.H
\*---------------------------------------------------------------------------*/


#ifndef HybridMatrixI_H
#define HybridMatrixI_H

#include "UIndirectList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * //

template<typename T>
HybridMatrix<T>::HybridMatrix()
:
    rows_(0),
    cols_(0),
    storeType_(UNKNOWN_STATE),
    data_(nullptr)
{}

template<typename T>
HybridMatrix<T>::HybridMatrix(const label rows, const STORE_TYPE storeType)
:
    rows_(rows),
    cols_(rows, 0),
    storeType_(storeType),
    data_(nullptr)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * //

template<typename T>
HybridMatrix<T>::~HybridMatrix()
{
    deallocate();
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<typename T>
void HybridMatrix<T>::allocate(const labelList& colsSize)
{
    // Deallocate existing memory
    deallocate();

    cols_ = colsSize;

    if(storeType_== STORE_ON_GPU)
    {
        // Allocate GPU memory
#ifdef CUDA_USE
        //_CUDA(cudaMalloc(reinterpret_cast<void***>(&data_), rows_ * sizeof(T*)));
#endif

        for(label i=0; i<rows_; ++i)
        {
#ifdef CUDA_USE
        //_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_[i]), cols_[i] * sizeof(T)));
#endif
        }
    }
    else if (storeType_==STORE_ON_HOST_GPU)
    {
        // Allocate host memory
        data_ = static_cast<T**>(malloc(rows_*sizeof(T*)));
        if(!data_)
        {
            FatalErrorInFunction
                << "Host memory allocation failed"
                << abort(FatalError);
        }
        for(label i=0; i<rows_; ++i)
        {
#ifdef CUDA_USE
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&data_[i]), cols_[i] * sizeof(T)));
#endif
        }

    }
    else if(storeType_==STORE_ON_HOST)
    {
        // Allocate host memory
        data_ = static_cast<T**>(malloc(rows_*sizeof(T*)));
        if(!data_)
        {
            FatalErrorInFunction
                << "Host memory allocation failed"
                << abort(FatalError);
        }

        for(label i=0; i<rows_; ++i)
        {
            data_[i] = static_cast<T*>(malloc(cols_[i]*sizeof(T)));
            if(!data_[i])
            {
                FatalErrorInFunction
                    << "Host allocation failed for row " << i
                    << abort(FatalError);
            }
        }
    }
}

template<typename T> 
void HybridMatrix<T>::allocate_rows(const label rows, const STORE_TYPE storeType)
{
    // Deallocate existing memory
    deallocate();
    cols_.setSize(rows,0);
    storeType_ = storeType;
    if(storeType_== STORE_ON_GPU)
    {
        // Allocate GPU memory
#ifdef CUDA_USE
        //_CUDA(cudaMalloc(reinterpret_cast<void***>(&data_), rows_ * sizeof(T*)));
#endif
    }
    else if(storeType_== STORE_ON_HOST || storeType_ == STORE_ON_HOST_GPU)
    {
        // Allocate host memory
        /*data_ = static_cast<T**>(malloc(rows_*sizeof(T*)));
        if(!data_)
        {
            FatalErrorInFunction
                << "Host memory allocation failed"
                << abort(FatalError);
        }*/
    }

}
template<typename T>
void HybridMatrix<T>::allocate_col(const label row, const label colSize)
{
    cols_[row] = colSize;
    if(storeType_== STORE_ON_GPU || storeType_== STORE_ON_HOST_GPU)
    {
#ifdef CUDA_USE
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&data_[row]), cols_[row] * sizeof(T)));
#endif
    }
    else if(storeType_== STORE_ON_HOST)
    {

            data_[row] = static_cast<T*>(malloc(cols_[row]*sizeof(T)));
            if(!data_[row])
            {
                FatalErrorInFunction
                    << "Host allocation failed for row " << row
                    << abort(FatalError);
            }
    }
	
}

template<typename T>
void HybridMatrix<T>::deallocate()
{
    if(data_)
    {
        if(storeType_== STORE_ON_GPU)
        {
#ifdef CUDA_USE
            for(label i=0; i<rows_; ++i)
            {
                if(data_[i]) cudaFree(data_[i]);
            }
            cudaFree(data_);
#endif
        }
        else if(storeType_== STORE_ON_HOST)
        {
            for(label i=0; i<rows_; ++i)
            {
                if(data_[i]) free(data_[i]);
            }
            free(data_);
        }
        else if(storeType_== STORE_ON_HOST_GPU)
        {
#ifdef CUDA_USE
            for(label i=0; i<rows_; ++i)
            {
                if(data_[i]) cudaFree(data_[i]);
            }
#endif
            free(data_);
        }
        data_ = nullptr;
    }
}

template<typename T>
void HybridMatrix<T>::copy(const T** src)
{
    if(storeType_== STORE_ON_GPU || storeType_== STORE_ON_HOST_GPU)
    {
        for(label i=0; i<rows_; ++i)
        {
#ifdef CUDA_USE
        _CUDA(cudaMemcpy(data_[i], src[i], cols_[i]*sizeof(T),cudaMemcpyHostToDevice));
#endif
        }
    }
    else if(storeType_== STORE_ON_HOST)
    {
        for(label i=0; i<rows_; ++i)
        {
            memcpy(data_[i], src[i], cols_[i]*sizeof(T));
        }
    }
}

//- copy src to row column
template<typename T>        
void HybridMatrix<T>::copy_col(const label row, const T* src)
{
    if(storeType_== STORE_ON_GPU || storeType_== STORE_ON_HOST_GPU)
    {
#ifdef CUDA_USE
        _CUDA(cudaMemcpy(data_[row], src, cols_[row]*sizeof(T),cudaMemcpyHostToDevice));
#endif
    }
    else if(storeType_== STORE_ON_HOST)
    {
        memcpy(data_[row], src, cols_[row]*sizeof(T));
    }
}

// * * * * * * * * * * * * * * Accessors * * * * * * * * * * * * * * * * * //

template<typename T>
label HybridMatrix<T>::rows() const noexcept
{
    return rows_;
}

template<typename T>
label HybridMatrix<T>::cols(const label row) const
{
    return cols_[row];
}

template<typename T>
STORE_TYPE HybridMatrix<T>::onDevice() const noexcept
{
    return storeType_;
}
template<typename T>
T** HybridMatrix<T>::Data() const 
{
    return data_;
}

// * * * * * * * * * * * * * * Operators * * * * * * * * * * * * * * * * * //

template<typename T>
T& HybridMatrix<T>::operator()(const label row, const label col)
{
    return data_[row][col];
}

template<typename T>
const T& HybridMatrix<T>::operator()(const label row, const label col) const
{
    return data_[row][col];
}

template<typename T>
void HybridMatrix<T>::set(const label row, const label col, const T value)
{
        if(row >= 0 && row < rows_ && col >= 0 && col < cols_[row])
        {
            data_[row][col] = value;
        }
        else
        {
            FatalErrorInFunction
                << "Row or column index out of bounds"
                << abort(FatalError);
        }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
#endif
