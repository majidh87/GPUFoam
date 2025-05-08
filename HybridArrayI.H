/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2015-2023 OpenCFD Ltd.
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
    Foam::HybridArrayI

Description
    A template class for dynamic array allocation supporting both host and GPU 
    memory using CUDA. 

SourceFiles
    HybridArrayI.H
\*---------------------------------------------------------------------------*/
#ifndef HybridArrayI_H
#define HybridArrayI_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class T>
HybridArray<T>::HybridArray()
:
    size_(0),
    data_(nullptr),
    onGPU_(false)
{}

template<class T>
HybridArray<T>::HybridArray(const Switch onGPU)
:
size_(0),
data_(nullptr),
onGPU_(onGPU)
{}

template<class T>
HybridArray<T>::HybridArray(const label size, const bool onGPU)
:
    size_(size),
    data_(nullptr),
    onGPU_(onGPU)
{
    allocate(size, onGPU);
}

template<class T>
HybridArray<T>::~HybridArray()
{
    deallocate();
}

template<class T>
void HybridArray<T>::allocate(const label size, const Switch onGPU)
{
    if (size <= 0)
    {
        FatalErrorIn("HybridArray<T>::allocate(const label, const bool)")
            << "Invalid size: " << size
            << abort(FatalError);
    }

    deallocate(); // Deallocate existing memory

    size_ = size;
    onGPU_ = onGPU;

    if (onGPU_)
    {
#ifdef CUDA_USE
        if constexpr(std::is_same<T,int>::value)
	{
	  printf("allocation int array in cuda size_:%d\n",size_);
	}
	_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
        if constexpr(std::is_same<T,int>::value)
	{
	//	printf("data[0]:%d \n",data_[0]);
	}
#endif
        if (!data_)
        {
            FatalErrorIn("HybridArray<T>::allocate(const label, const bool)")
                << "Failed to allocate GPU memory."
                << abort(FatalError);
        }
    }
    else
    {
        // Host allocation using malloc
        data_ = static_cast<T*>(std::malloc(size_ * sizeof(T)));
        if (!data_)
        {
            FatalErrorIn("HybridArray<T>::allocate(const label, const bool)")
                << "Failed to allocate memory."
                << abort(FatalError);
        }
    }
}

template<class T>
void HybridArray<T>::deallocate()
{
    if (data_)
    {
        if (onGPU_)
        {
            // GPU deallocation (placeholder for CUDA/OpenCL code)
#ifdef CUDA_USE
            _CUDA(cudaFree(data_)); 
#endif
        }
        else
        {
            // Host deallocation using free
            std::free(data_);
        }
        data_ = nullptr;
    }
    size_ = 0;
}

template<class T>
T& HybridArray<T>::operator[](const label i)
{
    if (i < 0 || i >= size_)
    {
        FatalErrorIn("HybridArray<T>::operator[](const label)")
            << "Index out of bounds: " << i
            << abort(FatalError);
    }
    return data_[i];
}

template<class T>
const T& HybridArray<T>::operator[](const label i) const
{
    if (i < 0 || i >= size_)
    {
        FatalErrorIn("HybridArray<T>::operator[](const label) const")
            << "Index out of bounds: " << i
            << abort(FatalError);
    }
    return data_[i];
}

template<class T>
label HybridArray<T>::size() const
{
    return size_;
}

template<class T>
bool HybridArray<T>::onGPU() const
{
    return onGPU_;
}

template<class T>
T* HybridArray<T>::Data() 
{
    return data_;
}


template<class T>
void HybridArray<T>::copy(const T* src,const Switch srcOnGPU = false)
{
    if (!src)
    {
        FatalErrorIn("HybridArray<T>::copyFromHost(const T*)")
            << "Source pointer is null."
            << abort(FatalError);
    }
    if(!onGPU_ && !srcOnGPU)
    {
        // Copy data from external pointer to Host memory
        std::memcpy(data_, src, size_ * sizeof(T));
    }
    else if(onGPU_ && !srcOnGPU)
    {
#ifdef CUDA_USE
        printf("copy of hybrid array size_:%d sizeof(T):%d \n",size_,sizeof(T));
        if constexpr(std::is_same<T,int>::value)
        {
            printf("src[0] %d \n",src[0]);
        }
        _CUDA(cudaMemcpy(data_, src, size_*sizeof(T),cudaMemcpyHostToDevice));
        if constexpr(std::is_same<T,int>::value)
        {
        //	printf("data[0] %d \n",data_[0]);
        }
#endif
    }
    else if(!onGPU_ && srcOnGPU)
    {
        _CUDA(cudaMemcpy(data_, src, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template<typename T>
void HybridArray<T>::set(const label i, T value)
{
        if(i >= 0 && i < size_ )
        {
            data_[i] = value;
        }
        else
        {
            FatalErrorInFunction
                << "array index out of bounds"
                << abort(FatalError);
        }
}

template<typename T>
template<class U>
void HybridArray<T>::copy(HybridArray<U>& rhs)
{
    if constexpr(std::is_same<T,U>::value)
    {

    allocate(rhs.size(),onGPU_);
    copy(rhs.Data(),rhs.onGPU());    
    }
    else
    {
            FatalErrorIn("HybridArray<T>::copy")
                << "T and U template type is not equal."
                << abort(FatalError);
    
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
