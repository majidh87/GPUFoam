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
    Foam::HybridArray

Description
    A template class for dynamic array allocation supporting both host and GPU 
    memory using CUDA. 

SourceFiles
    HybridArray.H
\*---------------------------------------------------------------------------*/
#ifndef HybridArray_H
#define HybridArray_H

#include "label.H"
#include "scalar.H"
#include "Ostream.H"
#include "IOstreams.H"
#include "error.H"
#include <cstdlib> // For malloc and free
//#define __DOXYGEN_ONLY__	//temporary code to cuda not get compilation error
//#define CUDA_USE
//#ifdef CUDA_USE
#include "cuda_error_handle.H"
//#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

template<class T>
class HybridArray
{
    // Private Data

        // Size of the array
        label size_;

        // Pointer to the allocated memory
        T* data_;

        // Memory location (Host or GPU)
        bool onGPU_;

public:

    // Constructors

        //- Default constructor
        HybridArray();

        //- Construct with size and memory location
        HybridArray(const label size, const bool onGPU = false);

        //- Destructor
        ~HybridArray();

    // Assignment

        //- No copy assignment from plain pointer (uncontrolled access)
        void operator=(T* p) = delete;

        //- No move assignment disguised as a copy assignment
        void operator=(const HybridArray<T>&) = delete;

        //- Move construct, transferring ownership from derived type.
        //  U must be derivable from T
        template<class U>
        explicit HybridArray(HybridArray<U>&& rhs) = delete;
		
    // Member Functions

        //- Allocate memory
        void allocate(const label size, const bool onGPU = false);

        //- Deallocate memory
        void deallocate();

        //- Access elements
        T& operator[](const label i);

        //- Const access elements
        const T& operator[](const label i) const;

        //- Return the size of the array
        label size() const;

        //- Check if memory is on GPU
        bool onGPU() const;

        //- Return the array data
        T* Data() const;

        //- copy src buffer 
        void copy(const T* src);
        
        //- set an special element of array
        void set(const label i, T value);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "HybridArrayI.H"

#endif
