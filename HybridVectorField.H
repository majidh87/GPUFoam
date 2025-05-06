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
    Foam::HybridVectorField

Description
    A template class for dynamic array allocation supporting both host and GPU 
    memory using CUDA. 

SourceFiles
    HybridVectorFieldI.H
\*---------------------------------------------------------------------------*/

#ifndef Foam_MiniVectorSpace_H
#define Foam_MiniVectorSpace_H

#include "direction.H"
#include "scalar.H"
#include "word.H"
#include "zero.H"
#include "HybridArray.H"
#include "MiniVectorSpace.H"
#ifdef CUDA_USE
#include "cuda_error_handle.H"
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class HybridVectorField Declaration
\*---------------------------------------------------------------------------*/
template<class Cmpt, direction Ncmpts>
class HybridVectorField
{
	
private:
    typedef Foam::MiniVectorSpace<Cmpt,Ncmpts> miniVector;
    Foam::HybridArray<miniVector> data_;

public:
    // Constructors

        //- Default constructor
        HybridVectorField();

        //- Construct with size and memory location
        HybridVectorField(const label size, const bool onGPU = false);

        //- Destructor
        ~HybridVectorField();

    //- Component labeling enumeration
    enum components { X, Y, Z };

    // Assignment

        //- No copy assignment from plain pointer (uncontrolled access)
        //void operator=(T* p) = delete;

        //- No move assignment disguised as a copy assignment
        void operator=(const HybridVectorField<Cmpt,Ncmpts>&) = delete;

        //- Move construct, transferring ownership from derived type.
        //  U must be derivable from T
        template<class Cmpt2, direction Ncmpts2>
        explicit HybridVectorField(HybridVectorField<Cmpt2,Ncmpts2>&& rhs) = delete;

    // Member Functions

        //- Allocate memory
        void allocate(const label size, const bool onGPU = false);

        //- Deallocate memory
        void deallocate();

        //- Access elements
        Foam::MiniVectorSpace<Cmpt,Ncmpts>& operator[](const direction d);

        //- Const access elements
        const Foam::MiniVectorSpace<Cmpt,Ncmpts>& operator[](const direction d) const;

        //- Return the size of the array
        label size() const;

        //- Check if memory is on GPU
        Foam::Switch onGPU() const;

        //- Return the array data
        Foam::MiniVectorSpace<Cmpt,Ncmpts>* Data() const;

    // Component Access

        //- Access to the vector x component
        const Cmpt& x(direction index) const noexcept;

        //- Access to the vector y component
        const Cmpt& y(direction index) const noexcept;

        //- Access to the vector z component
        const Cmpt& z(direction index) const noexcept;

        //- Access to the vector x component
        Cmpt& x(direction index) noexcept;

        //- Access to the vector y component
        Cmpt& y(direction index) noexcept;

        //- Access to the vector z component
        Cmpt& z(direction index) noexcept;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "HybridVectorFieldI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
