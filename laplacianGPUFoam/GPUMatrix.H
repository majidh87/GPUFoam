/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2025 OpenFOAM Foundation
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
    Foam::GPUMatrix

Description
    A template class for dynamic matrix allocation on GPU 

SourceFiles
    GPUMatrix.H
    GPUMatrixI.H

\*---------------------------------------------------------------------------*/

#ifndef Foam_GpuMatrix_H
#define Foam_GpuMatrix_H

#include "direction.H"
#include "scalar.H"
#include "word.H"
#include "zero.H"
#include "HybridArray.H"
#include "HybridMatrix.H"
#include "MiniVectorSpace.H"
#include "cuda_error_handle.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class HybridVectorField Declaration
\*---------------------------------------------------------------------------*/
template<class T>
class GPUMatrix
{
public:
    // Constructors

        //- Default constructor
        GPUMatrix();
        ~GPUMatrix();
        HybridMatrix<T> deviceIntermediate; // Intermediate host array for device data
        HybridArray<T *> deviceList;

        void copy(HybridMatrix<T>& hostMatrix);
        void copy(HybridArray<T *>& hostBuffer,HybridArray<Foam::label>& hostSize);
        void deallocate();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "GPUMatrixI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
