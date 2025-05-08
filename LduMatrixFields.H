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
    A template class for dynamic array allocation of output matrix fields. 

SourceFiles
    LduMatrixFields.H
\*---------------------------------------------------------------------------*/
#ifndef LduMatrixFields_H
#define LduMatrixFields_H

#include "label.H"
#include "scalar.H"
#include "HybridArray.H"

#ifdef CUDA_USE
#include "cuda_error_handle.H"
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{

    class LduMatrixFields
    {
    public:
        // Public Data

            // Device arrays for linear system (matrix and source terms)
            // Diagonal elements of the matrix
            HybridArray<Foam::scalar> diagonal; 
            // Source term array     
            HybridArray<Foam::scalar> source;
            // Lower triangular elements of the matrix    
            HybridArray<Foam::scalar> lower;
            // Upper triangular elements of the matrix     
            HybridArray<Foam::scalar> upper;     

    private:
            // Number of cells in the mesh
            label numCells_;

            // Number of internal faces in the mesh
            label numInternalFaces_;

            // Memory location (Host or GPU)
            Switch onGPU_;

    public:

        // Constructors

            //- Default constructor
            LduMatrixFields();

            //- Construct with size and memory location
            LduMatrixFields(const label numCells, const label numInternalFaces , const Switch onGPU);

            
            //- set member values and call allocate
            void init(const label numCells, const label numInternalFaces , const Switch onGPU);
            
            //- Destructor
            ~LduMatrixFields();

            // Initialize the class with buffers
            void allocate();

            //- Deallocate memory for the buffers
            void deallocate();

    };
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "LduMatrixFieldsI.H"

#endif
