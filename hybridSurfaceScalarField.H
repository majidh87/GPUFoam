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
    Foam::hybridSurfaceScalarField

Description
    A template class for dynamic matrix allocation on GPU 

SourceFiles
    hybridSurfaceScalarField.H
    hybridSurfaceScalarFieldI.H

\*---------------------------------------------------------------------------*/
#ifndef Foam_hybridSurfaceScalarField_H
#define Foam_hybridSurfaceScalarField_H

#include "direction.H"
#include "scalar.H"
#include "word.H"
#include "zero.H"
#include "HybridArray.H"
#include "HybridMatrix.H"
#include "cuda_error_handle.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class HybridVectorField Declaration
\*---------------------------------------------------------------------------*/
class hybridSurfaceScalarField
{
public:
    // Constructors

        //- Default constructor
        hybridSurfaceScalarField();
        ~hybridSurfaceScalarField();

        // Host: Patch field _ surface scalar field DT
        HybridArray<Foam::scalar *> hostBoundaryField;
        
        // Device: face area
        HybridArray<Foam::scalar> deviceInternalField;  

        // Device: Patch field _ surface scalar field DT
        Foam::GPUMatrix<Foam::scalar> deviceBoundaryField; 

    // Initialize the class with mesh data
    void init(fvMesh& mesh);

    // Handle function to process mesh and field data
    void handle(fvMesh& mesh,
        surfaceScalarField& sf_DT);

private:

        // Flag to check if initialization is done
        Foam::Switch isInitialized_;
        
        // Host: Number of cells adjacent to the faces of each patch
        HybridArray<Foam::label> hostPatchSizes_;        
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "hybridSurfaceScalarFieldI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
