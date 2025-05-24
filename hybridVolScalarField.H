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
    Foam::hybridVolScalarField

Description
    A template class for hybrid volume scalar field on GPU 

SourceFiles
    hybridVolScalarField.H
    hybridVolScalarFieldI.H

\*---------------------------------------------------------------------------*/
#ifndef Foam_hybridVolScalarField_H
#define Foam_hybridVolScalarField_H

#include "direction.H"
#include "scalar.H"
#include "word.H"
#include "zero.H"
#include "HybridArray.H"
#include "HybridMatrix.H"
#include "GPUMatrix.H"
#include "cuda_error_handle.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class hybridVolScalarField Declaration
\*---------------------------------------------------------------------------*/
class hybridVolScalarField
{
public:
    //- Constructors
    hybridVolScalarField();
    
    //- Deconstructors
    ~hybridVolScalarField();

    // Host: Boundary coefficients for gradient calculation
    HybridArray<Foam::scalar *> hostPatchBoundaryCoeffs;    

    // Host: Internal coefficients for gradient calculation
    HybridArray<Foam::scalar *> hostPatchInternalCoeffs;    

    // Device: Old time temperature field
    HybridArray<Foam::scalar> oldField;       

    // Device: Boundary coefficients for gradient calculation
    Foam::GPUMatrix<Foam::scalar> devicePatchBoundaryCoeffs;    
    
    // Device: Internal coefficients for gradient calculation
    Foam::GPUMatrix<Foam::scalar> devicePatchInternalCoeffs;    
    
    // device array for new temperature field
    Foam::scalar* currentField;
     
    // Initialize the class with mesh data
    void init(fvMesh& mesh);

    // Handle function to process mesh and field data
    void handle(fvMesh& mesh,
        volScalarField& T);
    
    //- copy currentField to oldField
    void update();

private:

    // Flag to check if initialization is done
    Foam::Switch isInitialized_;
    
    // Host: Number of cells adjacent to the faces of each patch
    HybridArray<Foam::label> hostPatchSizes_;        

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "hybridVolScalarFieldI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
