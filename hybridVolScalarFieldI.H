/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2019-2022 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

inline Foam::hybridVolScalarField::hybridVolScalarField() :
currentField(NULL),
isInitialized_(false)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::hybridVolScalarField::init(fvMesh& mesh)
{
    const polyBoundaryMesh& patches = mesh.boundaryMesh();
    
    Foam::label numCells          = static_cast<Foam::label>(mesh.cells().size());
    Foam::label numPatches        = static_cast<Foam::label>(patches.size());

    hostPatchSizes_.allocate(numPatches,false);

    hostPatchBoundaryCoeffs.allocate(numPatches,false);
    
    hostPatchInternalCoeffs.allocate(numPatches,false);

    
    // Allocate memory for host arrays
    //currentField = static_cast<Foam::scalar*>(malloc(numCells*sizeof(Foam::scalar)));
#ifdef CUDA_USE
    _CUDA(cudaMalloc(reinterpret_cast<void**>(&currentField), numCells * sizeof(Foam::scalar)));
#endif

    oldField.allocate(numCells,true);

    isInitialized_ = true;            

    hostPatchBoundaryCoeffs_.allocate_rows(numPatches,STORE_ON_HOST);
    
    hostPatchInternalCoeffs_.allocate_rows(numPatches,STORE_ON_HOST);

}

void Foam::hybridVolScalarField::handle(fvMesh& mesh,volScalarField& T)
{
    if (!isInitialized_)
    {
        init(mesh);
    }

    const polyBoundaryMesh& patches = mesh.boundaryMesh();

    //Foam::label numPatches        = static_cast<Foam::label>(patches.size());

    forAll(patches, patchI)
    {
        scalarField pF_BC_SF = T.boundaryField()[patchI].gradientBoundaryCoeffs();
        scalarField pF_IC_SF = T.boundaryField()[patchI].gradientInternalCoeffs();
        
        hostPatchSizes_.set(patchI,patches[patchI].faceCells().size());

        
        //hostPatchInternalCoeffs_.allocate_col(patchI,hostPatchSizes_[patchI]);
        
        
        //hostPatchBoundaryCoeffs_.allocate_col(patchI,hostPatchSizes_[patchI]);
        
        hostPatchInternalCoeffs.set(patchI,pF_IC_SF.data());

        
        hostPatchBoundaryCoeffs.set(patchI,pF_BC_SF.data());
        
        //hostPatchInternalCoeffs_.copy_col(patchI,pF_IC_SF.data());
        //hostPatchBoundaryCoeffs_.copy_col(patchI,pF_BC_SF.data());

    }    

    devicePatchBoundaryCoeffs.copy(hostPatchBoundaryCoeffs,hostPatchSizes_);

    devicePatchInternalCoeffs.copy(hostPatchInternalCoeffs,hostPatchSizes_);

    // Update old time temperature field to device
    const scalarList* ToldPtr = &T.oldTime().primitiveField();
    scalarList* Toldlist = const_cast<scalarList*>(ToldPtr);
    Foam::scalar* hostOldTemperature;	//Host : Old Temperature Time Field
    hostOldTemperature = Toldlist->begin();
        
    oldField.copy(hostOldTemperature);

    //devicePatchBoundaryCoeffs.copy(hostPatchBoundaryCoeffs_);
    
    //devicePatchInternalCoeffs.copy(hostPatchInternalCoeffs_);

}

void Foam::hybridVolScalarField::update()
{
    //copy device to device
    oldField.copy(currentField,true);
}


Foam::hybridVolScalarField::~hybridVolScalarField()
{
    isInitialized_ = false;
    hostPatchSizes_.deallocate();
    hostPatchBoundaryCoeffs.deallocate();
    hostPatchInternalCoeffs.deallocate();
    oldField.deallocate();
    devicePatchBoundaryCoeffs.deallocate();
    devicePatchInternalCoeffs.deallocate();

#ifdef CUDA_USE
    _CUDA(cudaFree(currentField)); 
#endif

}
