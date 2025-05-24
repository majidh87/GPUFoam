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
inline Foam::hybridSurfaceScalarField::hybridSurfaceScalarField()
{

}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::hybridSurfaceScalarField::init(fvMesh& mesh)
{
    const polyBoundaryMesh& patches = mesh.boundaryMesh();
    
    //Foam::label numCells          = static_cast<Foam::label>(mesh.cells().size());
    Foam::label numPatches        = static_cast<Foam::label>(patches.size());
    Foam::label numInternalFaces  = static_cast<Foam::label>(mesh.faceNeighbour().size());

    isInitialized_ = true;            

    hostPatchSizes_.allocate(numPatches,false);

    deviceInternalField.allocate(numInternalFaces,true);           

    hostBoundaryField.allocate(numPatches,false);

}

void Foam::hybridSurfaceScalarField::handle(fvMesh& mesh,surfaceScalarField& sf_DT)
{

    if (!isInitialized_)
    {
        init(mesh);
    }

    const polyBoundaryMesh& patches = mesh.boundaryMesh();

    Foam::label numPatches        = static_cast<Foam::label>(patches.size());

    forAll(patches, patchI)
    {
        hostPatchSizes_.set(patchI,patches[patchI].faceCells().size());

        const scalarList* pfSfDTptr= &sf_DT.boundaryField()[patchI];
        scalarList* pfSfDTlist = const_cast<scalarList*>(pfSfDTptr);

        hostBoundaryField.set(patchI,&pfSfDTlist->first());

    }
    deviceBoundaryField.copy(hostBoundaryField,hostPatchSizes_);

    const scalarList* sfDTPtr = &sf_DT.internalField();
    scalarList* sfDTlist = const_cast<scalarList*>(sfDTPtr);
    Foam::scalar *hostSurfDT = sfDTlist->begin();
    deviceInternalField.copy(hostSurfDT);

}

Foam::hybridSurfaceScalarField::~hybridSurfaceScalarField()
{
    hostPatchSizes_.deallocate();
    hostBoundaryField.deallocate();
    deviceInternalField.deallocate();
    deviceBoundaryField.deallocate();
    isInitialized_ = false;            

}