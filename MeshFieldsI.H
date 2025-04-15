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
    Foam::MeshFieldsI

Description
    A template class for dynamic mesh fields allocation supporting both host and GPU 
    memory using CUDA. 

SourceFiles
    MeshFieldsI.H
\*---------------------------------------------------------------------------*/


#ifndef HeshFieldsI_H
#define MeshFieldsI_H

#include "UIndirectList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

// * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * //

MeshFields::MeshFields()
:
    onGPU_(false),
    isInitialized_(false)
{}

MeshFields::MeshFields(const bool onGPU)
:
onGPU_(onGPU),
isInitialized_(false)
{}

MeshFields::~MeshFields()
{
    deallocate();
}

void MeshFields::deallocate()
{
    isInitialized_ = false;
    cellVolumes_.deallocate();
        
        
    
    deltaCellCenters_.deallocate();
    
    faceAreas_.deallocate();
    
    //gammaMagSf_.deallocate();
    
    lowerAddress_.deallocate();
    upperAddress_.deallocate();

}

void MeshFields::init(fvMesh& mesh)
{
    Foam::label numCells          = static_cast<Foam::label>(mesh.cells().size());
    Foam::label numInternalFaces  = static_cast<Foam::label>(mesh.faceNeighbour().size());


    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCellVolumes_)  ,numCells_*sizeof(double)));
    cellVolumes_.allocate(numCells,STORE_ON_GPU);
        
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceFaceAreas_)  ,numCells_*sizeof(double)));
    faceAreas_.allocate(numCells,STORE_ON_GPU);
        
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceDeltaCellCenters_)  ,numInternalFaces_*sizeof(double)));
	deltaCellCenters_.allocate(numInternalFaces,STORE_ON_GPU);

    //gammaMagSf_.allocate(numInternalFaces,STORE_ON_GPU);           
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceGammaMagSf_),numInternalFaces_*sizeof(double)));
    
    lowerAddress_.allocate(numInternalFaces,STORE_ON_GPU);
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceLowerAddress_)   ,numInternalFaces_*sizeof(int)));
    
    upperAddress_.allocate(numInternalFaces,STORE_ON_GPU);
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceUpperAddress_)   ,numInternalFaces_*sizeof(int)));

    isInitialized_ = true;            

}

void MeshFields::handle(fvMesh& mesh)
{
    if (!isInitialized_)
    {
         FatalErrorInFunction
            << "ERROR: handle is called before init "
            << endl
            << abort(FatalError);
            return;
    }

    // Mesh data
    const scalarList* listPtr       = &mesh.V();
    const scalarList* deltaCPtr     = &mesh.deltaCoeffs();
    const scalarList* magSfPtr      = &mesh.magSf();
    const labelUList* ownerPtr      = &mesh.owner();
    const labelUList* neighbourPtr  = &mesh.neighbour();

    scalarList* blist           = const_cast<scalarList*>(listPtr);
    scalarList* deltaClist      = const_cast<scalarList*>(deltaCPtr);
    scalarList* magSflist       = const_cast<scalarList*>(magSfPtr);
    labelUList* ownerlist       = const_cast<labelUList*>(ownerPtr);
    labelUList* neighbourlist   = const_cast<labelUList*>(neighbourPtr);
    
    scalar *hostCellVolumes = &blist->first();             // Host: Cell volumes
    scalar *hostDeltaCellCenters = &deltaClist->first();   // Host: Delta cell centers for each internal face
    scalar *hostFaceAreas = &magSflist->first();           // Host: Lower address (owner) for each face
    label *hostLowerAddress = &ownerlist->first();           // Host: Upper address (neighbour) for each face
    label *hostUpperAddress = &neighbourlist->first();       // Host: Face surface area magnitude
		
    cellVolumes_.copy(hostCellVolumes);
        
    deltaCellCenters_.copy(hostDeltaCellCenters);
        
    faceAreas_.copy(hostFaceAreas);
        
    lowerAddress_.copy(hostLowerAddress);
        
    upperAddress_.copy(hostUpperAddress);

    //surfaceScalarField gammaMagSf_ = sf_DT * mesh.magSf();

}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
