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
    numCells(0),
    numInternalFaces(0),
    maxPatchSize(0),
    numPatches(0),
    onGPU_(false),
    isInitialized_(false)
{}

MeshFields::MeshFields(const bool onGPU)
:
    numCells(0),
    numInternalFaces(0),
    maxPatchSize(0),
    numPatches(0),
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

    cellVolumes.deallocate();
        
        
    
    deltaCellCenters.deallocate();

    nonOrthdeltaCellCenters.deallocate();
    
    faceAreas.deallocate();

    faceWeights.deallocate();

    faceAreaVector_x.deallocate();
    faceAreaVector_y.deallocate();
    faceAreaVector_z.deallocate();
    
    
    lowerAddress.deallocate();
    upperAddress.deallocate();

    
    
    
    
    hostPatchSizes.deallocate();

    hostPatchAddr.deallocate();

    hostPatchMagSf.deallocate();

    
    devicePatchSizes.deallocate();
       
    devicePatchAddr.deallocate();
  
 
    devicePatchMagSf.deallocate();

}

void MeshFields::init(fvMesh& mesh,const bool onGPU )
{
    onGPU_ = onGPU;

    const polyBoundaryMesh& patches = mesh.boundaryMesh();
    
    numCells          = static_cast<Foam::label>(mesh.cells().size());
    numInternalFaces  = static_cast<Foam::label>(mesh.faceNeighbour().size());
    numPatches        = static_cast<Foam::label>(patches.size());


    Foam::Switch st = true;
    if(!onGPU_)
    {
        st = false;
    }
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCellVolumes_)  ,numCells_*sizeof(double)));
    cellVolumes.allocate(numCells,st);
        
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceFaceAreas_)  ,numCells_*sizeof(double))); ???
    faceAreas.allocate(numInternalFaces,st);

    faceWeights.allocate(numInternalFaces,st);

    faceAreaVector_x.allocate(numInternalFaces,st);
    faceAreaVector_y.allocate(numInternalFaces,st);
    faceAreaVector_z.allocate(numInternalFaces,st);

    corrVector_x.allocate(numInternalFaces,st);
    corrVector_y.allocate(numInternalFaces,st);
    corrVector_z.allocate(numInternalFaces,st);

    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceDeltaCellCenters_)  ,numInternalFaces_*sizeof(double)));
	deltaCellCenters.allocate(numInternalFaces,st);

    nonOrthdeltaCellCenters.allocate(numInternalFaces,st);


    //gammaMagSf_.allocate(numInternalFaces,STORE_ON_GPU);           
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceGammaMagSf_),numInternalFaces_*sizeof(double)));
    
    lowerAddress.allocate(numInternalFaces,st);
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceLowerAddress_)   ,numInternalFaces_*sizeof(int)));
    
    upperAddress.allocate(numInternalFaces,st);
    //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceUpperAddress_)   ,numInternalFaces_*sizeof(int)));

    hostPatchSizes.allocate(numPatches,false);
        
    hostPatchAddr.allocate(numPatches,false);
    
    
    hostPatchMagSf.allocate(numPatches,false);

    devicePatchSizes.allocate(numPatches,true); 

    isInitialized_ = true;            

}

void MeshFields::handle(fvMesh& mesh)
{
    if (!isInitialized_)
    {
        init(mesh); 
        /*
        FatalErrorInFunction
            << "ERROR: handle is called before init "
            << endl
            << abort(FatalError);
            return;
        */
    }

    // Mesh data
    const scalarList* listPtr       = &mesh.V();
    const scalarList* deltaCPtr     = &mesh.deltaCoeffs();
    const scalarList* nonOrthDeltaCPtr = &mesh.nonOrthDeltaCoeffs();
    const scalarList* magSfPtr      = &mesh.magSf();
    const labelUList* ownerPtr      = &mesh.owner();
    const labelUList* neighbourPtr  = &mesh.neighbour();

    scalarList* blist           = const_cast<scalarList*>(listPtr);
    scalarList* deltaClist      = const_cast<scalarList*>(deltaCPtr);
    scalarList* nonOrthDeltaClist = const_cast<scalarList*>(nonOrthDeltaCPtr);
    scalarList* magSflist       = const_cast<scalarList*>(magSfPtr);
    labelUList* ownerlist       = const_cast<labelUList*>(ownerPtr);
    labelUList* neighbourlist   = const_cast<labelUList*>(neighbourPtr);
    
    scalar *hostCellVolumes = &blist->first();             // Host: Cell volumes
    scalar *hostDeltaCellCenters = &deltaClist->first();   // Host: Delta cell centers for each internal face
    scalar *hostNonOrthDeltaCellCenters = &nonOrthDeltaClist->first(); // Host: Non-orthogonal delta cell centers for each internal face
    scalar *hostFaceAreas = &magSflist->first();           // Host: Lower address (owner) for each face
    label *hostLowerAddress = &ownerlist->first();           // Host: Upper address (neighbour) for each face
    label *hostUpperAddress = &neighbourlist->first();       // Host: Face surface area magnitude
		
    cellVolumes.copy(hostCellVolumes);
        
    deltaCellCenters.copy(hostDeltaCellCenters);
        
    nonOrthdeltaCellCenters.copy(hostNonOrthDeltaCellCenters);

    faceAreas.copy(hostFaceAreas);
        
    lowerAddress.copy(hostLowerAddress);
        
    upperAddress.copy(hostUpperAddress);

    
    // calculate face weights --- 

    scalarField weights(mesh.nInternalFaces());
    const vectorField& C = mesh.C();
    const vectorField& Cf = mesh.Cf();
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    forAll(mesh.faceNeighbour(), faceI)
    {

        const label own = owner[faceI];
        const label nei = neighbour[faceI];

        const vector& Ci = C[own];        // Owner cell center
        const vector& Cj = C[nei];        // Neighbour cell center
        const vector& Cf_i = Cf[faceI];   // Face center

        const scalar distToOwner = mag(Cf_i - Ci);
        const scalar distBetween = mag(Cj - Ci);

            if (distBetween > SMALL)  // avoid divide by zero
            {
                weights[faceI] = distToOwner / distBetween;
            }
            else
            {
                weights[faceI] = 0.5; // fallback (orthogonal)
            }
         //   printf("Face %d: Owner %d, Neighbour %d, Weight: %f\n", faceI, own, nei, weights[faceI]);
    }
    const scalarList* faceWeightsPtr       = &weights;
    scalarList* faceWeights_list       = const_cast<scalarList*>(faceWeightsPtr);
    scalar *hostFaceWeight = &faceWeights_list->first();
    faceWeights.copy(hostFaceWeight);

    // face area vectors for each internal face --- 
     const surfaceVectorField & Sf =  mesh.Sf();
    List<scalar> Sf_x(numInternalFaces);
    List<scalar> Sf_y(numInternalFaces);
    List<scalar> Sf_z(numInternalFaces);

    for (label i = 0; i < numInternalFaces; ++i)
    {
        const vector& faceVec = Sf[i];
        Sf_x[i] = faceVec.x();
        Sf_y[i] = faceVec.y();
        Sf_z[i] = faceVec.z();
    }

    const scalarList* Sf_xPtr       = &Sf_x;
    const scalarList* Sf_yPtr       = &Sf_y;
    const scalarList* Sf_zPtr       = &Sf_z;

    scalarList* faceAreaVector_xList       = const_cast<scalarList*>(Sf_xPtr);
    scalarList* faceAreaVector_yList       = const_cast<scalarList*>(Sf_yPtr);
    scalarList* faceAreaVector_zList       = const_cast<scalarList*>(Sf_zPtr);

    scalar *hostFaceAreaVector_x = &faceAreaVector_xList->first();
    scalar *hostFaceAreaVector_y = &faceAreaVector_yList->first();
    scalar *hostFaceAreaVector_z = &faceAreaVector_zList->first();

    faceAreaVector_x.copy(hostFaceAreaVector_x);
    faceAreaVector_y.copy(hostFaceAreaVector_y);
    faceAreaVector_z.copy(hostFaceAreaVector_z);

    // for (label i = 0; i < numInternalFaces; ++i){
    // printf("hostFaceAreaVector_x[%d]: %f\n", i, hostFaceAreaVector_x[i]);
    // }

    // correction vector 
    surfaceVectorField Sn(mesh.Sf() / mesh.magSf());

    surfaceVectorField corrVecDummy = mesh.delta() - (mesh.delta() & Sn) * Sn; // dummy 

    const surfaceScalarField deltaCoeff = mesh.nonOrthDeltaCoeffs();

    surfaceVectorField corrVec = corrVecDummy; 
    forAll(corrVec, faceI)
    {
        vector delta(C[neighbour[faceI]] - C[owner[faceI]]);
        corrVec[faceI] = Sn[faceI] - delta * deltaCoeff[faceI]; 
    }

    List<scalar> corrVec_x(numInternalFaces);
    List<scalar> corrVec_y(numInternalFaces);
    List<scalar> corrVec_z(numInternalFaces);

    for (label i = 0; i < numInternalFaces; ++i)
    {
        const vector& faceVec = corrVec[i];
        corrVec_x[i] = faceVec.x();
        corrVec_y[i] = faceVec.y();
        corrVec_z[i] = faceVec.z();
    }

    const scalarList* corrVec_xPtr       = &corrVec_x;
    const scalarList* corrVec_yPtr       = &corrVec_y;
    const scalarList* corrVec_zPtr       = &corrVec_z;

    scalarList* corrVec_xList       = const_cast<scalarList*>(corrVec_xPtr);
    scalarList* corrVec_yList       = const_cast<scalarList*>(corrVec_yPtr);
    scalarList* corrVec_zList       = const_cast<scalarList*>(corrVec_zPtr);

    scalar *hostCorrVector_x = &corrVec_xList->first();
    scalar *hostCorrVector_y = &corrVec_yList->first();
    scalar *hostCorrVector_z = &corrVec_zList->first();


    corrVector_x.copy(hostCorrVector_x);
    corrVector_y.copy(hostCorrVector_y);
    corrVector_z.copy(hostCorrVector_z);

    // printf("HostCorrVector_x: ");
    // for (label i = 0; i < numInternalFaces; ++i)
    // {
    //     printf("%f ", hostCorrVector_x[i]);
    // }
    // printf("\n");

    // printf("HostCorrVector_y: ");
    // for (label i = 0; i < numInternalFaces; ++i)
    // {
    //     printf("%f ", hostCorrVector_y[i]);
    // }
    // printf("\n");

    // printf("HostCorrVector_z: ");
    // for (label i = 0; i < numInternalFaces; ++i)
    // {
    //     printf("%f ", hostCorrVector_z[i]);
    // }
    // printf("\n");

    // Boundary condition data ----
    const polyBoundaryMesh& patches = mesh.boundaryMesh();

    forAll(patches, patchI)
    {
        const labelUList* pfCPtr = &patches[patchI].faceCells();
        labelUList* pfClist  = const_cast<labelUList*>(pfCPtr);
        const scalarList* pfMagSFptr= &mesh.magSf().boundaryField()[patchI];
        scalarList* pfMagSFlist = const_cast<scalarList*>(pfMagSFptr);


        hostPatchAddr.set(patchI,&pfClist->first());
            
        hostPatchSizes.set(patchI,patches[patchI].faceCells().size());
            
			
        hostPatchMagSf.set(patchI,&pfMagSFlist->first());

            
        maxPatchSize = (hostPatchSizes[patchI] > maxPatchSize) ? hostPatchSizes[patchI] : maxPatchSize;
    }
    if(onGPU_)
    {
        devicePatchSizes.copy(hostPatchSizes.Data());

        //devicePatchAddr_.copy(deviceIntermediatePatchAddr_.Data());
        devicePatchAddr.copy(hostPatchAddr,hostPatchSizes);

            
        //devicePatchMagSf__.copy(deviceIntermediatePatchMagSf__.Data());
        devicePatchMagSf.copy(hostPatchMagSf,hostPatchSizes);
            
    }
    // Mesh data
    scalar rDeltaT = 1.0/mesh.time().deltaTValue();
    invDeltaT = static_cast<Foam::scalar>(rDeltaT);



    
    
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
