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
    Foam::MeshFields

Description
    A template class for dynamic array allocation of mesh fields. 

SourceFiles
    meshFields.H
\*---------------------------------------------------------------------------*/
#ifndef MESHFIELDS_H
#define MESHFIELDS_H

#include "label.H"
#include "scalar.H"
#include "HybridArray.H"
#include "HybridMatrix.H"
#include "GPUMatrix.H"
#include "HybridVectorField.H"


#include "cuda_error_handle.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{

    class MeshFields
    {
        // Public Data
        public:    
            // Number of cells in the mesh
            Foam::label numCells;          
            
            // Number of internal faces in the mesh
            Foam::label numInternalFaces;  
            
            // Cell volumes
            HybridArray<Foam::scalar> cellVolumes;
            
            // Delta cell centers for each internal face        
            HybridArray<Foam::scalar> deltaCellCenters; 

            // face weigth factor for each internal face
            HybridArray<Foam::scalar> faceWeights; 

            // Non-orthogonal delta cell centers for each internal face        
            HybridArray<Foam::scalar> nonOrthdeltaCellCenters; 
            
            // Face surface area magnitude   
            HybridArray<Foam::scalar> faceAreas;     
            
            // face area vectors for each internal face
            HybridArray<Foam::scalar> faceAreaVector_x;
            HybridArray<Foam::scalar> faceAreaVector_y;
            HybridArray<Foam::scalar> faceAreaVector_z;
            // HybridVectorField<Foam::scalar, 3> faceAreaVectors;
            
            // correction Vector for each internal face
            //HybridArray<Foam::vector> corrVec;
            HybridArray<Foam::scalar> corrVector_x;
            HybridArray<Foam::scalar> corrVector_y;
            HybridArray<Foam::scalar> corrVector_z;

            // Lower address (owner) for each face
            HybridArray<Foam::label> lowerAddress;     
            
            // Upper address (neighbour) for each face
            HybridArray<Foam::label> upperAddress;        
    
            // Maximum number of cells adjacent to any patch
            Foam::label maxPatchSize;      
            
            // Number of patches in the domain
            Foam::label numPatches;        
        
            // Host arrays for patch-related data
            // Host: Number of cells adjacent to the faces of each patch
            HybridArray<Foam::label> hostPatchSizes;        

            // Host: Indices of cells adjacent to the faces of each patch
            HybridArray<Foam::label *> hostPatchAddr;        

            
            // Host: Patch field _ Magnitude of face area
            HybridArray<Foam::scalar *> hostPatchMagSf; 
        
            // Device arrays for patch-related data
            // Device: Number of cells adjacent to the faces of each patch
            HybridArray<Foam::label> devicePatchSizes;        
            
            // Device: Indices of cells adjacent to the faces of each patch
            Foam::GPUMatrix<Foam::label> devicePatchAddr; 

            // Device: Patch field _ mag of face area
            Foam::GPUMatrix<Foam::scalar> devicePatchMagSf; 

            // Reciprocal of delta t (time step)
            Foam::scalar invDeltaT;      


        private:

            // Memory location (Host or GPU)
            Foam::Switch onGPU_;

            // Flag to check if initialization is done
            Foam::Switch isInitialized_;

    public:

        // Constructors

            //- Default constructor
            MeshFields();

            //- Construct with size and memory location
            MeshFields(const bool onGPU);

            //- Destructor
            ~MeshFields();

            // Initialize the class with buffers
            //void allocate();

            //- Deallocate memory for the buffers
            void deallocate();

            // Initialize the class with mesh data
            void init(fvMesh& mesh,const bool onGPU = true);

            // Handle function to process mesh and field data
            void handle(fvMesh& mesh);
    };
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "MeshFieldsI.H"

#endif
