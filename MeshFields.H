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

#ifdef CUDA_USE
#include "cuda_error_handle.H"
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{

    class MeshFields
    {
        // Public Data
        public:    
            // Cell volumes
            HybridArray<Foam::scalar> cellVolumes;
            
            // Delta cell centers for each internal face        
            HybridArray<Foam::scalar> deltaCellCenters; 
            
            // Face surface area magnitude   
            HybridArray<Foam::scalar> faceAreas;        
            
            // Gamma multiplied by face area
            //depend DT Fields
            //HybridArray<Foam::scalar> gammaMagSf_;  

            // Lower address (owner) for each face
            HybridArray<Foam::label> lowerAddress;     
            
            // Upper address (neighbour) for each face
            HybridArray<Foam::label> upperAddress;        
    
            // Number of cells in the mesh
            //label numCells_;

            // Number of internal faces in the mesh
            //label numInternalFaces_;
        private:

            // Memory location (Host or GPU)
            bool onGPU_;

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
