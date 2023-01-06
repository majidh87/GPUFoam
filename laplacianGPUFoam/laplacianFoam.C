/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
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

Application
    laplacianFoam

Group
    grpBasicSolvers

Description
    Laplace equation solver for a scalar quantity.

    \heading Solver details
    The solver is applicable to, e.g. for thermal diffusion in a solid.  The
    equation is given by:

    \f[
        \ddt{T}  = \div \left( D_T \grad T \right)
    \f]

    Where:
    \vartable
        T     | Scalar field which is solved for, e.g. temperature
        D_T   | Diffusion coefficient
    \endvartable

    \heading Required fields
    \plaintable
        T     | Scalar field which is solved for, e.g. temperature
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "kernel0.h"
#include "/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h"
#include "/usr/local/cuda/targets/x86_64-linux/include/cuda.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Laplace equation solver for a scalar quantity."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

        //Added code
    #include "StopWatch.H"
    StopWatch DiscTime;
    StopWatch kernel1;
    // StopWatch kernel2;
    // StopWatch kernel3;



    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    /////////////////////////////////////////////////////////////// GPU Initialization/////////////////////////////////////////////////////
    // mesh data add MJD
    label nCells = mesh.cells().size();
    label nIFaces = mesh.faceNeighbour().size();
    label nPatches = mesh.boundaryMesh().size();
    Info<<"Total number of cells in mesh:  "<<nCells<< ", number of internal faces: "<<nIFaces<<", and number of patches: "<<nPatches<<endl;
   
    //Boundary parameters
    const polyBoundaryMesh& patches = mesh.boundaryMesh();
    int numberOfPatches=(int)mesh.boundaryMesh().size();
    int counter=0;
    
    int *patchCounter=(int*)malloc(numberOfPatches*sizeof(int));
    
    forAll(patches, patchI)
    {
        int pC_size= patches[patchI].faceCells().size();
        int i=(int)patchI;
        counter+=pC_size;
        patchCounter[i]=counter;
    }

    double *pDiag=(double*)malloc(counter*sizeof(double));
    double *pSource=(double*)malloc(counter*sizeof(double));
    int *pC_Array=(int*)malloc(counter*sizeof(double));
    int cj=0;
    surfaceScalarField gammaMagSf = -fvc::interpolate(DT) * mesh.magSf();

    forAll(patches, patchI)
    {
        fvPatchScalarField pTf = T.boundaryField()[patchI];
        scalarField patchT_d = T.boundaryField()[patchI].gradientInternalCoeffs();
        scalarField patchT_s = T.boundaryField()[patchI].gradientBoundaryCoeffs();
        
        forAll( pTf,j)
        {
            pC_Array[cj]= (int)(patches[patchI].faceCells()[j]);
            fvsPatchScalarField pGamma = gammaMagSf.boundaryField()[patchI];        
            pDiag[cj] = (double)(pGamma[j]*patchT_d[j]);
            pSource[cj] = (double)(-pGamma[j]*patchT_s[j]);
            cj++;
        }
    }


    // Input arrays
    //CPU inputs
    double *cV_C=(double*)malloc(nCells*sizeof(double));           
    double *Tot_C=(double*)malloc(nCells*sizeof(double));           
    
    double *deltaC_C=(double*)malloc(nIFaces*sizeof(double));           
    double *sf_C=(double*)malloc(nIFaces*sizeof(double));
    double *gammaMSF_C=(double*)malloc(nIFaces*sizeof(double));
    int *lAddr_C=(int*)malloc(nIFaces*sizeof(int));
    int *uAddr_C=(int*)malloc(nIFaces*sizeof(int));

    //GPU output arrays
    double *diag_G = (double*)malloc(nCells*sizeof(double));
    double *source_G = (double*)malloc(nCells*sizeof(double));
    double *lower_G = (double*)malloc(nIFaces*sizeof(double));
    double *upper_G = (double*)malloc(nIFaces*sizeof(double));
    
    //GPU memory pointer
    double *cV_G,*Tot_G; 
    double *deltaC_G, *sf_G, *gammaMSF_G;
    int *lAddr_G, *uAddr_G; 
    
    //GPU memory pointer for boundary
    int *pC_Array_G, *patchCounter_G; 
    double *pDiag_G, *pSource_G;

    //GPU memory allocation
    cudaMalloc(&Tot_G, nCells*sizeof(double));
    cudaMalloc(&cV_G, nCells*sizeof(double));
    cudaMalloc(&sf_G, nIFaces*sizeof(double));

    cudaMalloc(&deltaC_G, nIFaces*sizeof(double));           
    cudaMalloc(&gammaMSF_G, nIFaces*sizeof(double));
    cudaMalloc(&lAddr_G, nIFaces*sizeof(int));
    cudaMalloc(&uAddr_G, nIFaces*sizeof(int));

    cudaMalloc(&pC_Array_G, counter*sizeof(int));
    cudaMalloc(&patchCounter_G, numberOfPatches*sizeof(int));
    cudaMalloc(&pDiag_G, counter*sizeof(double));
    cudaMalloc(&pSource_G, counter*sizeof(double));
    
    //Copy to get double format
    for (int i=0; i<nCells; i++)
    {
        cV_C[i] = (double)(mesh.V()[i]);
    }

    for (int i=0; i<nIFaces; i++)
    {
        deltaC_C[i] = (double)(mesh.deltaCoeffs()[i]);
        sf_C[i]     = (double)(mesh.magSf()[i]);
        lAddr_C[i]  = (int)(mesh.owner()[i]);
        uAddr_C[i]  = (int)(mesh.neighbour()[i]);
    }
    
    //CPU to GPU memory copy
    cudaMemcpy(cV_G, cV_C, nCells*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(deltaC_G, deltaC_C, nIFaces*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(sf_G, sf_C, nIFaces*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(lAddr_G, lAddr_C, nIFaces*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(uAddr_G, uAddr_C, nIFaces*sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(pC_Array_G, pC_Array, counter*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(pDiag_G, pDiag, counter*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(pSource_G, pSource, counter*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(patchCounter_G, patchCounter, numberOfPatches*sizeof(int),cudaMemcpyHostToDevice);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Info<< "\nCalculating temperature distribution\n" << endl;
    
    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
               
        while (simple.correctNonOrthogonal())
        {
            DiscTime.start();
            fvScalarMatrix TEqn
            (
                fvm::ddt(T) 
                - fvm::laplacian(DT, T)
             ==
                fvOptions(T)
            );
            DiscTime.stop();


            fvScalarMatrix TEqn_d = TEqn;
            scalar rDeltaT = 1.0/mesh.time().deltaTValue();
            double rDelgaG= (double)rDeltaT;
            TEqn_d.diag() = rDeltaT*mesh.Vsc();
            TEqn_d.source() = rDeltaT*T.oldTime().primitiveField()*mesh.Vsc();
            
            surfaceScalarField sf_DT = -fvc::interpolate(DT);
            surfaceScalarField gammaMagSf_ = sf_DT * mesh.magSf();
            TEqn_d.upper() = mesh.deltaCoeffs().primitiveField()*gammaMagSf_.primitiveField();
            TEqn_d.negSumDiag();
            
            // Update T_oldTime() and copy to GPU
            for (int i=0; i<nCells; i++)
            {
                Tot_C[i]=(double)(T.oldTime().primitiveField()[i]);
            }
            cudaMemcpy(Tot_G, Tot_C, nCells*sizeof(double),cudaMemcpyHostToDevice);

            // Update gammaMSF()             
            for (int i=0; i<nIFaces; i++)
            {
                gammaMSF_C[i] = (double)(gammaMagSf_[i]);
            }
            cudaMemcpy(gammaMSF_G, gammaMSF_C, nIFaces*sizeof(double),cudaMemcpyHostToDevice);


            kernel1.start();
            // Execute GPU Kernel 
            laplasKernel( nCells,
                        nIFaces,
                        cV_G,
                        Tot_G,
                        deltaC_G,
                        gammaMSF_G,
                        uAddr_G,
                        lAddr_G,
                        patchCounter_G,
                        pC_Array_G,
                        pDiag_G,
                        pSource_G,
                        counter,
                        numberOfPatches,
                        rDelgaG,
                        diag_G,
                        source_G,
                        upper_G,
                        lower_G
                        );
            
            kernel1.stop();

            
            for(label i=0; i<TEqn.upper().size(); i++) 
            {
                scalar error = TEqn.upper()[i] - TEqn_d.upper()[i];
                if(error != 0)
                Info << "face ="<<i<<" error ="<<error<<",  ref TEqn ="<<TEqn.upper()[i]<< "  and TEqn_d ="<<TEqn_d.upper()[i]<<endl; 
            }

            Info<<"  Total number of cells in mesh: " <<endl;

            fvOptions.constrain(TEqn);
            TEqn.solve();
            fvOptions.correct(T);
        }

        #include "write.H"

        runTime.printExecutionTime(Info);
    }

     Info<<"Time Profile: "
        <<"\n\tDiscretization total time (TEqn) (s): " << DiscTime.getTotalTime()
        <<"\n\t\tKernel total time (s): " << kernel1.getTotalTime()<<endl;

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
