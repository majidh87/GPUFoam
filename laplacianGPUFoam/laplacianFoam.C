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
#include <cuda_runtime_api.h>
#include <cuda.h>
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

    //Add clocks to profile
    #include "StopWatch.H"
    StopWatch DiscTime;
    StopWatch kernel1;

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    /////// GPU Initialization /////////
    #include "createGPUFields.H"

    Info<< "\nCalculating temperature distribution\n" << endl;
    
    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
               
        while (simple.correctNonOrthogonal())
        {
            DiscTime.start(); // OF discretization on CPU
            fvScalarMatrix TEqn
            (
                fvm::ddt(T) 
                - fvm::laplacian(DT, T)
             ==
                fvOptions(T)
            );
            DiscTime.stop();
          
            
            scalar rDeltaT = 1.0/mesh.time().deltaTValue();
            double rDelgaG= static_cast<double>(rDeltaT);
            surfaceScalarField sf_DT = -fvc::interpolate(DT);
            surfaceScalarField gammaMagSf_ = sf_DT * mesh.magSf();
            
            // Update Told to device
            const scalarList* ToldPtr = &T.oldTime().primitiveField();
            scalarList* Toldlist = const_cast<scalarList*>(ToldPtr);
            Tot_C = &Toldlist->first();
            cudaMemcpy(Tot_G, Tot_C, nCells*sizeof(double),cudaMemcpyHostToDevice);

            // Update gammaMSF() to device
            const scalarList* gMgaSfPtr = &gammaMagSf_;
            scalarList* TgMagSflist = const_cast<scalarList*>(gMgaSfPtr);
            gammaMSF_C = &TgMagSflist->first();
            cudaMemcpy(gammaMSF_G, gammaMSF_C, nIFaces*sizeof(double),cudaMemcpyHostToDevice);


            
            // Execute GPU Kernel 
            kernel1.start();
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

            
            for(label i=0; i<TEqn.diag().size(); i++) 
            {
                scalar error = TEqn.diag()[i] - diag_G[i];
                if(error != 0)
                Info << "face ="<<i<<" error ="<<error<<",  ref TEqn ="<<TEqn.diag()[i]<< "  and TEqn_d ="<<diag_G[i]<<endl; 
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
