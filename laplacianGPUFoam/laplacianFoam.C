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
#include "discretizationKernel.h"
#include "ldu2csr.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "gpuFields.H"

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
    StopWatch solveTime;
    StopWatch kernel_Disc;
    StopWatch kernel_ldu2csr;
    StopWatch kernel_amgx;
    StopWatch kernel_amgx_solve;

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    /////// GPU Initialization /////////
    
    gpuFields g;
    g.init(mesh);
    g.handle(mesh,DT,T);
    
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

            g.update(mesh,DT,T);
            
            g.discKernel();

            g.ldu2csr ();

            Info<<"  Total number of cells in mesh: " <<endl;

        }

        #include "write.H"

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}
