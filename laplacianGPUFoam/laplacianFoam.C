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
#include "error.H"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <amgx_c.h>
#include <AmgXSolver.H>



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
    #include "discretizationKernel.h"
    #include "gpuFields.H"

    //Add clocks to profile
    #include "StopWatch.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    gpuFields g;
    AmgXSolver AmgSol;
    AmgXCSRMatrix Amat;
    MPI_Init(NULL, NULL);
    const MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;
    const std::string modeStr = "dDDI";
    const std::string cfgFile = "./system/config";

    g.init(mesh);
    g.handle(mesh,DT,T);
    Amat. initialiseComms(
    amgx_mpi_comm,
    0);

    AmgSol.initialize
        (
            amgx_mpi_comm,
            modeStr,
            cfgFile
        );
    
    bool firstIter = true;


    Info<< "\nCalculating temperature distribution\n" << endl;
    
    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
               
        while (simple.correctNonOrthogonal())
        {
            

            if (firstIter)
            {
                Info<< "discritization " <<endl;
                g.discKernel();
                Amat.setValuesLDU
                (
                g.nCells,
                g.nIFaces,
                0,
                0,
                0,
                g.d_uAddr,
                g.d_lAddr,
                0,
                NULL,
                NULL,
                g.d_diag,
                g.d_upper,
                g.d_lower,
                NULL
                ); 

            Info<< "amgx.setup " <<endl;
                AmgSol.setOperator
                (
                g.nCells,
                g.nCells,
                g.nCells+2*g.nIFaces,
                Amat
                );
                firstIter = false;
            }
            else
            {
                g.update();
                g.updateDisc();
                Amat.updateValues(  g.nCells,
                            g.nIFaces,
                            0,
                            g.d_diag,
                            g.d_upper,
                            g.d_lower,
                            NULL
                            );
                AmgSol.updateOperator(  g.nCells,
                                g.nCells+2*g.nIFaces,
                                Amat
                            );
            }
            Info<< "amgx.solve " <<endl;

            AmgSol.solve
            (
                g.nCells,
                &g.h_T_new[0],
                &g.d_source[0],
                Amat
            );
        
            /*for(label i=0; i<10; i++) 
            {  
                Info <<"REsult:"<<g.h_T_new[i]<<endl; 
            }*/
        }
        

        runTime.printExecutionTime(Info);
    }
    
    //MPI_Finalize();
    Info<< "End\n" << endl;

    return 0;
}
