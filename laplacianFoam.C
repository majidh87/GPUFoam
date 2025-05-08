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
#include "gpuFields.H"
#include "MeshFields.H"
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

//    #include "discretizationKernel.h"


    simpleControl simple(mesh);


    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating temperature distribution\n" << endl;

     gpuFields g; // create an object from gpuFields

     g.init(mesh);
    
     surfaceScalarField sf_DT = -fvc::interpolate(DT); // changed New
     g.handle(mesh,sf_DT,T);


     Foam::MeshFields gpuMesh;
     gpuMesh.handle(mesh);

    // Allocate host memory to hold the diagonal, source, and upper terms - testing
    int numCells_ = mesh.cells().size();
    int numInternalFaces_ = mesh.faceNeighbour().size();
    double* h_diag = new double[numCells_];
    double* h_source = new double[numCells_];
    double* h_upper = new double[numInternalFaces_];
     
    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {

            
            g.discKernel(h_diag, h_source, h_upper);

             // Loop over and print each entry
            for (int i = 0; i < numCells_; i++) {
                Info << "GPU diag[" << i << "] = " << h_diag[i] <<" -- source["<< i<<"] = "<< h_source[i]<< endl;
            }
            // for (int i = 0; i < numInternalFaces_; i++) {
            //     Info << "GPU upper[" << i << "] = " << h_upper[i]<< endl;
            // }
    
            fvScalarMatrix TEqn
            (
                fvm::ddt(T) - fvm::laplacian(DT, T)
             ==
                fvOptions(T)
            );
            
            scalarField& diag = TEqn.diag();
            scalarField& source = TEqn.source();
            const scalarField& upper = TEqn.upper();
            const scalarField& lower = TEqn.lower();


            // // correct diag and source with boundary conditions
            // forAll(T.boundaryField(), patchi)
            // {
            //     const fvPatch& p = T.boundaryField()[patchi].patch();
            //     const labelList& faceCells = p.faceCells();
             
            //     forAll(faceCells, i)
            //     {
            //         diag[faceCells[i]] += TEqn.internalCoeffs()[patchi][i];
            //         source[faceCells[i]] += TEqn.boundaryCoeffs()[patchi][i];
            //     }
            // }

            // const auto& internalCoeffs = TEqn.internalCoeffs();
            // const auto& boundaryCoeffs = TEqn.boundaryCoeffs();

            //Loop over and print each diagonal entry: 
            for (label i = 0; i < diag.size(); i++) { 
                Info << "OpenFOAM: diag[" << i << "] = " << diag[i] <<" -- source[" << i << "] = " << source[i] << endl; 
                }

            // for (label i = 0; i < upper.size(); i++) { 
            //     Info << "OpenFOAM: upper[" << i << "] = " << upper[i] <<" -- lower[" << i << "] = " << lower[i] << endl; 
            //     }

            // for (label i = 0; i < internalCoeffs.size(); i++) { 
            //     Info << "internalCoeffs[" << i << "] = " << internalCoeffs[i] <<"boundaryCoeffs[" << i << "] = " << boundaryCoeffs[i] << endl; 
            //     }
            double errorSumDiag = 0.0;
            double errorSumSource = 0.0;
            for (label i = 0; i < numCells_; i++) {
                errorSumDiag += fabs(diag[i] - h_diag[i]);
                errorSumSource += fabs(source[i] - h_source[i]);
            }
            double avgErrorDiag = errorSumDiag / numCells_;
            double avgErrorSource = errorSumSource / numCells_;
            
            Info << "Average error Diag = " << avgErrorDiag << "  -- source = "<< avgErrorSource<< endl;
           
            // fvOptions.constrain(TEqn);
            // TEqn.solve();
            // fvOptions.correct(T);
        }

        #include "write.H"

        runTime.printExecutionTime(Info);
    }
    // Clean up host memory
    delete[] h_diag;
    delete[] h_source;
    delete[] h_upper;

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
