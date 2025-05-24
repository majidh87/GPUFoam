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
//#include "gpuFields.H"
#include "discretizationKernel.h"
#include "MeshFields.H"
#include "hybridSurfaceScalarField.H"
#include "hybridVolScalarField.H"
#include "LduMatrixFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
bool isLog = true;    

#define Logger if (isLog) Info

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

    // Device arrays for mesh-related data
    Foam::MeshFields deviceMesh;

    Foam::hybridVolScalarField deviceT;

    Foam::hybridSurfaceScalarField deviceSDT;

    // Device arrays for linear system (matrix and source terms)
    Foam::LduMatrixFields deviceLdu;

    //gpuFields g; // create an object from gpuFields

     //g.init(mesh);

    deviceMesh.handle(mesh);
    
    surfaceScalarField sf_DT = -fvc::interpolate(DT); // changed New
    //g.handle(mesh,sf_DT,T);
    deviceT.handle(mesh,T);
    
    deviceSDT.handle(mesh,sf_DT);
    
    deviceLdu.init(deviceMesh.numCells,deviceMesh.numInternalFaces,true);

    

    // Allocate host memory to hold the diagonal, source, and upper terms - testing
    int numCells_ = mesh.cells().size();
    int numInternalFaces_ = mesh.faceNeighbour().size();
    HybridArray<scalar> h_diag(numCells_,false);
    HybridArray<scalar> h_source(numCells_,false);
    HybridArray<scalar> h_upper(numInternalFaces_,false);
    HybridArray<scalar> h_lower(numInternalFaces_,false);
    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {

            printf("discKernel before\n");
            //g.discKernel();
            printf("deviceMesh.invDeltaT %f" , deviceMesh.invDeltaT);

            /*
            cellKernelWrapper(
                deviceMesh.numCells,
                deviceMesh.cellVolumes.Data(),
                deviceT.oldField.Data(),
                deviceMesh.invDeltaT,
                deviceLdu.diagonal,
                deviceLdu.source
            );


            faceKernelWrapper(
                deviceMesh.numInternalFaces,
                deviceMesh.deltaCellCenters.Data(),
                deviceMesh.faceAreas.Data(),
                deviceSDT.deviceInternalField.Data(),
                deviceMesh.upperAddress.Data(),
                deviceMesh.lowerAddress.Data(),
                deviceLdu.upper,
                deviceLdu.lower,
                deviceLdu.diagonal
            );


            boundaryKernelWrapper(
                deviceMesh.numPatches,
                deviceMesh.maxPatchSize,
                deviceMesh.devicePatchSizes.Data(),
                deviceMesh.devicePatchAddr.deviceList.Data(),
                deviceT.devicePatchBoundaryCoeffs.deviceList.Data(),
                deviceT.devicePatchInternalCoeffs.deviceList.Data(),
                deviceMesh.devicePatchMagSf.deviceList.Data(),
                deviceSDT.deviceBoundaryField.deviceList.Data(),
                deviceLdu.diagonal,
                deviceLdu.source
            );
            */
           discKernelWrapper(       
            deviceMesh.numCells,
             deviceMesh.numInternalFaces,
             deviceMesh.cellVolumes.Data(),
             deviceT.oldField.Data(),
             deviceSDT.deviceInternalField.Data(),
             deviceMesh.deltaCellCenters.Data(),
             deviceMesh.faceAreas.Data(),
             deviceMesh.upperAddress.Data(),
             deviceMesh.lowerAddress.Data(),
             deviceMesh.numPatches,
             deviceMesh.maxPatchSize,
             deviceMesh.devicePatchSizes.Data(),
             deviceMesh.devicePatchAddr.deviceList.Data(),
             deviceT.devicePatchBoundaryCoeffs.deviceList.Data(),
             deviceT.devicePatchInternalCoeffs.deviceList.Data(),
             deviceMesh.devicePatchMagSf.deviceList.Data(),
             deviceSDT.deviceBoundaryField.deviceList.Data(),
             deviceMesh.invDeltaT,
             deviceLdu.diagonal,
             deviceLdu.source,
             deviceLdu.upper,
             deviceLdu.lower
             );

            Info<< "discKernel run is finished!"<<endl;

            // std::cout << "Press any key to continue...";
            // std::cin.get();  // Waits for a character input including Enter

	        h_diag.copy(deviceLdu.diagonal,true);
            h_source.copy(deviceLdu.source,true);
            h_upper.copy(deviceLdu.upper,true);
            h_lower.copy(deviceLdu.lower,true);
	    
            // std::cout << "Press any key to continue...";
            // std::cin.get();  // Waits for a character input including Enter

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


            // correct diag and source with boundary conditions
            forAll(T.boundaryField(), patchi)
            {
                const fvPatch& p = T.boundaryField()[patchi].patch();
                const labelList& faceCells = p.faceCells();
             
                forAll(faceCells, i)
                {
                    diag[faceCells[i]] += TEqn.internalCoeffs()[patchi][i];
                    source[faceCells[i]] += TEqn.boundaryCoeffs()[patchi][i];
                }
            }

            // const auto& internalCoeffs = TEqn.internalCoeffs();
            // const auto& boundaryCoeffs = TEqn.boundaryCoeffs();

            //Loop over and print each diagonal entry: 
            
	        // for (label i = 0; i < diag.size(); i++) { 
            //     Info << "OpenFOAM: diag[" << i << "] = " << diag[i] <<" -- source[" << i << "] = " << source[i] << endl; 
            //     }
	     
            // for (label i = 0; i < upper.size(); i++) { 
            //     Info << "OpenFOAM: upper[" << i << "] = " << upper[i] <<" -- lower[" << i << "] = " << lower[i] << endl; 
            //     }

            // for (label i = 0; i < internalCoeffs.size(); i++) { 
            //     Info << "internalCoeffs[" << i << "] = " << internalCoeffs[i] <<"boundaryCoeffs[" << i << "] = " << boundaryCoeffs[i] << endl; 
            //     }

            for (label i = 0; i < numCells_; i++) {
                scalar diff = fabs(diag[i] - h_diag[i]);
                scalar diff2 = fabs(source[i] - h_source[i]);
                if (diff > 1e-6) {
                    Logger << "diag error = OpenFOAM: diag[" << i << "] = " << diag[i] <<" -- GPU diag[" << i << "] = " << h_diag[i] << endl;
                }
                if (diff2 > 1e-6) {
                    Logger << "source error = OpenFOAM: source[" << i << "] = " << source[i] <<" -- GPU source[" << i << "] = " << h_source[i] << endl;
                }
            }

            for (label i = 0; i < numInternalFaces_; i++) {
                scalar diff = fabs(upper[i] - h_upper[i]);
                scalar diff2 = fabs(lower[i] - h_lower[i]);
                if (diff > 1e-6) {
                    Logger << "upper error = OpenFOAM: upper[" << i << "] = " << upper[i] <<" -- GPU upper[" << i << "] = " << h_upper[i] << endl;
                } 
                if (diff2 > 1e-6) {
                    Logger << "lower error = OpenFOAM: lower[" << i << "] = " << lower[i] <<" -- GPU lower[" << i << "] = " << h_lower[i] << endl;
                }
            }

            double errorSumDiag = 0.0;
            double errorSumSource = 0.0;
            double errorSumlower = 0.0;
            double errorSumupper = 0.0;
            
            for (label i = 0; i < numCells_; i++) {
                errorSumDiag += fabs(diag[i] - h_diag[i]);
                errorSumSource += fabs(source[i] - h_source[i]);
            }
            for (label i = 0; i < numInternalFaces_; i++) {
                errorSumlower += fabs(lower[i] - h_lower[i]);
                errorSumupper += fabs(upper[i] - h_upper[i]);
            }
            double avgErrorDiag = errorSumDiag / numCells_;
            double avgErrorSource = errorSumSource / numCells_;
            
            Logger << "Average error Diag = " << avgErrorDiag << "  -- source = "<< avgErrorSource<< endl;
            Logger << "Average error lower = " << errorSumlower / numInternalFaces_ << "  -- upper = "<< errorSumupper / numInternalFaces_<< endl;
           
            // fvOptions.constrain(TEqn);
            // TEqn.solve();
            // fvOptions.correct(T);
        }

        #include "write.H"

        runTime.printExecutionTime(Info);
    }
    // Clean up host memory
    h_diag.deallocate();
    h_source.deallocate();
    h_upper.deallocate();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
