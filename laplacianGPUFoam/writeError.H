

    h_diag.copy(deviceLdu.diagonal,true);
    h_source.copy(deviceLdu.source,true);
    h_upper.copy(deviceLdu.upper,true);
    h_lower.copy(deviceLdu.lower,true);

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
            int numCells = mesh.cells().size();
            int numInternalFaces = mesh.faceNeighbour().size();
            for (label i = 0; i < numCells; i++) {
                scalar diff = fabs(diag[i] - h_diag[i]);
                scalar diff2 = fabs(source[i] - h_source[i]);
                if (diff > 1e-6) {
                    Logger << "diag error ="<< diff <<" OpenFOAM: diag[" << i << "] = " << diag[i] <<" -- GPU diag[" << i << "] = " << h_diag[i] << endl;
                }
                if (diff2 > 1e-6) {
                    Logger << "source error ="<< diff2 <<" OpenFOAM: source[" << i << "] = " << source[i] <<" -- GPU source[" << i << "] = " << h_source[i] << endl;
                }
            }

            for (label i = 0; i < numInternalFaces; i++) {
                scalar diff = fabs(upper[i] - h_upper[i]);
                scalar diff2 = fabs(lower[i] - h_lower[i]);
                if (diff > 1e-6) {
                    Logger << "upper error ="<< diff <<" OpenFOAM: upper[" << i << "] = " << upper[i] <<" -- GPU upper[" << i << "] = " << h_upper[i] << endl;
                } 
                if (diff2 > 1e-6) {
                    Logger << "lower error ="<< diff2 <<" OpenFOAM: lower[" << i << "] = " << lower[i] <<" -- GPU lower[" << i << "] = " << h_lower[i] << endl;
                }
            }

            double errorSumDiag = 0.0;
            double errorSumSource = 0.0;
            double errorSumlower = 0.0;
            double errorSumupper = 0.0;
            
            for (label i = 0; i < numCells; i++) {
                errorSumDiag += fabs(diag[i] - h_diag[i]);
                errorSumSource += fabs(source[i] - h_source[i]);
            }
            for (label i = 0; i < numInternalFaces; i++) {
                errorSumlower += fabs(lower[i] - h_lower[i]);
                errorSumupper += fabs(upper[i] - h_upper[i]);
            }
            double avgErrorDiag = errorSumDiag / numCells;
            double avgErrorSource = errorSumSource / numCells;

            Logger << "Average error Diag = " << avgErrorDiag << "  -- source = "<< avgErrorSource<< endl;
            Logger << "Average error lower = " << errorSumlower / numInternalFaces << "  -- upper = "<< errorSumupper / numInternalFaces<< endl;

    fvOptions.constrain(TEqn);
    TEqn.solve();
    fvOptions.correct(T);

    // forAll(T, celli){
    //     Info << "T[" << celli << "] = " << T[celli] << endl;
    // }