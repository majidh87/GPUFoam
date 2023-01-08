
    // Mesh data
    label nCells = mesh.cells().size();
    label nIFaces = mesh.faceNeighbour().size();
    label nPatches = mesh.boundaryMesh().size();
    Info<<"Total number of cells in mesh:  "<<nCells<< ", number of internal faces: "<<nIFaces<<", and number of patches: "<<nPatches<<endl;
   
    //Boundary condition fields
    const polyBoundaryMesh& patches = mesh.boundaryMesh();
    int numberOfPatches=static_cast<int>(mesh.boundaryMesh().size());
    int counter=0;
    
    int *patchCounter=static_cast<int*>(malloc(numberOfPatches*sizeof(int)));
    
    forAll(patches, patchI)
    {
        int pC_size= patches[patchI].faceCells().size();
        int i=static_cast<int>(patchI);
        counter+=pC_size;
        patchCounter[i]=counter;
    }

    double *pDiag=static_cast<double*>(malloc(counter*sizeof(double)));
    double *pSource=static_cast<double*>(malloc(counter*sizeof(double)));
    int *pC_Array=static_cast<int*>(malloc(counter*sizeof(double)));
    int cj=0;
    surfaceScalarField gammaMagSf = -fvc::interpolate(DT) * mesh.magSf();

    forAll(patches, patchI)
    {
        fvPatchScalarField pTf = T.boundaryField()[patchI];
        scalarField patchT_d = T.boundaryField()[patchI].gradientInternalCoeffs();
        scalarField patchT_s = T.boundaryField()[patchI].gradientBoundaryCoeffs();
        forAll( pTf,j)
        {
	        pC_Array[cj]= static_cast<int>(patches[patchI].faceCells()[j]);
            fvsPatchScalarField pGamma = gammaMagSf.boundaryField()[patchI];        
            pDiag[cj] = static_cast<double>((pGamma[j]*patchT_d[j]));
            pSource[cj] = static_cast<double>((-pGamma[j]*patchT_s[j]));
            cj++;
        }
    }

    // Input arrays
    //CPU inputs
    double *cV_C=static_cast<double*>(malloc(nCells*sizeof(double)));           
    double *Tot_C=static_cast<double*>(malloc(nCells*sizeof(double)));           
    
    double *deltaC_C=static_cast<double*>(malloc(nIFaces*sizeof(double)));           
    double *sf_C=static_cast<double*>(malloc(nIFaces*sizeof(double)));
    double *gammaMSF_C=static_cast<double*>(malloc(nIFaces*sizeof(double)));
    
    int *lAddr_C=static_cast<int*>(malloc(nIFaces*sizeof(int)));
    int *uAddr_C=static_cast<int*>(malloc(nIFaces*sizeof(int)));

    //GPU output arrays
    double *diag_G   = static_cast<double*>(malloc(nCells*sizeof(double)));
    double *source_G = static_cast<double*>(malloc(nCells*sizeof(double)));
    double *lower_G  = static_cast<double*>(malloc(nIFaces*sizeof(double)));
    double *upper_G  = static_cast<double*>(malloc(nIFaces*sizeof(double)));
    
    //GPU memory pointer
    double *cV_G,*Tot_G; 
    double *deltaC_G, *sf_G, *gammaMSF_G;
    int *lAddr_G, *uAddr_G; 
    
    //GPU memory pointer for boundary
    int *pC_Array_G, *patchCounter_G; 
    double *pDiag_G, *pSource_G;

    //GPU memory allocation
    cudaMalloc(reinterpret_cast<void**>(&Tot_G), nCells*sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&cV_G),  nCells*sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&sf_G),  nIFaces*sizeof(double));

    cudaMalloc(reinterpret_cast<void**>(&deltaC_G),  nIFaces*sizeof(double));           
    cudaMalloc(reinterpret_cast<void**>(&gammaMSF_G),nIFaces*sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&lAddr_G),   nIFaces*sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&uAddr_G),   nIFaces*sizeof(int));

    cudaMalloc(reinterpret_cast<void**>(&pC_Array_G),   counter*sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&patchCounter_G), numberOfPatches*sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&pDiag_G),      counter*sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&pSource_G),    counter*sizeof(double));
    
    const scalarList* listPtr = &mesh.V();
    scalarList* blist = const_cast<scalarList*>(listPtr);
    cV_C = &blist->first();

    const scalarList* deltaCPtr = &mesh.deltaCoeffs();
    const scalarList* magSfPtr = &mesh.magSf();
    const labelUList* ownerPtr = &mesh.owner();
    const labelUList* neighbourPtr = &mesh.neighbour();

    scalarList* deltaClist =  const_cast<scalarList*>(deltaCPtr);
    scalarList* magSflist =  const_cast<scalarList*>(magSfPtr);
    labelUList* ownerlist =  const_cast<labelUList*>(ownerPtr);
    labelUList* neighbourlist =  const_cast<labelUList*>(neighbourPtr);
    
    deltaC_C = &deltaClist->first();
    sf_C = &magSflist->first();
    lAddr_C = &ownerlist->first();
    uAddr_C = &neighbourlist->first();
    
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