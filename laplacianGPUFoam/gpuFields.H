#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"



#define checkCudaErrors(call)                                       \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#define _CUDA(x) checkCudaErrors(x)


class gpuFields {       
  public:    
    int nCells; 
    int nIFaces;

    double rDelgaG;

    int maxPatches;
    int numberOfPatches;
    int *h_pSize;
    int **h_pAdrr;
    double **h_pf_BC;
    double **h_pf_IC;
    double **h_pf_GammaSf;
        

    int* d_pSize;
    int ** d_pAdrr;
    int ** dInh_pAdrr;
    double **d_pf_BC;
    double **dInh_pf_BC;
    double **d_pf_IC;
    double **dInh_pf_IC;
    double **d_pf_GammaSf;
    double **dInh_pf_GammaSf;


    double *h_cV; ;          
    double *h_Tot;            
    
    double *h_deltaC;
    double *h_sf; 
    double *h_gammaMSF; 
    
    int *h_lAddr; 
    int *h_uAddr;

    double *d_cV;
    double *d_Tot; 
    double *d_deltaC;
    double *d_sf;
    double *d_gammaMSF;
    int *d_lAddr;
    int *d_uAddr; 

    double *d_diag;
    double *d_source;
    double *d_lower;
    double *d_upper;

    int *d_csr_rPtr;
    int *d_csr_col;
	double *d_csr_val;

    bool initDone;


    //Constructor:
    gpuFields():
    nCells(0),
    nIFaces(0),
    maxPatches(0),
    numberOfPatches(0)
    {
        initDone = false;
    }
    ~gpuFields()
    {
        reset();
    }

    void init(fvMesh& mesh)
    {
        const polyBoundaryMesh& patches = mesh.boundaryMesh();
    
        nCells          = static_cast<int>(mesh.cells().size());
        nIFaces         = static_cast<int>(mesh.faceNeighbour().size());
        numberOfPatches = static_cast<int>(patches.size());

        h_pSize         = static_cast<int*>(malloc(numberOfPatches*sizeof(int)));
        h_pAdrr         = static_cast<int**>(malloc(numberOfPatches*sizeof(int*)));
        h_pf_BC         = static_cast<double**>(malloc(numberOfPatches*sizeof(double*)));
        h_pf_IC         = static_cast<double**>(malloc(numberOfPatches*sizeof(double*)));
        h_pf_GammaSf    = static_cast<double**>(malloc(numberOfPatches*sizeof(double*)));

        dInh_pAdrr      = static_cast<int**>(malloc(sizeof(int*)*numberOfPatches));
        dInh_pf_BC      = static_cast<double**>(malloc(sizeof(double*)*numberOfPatches));
        dInh_pf_IC      = static_cast<double**>(malloc(sizeof(double*)*numberOfPatches));
        dInh_pf_GammaSf = static_cast<double**>(malloc(sizeof(double*)*numberOfPatches));

        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_pSize)       ,numberOfPatches*sizeof(int)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_pAdrr)       ,numberOfPatches*sizeof(int*)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_pf_BC)       ,numberOfPatches*sizeof(double*)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_pf_IC)       ,numberOfPatches*sizeof(double*)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_pf_GammaSf)  ,numberOfPatches*sizeof(double*)));
             
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_diag)    ,nCells*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_source)  ,nCells*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_lower)   ,nIFaces*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_upper)   ,nIFaces*sizeof(double)));
    
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_Tot) ,nCells*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_cV)  ,nCells*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_sf)  ,nIFaces*sizeof(double)));

        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_deltaC)  ,nIFaces*sizeof(double)));           
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_gammaMSF),nIFaces*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_lAddr)   ,nIFaces*sizeof(int)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_uAddr)   ,nIFaces*sizeof(int)));

        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_csr_rPtr),(nCells+1)*sizeof(int)));
	    _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_csr_col) ,(nCells+2*nIFaces)*sizeof(int)));
	    _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_csr_val) ,(nCells+2*nIFaces)*sizeof(double)));
        _CUDA(cudaMemset(d_csr_rPtr, 0, (nCells+1)*sizeof(int)));
    
    }
    void handle(fvMesh& mesh,
                volScalarField& DT,
                volScalarField& T)
    {
         const polyBoundaryMesh& patches = mesh.boundaryMesh();
        surfaceScalarField gammaMagSf = -fvc::interpolate(DT) * mesh.magSf();
        forAll(patches, patchI)
        {
            const labelUList* pfCPtr = &patches[patchI].faceCells();
            labelUList* pfClist  = const_cast<labelUList*>(pfCPtr);
            scalarField pF_BC_SF = T.boundaryField()[patchI].gradientBoundaryCoeffs();
            scalarField pF_IC_SF = T.boundaryField()[patchI].gradientInternalCoeffs();
            const scalarList* pfGammaSFptr= &gammaMagSf.boundaryField()[patchI];
            scalarList* pfGammaSFlist = const_cast<scalarList*>(pfGammaSFptr);

            h_pAdrr[patchI]         = &pfClist->first();
            h_pSize[patchI]         = patches[patchI].faceCells().size();
            //h_pf_BC[patchI]         = &pF_BC_SF.first();
            //h_pf_IC[patchI]         = &pF_IC_SF.first();
            h_pf_IC[patchI]  = static_cast<double*>(malloc(h_pSize[patchI]*sizeof(double)));  ////////////////////
            h_pf_BC[patchI]  = static_cast<double*>(malloc(h_pSize[patchI]*sizeof(double)));  ////////////////////
            for (int i=0 ;i<h_pSize[patchI];i++)                                              ////////// TO    ///
            {                                                                                 ////////// BE    ///
                h_pf_IC[patchI][i] = pF_IC_SF[i];                                             ////////// DONE! ///
                h_pf_BC[patchI][i] = pF_BC_SF[i];                                             ////////////////////
            }                                                                                 ////////////////////
            h_pf_GammaSf[patchI]    = &pfGammaSFlist->first();
            maxPatches=(h_pSize[patchI]>maxPatches)?h_pSize[patchI]:maxPatches;
        }

        for(int i=0; i<numberOfPatches; i++)
        {
            _CUDA(cudaMalloc(reinterpret_cast<void**>(&dInh_pAdrr[i])     ,h_pSize[i]*sizeof(int)));
            _CUDA(cudaMalloc(reinterpret_cast<void**>(&dInh_pf_BC[i])     ,h_pSize[i]*sizeof(double)));
            _CUDA(cudaMalloc(reinterpret_cast<void**>(&dInh_pf_IC[i])     ,h_pSize[i]*sizeof(double)));
            _CUDA(cudaMalloc(reinterpret_cast<void**>(&dInh_pf_GammaSf[i]) ,h_pSize[i]*sizeof(double)));
        }

        _CUDA(cudaMemcpy(d_pSize      ,h_pSize, sizeof(int)*numberOfPatches, cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_pAdrr      ,dInh_pAdrr, sizeof(int*)*numberOfPatches, cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_pf_BC      ,dInh_pf_BC, sizeof(double*)*numberOfPatches, cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_pf_IC      ,dInh_pf_IC, sizeof(double*)*numberOfPatches, cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_pf_GammaSf ,dInh_pf_GammaSf, sizeof(double*)*numberOfPatches, cudaMemcpyHostToDevice));

        for( int i=0; i<numberOfPatches; i++ )
        {
            _CUDA(cudaMemcpy(dInh_pAdrr[i]        ,h_pAdrr[i], h_pSize[i]*sizeof(int), cudaMemcpyHostToDevice));
            _CUDA(cudaMemcpy(dInh_pf_BC[i]        ,h_pf_BC[i], h_pSize[i]*sizeof(double), cudaMemcpyHostToDevice));
            _CUDA(cudaMemcpy(dInh_pf_IC[i]        ,h_pf_IC[i], h_pSize[i]*sizeof(double), cudaMemcpyHostToDevice));
            _CUDA(cudaMemcpy(dInh_pf_GammaSf[i]   ,h_pf_GammaSf[i], h_pSize[i]*sizeof(double), cudaMemcpyHostToDevice));
        }
        
        const scalarList* listPtr       = &mesh.V();
        const scalarList* deltaCPtr     = &mesh.deltaCoeffs();
        const scalarList* magSfPtr      = &mesh.magSf();
        const labelUList* ownerPtr      = &mesh.owner();
        const labelUList* neighbourPtr  = &mesh.neighbour();

        scalarList* blist           = const_cast<scalarList*>(listPtr);
        scalarList* deltaClist      = const_cast<scalarList*>(deltaCPtr);
        scalarList* magSflist       = const_cast<scalarList*>(magSfPtr);
        labelUList* ownerlist       = const_cast<labelUList*>(ownerPtr);
        labelUList* neighbourlist   = const_cast<labelUList*>(neighbourPtr);
    
        h_cV        = &blist->first();
        h_deltaC    = &deltaClist->first();
        h_sf        = &magSflist->first();
        h_lAddr     = &ownerlist->first();
        h_uAddr     = &neighbourlist->first();

        _CUDA(cudaMemcpy(d_cV,        h_cV,       nCells*sizeof(double),cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_deltaC,    h_deltaC,   nIFaces*sizeof(double),cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_sf,        h_sf,       nIFaces*sizeof(double),cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_lAddr,     h_lAddr,    nIFaces*sizeof(int),cudaMemcpyHostToDevice));
        _CUDA(cudaMemcpy(d_uAddr,     h_uAddr,    nIFaces*sizeof(int),cudaMemcpyHostToDevice));
    }
    void update(fvMesh& mesh,
                volScalarField& DT,
                volScalarField& T)
    {
        scalar rDeltaT = 1.0/mesh.time().deltaTValue();
        rDelgaG= static_cast<double>(rDeltaT);
        surfaceScalarField sf_DT = -fvc::interpolate(DT);
        surfaceScalarField gammaMagSf_ = sf_DT * mesh.magSf();
            
        // Update Told to device
        const scalarList* ToldPtr = &T.oldTime().primitiveField();
        scalarList* Toldlist = const_cast<scalarList*>(ToldPtr);
        h_Tot = &Toldlist->first();
        _CUDA(cudaMemcpy(d_Tot, h_Tot, nCells*sizeof(double),cudaMemcpyHostToDevice));

        // Update gammaMSF() to device
        const scalarList* gMgaSfPtr = &gammaMagSf_;
        scalarList* TgMagSflist = const_cast<scalarList*>(gMgaSfPtr);
        h_gammaMSF = &TgMagSflist->first();
        _CUDA(cudaMemcpy(d_gammaMSF, h_gammaMSF, nIFaces*sizeof(double),cudaMemcpyHostToDevice));
    }
    void reset()
    {
        if (h_pf_BC)            free(h_pf_BC);
        if (h_pf_IC)            free(h_pf_IC);
        if (h_pf_GammaSf)       free(h_pf_GammaSf);
        if (h_pAdrr)            free(h_pAdrr);
        if (h_pSize)            free(h_pSize);
        if (dInh_pAdrr)         free(dInh_pAdrr);
        if (dInh_pf_BC)         free(dInh_pf_BC);
        if (dInh_pf_IC)         free(dInh_pf_IC);
        if (dInh_pf_GammaSf)    free(dInh_pf_GammaSf);

        if (d_pSize)            _CUDA(cudaFree(d_pSize));
        if (d_pAdrr)            _CUDA(cudaFree(d_pAdrr));
        if (d_pf_BC)            _CUDA(cudaFree(d_pf_BC));
        if (d_pf_IC)            _CUDA(cudaFree(d_pf_IC));
        if (d_pf_GammaSf)       _CUDA(cudaFree(d_pf_GammaSf));
        if (d_cV)               _CUDA(cudaFree(d_cV));
        if (d_Tot)              _CUDA(cudaFree(d_Tot));
        if (d_deltaC)           _CUDA(cudaFree(d_deltaC));
        if (d_sf)               _CUDA(cudaFree(d_sf));
        if (d_gammaMSF)         _CUDA(cudaFree(d_gammaMSF));
        if (d_lAddr)            _CUDA(cudaFree(d_lAddr));
        if (d_uAddr)            _CUDA(cudaFree(d_uAddr));
        if (d_csr_rPtr)         _CUDA(cudaFree(d_csr_rPtr));
        if (d_csr_col)          _CUDA(cudaFree(d_csr_col));
        if (d_csr_val)          _CUDA(cudaFree(d_csr_val));
    } 
    void discKernel()
    {
    
    discKernelWrapper(nCells,
                        nIFaces,
                        d_cV,
                        d_Tot,
                        d_deltaC,
                        d_gammaMSF,
                        d_uAddr,
                        d_lAddr,
                        numberOfPatches,
                        maxPatches,
                        d_pSize,
                        d_pAdrr,
                        d_pf_BC,
                        d_pf_IC,
                        d_pf_GammaSf,
                        rDelgaG,
                        d_diag,
                        d_source,
                        d_upper,
                        d_lower
                        );
    }
    void ldu2csr ()
    {
        ldu2csrWrapper (    d_diag,
                            d_lower,
                            d_upper,
                            d_lAddr,
                            d_uAddr,
                            d_csr_rPtr ,
                            d_csr_col,
                            d_csr_val,
                            nCells,
                            nIFaces
                            );
    }

};
    
   
    

   
   



    

    

   
    


   

    


    