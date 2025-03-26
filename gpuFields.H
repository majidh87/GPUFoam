#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
//#include <AmgXCSRMatrix.H>
#include "label.H"
#include "scalar.H"
#include "HybridArray.H"
// #define checkCudaErrors(call)                                       \
//     do {                                                            \
//         cudaError_t err = call;                                     \
//         if (err != cudaSuccess) {                                   \
//             printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
//                          cudaGetErrorString(err));                  \
//             exit(EXIT_FAILURE);                                     \
//         }                                                           \
//     } while (0)

// #define _CUDA(x) checkCudaErrors(x)

class gpuFields { 

  private:    
    int numCells_;          // Number of cells in the mesh
    int numInternalFaces_;  // Number of internal faces in the mesh
    double invDeltaT_;      // Reciprocal of delta t (time step)

    int maxPatchSize_;      // Maximum number of cells adjacent to any patch
    int numPatches_;        // Number of patches in the domain

    // Host arrays for patch-related data
    int *hostPatchSizes_;        // Host: Number of cells adjacent to the faces of each patch
    int **hostPatchAddr_;        // Host: Indices of cells adjacent to the faces of each patch
    double **hostPatchBoundaryCoeffs_;    // Host: Boundary coefficients for gradient calculation

    double **hostPatchInternalCoeffs_;    // Host: Internal coefficients for gradient calculation
    double **hostPatchGammaSf_; // Host: Patch field coefficients multiplied by face area

    // Device arrays for patch-related data
    HybridArray<Foam::label> devicePatchSizes_;        // Device: Number of cells adjacent to the faces of each patch
    int ** devicePatchAddr_;      // Device: Indices of cells adjacent to the faces of each patch
    int ** deviceIntermediatePatchAddr_;   // Intermediate host array for device data
    double **devicePatchBoundaryCoeffs_;    // Device: Boundary coefficients for gradient calculation
    double **deviceIntermediatePatchBoundaryCoeffs_; // Intermediate host array for device data
    double **devicePatchInternalCoeffs_;    // Device: Internal coefficients for gradient calculation
    double **deviceIntermediatePatchInternalCoeffs_; // Intermediate host array for device data
    double **devicePatchGammaSf_; // Device: Patch field coefficients multiplied by face area

    double **deviceIntermediatePatchGammaSf_; // Intermediate host array for device data

    double* hostOldTemperature_;	//Host : Old Temperature Time Field

    // Device arrays for mesh-related data
    double *deviceCellVolumes_;        // Device: Cell volumes
    double *deviceOldTemperature_;       // Device: Old time temperature field
    double *deviceDeltaCellCenters_;    // Device: Delta cell centers for each internal face
    double *deviceFaceAreas_;        // Device: Face surface area magnitude
    HybridArray<Foam::scalar> deviceGammaMagSf_;  // Device: Gamma multiplied by face area
    HybridArray<Foam::label> deviceLowerAddress_;        // Device: Lower address (owner) for each face
    HybridArray<Foam::label> deviceUpperAddress_;        // Device: Upper address (neighbour) for each face

    // Device arrays for linear system (matrix and source terms)
    double *deviceDiagonal_;      // Device: Diagonal elements of the matrix
    double *deviceSource_;    // Device: Source term array
    double *deviceLower_;     // Device: Lower triangular elements of the matrix
    double *deviceUpper_;     // Device: Upper triangular elements of the matrix

    // Device arrays for CSR (Compressed Sparse Row) format matrix
    int *deviceCsrRowPtr_;     // Device: Row pointers for CSR format
    int *deviceCsrColInd_;      // Device: Column indices for CSR format
    double *deviceCsrValues_;   // Device: Values for CSR format

    // Host array for new temperature field
    double* hostNewTemperature_;

    // Flag to check if initialization is done
    bool isInitialized_;
public:
    // Constructor: Initialize all pointers to NULL and set initDone to false
    gpuFields():
    numCells_(0),
    numInternalFaces_(0),
    maxPatchSize_(0),
    numPatches_(0),
    hostPatchSizes_(NULL),
    hostPatchAddr_(NULL),
    hostPatchBoundaryCoeffs_(NULL),
    hostPatchInternalCoeffs_(NULL),
    hostPatchGammaSf_(NULL),
    devicePatchAddr_(NULL),
    deviceIntermediatePatchAddr_(NULL),
    devicePatchBoundaryCoeffs_(NULL),
    deviceIntermediatePatchBoundaryCoeffs_(NULL),
    devicePatchInternalCoeffs_(NULL),
    deviceIntermediatePatchInternalCoeffs_(NULL),
    devicePatchGammaSf_(NULL),
    deviceIntermediatePatchGammaSf_(NULL),
    hostOldTemperature_(NULL),            
    deviceCellVolumes_(NULL),
    deviceOldTemperature_(NULL), 
    deviceDeltaCellCenters_(NULL),
    deviceFaceAreas_(NULL),
    deviceDiagonal_(NULL),
    deviceSource_(NULL),
    deviceLower_(NULL),
    deviceUpper_(NULL),
    deviceCsrRowPtr_(NULL),
    deviceCsrColInd_(NULL),
    deviceCsrValues_(NULL),
    hostNewTemperature_(NULL),
    isInitialized_(false)
    {
    }


	// Destructor: Free all allocated memory
    ~gpuFields()
    {
        reset();
    }
    // Initialize the class with mesh data
    void init(fvMesh& mesh)
    {
        printf("init() is called \n");
        const polyBoundaryMesh& patches = mesh.boundaryMesh();
    
        numCells_          = static_cast<int>(mesh.cells().size());
        numInternalFaces_  = static_cast<int>(mesh.faceNeighbour().size());
        numPatches_        = static_cast<int>(patches.size());
        // Allocate memory for host arrays
        hostNewTemperature_ = static_cast<double*>(malloc(numCells_*sizeof(double)));
        hostPatchSizes_     = static_cast<int*>(malloc(numPatches_*sizeof(int)));
        hostPatchAddr_      = static_cast<int**>(malloc(numPatches_*sizeof(int*)));
        hostPatchBoundaryCoeffs_ = static_cast<double**>(malloc(numPatches_*sizeof(double*)));
        hostPatchInternalCoeffs_ = static_cast<double**>(malloc(numPatches_*sizeof(double*)));
        hostPatchGammaSf_   = static_cast<double**>(malloc(numPatches_*sizeof(double*)));
        // Allocate memory for intermediate host arrays
        deviceIntermediatePatchAddr_ = static_cast<int**>(malloc(sizeof(int*)*numPatches_));
        deviceIntermediatePatchBoundaryCoeffs_ = static_cast<double**>(malloc(sizeof(double*)*numPatches_));
        deviceIntermediatePatchInternalCoeffs_ = static_cast<double**>(malloc(sizeof(double*)*numPatches_));
        deviceIntermediatePatchGammaSf_ = static_cast<double**>(malloc(sizeof(double*)*numPatches_));
		
        devicePatchSizes_.allocate(numPatches_,true); 
        /*        instead of          
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&)       ,numPatches_*sizeof(int)));
        */
         deviceGammaMagSf_.allocate(numInternalFaces_,true);           
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceGammaMagSf_),numInternalFaces_*sizeof(double)));
         deviceLowerAddress_.allocate(numInternalFaces_,true);
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceLowerAddress_)   ,numInternalFaces_*sizeof(int)));
         deviceUpperAddress_.allocate(numInternalFaces_,true);
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceUpperAddress_)   ,numInternalFaces_*sizeof(int)));

        /*
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchAddr_)       ,numPatches_*sizeof(int*)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchBoundaryCoeffs_)       ,numPatches_*sizeof(double*)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchInternalCoeffs_)       ,numPatches_*sizeof(double*)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchGammaSf_)  ,numPatches_*sizeof(double*)));
             
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceDiagonal_)    ,numCells_*sizeof(double)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceSource_)  ,numCells_*sizeof(double)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceLower_)   ,numInternalFaces_*sizeof(double)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceUpper_)   ,numInternalFaces_*sizeof(double)));
    
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceOldTemperature_) ,numCells_*sizeof(double)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCellVolumes_)  ,numCells_*sizeof(double)));
         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceFaceAreas_)  ,numCells_*sizeof(double)));

         _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceDeltaCellCenters_)  ,numInternalFaces_*sizeof(double)));

        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_T_new)   ,numCells_*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCsrRowPtr_) ,(numCells_+1)*sizeof(int)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCsrColInd_) ,(numCells_+2*nIFaces)*sizeof(int)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCsrValues_) ,(numCells_+2*nIFaces)*sizeof(double)));
        */
        //f2c.init();
        // Set initialization flag to true
        isInitialized_ = true;            
    }
    // Handle function to process mesh and field data
    void handle(fvMesh& mesh,
                volScalarField& DT,
                volScalarField& T)
    {
        printf("handle() is called \n");
        if (!isInitialized_)
        {
             FatalErrorInFunction
                << "ERROR: handle is called before init "
                << endl
                << abort(FatalError);
        }

        // Boundary condition data
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

            hostPatchAddr_[patchI] = &pfClist->first();
            hostPatchSizes_[patchI] = patches[patchI].faceCells().size();
            hostPatchInternalCoeffs_[patchI] = static_cast<double*>(malloc(hostPatchSizes_[patchI]*sizeof(double)));
            hostPatchBoundaryCoeffs_[patchI] = static_cast<double*>(malloc(hostPatchSizes_[patchI]*sizeof(double)));
            for (int i=0 ;i<hostPatchSizes_[patchI];i++)
            {
                hostPatchInternalCoeffs_[patchI][i] = pF_IC_SF[i];
                hostPatchBoundaryCoeffs_[patchI][i] = pF_BC_SF[i];
            }
            hostPatchGammaSf_[patchI] = &pfGammaSFlist->first();
            maxPatchSize_ = (hostPatchSizes_[patchI] > maxPatchSize_) ? hostPatchSizes_[patchI] : maxPatchSize_;
        }


        for(int i=0; i<numPatches_; i++)
        {
        //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchAddr_[i])     ,hostPatchSizes_[i]*sizeof(int)));
        //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchBoundaryCoeffs_[i])     ,hostPatchSizes_[i]*sizeof(double)));
        //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchInternalCoeffs_[i])     ,hostPatchSizes_[i]*sizeof(double)));
        //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchGammaSf_[i]) ,hostPatchSizes_[i]*sizeof(double)));
        // }
        
        devicePatchSizes_.copy(hostPatchSizes_);
        /*
        instead of 
        _CUDA(cudaMemcpy(devicePatchSizes_      ,hostPatchSizes_, sizeof(int)*numPatches_, cudaMemcpyHostToDevice));
        */

        // _CUDA(cudaMemcpy(devicePatchAddr_      ,deviceIntermediatePatchAddr_, sizeof(int*)*numPatches_, cudaMemcpyHostToDevice));
        // _CUDA(cudaMemcpy(devicePatchBoundaryCoeffs_      ,deviceIntermediatePatchBoundaryCoeffs_, sizeof(double*)*numPatches_, cudaMemcpyHostToDevice));
        // _CUDA(cudaMemcpy(devicePatchInternalCoeffs_      ,deviceIntermediatePatchInternalCoeffs_, sizeof(double*)*numPatches_, cudaMemcpyHostToDevice));
        // _CUDA(cudaMemcpy(devicePatchGammaSf_ ,deviceIntermediatePatchGammaSf_, sizeof(double*)*numPatches_, cudaMemcpyHostToDevice));

        // for( int i=0; i<numPatches_; i++ )
        // {
        //     _CUDA(cudaMemcpy(deviceIntermediatePatchAddr_[i]        ,h_pAdrr[i], hostPatchSizes_[i]*sizeof(int), cudaMemcpyHostToDevice));
        //     _CUDA(cudaMemcpy(deviceIntermediatePatchBoundaryCoeffs_[i]        ,h_pf_BC[i], hostPatchSizes_[i]*sizeof(double), cudaMemcpyHostToDevice));
        //     _CUDA(cudaMemcpy(deviceIntermediatePatchInternalCoeffs_[i]        ,h_pf_IC[i], hostPatchSizes_[i]*sizeof(double), cudaMemcpyHostToDevice));
        //     _CUDA(cudaMemcpy(deviceIntermediatePatchGammaSf_[i]   ,h_pf_GammaSf[i], hostPatchSizes_[i]*sizeof(double), cudaMemcpyHostToDevice));
         }
        
        // Mesh data
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
    
        double *hostCellVolumes = &blist->first();             // Host: Cell volumes
        double *hostDeltaCellCenters = &deltaClist->first();   // Host: Delta cell centers for each internal face
        double *hostFaceAreas = &magSflist->first();           // Host: Lower address (owner) for each face
        int *hostLowerAddress = &ownerlist->first();           // Host: Upper address (neighbour) for each face
        int *hostUpperAddress = &neighbourlist->first();       // Host: Face surface area magnitude
		
        // _CUDA(cudaMemcpy(deviceCellVolumes_,        hostCellVolumes_,       nCells*sizeof(double),cudaMemcpyHostToDevice));
        // _CUDA(cudaMemcpy(deviceDeltaCellCenters_,    hostDeltaCellCenters_,   nIFaces*sizeof(double),cudaMemcpyHostToDevice));
        // _CUDA(cudaMemcpy(deviceFaceAreas_,        hostFaceAreas_,       nIFaces*sizeof(double),cudaMemcpyHostToDevice));
        
        deviceLowerAddress_.copy(hostLowerAddress);
        // _CUDA(cudaMemcpy(deviceLowerAddress_,     hostLowerAddress_,    nIFaces*sizeof(int),cudaMemcpyHostToDevice));
        
        deviceUpperAddress_.copy(hostUpperAddress);
        // _CUDA(cudaMemcpy(deviceUpperAddress_,     hostUpperAddress_,    nIFaces*sizeof(int),cudaMemcpyHostToDevice));

        // Update time step and gammaMagSf
        scalar rDeltaT = 1.0/mesh.time().deltaTValue();
        invDeltaT_ = static_cast<double>(rDeltaT);
        surfaceScalarField sf_DT = -fvc::interpolate(DT); 
        surfaceScalarField gammaMagSf_ = sf_DT * mesh.magSf();
		
        // Update old time temperature field to device
        const scalarList* ToldPtr = &T.oldTime().primitiveField();
        scalarList* Toldlist = const_cast<scalarList*>(ToldPtr);
        hostOldTemperature_ = &Toldlist->first();
        // _CUDA(cudaMemcpy(deviceOldTemperature_, hostOldTemperature_, nCells*sizeof(double),cudaMemcpyHostToDevice));

        // Update gammaMSF() to device
        const scalarList* gMgaSfPtr = &gammaMagSf_;
        scalarList* TgMagSflist = const_cast<scalarList*>(gMgaSfPtr);

            // Host arrays for mesh-related data
        double *hostGammaMagSf = &TgMagSflist->first();    // Host: Gamma multiplied by face area , not need to be class member
        deviceGammaMagSf_.copy(hostGammaMagSf);
        // _CUDA(cudaMemcpy(deviceGammaMagSf_, hostGammaMagSf_, nIFaces*sizeof(double),cudaMemcpyHostToDevice));

    }

    //  void updateF()
    // {

    // }
    // void update()
    // {
    //     // _CUDA(cudaMemcpy(deviceOldTemperature_, h_T_new, nCells*sizeof(double),cudaMemcpyHostToDevice));
    // }
    // Reset function to free all allocated memory
    void reset()
    {
        isInitialized_ = false;
        if (hostPatchBoundaryCoeffs_) free(hostPatchBoundaryCoeffs_);
        if (hostPatchInternalCoeffs_) free(hostPatchInternalCoeffs_);
        if (hostPatchGammaSf_) free(hostPatchGammaSf_);
        if (hostPatchAddr_) free(hostPatchAddr_);
        if (hostPatchSizes_) free(hostPatchSizes_);
        if (deviceIntermediatePatchAddr_) free(deviceIntermediatePatchAddr_);
        if (deviceIntermediatePatchBoundaryCoeffs_) free(deviceIntermediatePatchBoundaryCoeffs_);
        if (deviceIntermediatePatchInternalCoeffs_) free(deviceIntermediatePatchInternalCoeffs_);
        if (deviceIntermediatePatchGammaSf_) free(deviceIntermediatePatchGammaSf_);
		
        // if (devicePatchSizes_)            _CUDA(cudaFree(devicePatchSizes_));
        // if (devicePatchAddr_)            _CUDA(cudaFree(devicePatchAddr_));
        // if (devicePatchBoundaryCoeffs_)            _CUDA(cudaFree(devicePatchBoundaryCoeffs_));
        // if (devicePatchInternalCoeffs_)            _CUDA(cudaFree(devicePatchInternalCoeffs_));
        // if (devicePatchGammaSf_)       _CUDA(cudaFree(devicePatchGammaSf_));
        // if (deviceCellVolumes_)               _CUDA(cudaFree(deviceCellVolumes_));
        // if (deviceOldTemperature_)              _CUDA(cudaFree(deviceOldTemperature_));
        // if (deviceDeltaCellCenters_)           _CUDA(cudaFree(deviceDeltaCellCenters_));
        // if (deviceFaceAreas_)               _CUDA(cudaFree(deviceFaceAreas_));
        // if (deviceGammaMagSf_)         _CUDA(cudaFree(deviceGammaMagSf_));
        // if (deviceLowerAddress_)            _CUDA(cudaFree(deviceLowerAddress_));
        // if (deviceUpperAddress_)            _CUDA(cudaFree(deviceUpperAddress_));   
        // //if (deviceCsrRowPtr_)            _CUDA(cudaFree(deviceCsrRowPtr_));      
        //if (deviceCsrColInd_)            _CUDA(cudaFree(deviceCsrColInd_));
        //if (deviceCsrValues_)            _CUDA(cudaFree(deviceCsrValues_)); 
    } 
    // Function to call the discretization kernel (currently just a placeholder)
    void discKernel()
    {
        printf("discKernel is called \n");

        // discKernelWrapper(  nCells,
        //                     nIFaces,
        //                     deviceCellVolumes_,
        //                     deviceOldTemperature_,
        //                     deviceDeltaCellCenters_,
        //                     deviceGammaMagSf_,
        //                     deviceUpperAddress_,
        //                     deviceLowerAddress_,
        //                     numPatches_,
        //                     maxPatches,
        //                     devicePatchSizes_,
        //                     devicePatchAddr_,
        //                     devicePatchBoundaryCoeffs_,
        //                     devicePatchInternalCoeffs_,
        //                     devicePatchGammaSf_,
        //                     rDelgaG,
        //                     deviceDiagonal_,
        //                     deviceSource_,
        //                     deviceUpper_,
        //                     deviceLower_
        //                     );
    }


};
    
   
    

   
   



    

    

   
    


   

    


    
