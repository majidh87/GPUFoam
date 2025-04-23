#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
//#include <AmgXCSRMatrix.H>
#include "label.H"
#include "scalar.H"
#include "HybridArray.H"
#include "HybridMatrix.H"
#include "LduMatrixFields.H"
#include "MeshFields.H"
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

  public:    
    Foam::label numCells_;          // Number of cells in the mesh
    Foam::label numInternalFaces_;  // Number of internal faces in the mesh
    Foam::scalar invDeltaT_;      // Reciprocal of delta t (time step)

    Foam::label maxPatchSize_;      // Maximum number of cells adjacent to any patch
    Foam::label numPatches_;        // Number of patches in the domain

    // Host arrays for patch-related data
    HybridArray<Foam::label> hostPatchSizes_;        // Host: Number of cells adjacent to the faces of each patch
    HybridArray<Foam::label *> hostPatchAddr_;        // Host: Indices of cells adjacent to the faces of each patch
    HybridMatrix<Foam::scalar> hostPatchBoundaryCoeffs_;    // Host: Boundary coefficients for gradient calculation

    HybridMatrix<Foam::scalar> hostPatchInternalCoeffs_;    // Host: Internal coefficients for gradient calculation
    
    HybridArray<Foam::scalar *> hostPatchGammaSf_; // Host: Patch field coefficients multiplied by face area

    // Device arrays for patch-related data
    HybridArray<Foam::label> devicePatchSizes_;        // Device: Number of cells adjacent to the faces of each patch
    
    HybridMatrix<Foam::label> devicePatchAddr_;      // Device: Indices of cells adjacent to the faces of each patch
    HybridMatrix<Foam::label> deviceIntermediatePatchAddr_;   // Intermediate host array for device data
    HybridMatrix<Foam::scalar> devicePatchBoundaryCoeffs_;    // Device: Boundary coefficients for gradient calculation
    HybridMatrix<Foam::scalar> deviceIntermediatePatchBoundaryCoeffs_; // Intermediate host array for device data
    HybridMatrix<Foam::scalar> devicePatchInternalCoeffs_;    // Device: Internal coefficients for gradient calculation
    HybridMatrix<Foam::scalar> deviceIntermediatePatchInternalCoeffs_; // Intermediate host array for device data
    HybridMatrix<Foam::scalar> devicePatchGammaSf_; // Device: Patch field coefficients multiplied by face area

    HybridMatrix<Foam::scalar> deviceIntermediatePatchGammaSf_; // Intermediate host array for device data

    

    // Device arrays for mesh-related data
    Foam::MeshFields deviceMesh;

    HybridArray<Foam::scalar> deviceOldTemperature_;       // Device: Old time temperature field
    HybridArray<Foam::scalar> deviceGammaMagSf_;  // Device: Gamma multiplied by face area

    // Device arrays for linear system (matrix and source terms)
    Foam::LduMatrixFields deviceLdu;

    // Device arrays for CSR (Compressed Sparse Row) format matrix
    Foam::label *deviceCsrRowPtr_;     // Device: Row pointers for CSR format
    Foam::label *deviceCsrColInd_;      // Device: Column indices for CSR format
    Foam::scalar *deviceCsrValues_;   // Device: Values for CSR format

    // Host array for new temperature field
    Foam::scalar* hostNewTemperature_;

    // Flag to check if initialization is done
    Foam::Switch isInitialized_;
public:
    // Constructor: Initialize all pointers to NULL and set initDone to false
    gpuFields():
    numCells_(0),
    numInternalFaces_(0),
    maxPatchSize_(0),
    numPatches_(0),
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
        
        //hostPatchSizes_     = static_cast<int*>(malloc(numPatches_*sizeof(int)));
        hostPatchSizes_.allocate(numPatches_,STORE_ON_HOST);
        
        //hostPatchAddr_      = static_cast<int**>(malloc(numPatches_*sizeof(int*)));
        hostPatchAddr_.allocate(numPatches_,STORE_ON_HOST);
        
        //hostPatchBoundaryCoeffs_ = static_cast<double**>(malloc(numPatches_*sizeof(double*)));
        hostPatchBoundaryCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST);
        
        //hostPatchInternalCoeffs_ = static_cast<double**>(malloc(numPatches_*sizeof(double*)));
        hostPatchInternalCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST);
        
        //hostPatchGammaSf_   = static_cast<double**>(malloc(numPatches_*sizeof(double*)));
        hostPatchGammaSf_.allocate(numPatches_,STORE_ON_HOST);
        
        // Allocate memory for intermediate host arrays
        
        //deviceIntermediatePatchAddr_ = static_cast<int**>(malloc(sizeof(int*)*numPatches_));
        deviceIntermediatePatchAddr_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        //deviceIntermediatePatchBoundaryCoeffs_ = static_cast<double**>(malloc(sizeof(double*)*numPatches_));
        deviceIntermediatePatchBoundaryCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);
        //deviceIntermediatePatchInternalCoeffs_ = static_cast<double**>(malloc(sizeof(double*)*numPatches_));
        deviceIntermediatePatchInternalCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);
        //deviceIntermediatePatchGammaSf_ = static_cast<double**>(malloc(sizeof(double*)*numPatches_));
		deviceIntermediatePatchGammaSf_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);
        
        devicePatchSizes_.allocate(numPatches_,STORE_ON_GPU); 
        /*        instead of          
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&)       ,numPatches_*sizeof(int)));
        */
         deviceGammaMagSf_.allocate(numInternalFaces_,STORE_ON_GPU);           
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceGammaMagSf_),numInternalFaces_*sizeof(double)));

        
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchAddr_)       ,numPatches_*sizeof(int*)));
         devicePatchAddr_.allocate_rows(numPatches_,STORE_ON_GPU);
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchBoundaryCoeffs_)       ,numPatches_*sizeof(double*)));
         devicePatchBoundaryCoeffs_.allocate_rows(numPatches_,STORE_ON_GPU);
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchInternalCoeffs_)       ,numPatches_*sizeof(double*)));
         devicePatchInternalCoeffs_.allocate_rows(numPatches_,STORE_ON_GPU);
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&devicePatchGammaSf_)  ,numPatches_*sizeof(double*)));
         devicePatchGammaSf_.allocate_rows(numPatches_,STORE_ON_GPU);
         
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceDiagonal_)    ,numCells_*sizeof(double)));
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceSource_)  ,numCells_*sizeof(double)));
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceLower_)   ,numInternalFaces_*sizeof(double)));
         //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceUpper_)   ,numInternalFaces_*sizeof(double)));
         deviceLdu.init(numCells_,numInternalFaces_,true);
         /*     
    


        _CUDA(cudaMalloc(reinterpret_cast<void**>(&d_T_new)   ,numCells_*sizeof(double)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCsrRowPtr_) ,(numCells_+1)*sizeof(int)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCsrColInd_) ,(numCells_+2*nIFaces)*sizeof(int)));
        _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceCsrValues_) ,(numCells_+2*nIFaces)*sizeof(double)));
        */
        deviceMesh.init(mesh,true);
        
        //_CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceOldTemperature_) ,numCells_*sizeof(double)));
        deviceOldTemperature_.allocate(numCells_,STORE_ON_GPU);

        
	
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

            //hostPatchAddr_[patchI] = &pfClist->first();
            hostPatchAddr_.set(patchI,&pfClist->first());
            
            //hostPatchSizes_[patchI] = patches[patchI].faceCells().size();
            hostPatchSizes_.set(patchI,patches[patchI].faceCells().size());
            
            //hostPatchInternalCoeffs_[patchI] = static_cast<double*>(malloc(hostPatchSizes_[patchI]*sizeof(double)));
            hostPatchInternalCoeffs_.allocate_col(patchI,hostPatchSizes_[patchI]);
            
            //hostPatchBoundaryCoeffs_[patchI] = static_cast<double*>(malloc(hostPatchSizes_[patchI]*sizeof(double)));
            hostPatchBoundaryCoeffs_.allocate_col(patchI,hostPatchSizes_[patchI]);
            
            /*for (int i=0 ;i<hostPatchSizes_[patchI];i++)
            {
                //hostPatchInternalCoeffs_[patchI][i] = pF_IC_SF[i];
                hostPatchInternalCoeffs_.set(patchI,i,pF_IC_SF[i]);
                
                //hostPatchBoundaryCoeffs_[patchI][i] = pF_BC_SF[i];
                hostPatchBoundaryCoeffs_.set(patchI,i,pF_BC_SF[i]);
            }*/
			hostPatchInternalCoeffs_.copy_col(patchI,pF_IC_SF.data());
			hostPatchBoundaryCoeffs_.copy_col(patchI,pF_BC_SF.data());
			
			
            //hostPatchGammaSf_[patchI] = &pfGammaSFlist->first();
            hostPatchGammaSf_.set(patchI,&pfGammaSFlist->first());
            
            maxPatchSize_ = (hostPatchSizes_[patchI] > maxPatchSize_) ? hostPatchSizes_[patchI] : maxPatchSize_;
        }


        for(int i=0; i<numPatches_; i++)
        {
            //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchAddr_[i])     ,hostPatchSizes_[i]*sizeof(int)));
            deviceIntermediatePatchAddr_.allocate_col(i,hostPatchSizes_[i]);
            //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchBoundaryCoeffs_[i])     ,hostPatchSizes_[i]*sizeof(double)));
            deviceIntermediatePatchBoundaryCoeffs_.allocate_col(i,hostPatchSizes_[i]);
            //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchInternalCoeffs_[i])     ,hostPatchSizes_[i]*sizeof(double)));
            deviceIntermediatePatchInternalCoeffs_.allocate_col(i,hostPatchSizes_[i]);
            //     _CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceIntermediatePatchGammaSf_[i]) ,hostPatchSizes_[i]*sizeof(double)));
            deviceIntermediatePatchGammaSf_.allocate_col(i,hostPatchSizes_[i]);
        }
        
        devicePatchSizes_.copy(hostPatchSizes_.Data());
        /*
        instead of 
        _CUDA(cudaMemcpy(devicePatchSizes_      ,hostPatchSizes_, sizeof(int)*numPatches_, cudaMemcpyHostToDevice));
        */
        
        //???? MN
        // _CUDA(cudaMemcpy(devicePatchAddr_      ,deviceIntermediatePatchAddr_, sizeof(int*)*numPatches_, cudaMemcpyHostToDevice));
        //devicePatchAddr_.

        // _CUDA(cudaMemcpy(devicePatchBoundaryCoeffs_      ,deviceIntermediatePatchBoundaryCoeffs_, sizeof(double*)*numPatches_, cudaMemcpyHostToDevice));
        // _CUDA(cudaMemcpy(devicePatchInternalCoeffs_      ,deviceIntermediatePatchInternalCoeffs_, sizeof(double*)*numPatches_, cudaMemcpyHostToDevice));
        // _CUDA(cudaMemcpy(devicePatchGammaSf_ ,deviceIntermediatePatchGammaSf_, sizeof(double*)*numPatches_, cudaMemcpyHostToDevice));

         for( int i=0; i<numPatches_; i++ )
         {
            //     _CUDA(cudaMemcpy(deviceIntermediatePatchAddr_[i]        ,h_pAdrr[i], hostPatchSizes_[i]*sizeof(int), cudaMemcpyHostToDevice));
            deviceIntermediatePatchAddr_.copy_col(i,hostPatchAddr_[i]);
            //     _CUDA(cudaMemcpy(deviceIntermediatePatchBoundaryCoeffs_[i]        ,h_pf_BC[i], hostPatchSizes_[i]*sizeof(double), cudaMemcpyHostToDevice));
            deviceIntermediatePatchBoundaryCoeffs_.copy_col(i,hostPatchInternalCoeffs_.Data()[i]);
            //     _CUDA(cudaMemcpy(deviceIntermediatePatchInternalCoeffs_[i]        ,h_pf_IC[i], hostPatchSizes_[i]*sizeof(double), cudaMemcpyHostToDevice));
            deviceIntermediatePatchInternalCoeffs_.copy_col(i,hostPatchBoundaryCoeffs_.Data()[i]);
            //     _CUDA(cudaMemcpy(deviceIntermediatePatchGammaSf_[i]   ,h_pf_GammaSf[i], hostPatchSizes_[i]*sizeof(double), cudaMemcpyHostToDevice));
            deviceIntermediatePatchGammaSf_.copy_col(i,hostPatchGammaSf_[i]);
        }
        
        // Mesh data
        deviceMesh.handle(mesh);
        // Update time step and gammaMagSf
        scalar rDeltaT = 1.0/mesh.time().deltaTValue();
        invDeltaT_ = static_cast<double>(rDeltaT);
        surfaceScalarField sf_DT = -fvc::interpolate(DT); 
        surfaceScalarField gammaMagSf_ = sf_DT * mesh.magSf();
		
        // Update old time temperature field to device
        const scalarList* ToldPtr = &T.oldTime().primitiveField();
        scalarList* Toldlist = const_cast<scalarList*>(ToldPtr);
        double* hostOldTemperature;	//Host : Old Temperature Time Field
        hostOldTemperature = &Toldlist->first();
        
        // _CUDA(cudaMemcpy(deviceOldTemperature_, hostOldTemperature, nCells*sizeof(double),cudaMemcpyHostToDevice));
        deviceOldTemperature_.copy(hostOldTemperature);

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
        //if (hostPatchBoundaryCoeffs_) free(hostPatchBoundaryCoeffs_);
        hostPatchBoundaryCoeffs_.deallocate();
        
        //if (hostPatchInternalCoeffs_) free(hostPatchInternalCoeffs_);
        hostPatchInternalCoeffs_.deallocate();
        
        //if (hostPatchGammaSf_) free(hostPatchGammaSf_);
        hostPatchGammaSf_.deallocate();
        
        //if (hostPatchAddr_) free(hostPatchAddr_);
        hostPatchAddr_.deallocate();
        
        //if (hostPatchSizes_) free(hostPatchSizes_);
        hostPatchSizes_.deallocate();
        
        //if (deviceIntermediatePatchAddr_) free(deviceIntermediatePatchAddr_);
        deviceIntermediatePatchAddr_.deallocate();
        //if (deviceIntermediatePatchBoundaryCoeffs_) free(deviceIntermediatePatchBoundaryCoeffs_);
        deviceIntermediatePatchBoundaryCoeffs_.deallocate();
        //if (deviceIntermediatePatchInternalCoeffs_) free(deviceIntermediatePatchInternalCoeffs_);
        deviceIntermediatePatchInternalCoeffs_.deallocate();
        //if (deviceIntermediatePatchGammaSf_) free(deviceIntermediatePatchGammaSf_);
		deviceIntermediatePatchGammaSf_.deallocate();
        // if (devicePatchSizes_)            _CUDA(cudaFree(devicePatchSizes_));
        devicePatchSizes_.deallocate();
        // if (devicePatchAddr_)            _CUDA(cudaFree(devicePatchAddr_));
        devicePatchAddr_.deallocate();
        // if (devicePatchBoundaryCoeffs_)            _CUDA(cudaFree(devicePatchBoundaryCoeffs_));
        devicePatchBoundaryCoeffs_.deallocate();
        // if (devicePatchInternalCoeffs_)            _CUDA(cudaFree(devicePatchInternalCoeffs_));
        devicePatchInternalCoeffs_.deallocate();
        // if (devicePatchGammaSf_)       _CUDA(cudaFree(devicePatchGammaSf_));
        devicePatchGammaSf_.deallocate();
        
        
        // if (deviceOldTemperature_)              _CUDA(cudaFree(deviceOldTemperature_));
        deviceOldTemperature_.deallocate();
        
        
        // if (deviceGammaMagSf_)         _CUDA(cudaFree(deviceGammaMagSf_));
        deviceMesh.deallocate();
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
        //                     deviceOldTemperature_,
        //                     deviceGammaMagSf_,
        //                     deviceMesh,
        //                     numPatches_,
        //                     maxPatches,
        //                     devicePatchSizes_,
        //                     devicePatchAddr_,
        //                     devicePatchBoundaryCoeffs_,
        //                     devicePatchInternalCoeffs_,
        //                     devicePatchGammaSf_,
        //                     rDelgaG,
        //                        deviceLdu
        //                     );
    }


};
    
   
    

   
   



    

    

   
    


   

    


    
