#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
//#include <AmgXCSRMatrix.H>
#include "label.H"
#include "scalar.H"
#include "discretizationKernel.h"
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
    
    HybridArray<Foam::scalar *> hostPatchMagSf__; // Host: Patch field _ Magnitude of face area

    HybridArray<Foam::scalar *> hostPatchSfDT__; // Host: Patch field _ surface scalar field DT

    // Device arrays for patch-related data
    HybridArray<Foam::label> devicePatchSizes_;        // Device: Number of cells adjacent to the faces of each patch
    
    HybridArray<Foam::label *> devicePatchAddr_;      // Device: Indices of cells adjacent to the faces of each patch
    HybridMatrix<Foam::label> deviceIntermediatePatchAddr_;   // Intermediate host array for device data
    HybridArray<Foam::scalar *> devicePatchBoundaryCoeffs_;    // Device: Boundary coefficients for gradient calculation
    HybridMatrix<Foam::scalar> deviceIntermediatePatchBoundaryCoeffs_; // Intermediate host array for device data
    HybridArray<Foam::scalar *> devicePatchInternalCoeffs_;    // Device: Internal coefficients for gradient calculation
    HybridMatrix<Foam::scalar> deviceIntermediatePatchInternalCoeffs_; // Intermediate host array for device data
    HybridArray<Foam::scalar *> devicePatchMagSf__; // Device: Patch field _ mag of face area
    HybridArray<Foam::scalar *> devicePatchSfDT__; // Device: Patch field _ surface scalar field DT

    HybridMatrix<Foam::scalar> deviceIntermediatePatchMagSf__; // Intermediate host array for device data

    HybridMatrix<Foam::scalar> deviceIntermediatePatchSfDT__; // Intermediate host array for device data

    // Device arrays for mesh-related data
    Foam::MeshFields deviceMesh;

    HybridArray<Foam::scalar> deviceOldTemperature_;       // Device: Old time temperature field
    HybridArray<Foam::scalar> deviceSurfDT__;  // Device: face area
    HybridArray<Foam::scalar> deviceMagSf__;  // Device: face area

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
        
        hostPatchSizes_.allocate(numPatches_,false);
        
        hostPatchAddr_.allocate(numPatches_,false);
        
        hostPatchBoundaryCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST);
        
        hostPatchInternalCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST);
        
        hostPatchMagSf__.allocate(numPatches_,false);

        hostPatchSfDT__.allocate(numPatches_,false);
        
        // Allocate memory for intermediate host arrays
        
        deviceIntermediatePatchAddr_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        deviceIntermediatePatchBoundaryCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        deviceIntermediatePatchInternalCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        deviceIntermediatePatchMagSf__.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        deviceIntermediatePatchSfDT__.allocate_rows(numPatches_,STORE_ON_HOST_GPU);
        
        devicePatchSizes_.allocate(numPatches_,true); 

        deviceMagSf__.allocate(numInternalFaces_,true);     

        deviceSurfDT__.allocate(numInternalFaces_,true);           

        devicePatchAddr_.allocate(numPatches_,true);

        devicePatchBoundaryCoeffs_.allocate(numPatches_,true);
 
        devicePatchInternalCoeffs_.allocate(numPatches_,true);
 
        devicePatchMagSf__.allocate(numPatches_,true);

        devicePatchSfDT__.allocate(numPatches_,true);
         
        deviceLdu.init(numCells_,numInternalFaces_,true);

        deviceMesh.init(mesh,true);
        
        deviceOldTemperature_.allocate(numCells_,true);

        isInitialized_ = true;            
    }
    // Handle function to process mesh and field data
    void handle(fvMesh& mesh,
                surfaceScalarField& sf_DT,
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

        forAll(patches, patchI)
        {
            const labelUList* pfCPtr = &patches[patchI].faceCells();
            labelUList* pfClist  = const_cast<labelUList*>(pfCPtr);
            scalarField pF_BC_SF = T.boundaryField()[patchI].gradientBoundaryCoeffs();
            scalarField pF_IC_SF = T.boundaryField()[patchI].gradientInternalCoeffs();
            const scalarList* pfMagSFptr= &mesh.magSf().boundaryField()[patchI];
            scalarList* pfMagSFlist = const_cast<scalarList*>(pfMagSFptr);

            const scalarList* pfSfDTptr= &sf_DT.boundaryField()[patchI];
            scalarList* pfSfDTlist = const_cast<scalarList*>(pfSfDTptr);

            hostPatchAddr_.set(patchI,&pfClist->first());
            
            hostPatchSizes_.set(patchI,patches[patchI].faceCells().size());
            
            hostPatchInternalCoeffs_.allocate_col(patchI,hostPatchSizes_[patchI]);
            
            
            hostPatchBoundaryCoeffs_.allocate_col(patchI,hostPatchSizes_[patchI]);
            
 	        hostPatchInternalCoeffs_.copy_col(patchI,pF_IC_SF.data());
	        hostPatchBoundaryCoeffs_.copy_col(patchI,pF_BC_SF.data());
			
            hostPatchMagSf__.set(patchI,&pfMagSFlist->first());

            hostPatchSfDT__.set(patchI,&pfSfDTlist->first());
            
            maxPatchSize_ = (hostPatchSizes_[patchI] > maxPatchSize_) ? hostPatchSizes_[patchI] : maxPatchSize_;
        }


        for(int i=0; i<numPatches_; i++)
        {
            deviceIntermediatePatchAddr_.allocate_col(i,hostPatchSizes_[i]);
 
            deviceIntermediatePatchBoundaryCoeffs_.allocate_col(i,hostPatchSizes_[i]);

            deviceIntermediatePatchInternalCoeffs_.allocate_col(i,hostPatchSizes_[i]);

            deviceIntermediatePatchMagSf__.allocate_col(i,hostPatchSizes_[i]);

            deviceIntermediatePatchSfDT__.allocate_col(i,hostPatchSizes_[i]);
        }
        
        
	    devicePatchSizes_.copy(hostPatchSizes_.Data());
 
        
         for( int i=0; i<numPatches_; i++ )
         {
            deviceIntermediatePatchAddr_.copy_col(i,hostPatchAddr_[i]);

            deviceIntermediatePatchBoundaryCoeffs_.copy_col(i,hostPatchBoundaryCoeffs_.Data()[i]);

            deviceIntermediatePatchInternalCoeffs_.copy_col(i,hostPatchInternalCoeffs_.Data()[i]);

            deviceIntermediatePatchMagSf__.copy_col(i,hostPatchMagSf__[i]);

            deviceIntermediatePatchSfDT__.copy_col(i,hostPatchSfDT__[i]);
        }
        
        devicePatchAddr_.copy(deviceIntermediatePatchAddr_.Data());
 
        devicePatchBoundaryCoeffs_.copy(deviceIntermediatePatchBoundaryCoeffs_.Data());

        devicePatchInternalCoeffs_.copy(deviceIntermediatePatchInternalCoeffs_.Data());
 
        devicePatchMagSf__.copy(deviceIntermediatePatchMagSf__.Data());

        devicePatchSfDT__.copy(deviceIntermediatePatchSfDT__.Data());

        // Mesh data
        deviceMesh.handle(mesh);
        
        scalar rDeltaT = 1.0/mesh.time().deltaTValue();
        invDeltaT_ = static_cast<double>(rDeltaT);
 
        // Update old time temperature field to device
        const scalarList* ToldPtr = &T.oldTime().primitiveField();
        scalarList* Toldlist = const_cast<scalarList*>(ToldPtr);
        double* hostOldTemperature;	//Host : Old Temperature Time Field
        hostOldTemperature = &Toldlist->first();
        
        deviceOldTemperature_.copy(hostOldTemperature);

        // Update surface magnitude() to device
        const scalarList* gMgaSfPtr = &mesh.magSf();
        scalarList* TgMagSflist = const_cast<scalarList*>(gMgaSfPtr);

        // Host arrays for mesh-related data
        double *hostMagSf_ = &TgMagSflist->first();    
        deviceMagSf__.copy(hostMagSf_);
 
        // Update surface scalar field to device
        const scalarList* sfDTPtr = &sf_DT.internalField();
        scalarList* sfDTlist = const_cast<scalarList*>(sfDTPtr);
        double *hostSurfDT_ = &sfDTlist->first();
        deviceSurfDT__.copy(hostSurfDT_);
    
    }

   
    // Reset function to free all allocated memory
    void reset()
    {
        isInitialized_ = false;

        hostPatchBoundaryCoeffs_.deallocate();

        hostPatchInternalCoeffs_.deallocate();
        
        hostPatchMagSf__.deallocate();
        
        hostPatchAddr_.deallocate();
        
        
        hostPatchSizes_.deallocate();
        
        
        deviceIntermediatePatchAddr_.deallocate();
       
        deviceIntermediatePatchBoundaryCoeffs_.deallocate();
      
        deviceIntermediatePatchInternalCoeffs_.deallocate();
        
		deviceIntermediatePatchMagSf__.deallocate();

        deviceIntermediatePatchSfDT__.deallocate();
       
        devicePatchSizes_.deallocate();
       
        devicePatchAddr_.deallocate();
      
        devicePatchBoundaryCoeffs_.deallocate();
    
        devicePatchInternalCoeffs_.deallocate();
     
        devicePatchMagSf__.deallocate();
        
        devicePatchSfDT__.deallocate();
        
        deviceOldTemperature_.deallocate();
        
        
        deviceMesh.deallocate();
    } 
    // Function to call the discretization kernel (currently just a placeholder)
    void discKernel()
        {
        // HybridArray<int> hostPatchSizes2;
        // hostPatchSizes2.allocate(devicePatchSizes_.size(),false);
        // hostPatchSizes2.copy(devicePatchSizes_);
        // for (label i = 0; i < hostPatchSizes2.size(); i++) 
        // { 
        //     printf("hostPathSizes2 %d %d\n",i,hostPatchSizes2[i]); 
        // }

    //     HybridArray<scalar> hostPatchInternalCoeff2;
    //     hostPatchInternalCoeff2.allocate(deviceIntermediatePatchInternalCoeffs_.cols(0),false);
    //     printf("cols[0]:%d \n",deviceIntermediatePatchInternalCoeffs_.cols(0));
	//     hostPatchInternalCoeff2.copy(deviceIntermediatePatchInternalCoeffs_.Data()[0],true);
    //     //not accessible
	// //hostPatchInternalCoeff2.copy(devicePatchInternalCoeffs_.Data()[0],true);
	//     for (label i = 0; i < hostPatchInternalCoeff2.size(); i++) 
    //     { 
    //         printf("hostPatchInternalCoeff2 %d %f %f\n",i,hostPatchInternalCoeff2[i],hostPatchInternalCoeffs_.Data()[0][i]); 
    //     }

	discKernelWrapper(       numCells_,
                             numInternalFaces_,
			                 deviceMesh.cellVolumes.Data(),
                             deviceOldTemperature_.Data(),
                             deviceSurfDT__.Data(),
			                 deviceMesh.deltaCellCenters.Data(),
		                     deviceMagSf__.Data(),
                             deviceMesh.upperAddress.Data(),
			                 deviceMesh.lowerAddress.Data(),
                             numPatches_,
                             maxPatchSize_,
                             devicePatchSizes_.Data(),
                             devicePatchAddr_.Data(),
                             devicePatchBoundaryCoeffs_.Data(),
                             devicePatchInternalCoeffs_.Data(),
                             devicePatchMagSf__.Data(),
                             devicePatchSfDT__.Data(),
                             invDeltaT_,
                             deviceLdu.diagonal,
			                 deviceLdu.source,
			                 deviceLdu.upper,
			                 deviceLdu.lower
                             );
                             
    }


};
    
   
    

   
   



    

    

   
    


   

    


    
