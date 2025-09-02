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




    // Device arrays for mesh-related data
    Foam::MeshFields deviceMesh;

    //HybridArray<Foam::scalar> deviceMagSf__;  // Device: face area


    //HybridMatrix<Foam::scalar> deviceIntermediatePatchBoundaryCoeffs_; // Intermediate host array for device data
    
    //HybridArray<Foam::scalar *> devicePatchInternalCoeffs_;    // Device: Internal coefficients for gradient calculation
    //HybridMatrix<Foam::scalar> deviceIntermediatePatchInternalCoeffs_; // Intermediate host array for device data

    //HybridArray<Foam::scalar *> devicePatchSfDT__; // Device: Patch field _ surface scalar field DT
    //HybridMatrix<Foam::scalar> deviceIntermediatePatchSfDT__; // Intermediate host array for device data

    // Device arrays for linear system (matrix and source terms)
    Foam::LduMatrixFields deviceLdu;

    // Device arrays for CSR (Compressed Sparse Row) format matrix
    Foam::label *deviceCsrRowPtr_;     // Device: Row pointers for CSR format
    Foam::label *deviceCsrColInd_;      // Device: Column indices for CSR format
    Foam::scalar *deviceCsrValues_;   // Device: Values for CSR format


    // Flag to check if initialization is done
    Foam::Switch isInitialized_;
public:
    // Constructor: Initialize all pointers to NULL and set initDone to false
    gpuFields():
    deviceCsrRowPtr_(NULL),
    deviceCsrColInd_(NULL),
    deviceCsrValues_(NULL),
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

        deviceMesh.init(mesh,true);


        
        
        // Allocate memory for intermediate host arrays
        
        //deviceIntermediatePatchAddr_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        //deviceIntermediatePatchBoundaryCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        //deviceIntermediatePatchInternalCoeffs_.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        //deviceIntermediatePatchMagSf__.allocate_rows(numPatches_,STORE_ON_HOST_GPU);

        //deviceIntermediatePatchSfDT__.allocate_rows(numPatches_,STORE_ON_HOST_GPU);
        

        //deviceMagSf__.allocate(numInternalFaces_,true);     


        //devicePatchAddr_.allocate(numPatches_,true);

        //devicePatchBoundaryCoeffs_.allocate(numPatches_,true);
 
        //devicePatchInternalCoeffs_.allocate(numPatches_,true);
 
        //devicePatchMagSf__.allocate(numPatches_,true);

        //devicePatchSfDT__.allocate(numPatches_,true);
         
        deviceLdu.init(deviceMesh.numCells,deviceMesh.numInternalFaces,true);

        


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

        deviceMesh.handle(mesh);

        const polyBoundaryMesh& patches = mesh.boundaryMesh();
        
        forAll(patches, patchI)
        {

        }    

        //for(int i=0; i<numPatches_; i++)
        //{
            //deviceIntermediatePatchAddr_.allocate_col(i,hostPatchSizes_[i]);
 
            //deviceIntermediatePatchBoundaryCoeffs_.allocate_col(i,hostPatchSizes_[i]);

            //deviceIntermediatePatchInternalCoeffs_.allocate_col(i,hostPatchSizes_[i]);

            //deviceIntermediatePatchMagSf__.allocate_col(i,hostPatchSizes_[i]);

            //deviceIntermediatePatchSfDT__.allocate_col(i,hostPatchSizes_[i]);
        //}
        
        
 
        
         //for( int i=0; i<numPatches_; i++ )
         //{
            //deviceIntermediatePatchAddr_.copy_col(i,hostPatchAddr_[i]);

            //deviceIntermediatePatchBoundaryCoeffs_.copy_col(i,hostPatchBoundaryCoeffs_.Data()[i]);

            //deviceIntermediatePatchInternalCoeffs_.copy_col(i,hostPatchInternalCoeffs_.Data()[i]);

            //deviceIntermediatePatchMagSf__.copy_col(i,hostPatchMagSf__[i]);

            //deviceIntermediatePatchSfDT__.copy_col(i,hostPatchSfDT__[i]);
        //}
        
        

        //devicePatchSfDT__.copy(deviceIntermediatePatchSfDT_);


 

        // Update surface magnitude() to device
        //const scalarList* gMgaSfPtr = &mesh.magSf();
        //scalarList* TgMagSflist = const_cast<scalarList*>(gMgaSfPtr);

        // Host arrays for mesh-related data
        //double *hostMagSf_ = TgMagSflist->begin();    
        //deviceMagSf__.copy(hostMagSf_);
 
        // Update surface scalar field to device
    
    }

   
    // Reset function to free all allocated memory
    void reset()
    {
        isInitialized_ = false;

        
        
        //deviceIntermediatePatchAddr_.deallocate();
       
        //deviceIntermediatePatchBoundaryCoeffs_.deallocate();
      
        //deviceIntermediatePatchInternalCoeffs_.deallocate();
        
		//deviceIntermediatePatchMagSf__.deallocate();

        //deviceIntermediatePatchSfDT__.deallocate();
       
        
        
        devicePatchSfDT_.deallocate();
        
        deviceOldTemperature_.deallocate();
        
        devicePatchBoundaryCoeffs_.deallocate();

        devicePatchInternalCoeffs_.deallocate();

        hostPatchBoundaryCoeffs_.deallocate();

        hostPatchInternalCoeffs_.deallocate();
            
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

	discKernelWrapper(       
                            deviceMesh.numCells,
                             deviceMesh.numInternalFaces,
			                 deviceMesh.cellVolumes.Data(),
                             deviceOldTemperature_.Data(),
                             deviceSurfDT_.Data(),
			                 deviceMesh.deltaCellCenters.Data(),
		                     deviceMesh.faceAreas.Data(),
                             deviceMesh.upperAddress.Data(),
			                 deviceMesh.lowerAddress.Data(),
                             deviceMesh.numPatches,
                             deviceMesh.maxPatchSize,
                             deviceMesh.devicePatchSizes.Data(),
                             deviceMesh.devicePatchAddr.deviceList.Data(),
                             devicePatchBoundaryCoeffs_.deviceList.Data(),
                             devicePatchInternalCoeffs_.deviceList.Data(),
                             deviceMesh.devicePatchMagSf.deviceList.Data(),
                             devicePatchSfDT_.deviceList.Data(),
                             deviceMesh.invDeltaT,
                             deviceLdu.diagonal,
			                 deviceLdu.source,
			                 deviceLdu.upper,
			                 deviceLdu.lower
                             );
                             
    }


};
    
   
    

   
   



    

    

   
    


   

    


    
