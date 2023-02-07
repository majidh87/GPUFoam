wclean
source envAMGX 
nvcc -c -arch=sm_70  discretizationKernel.cu -o discretizationKernel.o
nvcc -c -arch=sm_70  ldu2csr.cu -o ldu2csr.o
wmake
