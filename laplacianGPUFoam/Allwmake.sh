wclean
source envAMGX 
nvcc -c -arch=sm_70  discretizationKernel.cu -o discretizationKernel.o
wmake
