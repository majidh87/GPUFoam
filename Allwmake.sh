wclean
#source envAMGX 
nvcc -c -arch=$CUDA_ARC  discretizationKernel.cu -o discretizationKernel.o
wmake
