wclean
#source envAMGX 
nvcc -c -arch=sm_90  discretizationKernel.cu -o discretizationKernel.o
wmake
