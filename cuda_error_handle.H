/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/
#include "cuda_runtime_api.h"

#ifndef CUDA__ERROR_HANDLE_H
#define CUDA_ERROR_HANDLE_H

#define checkCudaErrors(call)                                      \
    {                                                             \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

 // Macro to wrap CUDA calls with error checking
#define _CUDA(x) checkCudaErrors(x)

#endif
