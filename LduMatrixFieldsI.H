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

Class
    Foam::LduMatrixFields

Description
    A template class for dynamic array allocation of output LDU matrix fields. 

SourceFiles
    LduMatrixFieldsI.H
\*---------------------------------------------------------------------------*/
#ifndef LduMatrixFieldsI_H
#define LduMatrixFieldsI_H

#include "UIndirectList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

// * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * //

LduMatrixFields::LduMatrixFields()
:
    numCells_(0),
    numInternalFaces_(0),
    onGPU_(false)
    {}

LduMatrixFields::LduMatrixFields(const label numCells, const label numInternalFaces , const Switch onGPU)
:
    numCells_(numCells),
    numInternalFaces_(numInternalFaces),
    onGPU_(onGPU)
{

}

LduMatrixFields::~LduMatrixFields()
{
    deallocate();
}

void LduMatrixFields::allocate()
{
    deallocate();
    diagonal.allocate(numCells_,onGPU_);
    source.allocate(numCells_,onGPU_);
    lower.allocate(numInternalFaces_,onGPU_);
    upper.allocate(numInternalFaces_,onGPU_);
}

void LduMatrixFields::deallocate()
{
    diagonal.deallocate();
    source.deallocate();
    upper.deallocate();
    lower.deallocate();

}

void LduMatrixFields::init(const label numCells, const label numInternalFaces , const Switch onGPU)

{
    numCells_ = numCells;
    numInternalFaces_ = numInternalFaces;
    onGPU_ = onGPU;
	allocate();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif
