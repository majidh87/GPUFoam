/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2019-2022 OpenCFD Ltd.
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

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class T>
inline Foam::GPUMatrix<T>::GPUMatrix()
{
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class T>
void Foam::GPUMatrix<T>::copy(HybridMatrix<T>& hostMatrix)
{
    deviceList.allocate(hostMatrix.rows(),true);

    deviceIntermediate.allocate_rows(hostMatrix.rows(),STORE_ON_HOST_GPU);

    for(int i=0; i<hostMatrix.rows(); i++)
    {

        deviceIntermediate.allocate_col(i,hostMatrix.cols(i));


        deviceIntermediate.copy_col(i,hostMatrix.Data()[i]);
    
    }

    deviceList.copy(deviceIntermediate.Data());

}
template<class T>
void Foam::GPUMatrix<T>::copy(HybridArray<T *>& hostBuffer,HybridArray<Foam::label>& hostSize)
{
    deviceList.allocate(hostBuffer.size(),true);

    deviceIntermediate.allocate_rows(hostBuffer.size(),STORE_ON_HOST_GPU);

    for(int i=0; i<hostBuffer.size(); i++)
    {

        deviceIntermediate.allocate_col(i,hostSize[i]);


        deviceIntermediate.copy_col(i,hostBuffer[i]);
    
    }

    deviceList.copy(deviceIntermediate.Data());

}

template<class T>
void Foam::GPUMatrix<T>::deallocate()
{
    deviceIntermediate.deallocate();
    deviceList.deallocate();
}


template<class T>
Foam::GPUMatrix<T>::~GPUMatrix()
{
    deallocate();
}

