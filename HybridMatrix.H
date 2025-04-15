/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2025 OpenFOAM Foundation
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
    Foam::HybridMatrix

Description
    A template class for dynamic matrix allocation supporting both host and GPU 
    memory using CUDA. The class allows for variable column sizes per row.

SourceFiles
    HybridMatrix.H

\*---------------------------------------------------------------------------*/

#ifndef Foam_HybridMatrix_H
#define Foam_HybridMatrix_H

#include "label.H"          //- OpenFOAM label type for indexing
#include "error.H"          //- OpenFOAM error handling macros
#include <cstdlib>          //- For malloc and free
#ifdef CUDA_USE
#include "cuda_error_handle.H"
#endif

#include "labelList.H"

namespace Foam {

//- Enum for determine store type
enum STORE_TYPE
{
    UNKNOWN_STATE,
    STORE_ON_HOST,
    STORE_ON_GPU,
    STORE_ON_HOST_GPU
};
//- Template class for dynamic matrix allocation supporting host/GPU memory.
template<typename T>
class HybridMatrix
{
private:

    //- Pointer to data (host or GPU memory)
    T** data_;

    //- Number of rows in the matrix
    label rows_;

    //- List storing column counts per row
    labelList cols_;

    //- Flag indicating whether data is stored on GPU (true) or host (false)
    STORE_TYPE storeType_;

public:

    // Constructors

        //- Default constructor
        HybridMatrix();

        //- Construct with row count and storage location flag (host or GPU)
        HybridMatrix(const label rows, const STORE_TYPE storeType);

        //- No copy assignment from plain pointer (uncontrolled access)
        void operator=(T** p) = delete;

        //- No move assignment disguised as a copy assignment
        void operator=(const HybridMatrix<T>&) = delete;

        //- Deleted copy constructor to prevent accidental copying
        HybridMatrix(const HybridMatrix&) = delete;

        //- Destructor to release allocated memory
        ~HybridMatrix();


    // Member Functions

        //- Allocate memory for the matrix using specified column sizes
        void allocate(const labelList& colsSize);

        //- Allocate memory for the rows of matrix 
        void allocate_rows(const label rows, const STORE_TYPE storeType);

        //- Allocate memory for the cols of special row of matrix 
        void allocate_col(const label row, const label colSize);
        
        //- Deallocate memory for the matrix
        void deallocate();

    	//- copy src buffer 
	    void copy(const T** src);

        //- copy src to row column
        void copy_col(const label row, const T* src);
        // Accessors

            //- Return number of rows in the matrix
            label rows() const noexcept;

            //- Return number of columns in a specific row
            label cols(const label row) const;

            //- Check if data is stored on GPU or host
            STORE_TYPE onDevice() const noexcept;

            //- Return matrix data
            T** Data() const;
        
        // Operators

            //- Access element at (row, col) for host-stored matrices only
            T& operator()(const label row, const label col);

            //- Const access element at (row, col) for host-stored matrices only
            const T& operator()(const label row, const label col) const;

            //- set an special element of matrix
            void set(const label row, const label col, const T value);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#include "HybridMatrixI.H"


#endif

// ************************************************************************* //
