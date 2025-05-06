/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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
    Foam::MiniVectorSpace

Description
    Templated vector space.

    Template arguments are the Form the vector space will be used to create,
    the type of the elements and the number of elements.

SourceFiles
    MiniVectorSpaceI.H

\*---------------------------------------------------------------------------*/

#ifndef Foam_MiniVectorSpaceUser_H
#define Foam_MiniVectorSpaceUser_H

#include "direction.H"
#include "scalar.H"
#include "word.H"
#include "zero.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class MiniVectorSpace Declaration
\*---------------------------------------------------------------------------*/

template<class Cmpt, direction Ncmpts>
class MiniVectorSpace
{
public:

    //- The components of this vector space
    Cmpt v_[Ncmpts];

        // Static Constants

        //- Dimensionality of space
        static constexpr direction dim = 3;

        //- Number of components in this vector space
        static constexpr direction nComponents = Ncmpts;


        // MiniVectorSpace currently defaults to a column-vector
        // This will be removed when column-vector is introduced
        // as a specialization
        static constexpr direction mRows = Ncmpts;
        static constexpr direction nCols = 1;

    // Generated Methods

        //- Default construct
        MiniVectorSpace() = default;


    // Constructors

        //- Construct initialized to zero
        inline MiniVectorSpace(const Foam::zero);

        //- Copy construct
        inline MiniVectorSpace(const MiniVectorSpace<Cmpt, Ncmpts>& vs);


    // Member Functions

        //- The number of elements in the VectorSpace = Ncmpts.
        static constexpr direction size() noexcept
        {
            return Ncmpts;
        }

        inline const Cmpt& component(const direction) const;
        inline Cmpt& component(const direction);

        inline void component(Cmpt&, const direction) const;
        inline void replace(const direction, const Cmpt&);

        //- Return const pointer to the first data element
        inline const Cmpt* cdata() const noexcept;

        //- Return pointer to the first data element
        inline Cmpt* data() noexcept;

    // Member Operators

        inline const Cmpt& operator[](const direction) const;
        inline Cmpt& operator[](const direction);

        inline void operator=(const MiniVectorSpace<Cmpt, Ncmpts>&);
        inline void operator+=(const MiniVectorSpace<Cmpt, Ncmpts>&);
        inline void operator-=(const MiniVectorSpace<Cmpt, Ncmpts>&);

        inline void operator=(const Foam::zero);
        inline void operator*=(const scalar);
        inline void operator/=(const scalar);

    // Iterators

        //- Random access iterator for traversing VectorSpace
        typedef Cmpt* iterator;

        //- Random access iterator for traversing VectorSpace
        typedef const Cmpt* const_iterator;


    // Random access iterator (non-const)

        //- Return an iterator to begin of VectorSpace
        inline iterator begin() noexcept;

        //- Return an iterator to end of VectorSpace
        inline iterator end() noexcept;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "MiniVectorSpaceI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
