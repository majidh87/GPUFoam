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
template<class Cmpt, Foam::direction Ncmpts>
inline Foam::HybridVectorField<Cmpt, Ncmpts>::HybridVectorField()
{
}

template<class Cmpt, Foam::direction Ncmpts>
Foam::HybridVectorField<Cmpt, Ncmpts>::HybridVectorField(const label size, const bool onGPU)
{
    data_.allocate(size, onGPU);
}

template<class Cmpt, Foam::direction Ncmpts>
Foam::HybridVectorField<Cmpt, Ncmpts>::~HybridVectorField()
{
    data_.deallocate();
}


template<class Cmpt, Foam::direction Ncmpts>
void Foam::HybridVectorField<Cmpt, Ncmpts>::allocate(const label size, const bool onGPU)
{
    if (size <= 0)
    {
        FatalErrorIn("HybridVectorField<Cmpt, Ncmpts>::allocate(const label, const bool)")
            << "Invalid size: " << size
            << abort(FatalError);
    }
    data_.allocate(size, onGPU);

}

template<class Cmpt, Foam::direction Ncmpts>
void Foam::HybridVectorField<Cmpt, Ncmpts>::deallocate()
{
    data_.deallocate();
}
// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
template<class Cmpt, Foam::direction Ncmpts>
label Foam::HybridVectorField<Cmpt, Ncmpts>::size()	const
{
    return data_.size();
}

template<class Cmpt, Foam::direction Ncmpts>
Foam::Switch Foam::HybridVectorField<Cmpt, Ncmpts>::onGPU()	const
{
    return data_.onGPU();
}

template<class Cmpt, Foam::direction Ncmpts>
Foam::MiniVectorSpace<Cmpt,Ncmpts>* Foam::HybridVectorField<Cmpt, Ncmpts>::Data() const
{
    return  data_.Data();
}

template<class Cmpt, Foam::direction Ncmpts>
const Cmpt& Foam::HybridVectorField<Cmpt, Ncmpts>::x(direction index) const noexcept
{
    #ifdef FULLDEBUG
    if (index >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[index].cdata()[X];
}

template<class Cmpt, Foam::direction Ncmpts>
Cmpt& Foam::HybridVectorField<Cmpt, Ncmpts>::x(direction index) noexcept
{
    #ifdef FULLDEBUG
    if (index >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[index].data()[X];
}

template<class Cmpt, Foam::direction Ncmpts>
const Cmpt& Foam::HybridVectorField<Cmpt, Ncmpts>::y(direction index) const noexcept
{
    #ifdef FULLDEBUG
    if (index >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[index].cdata()[Y];
}

template<class Cmpt, Foam::direction Ncmpts>
Cmpt& Foam::HybridVectorField<Cmpt, Ncmpts>::y(direction index) noexcept
{
    #ifdef FULLDEBUG
    if (index >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[index].data()[Y];
}

template<class Cmpt, Foam::direction Ncmpts>
const Cmpt& Foam::HybridVectorField<Cmpt, Ncmpts>::z(direction index)  const noexcept
{
    #ifdef FULLDEBUG
    if (index >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[index].cdata()[Z];
}

template<class Cmpt, Foam::direction Ncmpts>
Cmpt& Foam::HybridVectorField<Cmpt, Ncmpts>::z(direction index)  noexcept
{
    #ifdef FULLDEBUG
    if (index >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[index].data()[Z];
}

// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template<class Cmpt, Foam::direction Ncmpts>
inline const Foam::MiniVectorSpace<Cmpt,Ncmpts>& Foam::HybridVectorField<Cmpt, Ncmpts>::operator[]
(
    const direction d
) const
{
    #ifdef FULLDEBUG
    if (d >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[d];
}


template<class Cmpt, Foam::direction Ncmpts>
inline Foam::MiniVectorSpace<Cmpt,Ncmpts>& Foam::HybridVectorField<Cmpt, Ncmpts>::operator[]
(
    const direction d
)
{
    #ifdef FULLDEBUG
    if (d >= data_.size())
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return data_[d];
}
