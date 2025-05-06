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
inline Foam::MiniVectorSpace<Cmpt, Ncmpts>::MiniVectorSpace(const Foam::zero)
{
    forAll(v_, i)
    {
        // Access element with array[i]
        v_[i] = 0;
    }
    //VectorSpaceOps<Ncmpts,0>::eqOpS(*this, Zero, eqOp<Cmpt>());
}


template<class Cmpt, Foam::direction Ncmpts>
inline Foam::MiniVectorSpace<Cmpt, Ncmpts>::MiniVectorSpace
(
    const MiniVectorSpace<Cmpt, Ncmpts>& vs
)
{
    forAll(v_, i)
    {
        v_[i] = vs[i];
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Cmpt, Foam::direction Ncmpts>
inline const Cmpt& Foam::MiniVectorSpace<Cmpt, Ncmpts>::component
(
    const direction d
) const
{
    #ifdef FULLDEBUG
    if (d >= Ncmpts)
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return v_[d];
}


template<class Cmpt, Foam::direction Ncmpts>
inline Cmpt& Foam::MiniVectorSpace<Cmpt, Ncmpts>::component
(
    const direction d
)
{
    #ifdef FULLDEBUG
    if (d >= Ncmpts)
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return v_[d];
}

template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::component
(
    Cmpt& c,
    const direction d
) const
{
    #ifdef FULLDEBUG
    if (d >= Ncmpts)
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    c = v_[d];
}


template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::replace
(
    const direction d,
    const Cmpt& c
)
{
    #ifdef FULLDEBUG
    if (d >= Ncmpts)
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    v_[d] = c;
}

// * * * * * * * * * * * * * * * * Iterator  * * * * * * * * * * * * * * * * //

template<class Cmpt, Foam::direction Ncmpts>
inline Cmpt* Foam::MiniVectorSpace<Cmpt, Ncmpts>::data() noexcept
{
    return v_;
}


template<class Cmpt, Foam::direction Ncmpts>
inline const Cmpt* Foam::MiniVectorSpace<Cmpt, Ncmpts>::cdata() const noexcept
{
    return v_;
}

// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template<class Cmpt, Foam::direction Ncmpts>
inline const Cmpt& Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator[]
(
    const direction d
) const
{
    #ifdef FULLDEBUG
    if (d >= Ncmpts)
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return v_[d];
}


template<class Cmpt, Foam::direction Ncmpts>
inline Cmpt& Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator[]
(
    const direction d
)
{
    #ifdef FULLDEBUG
    if (d >= Ncmpts)
    {
        FatalErrorInFunction
            << "index out of range"
            << abort(FatalError);
    }
    #endif

    return v_[d];
}

template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator=
(
    const MiniVectorSpace<Cmpt, Ncmpts>& vs
)
{
    forAll(v_, i)
    {
        // Access element with array[i]
        v_[i] = vs[i];
    }

    //VectorSpaceOps<Ncmpts,0>::eqOp(*this, vs, eqOp<Cmpt>());
}


template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator+=
(
    const MiniVectorSpace<Cmpt, Ncmpts>& vs
)
{
    //VectorSpaceOps<Ncmpts,0>::eqOp(*this, vs, plusEqOp<Cmpt>());
    forAll(v_, i)
    {
        // Access element with array[i]
        v_[i] += vs[i];
    }
}


template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator-=
(
    const MiniVectorSpace<Cmpt, Ncmpts>& vs
)
{
    //VectorSpaceOps<Ncmpts,0>::eqOp(*this, vs, minusEqOp<Cmpt>());
    forAll(v_, i)
    {
        // Access element with array[i]
        v_[i] -= vs[i];
    }
}


template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator=(const Foam::zero)
{
    //VectorSpaceOps<Ncmpts,0>::eqOpS(*this, 0, eqOp<Cmpt>());
    forAll(v_, i)
    {
        // Access element with array[i]
        v_[i] = 0;
    }
}


template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator*=
(
    const scalar s
)
{
    //VectorSpaceOps<Ncmpts,0>::eqOpS(*this, s, multiplyEqOp2<Cmpt, scalar>());
    forAll(v_, i)
    {
        // Access element with array[i]
        v_[i] *= s;
    }
}


template<class Cmpt, Foam::direction Ncmpts>
inline void Foam::MiniVectorSpace<Cmpt, Ncmpts>::operator/=
(
    const scalar s
)
{
    forAll(v_, i)
    {
        // Access element with array[i]
        v_[i] /= s;
    }
    //VectorSpaceOps<Ncmpts,0>::eqOpS(*this, s, divideEqOp2<Cmpt, scalar>());
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
