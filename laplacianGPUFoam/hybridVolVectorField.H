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
    Foam::hybridVolVectorField

Description
    A template class for hybrid vector field on GPU 

SourceFiles
    hybridVolVectorField.H
    hybridVolVectorFieldI.H

\*---------------------------------------------------------------------------*/
#ifndef Foam_hybridVolVectorField_H
#define Foam_hybridVolVectorField_H

#include "HybridVectorField.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class hybridVolScalarField Declaration
\*---------------------------------------------------------------------------*/
template<direction Ncmpts>
class hybridVolVectorField
{
public:
    //- Constructors
    hybridVolVectorField();
    
    //- Deconstructors
    ~hybridVolVectorField();

    // Initialize the class with mesh data
    void init(fvMesh& mesh);

    // Handle function to process mesh and field data
    void handle(fvMesh& mesh,
        volVectorField& U);

    //- The number of elements in the VectorSpace = Ncmpts.
    static constexpr direction size() noexcept
    {
        return Ncmpts;
    }

    //- Component labeling enumeration
    enum components { X, Y, Z };

    //- Return const pointer to the first data element
    inline const hybridVolScalarField& cdata(const direction c) const noexcept
    {
        if(c<nComponents)
        {
            return components_[c];
        }
        else
        {
            FatalErrorIn("hybridVolVectorField<T>::cdata(const direction c)")
                << "Invalid component: " << c
                << abort(FatalError);            
        }
    }

private:

    // Flag to check if initialization is done
    Foam::Switch isInitialized_;

    // internal Field
    hybridVolScalarField components_[Ncmpts];
    
    //- Number of components in this vector space
    static constexpr direction nComponents = Ncmpts;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "hybridVolVectorFieldI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
