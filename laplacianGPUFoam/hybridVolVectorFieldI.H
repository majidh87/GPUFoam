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

inline Foam::hybridVolVectorField<3>::hybridVolVectorField() :
isInitialized_(false)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::hybridVolVectorField::init(fvMesh& mesh)
{

    /*
    const Foam::Field& internal = U.internalField();
    
    internalField_.allocate(internal.size(),true);

    printf("sizeof scalar : %d " , sizeof(Foam::scalar));
    for (label i = 0; i < internal.size(); i++)
    {
      const Vector& vel = internal[i];
      printf("sizeof Vector : %d " , sizeof(internal[i]));
      double ux = vel.x();
      double uy = vel.y();
      double uz = vel.z();
      // استفاده یا ذخیره‌سازی داده‌ها
    }    
    
    const fvPatchVectorField& boundary = U.boundaryField();
    
    boundaryField_.allocate(boundary.size(),true);

    for (label patchI = 0; patchI < boundary.size(); patchI++)
    {
      const fvPatchVectorField& patchField = boundary[patchI];
      for (label faceI = 0; faceI < patchField.size(); faceI++)
      {
          const Vector& vel = patchField[faceI];
          double ux = vel.x();
          double uy = vel.y();
          double uz = vel.z();
          // استفاده یا ذخیره‌سازی داده‌ها
      }
    }
    */
    /////////////////////////////
    //const Foam::Field<Foam::vector>& internalField = U.internalField();
    //const Foam::vector* dataPtr = internalField.cdata();
    //////////////////////////////////////////////////
    // Assume U is a volVectorField
    
    isInitialized_ = true;            
}

void Foam::hybridVolVectorField::handle(fvMesh& mesh,volVectorField& U)
{
    if (!isInitialized_)
    {
        init(mesh);
    }
    printf("U.size:%d\n",U.size());
    const Foam::volScalarField& Ux = U.component(X);
    components_[X].handle(mesh,Ux);
    const Foam::volScalarField& Uy = U.component(Y);
    components_[Y].handle(mesh,Uy);
    const Foam::volScalarField& Ux = U.component(Z);
    components_[Z].handle(mesh,Uz);

    // Access the internalField of the x-component scalar field
    //const Foam::scalar* xDataPtr = Ux.internalField().cdata();

}

Foam::hybridVolVectorField::~hybridVolVectorField()
{
    isInitialized_ = false;
    /*forall(n,nComponents)
    {
        components_[n].
    }*/
}
