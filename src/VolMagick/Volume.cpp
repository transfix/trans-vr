/*
  Copyright 2007-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolMagick.

  VolMagick is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolMagick is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: Volume.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/Volume.h>
#include <VolMagick/VoxelOperationStatusMessenger.h>
#include <VolMagick/Utility.h>

#include <CVC/App.h>

#include <boost/format.hpp>

namespace VolMagick
{
  Volume& Volume::copy(const Volume& vol)
  {
    if(this == &vol)
      return *this;

    Voxels::copy(vol);
    boundingBox(vol.boundingBox());
    desc(vol.desc());
    return *this;
  }

  Volume& Volume::sub(uint64 off_x, uint64 off_y, uint64 off_z,
		      const Dimension& subvoldim
#ifdef _MSC_VER 
			, int brain_damage //avoiding VC++ error C2555
#endif
			  )
  {
    Voxels::sub(off_x,off_y,off_z,subvoldim);
    boundingBox().setMin(XMin()+XSpan()*off_x,
			 YMin()+YSpan()*off_y,
			 ZMin()+ZSpan()*off_z);
    boundingBox().setMax(XMin()+XSpan()*(off_x+XDim()-1),
			 YMin()+YSpan()*(off_y+YDim()-1),
			 ZMin()+YSpan()*(off_z+ZDim()-1));
    return *this;
  }

#if 0
  Volume& Volume::compositeObj(const Volume& compVol, double off_x, double off_y, double off_z, const CompositeFunction& func)
  {
    return *this; // finish me!
  }
#endif

  Volume& Volume::sub(const BoundingBox& subvolbox)
  {
    //keep the span of the subvolume as close as possible to the original volume
    if(!subvolbox.isWithin(boundingBox()))
      throw SubVolumeOutOfBounds("The subvolume bounding box must be within the file's bounding box.");
    uint64 off_x = uint64((subvolbox.minx - XMin())/XSpan());
    uint64 off_y = uint64((subvolbox.miny - YMin())/YSpan());
    uint64 off_z = uint64((subvolbox.minz - ZMin())/ZSpan());
    Dimension dim;
    dim[0] = uint64((subvolbox.maxx - subvolbox.minx)/XSpan())+1;
    dim[1] = uint64((subvolbox.maxy - subvolbox.miny)/YSpan())+1;
    dim[2] = uint64((subvolbox.maxz - subvolbox.minz)/ZSpan())+1;
    for(int i = 0; i < 3; i++) if(dim[i] == 0) dim[i]=1;
    if(dim[0] + off_x > XDim()) dim[0] = XDim() - off_x;
    if(dim[1] + off_y > YDim()) dim[1] = YDim() - off_y;
    if(dim[2] + off_z > ZDim()) dim[2] = ZDim() - off_z;
    sub(off_x,off_y,off_z,dim);

    //just force the bounding box for now.. this might lead to aliasing errors
    boundingBox(subvolbox);    

    return *this;
  }

  Volume& Volume::sub(const BoundingBox& subvolbox, const Dimension& subvoldim)
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    if(!subvolbox.isWithin(boundingBox()))
      throw SubVolumeOutOfBounds("Subvolume bounding box must be within the bounding box of the original volume.");

    Volume subvol(subvoldim,
		  voxelType(),
		  subvolbox);

    subvol.desc(desc());

    if(_vosm) _vosm->start(this, VoxelOperationStatusMessenger::SubvolumeExtraction, subvoldim[2]);

    uint64 i,j,k;
    for(k=0; k<subvol.ZDim(); k++)
      {
	for(j=0; j<subvol.YDim(); j++)
	  for(i=0; i<subvol.XDim(); i++)
	    {
	      try
		{
		  subvol(i,j,k, this->interpolate(subvolbox.minx + i*subvol.XSpan(),
						  subvolbox.miny + j*subvol.YSpan(),
						  subvolbox.minz + k*subvol.ZSpan()));
		}
	      catch(IndexOutOfBounds& e)
		{
		  //Some really slight floating point errors pop up sometimes...
		  //		  fprintf(stderr,"Volume::sub(): caught IndexOutOfBounds exception (%s)."
		  //			  " Possible floating point error\n",e.what());
		}
	    }
	if(_vosm) _vosm->step(this, VoxelOperationStatusMessenger::SubvolumeExtraction, k);
      }

    if(_vosm) _vosm->end(this, VoxelOperationStatusMessenger::SubvolumeExtraction);

    copy(subvol);
    return *this;
  }

  double Volume::interpolate(double obj_x, double obj_y, double obj_z) const
  {
    using namespace boost;

    //double inSpaceX, inSpaceY, inSpaceZ;
    double val[8];
    uint64 resXIndex = 0, resYIndex = 0, resZIndex = 0;
    uint64 ValIndex[8];
    double xPosition = 0, yPosition = 0, zPosition = 0;
    double xRes = 1, yRes = 1, zRes = 1;
    /*
      uint64 i,j,k;
      double x,y,z;
    */

    if(!boundingBox().contains(obj_x,obj_y,obj_z)) 
      throw IndexOutOfBounds(str(format("Coordinates are outside of bounding box :: "
                                        "bbox (%f,%f,%f),(%f,%f,%f) coord (%f,%f,%f)")
                                 % boundingBox().minx
                                 % boundingBox().miny
                                 % boundingBox().minz
                                 % boundingBox().maxx
                                 % boundingBox().maxy
                                 % boundingBox().maxz
                                 % obj_x
                                 % obj_y
                                 % obj_z));

    resXIndex = uint64((obj_x - XMin())/XSpan());
    resYIndex = uint64((obj_y - YMin())/YSpan());
    resZIndex = uint64((obj_z - ZMin())/ZSpan());

    // find index to get eight voxel values
    ValIndex[0] = resZIndex*dimension()[0]*dimension()[1] + resYIndex*dimension()[0] + resXIndex;
    ValIndex[1] = ValIndex[0] + 1;
    ValIndex[2] = resZIndex*dimension()[0]*dimension()[1] + (resYIndex+1)*dimension()[0] + resXIndex;
    ValIndex[3] = ValIndex[2] + 1;
    ValIndex[4] = (resZIndex+1)*dimension()[0]*dimension()[1] + resYIndex*dimension()[0] + resXIndex;
    ValIndex[5] = ValIndex[4] + 1;
    ValIndex[6] = (resZIndex+1)*dimension()[0]*dimension()[1] + (resYIndex+1)*dimension()[0] + resXIndex;
    ValIndex[7] = ValIndex[6] + 1;
    
    if(resXIndex>=dimension()[0]-1)
      {
	ValIndex[1] = ValIndex[0];
	ValIndex[3] = ValIndex[2];
	ValIndex[5] = ValIndex[4];
	ValIndex[7] = ValIndex[6];
      }
    if(resYIndex>=dimension()[1]-1)
      {
	ValIndex[2] = ValIndex[0];
	ValIndex[3] = ValIndex[1];
	ValIndex[6] = ValIndex[4];
	ValIndex[7] = ValIndex[5];
      }
    if(resZIndex>=dimension()[2]-1) 
      {
	ValIndex[4] = ValIndex[0];
	ValIndex[5] = ValIndex[1];
	ValIndex[6] = ValIndex[2];
	ValIndex[7] = ValIndex[3];
      }
    
    for(int Index = 0; Index < 8; Index++) 
      val[Index] = (*this)(ValIndex[Index]);

    xPosition = obj_x - (double(resXIndex)*XSpan() + XMin());
    yPosition = obj_y - (double(resYIndex)*YSpan() + YMin());
    zPosition = obj_z - (double(resZIndex)*ZSpan() + ZMin());

    xRes = XSpan();
    yRes = YSpan();
    zRes = ZSpan();

    return getTriVal(val, xPosition, yPosition, zPosition, xRes, yRes, zRes);
  }

  Volume& Volume::combineWith(const Volume& vol, const Dimension& dim)
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    BoundingBox combbox = boundingBox() + vol.boundingBox();
    Volume combvol(dim,voxelType(),combbox);

    if(_vosm) _vosm->start(this, VoxelOperationStatusMessenger::CombineWith, dim[2]);

    for(uint64 k = 0; k < combvol.ZDim(); k++)
      {
	for(uint64 j = 0; j < combvol.YDim(); j++)
	  for(uint64 i = 0; i < combvol.XDim(); i++)
	    {
	      double x = i*combvol.XSpan() + XMin();
	      double y = j*combvol.YSpan() + YMin();
	      double z = k*combvol.ZSpan() + ZMin();
	      
	      //TODO: consider using composite algoritms or average these or something...
	      if(vol.boundingBox().contains(x,y,z))
		combvol(i,j,k, vol.interpolate(x,y,z));
	      else if(boundingBox().contains(x,y,z))
		combvol(i,j,k, interpolate(x,y,z));
	    }

        cvcapp.threadProgress(float(k)/float(dim[2]));
	if(_vosm) _vosm->step(this, VoxelOperationStatusMessenger::CombineWith, k);
      }

    (*this) = combvol;

    cvcapp.threadProgress(1.0f);
    if(_vosm) _vosm->end(this, VoxelOperationStatusMessenger::CombineWith);

    return (*this);
  }

  Volume& Volume::combineWith(const Volume& vol)
  {
    return combineWith(vol,dimension());
  }
}
