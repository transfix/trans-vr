/*
  Copyright 2005-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolUtils.

  VolUtils is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolUtils is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>

int main(int argc, char **argv)
{
  if(argc < 4)
    {
      std:: cerr << 
	"Usage: " << argv[0] << 

	
	"<input vol 1> <input vol 2> <output vol> \n";

      return 1;
    }

  try
    {
      VolMagick::Volume inputVol;

      VolMagick::Volume inputVol2;
      
      VolMagick::Volume outputVol;

      VolMagick::VolumeFileInfo volinfo;
      volinfo.read(argv[1]);


      VolMagick::readVolumeFile(inputVol,argv[1]); ///first argument is input volume
      
      VolMagick::readVolumeFile(inputVol2, argv[2]); /// second argument is mask volume
      

      VolMagick::VolumeFileInfo volinfo1;
      volinfo1.read(argv[1]);
      std::cout << volinfo1.filename() << ":" <<std::endl;

      VolMagick::VolumeFileInfo volinfo2;
      volinfo2.read(argv[2]);
      std::cout << volinfo2.filename() << ":" <<std::endl;
      
      std::cout<<"minVol1 , maxVol1: "<<volinfo1.min()<<" "<<volinfo1.max()<<std::endl;;

      std::cout<<"minVol2 , maxVol2: "<<volinfo2.min()<<" "<<volinfo2.max()<<std::endl;

           

      outputVol.voxelType(inputVol.voxelType());
      outputVol.dimension(inputVol.dimension());
      outputVol.boundingBox(inputVol.boundingBox());
      
      
      std::cout<<"voxeltype "<<inputVol.voxelType()<<std::endl;
      

      double min=100000000.0;
      double max=-100000000.0;
      
      int dx,dy,dz;
      dx = (inputVol.XDim() - inputVol2.XDim())/2;
      dy = (inputVol.YDim() - inputVol2.YDim())/2;
      dz = (inputVol.ZDim() - inputVol2.ZDim())/2;

      for( int kz = 0; kz<inputVol.ZDim(); kz++)
	{
	  std::cout<<kz<<"..";
	  int kz1 = kz - dz;
	  for( int jy = 0; jy<inputVol.YDim(); jy++) {
	      int jy1 = jy - dz;
	      for( int ix = 0; ix<inputVol.XDim(); ix++)
		{
		  int ix1 = ix - dz;
		  
		  if (kz1 >= 0 && kz1 < inputVol2.ZDim() && 
		      jy1 >= 0 && jy1 < inputVol2.YDim() && 
		      ix1 >= 0 && ix1 < inputVol2.XDim() ) {

		    outputVol(ix,jy,kz, inputVol(ix,jy,kz)+inputVol2(ix1,jy1,kz1));
		    
		    if (inputVol(ix,jy,kz)+inputVol2(ix1,jy1,kz1) < min) {
		      min = inputVol(ix,jy,kz)+inputVol2(ix1,jy1,kz1);
		    }
		    
		    if (inputVol(ix,jy,kz)+inputVol2(ix1,jy1,kz1) >max) {
		      max = inputVol(ix,jy,kz)+inputVol2(ix1,jy1,kz1);
		    }
		  } else {
		    outputVol(ix,jy,kz, inputVol(ix,jy,kz));
		    
		    if (inputVol(ix,jy,kz) < min) {
		      min = inputVol(ix,jy,kz);
		    }
		    
		    if (inputVol(ix,jy,kz) >max) {
		      max = inputVol(ix,jy,kz);
		    }
		  }


		}
	  }
	}	
      

      std::cout << std::endl << "New Min/Max " << min << " " << max << std::endl;

      VolMagick::createVolumeFile(argv[3],volinfo);

      VolMagick::writeVolumeFile(outputVol, argv[3]);

    }

  catch(VolMagick::Exception &e)
    {
      std:: cerr << e.what() << std::endl;
    }
  catch(std::exception &e)
    {
      std::cerr << e.what() << std::endl;
    }

  return 0;
}
