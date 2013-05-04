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
  if(argc < 3)
    {
      std:: cerr << 
	"Usage: " << argv[0] <<  "<input file> <output file>\n";
      return 1;
    }

  try
    {
      VolMagick::Volume inputVol;

      VolMagick::Volume outputVol;

      VolMagick::readVolumeFile(inputVol,argv[1]); ///first argument is input volume
      
      
      outputVol.voxelType(inputVol.voxelType());
      outputVol.dimension(inputVol.dimension());
      outputVol.boundingBox(inputVol.boundingBox());
      	
      for( int kz = 0; kz<inputVol.ZDim(); kz++)
	{
	  std::cout<<kz<<"..";
	    for( int jy = 0; jy<inputVol.YDim(); jy++)
	      for( int ix = 0; ix<inputVol.XDim(); ix++)
		{
		  
		  if (inputVol(ix,jy,kz) > 0) {
		    float val = ((float)(rand()%10000))/10000.0-0.33;
		    while (val > -0.1 && val < 0) {
		      val = ((float)(rand()%10000))/10000.0-0.33;
		    }
		    outputVol(ix,jy,kz, val );
		  } else {
		    outputVol(ix,jy,kz, 0.05);
		  }
		}
	}	
      

      VolMagick::createVolumeFile(argv[2], outputVol);

      VolMagick::writeVolumeFile(outputVol, argv[2]);

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
