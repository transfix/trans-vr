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



using namespace std;

class VolMagickOpStatus : public VolMagick::VoxelOperationStatusMessenger
{
public:
  void start(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 numSteps) const
  {
    _numSteps = numSteps;
  }

  void step(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 curStep) const
  {
    const char *opStrings[] = { "CalculatingMinMax", "CalculatingMin", "CalculatingMax",
				"SubvolumeExtraction", "Fill", "Map", "Resize", "Composite",
				"BilateralFilter", "ContrastEnhancement"};

    fprintf(stderr,"%s: %5.2f %%\r",opStrings[op],(((float)curStep)/((float)((int)(_numSteps-1))))*100.0);
  }

  void end(const VolMagick::Voxels *vox, Operation op) const
  {
    printf("\n");
  }

private:
  mutable VolMagick::uint64 _numSteps;
};

typedef boost::tuple<double, double, double> Color;

int main(int argc, char **argv)
{
  if(argc < 4)
    {
      cerr << "Usage: volSetMin <volume> <min> <output>" << endl 
	   << "This program sets minimum value of a volume." << endl;

      return 1;
    }

  try
    {
      VolMagick::Volume inputVol;
      VolMagick::Volume outputVol;
   
      VolMagick::readVolumeFile(inputVol,argv[1]); ///first argument is input volume

   
      double minval = atof(argv[2]);

      int xsize, ysize, zsize;
      xsize = inputVol.XDim();
      ysize = inputVol.YDim();
      zsize = inputVol.ZDim();

      cout << "Volume Size: " << xsize << " x " << ysize << " x " << zsize << endl;
      int count = 0;
      
      outputVol.voxelType(inputVol.voxelType());
      outputVol.dimension(inputVol.dimension());
      outputVol.boundingBox(inputVol.boundingBox());
      

      for(unsigned int i=0; i<xsize; i++) {
	for(unsigned int j=0; j<ysize; j++) {
	  for(unsigned int k=0; k<zsize; k++)
	    {
	      if(inputVol(i,j,k) < minval) {

		count++;
		outputVol(i,j,k,minval);

	      } else {
		outputVol(i,j,k,inputVol(i,j,k));
	      }
		 
	    }    
	}
      }


      VolMagick::createVolumeFile(argv[3],outputVol);
      VolMagick::writeVolumeFile(outputVol,argv[3]);

              
      cout << "New Minimum: " << minval << endl;
      cout << "Voxels changed: " << count << endl;

    }
  catch(VolMagick::Exception &e)
    {
      cerr << e.what() << endl;
    }
  catch(std::exception &e)
    {
      cerr << e.what() << endl;
    }
  
  return 0;

}
