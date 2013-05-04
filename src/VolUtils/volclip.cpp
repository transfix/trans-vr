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
  if(argc < 3)
    {
      cerr << 
	"Usage: " << argv[0] << 
	//" <input rawv with >=4 variables> <red var> <green var> <blue var> <alpha var> <timestep> <output rawiv file prefix>" << endl;
	" <input raw volume> <mask volume> <output volume>" << endl;
	
	/// first volume is the segmented mask, second volume is the original unprocessed data. Last volume is the ouput.

      return 1;
    }

  try
    {
      VolMagick::Volume inputVol;



      VolMagick::Volume outputVol;   




      //      for(unsigned int i = 0; i<argc-2; i++)
      VolMagick::readVolumeFile(inputVol,argv[1]); ///first argument is input volume

  
      outputVol.voxelType(inputVol.voxelType());
      outputVol.dimension(inputVol.dimension());
      outputVol.boundingBox(inputVol.boundingBox());

      int Zlbound = 61;
      int Zubound = 133;

      cout<<"Zlbound: "<<Zlbound<<endl;
      
      cout<<"Zubound: "<<Zubound<<endl;

      
      cout<<"voxeltype "<<inputVol.voxelType()<<endl;

      cout<<"begin volume copy"<<endl;
    


	for( int kz = 0; kz<inputVol.ZDim(); kz++)
	  {
	    cout<<kz<<"..";
	for( int jy = 0; jy<inputVol.YDim(); jy++)
	  for( int ix = 0; ix<inputVol.XDim(); ix++)
	    {

	    
	      if(kz>=Zlbound && kz <= Zubound)
		{
		  outputVol(ix,jy,kz, (inputVol(ix,jy,kz)) ); /// set values to inputVol only when it lies in teh bounds

		}
	      else
		{
		  outputVol(ix,jy,kz, 0); //ellse set iit to zero

		}
	      
	      
	  
	    }
	  }	





      cout<<"done volume copy"<<endl;

      
      //   cout<<"Performing Erosion here using a 3x3x3 cross shape kernel"<<endl;


	cout<<"dimensions are "<<inputVol.XDim()<<" "<<inputVol.YDim()<<" "<<inputVol.ZDim()<<endl;



      VolMagick::writeVolumeFile(outputVol, argv[2]);



      cout << endl;

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
