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

/* $Id: RawVRGBA2MultiRawIV.cpp 4742 2011-10-21 22:09:44Z transfix $ */

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
  if(argc != 8)
    {
      cerr << 
	"Usage: " << argv[0] << 
	" <input rawv with >=4 variables> <red var> <green var> <blue var> <alpha var> <timestep> <output rawiv file prefix>" << endl;
      return 1;
    }

  try
    {
      unsigned int red_var,green_var,blue_var,alpha_var,timestep, i,j,k;
      VolMagick::VolumeFileInfo volinfo;
      VolMagickOpStatus status;
      VolMagick::setDefaultMessenger(&status);

      volinfo.read(argv[1]);
      if(volinfo.numVariables() < 4) 
	throw VolMagick::UnsupportedVolumeFileType("Input volume must have more than 4 variables");

      red_var = atoi(argv[2]);
      green_var = atoi(argv[3]);
      blue_var = atoi(argv[4]);
      alpha_var = atoi(argv[5]);
      timestep = atoi(argv[6]);

      volinfo.min(alpha_var); //make sure the alpha var minimum has been calculated

      /* [0-3] == rgba slices, 5 == output slice */
      VolMagick::Volume slice[5];
      unsigned int var_ind[4] = { red_var, green_var, blue_var, alpha_var };
      
      std::set<Color> colors; /* list of all colors in the volume... hopefully there aren't too many!! */

      for(k=0; k<volinfo.ZDim(); k++)
	{
	  /* get the colors for this slice (ignoring alpha for now) */
	  VolMagick::readVolumeFile(slice[0],argv[1],red_var,timestep,
				    0,0,k,VolMagick::Dimension(volinfo.XDim(),volinfo.YDim(),1));
	  VolMagick::readVolumeFile(slice[1],argv[1],green_var,timestep,
				    0,0,k,VolMagick::Dimension(volinfo.XDim(),volinfo.YDim(),1));
	  VolMagick::readVolumeFile(slice[2],argv[1],blue_var,timestep,
				    0,0,k,VolMagick::Dimension(volinfo.XDim(),volinfo.YDim(),1));

	  /* add all new colors to the list */
	  for(j=0; j<volinfo.YDim(); j++)
	    for(i=0; i<volinfo.XDim(); i++)
	      colors.insert(boost::make_tuple(slice[0](i,j,0),slice[1](i,j,0),slice[2](i,j,0)));
	  
	  fprintf(stderr,"Determining set of colors... %5.2f %%   \r",
		  (((float)k)/((float)((int)(volinfo.ZDim()-1))))*100.0);
	}
      
      cout << "\nNumber of colors: " << colors.size() << endl;
      cout << "Colors: ";
      for(std::set<Color>::iterator cur = colors.begin();
	  cur != colors.end();
	  cur++)
	cout << *cur << " ";
      cout << endl;

      /* create the output volume files */
      std::stringstream ss;
      for(std::set<Color>::iterator cur = colors.begin();
	  cur != colors.end();
	  cur++)
	{
	  ss.str("");
	  ss << boost::tuples::set_delimiter('.') 
	     << boost::tuples::set_open('_')
	     << boost::tuples::set_close('_')
	     << *cur;
	  VolMagick::createVolumeFile(std::string(argv[7]) + ss.str() + std::string(".rawiv"),
				      volinfo.boundingBox(),
				      volinfo.dimension(),
				      std::vector<VolMagick::VoxelType>(1, volinfo.voxelTypes(alpha_var)));
	}

      /* write the output volumes */
      int count;
      std::vector<VolMagick::Volume> outslices(colors.size());
      for(k=0; k<volinfo.ZDim(); k++)
	{
	  /* get the colors for this slice */
	  VolMagick::readVolumeFile(slice[0],argv[1],red_var,timestep,
				    0,0,k,VolMagick::Dimension(volinfo.XDim(),volinfo.YDim(),1));
	  VolMagick::readVolumeFile(slice[1],argv[1],green_var,timestep,
				    0,0,k,VolMagick::Dimension(volinfo.XDim(),volinfo.YDim(),1));
	  VolMagick::readVolumeFile(slice[2],argv[1],blue_var,timestep,
				    0,0,k,VolMagick::Dimension(volinfo.XDim(),volinfo.YDim(),1));
	  VolMagick::readVolumeFile(slice[3],argv[1],alpha_var,timestep,
				    0,0,k,VolMagick::Dimension(volinfo.XDim(),volinfo.YDim(),1));
	  
	  /* make sure to use the precalculated min/max values */
	  for(i=0; i<4; i++)
	    {
	      slice[i].min(volinfo.min(var_ind[i],timestep));
	      slice[i].max(volinfo.max(var_ind[i],timestep));
	    }

	  /* copy the alpha slice to all the output slices */
	  std::fill(outslices.begin(), outslices.end(), slice[3]);
	    
	  /* lookup each color to determine it's output value */
	  for(j=0; j<volinfo.YDim(); j++)
	    for(i=0; i<volinfo.XDim(); i++)
	      {
		count = 0;
		for(std::set<Color>::iterator cur = colors.begin();
		    cur != colors.end();
		    cur++)
		  {
		    if(*cur == boost::make_tuple(slice[0](i,j,0),
						 slice[1](i,j,0),
						 slice[2](i,j,0)))
		      outslices[count++](i,j,0, slice[3](i,j,0));
		    else
		      outslices[count++](i,j,0, volinfo.min(alpha_var,timestep));
		  }
	      }
	  
	  /* write the alpha values to the appropriate volume */
	  count = 0;
	  for(std::set<Color>::iterator cur = colors.begin(); 
	      cur != colors.end(); 
	      cur++) 
	    {
	      ss.str(""); 
	      ss << boost::tuples::set_delimiter('.')
		 << boost::tuples::set_open('_')
		 << boost::tuples::set_close('_') 
		 << *cur; 
	      VolMagick::writeVolumeFile(outslices[count++],std::string(argv[7]) + ss.str() + std::string(".rawiv"),
					 0,0,0,0,k);
	    }
	  
	  fprintf(stderr,"Writing output alpha values... %5.2f %%   \r",
		  (((float)k)/((float)((int)(volinfo.ZDim()-1))))*100.0);
	}

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
