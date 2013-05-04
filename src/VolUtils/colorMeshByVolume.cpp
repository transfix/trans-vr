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
#include <string>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>

#include <cvcraw_geometry/cvcraw_geometry.h>


typedef cvcraw_geometry::geometry_t geometry;
typedef cvcraw_geometry::geometry_t::color_t color_t;
using namespace std;



typedef struct {
    double r,g,b;
} COLOUR;

COLOUR GetColour(double v,double vmin,double vmax)
{
   COLOUR c = {1.0,1.0,1.0}; // white
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4 * (v - vmin - 0.5 * dv) / dv;
      c.b = 0;
   } else {
      c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c.b = 0;
   }

   return(c);
}





int main(int argc, char **argv)
{
  if(argc < 4)
    {
      std:: cerr << 
	"Usage: " << argv[0] << 

"  inputmesh inputvol outputMesh \n";
      return 1;
    }

  try
    {

	  geometry geom;
	  cvcraw_geometry::read(geom, string(argv[1]));

      VolMagick::Volume inputVol;


      VolMagick::readVolumeFile(inputVol,argv[2]); ///input volume
      
      
	  float x,y,z;
      float max=inputVol.min(), min=inputVol.max();



	  float value; 
	  
	  color_t color;

	  COLOUR c;

	  for(int i=0; i< geom.points.size(); i++)
	  {
	  		x = geom.points[i][0];
			y = geom.points[i][1];
			z = geom.points[i][2];


	  		value = inputVol.interpolate(x,y,z);
			if(max<value) max = value;
			if(min>value) min = value;
	   }
	 
std::cout<<"Max and Min are: "<< max <<", "<< min << std::endl;
	  for(int i=0; i< geom.points.size(); i++)
	  {
	  		x = geom.points[i][0];
			y = geom.points[i][1];
			z = geom.points[i][2];


	  		value = inputVol.interpolate(x,y,z);
  	
	        c=GetColour(value, min, max);
			
	        color[0] = c.r;
			color[1] = c.g;
			color[2] = c.b;

			geom.colors.push_back(color);
	  }

	cvcraw_geometry::write(geom, string(argv[3]));

     std::cout<< "done. "<<  std::endl;

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
