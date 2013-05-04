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

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>

using namespace std;

int main(int argc, char **argv)
{
    typedef VolMagick::uint64 uint;

  if(argc < 4)
    {
      cerr << 
	"Usage: inputfile, outputfile, pointList.txt" << argv[0] << endl;


      return 1;
    }

  try
    {
      VolMagick::Volume inputVol;



      VolMagick::Volume outputVol;   

      
        FILE *ipfile;

	ipfile = fopen(argv[3], "r");

	 uint xindex, yindex, zindex;

	 std::cout<<"\nReading points file\n";

	 typedef std::vector<float> vecDouble;
	 
	 vecDouble Points[3], indices[3];

	 while (fscanf(ipfile, "%u %u %u", &xindex, &yindex, &zindex) && !feof(ipfile)) {
	   /*Transform indices to coordinates and push them into the respective vectors*/
	   
	   indices[0].push_back(xindex);
      
	   indices[1].push_back(yindex);
      
	   indices[2].push_back(zindex);
      
	 }

	 cout<<"Read points, size: "<<indices[0].size()<<endl;

	 //      for(unsigned int i = 0; i<argc-2; i++)
	 VolMagick::readVolumeFile(inputVol,argv[1]); ///first argument is input volume
	 

	 

	 
      outputVol.voxelType(inputVol.voxelType());
      outputVol.dimension(inputVol.dimension());
      outputVol.boundingBox(inputVol.boundingBox());
      
      for(unsigned int i=0; i<outputVol.XDim(); i++)
	for(unsigned int j=0; j<outputVol.YDim(); j++)
	  for(unsigned int k=0; k<outputVol.ZDim(); k++)
	    
	    {

		outputVol(i,j,k, inputVol(i,j,k));
	       
	    }


      for(unsigned count = 0; count<indices[0].size(); count++){
	cout<<indices[0][count]<<" "<<indices[1][count]<<" "<<indices[2][count]<<"\n";
	
	int ptx = indices[0][count];
	int pty = indices[1][count];
	int ptz = indices[2][count];
	
	for(int xtemp =-1; xtemp <2;xtemp++)
	  for(int ytemp =-1; ytemp <2;ytemp++)
	    for(int ztemp =-1; ztemp <2;ztemp++)
	      outputVol(ptx+xtemp, pty+ytemp, ptz+ztemp, 255);
	
      }
      
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
