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
#include <limits>

#include <boost/cstdint.hpp>

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>

#include <omp.h>
#include <time.h>


using namespace std;

int main(int argc, char **argv)
{
  if(argc<4)
    {
      cerr <<
 
	"Usage: inputfile, projectionAngles, outputfile" << endl;


      return 1;
    }


  try
    {    
      typedef boost::int64_t int64_t;

      VolMagick::Volume inputVol;

      VolMagick::Volume outputVol;   

      VolMagick::Volume startPtVol;

     VolMagick::Volume endPtVol;

      VolMagick::Volume rotatedVol;

      VolMagick::readVolumeFile(inputVol,argv[1]); ///first argument is input volume
    
      
	 
      static double PI = 3.14159265;
 
      std::cout<<"vol read\n";



       int numProcs = omp_get_num_procs();
      std::cout<<"Num of processors: "<< numProcs<<"\n";

      int64_t numAngles = 0;
	
      FILE* fp = fopen(argv[2],"r");
     
      std::cout<<"Counting angles \n";
	
      float tmp;	
	

	while(fscanf(fp, "%f", &tmp) != EOF)
		numAngles++;

	fclose(fp);


   // initialize projection output volume

//	 outputVol.dimension(VolMagick::Dimension(inputVol.XDim(), inputVol.YDim(), numAngles));

	 VolMagick::Dimension dim;

	 dim.xdim = inputVol.XDim();

	 dim.ydim = inputVol.YDim();

	 dim.zdim = numAngles;
	
	 VolMagick::VoxelType vox = VolMagick::Float;

	 outputVol.voxelType(vox);

         VolMagick::BoundingBox outputBB;

	 outputBB.minx = inputVol.boundingBox().minx;

	 outputBB.maxx = inputVol.boundingBox().maxx;
	 
	 outputBB.maxy = inputVol.boundingBox().maxy;	

	 outputBB.miny = inputVol.boundingBox().miny;

	 outputBB.minz = 0;

	 outputBB.maxz = numAngles-1;

	 outputVol.boundingBox(outputBB);

	 outputVol.dimension(dim);
	
		
	// check
	std::cout<<outputVol.XDim()<<" "<<outputVol.YDim()<<" "<<outputVol.ZDim()<<"\n";
	
	std::cout<<outputVol.boundingBox().maxx<<" "<<outputVol.boundingBox().maxy<<" "<<outputVol.boundingBox().maxz<<"\n";

	std::cout<<outputVol.boundingBox().minx<<" "<<outputVol.boundingBox().miny<<" "<<outputVol.boundingBox().minz<<"\n";

	std::cout<<outputVol.voxelType()<<"\n";
	
//	
//
//	VolMagick::createVolumeFile(argv[3], outputBB, dim, inputVol.voxelType());

      	VolMagick::createVolumeFile(argv[3], outputVol);
        
	std::cout<<"Total number of angles: "<<numAngles<<"\n";

    float *angles = new float[numAngles];
	
	FILE* fp1 = fopen(argv[2], "r");
	
	for(int64_t i=0; i<numAngles; i++) {

	float tempangle;
	
	fscanf(fp1, "%f", &tempangle);

	tempangle = tempangle*PI/180;

	angles[i] = tempangle;

	}

	fclose(fp1);

//        int startTime;
//        startTime = clock();
	time_t t0, t1;
	
	t0 = time(NULL);	


	/// Collect start points and end points for each ray. Start point will always be along the X axis and the End point will be along the Z axis.
	// The direction of the line does not matter since integration is the same startpoint to end point or vice versa.

	// The start point is -ve then it imples that the "Z" coordinate is at zero. Else it is assumed that the Z coordinate is at max ZDim and the 
	// only inded needed to be stored is the "X" coordinate.
	// Similarly if the end point (Z) is -ve then it implies that the "X" axis is at zero. Else if it is positive then the X axis is at max XDim.
	
	/// step size for search
	float delta = 0.5;

	#pragma omp parallel for schedule(static, numAngles/numProcs)           
	for(int64_t anum = 0; anum<numAngles; anum++)
	{
	std::cout<<"Current angle: "<<angles[anum]<<"\n"<<"Processor ID: "<<omp_get_thread_num()<<"\n";	
//	int anum = 1; 
	    for(int64_t y=0; y<inputVol.YDim(); y++)
		for(int64_t r=0; r<inputVol.XDim(); r++)
              {
		
		float tempangle = angles[anum];
//		float tempangle = 0;


		float MaxS = sqrt(inputVol.XDim()*inputVol.XDim()/4.0 + inputVol.ZDim()*inputVol.ZDim()/4.0);
		
		float MinS = -1*MaxS;

		for(float s = MinS; s<MaxS; s=s+delta)
		{
			float r_origShift =  r - (float)inputVol.XDim()/2;
			
			float xx = r_origShift*cos(tempangle) - s*sin(tempangle) + inputVol.XDim()/2;
		
			//std::cout<<r_origShift<<"..";
	
			float zz = r_origShift*sin(tempangle) + s*cos(tempangle) + inputVol.ZDim()/2;			
		
			int in_x = (int) xx;
			int in_z = (int) zz;
	
//			if(xx>inputVol.boundingBox().minx && xx<inputVol.boundingBox().maxx  &&  zz>inputVol.boundingBox().minz && zz<inputVol.boundingBox().maxz)
			if(in_x>0 && in_x<inputVol.XDim() && in_z>0 && in_z<inputVol.ZDim())
			
//			outputVol(r, y, anum, (outputVol(r, y, anum) + inputVol.interpolate(xx, y, zz)*delta ));
                        outputVol(r, y, anum, (outputVol(r, y, anum) + inputVol(in_x, y, in_z)*delta ));



		}

	     }
	
	}	
	
//        int endTime;

 //       endTime = clock();
		
	t1 = time(NULL);
	
	std::cout<<"Time elapsed: "<<(long)(t1-t0);

//        std::cout<<"Time elapsed: "<<(endTime-startTime)/(1000);
        std::cout<<"\n";

	
	std::cout<<"Created outputfile\n";
         
     VolMagick::writeVolumeFile(outputVol, argv[3]);

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
