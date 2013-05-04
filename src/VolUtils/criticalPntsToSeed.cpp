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

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <memory>
#include <iostream>
#include <VolMagick/VolMagick.h>

#include <fstream>

//#if defined (__APPLE__)
typedef unsigned int uint;
//#endif


#define max2(x, y)     ((x>y) ? (x):(y))      //Sorry about the macros.  I didn't read Lippman.
#define min2(x, y)     ((x<y) ? (x):(y))
#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))

namespace SegSubunit {

typedef struct CriticalPoint CPNT;
struct CriticalPoint{
  unsigned short x;
  unsigned short y;
  unsigned short z;
  CPNT *next;
};

static int XDIM, YDIM, ZDIM;
static float *dataset;

float GetImgGradient(int, int, int);


void FindCriticalPoints(int xd,int yd,int zd,float *data,
		 CPNT **critical, float tlow,int h_num, int k_num)
{
  int i,j,k;
  int x,y,z;
  int u,v;
  CPNT *critical_end;
  CPNT *critical_start;
  float tmp;
  unsigned char *temp;
  float *temp_float;
  float max_grad,min_grad;
  unsigned long histogram[256];
  unsigned long number, tri_num;
  

  dataset = data;
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;


  temp = (unsigned char*)malloc(sizeof(unsigned char)*XDIM*YDIM*ZDIM); //matrix used to flag down critical points
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++) 
	temp[IndexVect(i,j,k)] = 0;
  for (k=0; k<256; k++)
    histogram[k] = 0;

  number = 0;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++)  { 
	tmp = dataset[IndexVect(i,j,k)]; //tmp has NOTHING TO DO WITH temp. 
	if (tmp > tlow) { //tlow is user input; lower intensity threshold
	  u = 0; ///u is another flag. I know, I suck at this.
	  for (z=max2(k-1,0); z<=min2(k+1,ZDIM-1); z++)
	    for (y=max2(j-1,0); y<=min2(j+1,YDIM-1); y++)
	      for (x=max2(i-1,0); x<=min2(i+1,XDIM-1); x++){
		if (tmp < dataset[IndexVect(x,y,z)])
		  u = 1;
	      }
	  
	  if (u == 0) {
	    temp[IndexVect(i,j,k)] = 1;  //local maximum
	    v = (int)(tmp+0.5);
	    histogram[v] += 1;
	    number++;
	  }
	}
      }

  tri_num = 3000*(h_num*h_num + k_num*k_num + h_num*k_num); //3000 is magic! It really is. I'm superstitious. I have horseshoes on my apartment door.And skeletons in my closet.
  number = 0;
  for (k=255; k>=0; k--) {
    number += histogram[k];
    if (number > tri_num)
      break;
  }
  tmp = (float)k;
  if (tmp < tlow+20.0f)
    tmp = tlow+20.0f;
  printf("thigh = %f \n",tmp); //tmp == thigh.  

  std::ofstream outfile("maxcrit.vgr");

  if(!outfile.is_open())
    std::cout<<"\nUnable to open file\n";

  outfile<<"<!DOCTYPE pointclassdoc>"<<std::endl;
  outfile<<"<pointclassdoc>" <<std::endl;
  outfile<<" <pointclass timestep=\"0\" name=\"Class 0\" color="<<"\"#ffff00\" variable=\"0\" >"<<std::endl;

  critical_start = NULL;
  critical_end = NULL;
  number = 0;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++) 
      for (i=0; i<XDIM; i++)  { 
	if (temp[IndexVect(i,j,k)] == 1 && //if local maximum
	    dataset[IndexVect(i,j,k)] > tmp) {/// and > thigh
	  number++; //counts the number of maximal critical points
	  if (critical_start == NULL) { //linked list fundaes
	    critical_end = (CPNT*)malloc(sizeof(CPNT));
	    critical_start = critical_end;
	  }
	  else {
	    critical_end->next = (CPNT*)malloc(sizeof(CPNT));
	    critical_end = critical_end->next;
	  }
	  
	  critical_end->x = (unsigned short)i;
	  critical_end->y = (unsigned short)j;
	  critical_end->z = (unsigned short)k;
//	 outfile<<i<<"  "<<j<<"  "<<k<<std::endl;

	  outfile<<"  <point>"<< i<<"  "<<j<<"  "<<k<<"</point>" << std::endl;
	}
 }
  outfile<<" </pointclass>"<<std::endl;

  outfile<<"</pointclassdoc>"<<std::endl;

  outfile.close();

  printf("number of maximal critical points jeez: %ld \n",number);
  
  if (number < min2(50000,tri_num/5)) {
    temp_float = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM);
    
    min_grad = 999999.0f;
    max_grad = 0.0f;
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++) {
	  if (dataset[IndexVect(i,j,k)] == 0)
	    temp_float[IndexVect(i,j,k)] = 0.0f;
	  else {
	    u = 0;
	    for (z=max2(k-1,0); z<=min2(k+1,ZDIM-1); z++)
	      for (y=max2(j-1,0); y<=min2(j+1,YDIM-1); y++)
		for (x=max2(i-1,0); x<=min2(i+1,XDIM-1); x++){
		  if (dataset[IndexVect(x,y,z)] == 0)
		    u = 1;
		}
	    if (u == 0) 
	      temp_float[IndexVect(i,j,k)] = GetImgGradient(i,j,k);
	    else
	      temp_float[IndexVect(i,j,k)] = 0.0f;
	  }
	  if (temp_float[IndexVect(i,j,k)] > max_grad)
	    max_grad = temp_float[IndexVect(i,j,k)];
	  if (temp_float[IndexVect(i,j,k)] < min_grad)
	    min_grad = temp_float[IndexVect(i,j,k)];
	}
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++) 
	  temp_float[IndexVect(i,j,k)] = 255.0f*(temp_float[IndexVect(i,j,k)]-min_grad)
	    /(max_grad-min_grad);
    

    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++) 
	  temp[IndexVect(i,j,k)] = 0;
    for (k=0; k<256; k++)
      histogram[k] = 0;
    
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++)  { 
	  if (dataset[IndexVect(i,j,k)] > tlow) {
	    u = 0;
	    tmp = temp_float[IndexVect(i,j,k)];
	    for (z=max2(k-1,0); z<=min2(k+1,ZDIM-1); z++)
	      for (y=max2(j-1,0); y<=min2(j+1,YDIM-1); y++)
		for (x=max2(i-1,0); x<=min2(i+1,XDIM-1); x++){
		  if (tmp < temp_float[IndexVect(x,y,z)])
		    u = 1;
		}
	    
	    if (u == 0) {
	      temp[IndexVect(i,j,k)] = 1;
	      v = (int)(tmp+0.5);
	      histogram[v] += 1;
	    }
	  }
	}
    
    tri_num = number;
    number = 0;
    for (k=255; k>=0; k--) {
      number += histogram[k];
      if (number > tri_num)
	break;
    }
    tmp = (float)k;
    if (tmp < tlow+20.0f)
      tmp = tlow+20.0f;
    printf("thigh = %f \n",tmp);
    

	std::ofstream outfile("saddlecrit.vgr");

    if(!outfile.is_open())
    std::cout<<"\nUnable to open file\n";

   outfile<<"<!DOCTYPE pointclassdoc>"<<std::endl;
   outfile<<"<pointclassdoc>" <<std::endl;
   outfile<<" <pointclass timestep=\"0\" name=\"Class 0\" color="<<"\"#ffff00\" variable=\"0\" >"<<std::endl;

 
    number = 0;
    for (k=0; k<ZDIM; k++)
      for (j=0; j<YDIM; j++) 
	for (i=0; i<XDIM; i++)  { 
	  if (temp[IndexVect(i,j,k)] == 1 &&
	      temp_float[IndexVect(i,j,k)] > tmp) {
	    number++;
	    if (critical_start == NULL) {
	      critical_end = (CPNT*)malloc(sizeof(CPNT));
	      critical_start = critical_end;
	    }
	    else {
	      critical_end->next = (CPNT*)malloc(sizeof(CPNT));
	      critical_end = critical_end->next;
	    }
	    
	    critical_end->x = (unsigned short)i;
	    critical_end->y = (unsigned short)j;
	    critical_end->z = (unsigned short)k;
	    outfile<<"  <point>"<< i<<"  "<<j<<"  "<<k<<"</point>" << std::endl;
	  }
    }
    outfile<<" </pointclass>"<<std::endl;

    outfile<<"</pointclassdoc>"<<std::endl;
    outfile.close();
    printf("number of saddle critical points: %ld \n",number);

    free(temp_float);
  }

  if (critical_end != NULL)
    critical_end->next = NULL;
  *critical = critical_start;
  
  free(temp);
}





float GetImgGradient(int x, int y, int z)
{
  int i,j,k;
  float grad_x,grad_y,grad_z;
  float gradient;


  grad_x=0.0;
  for (j=max2(y-1,0); j<=min2(y+1,YDIM-1); j++) 
    for (k=max2(z-1,0); k<=min2(z+1,ZDIM-1); k++) {
      grad_x += dataset[IndexVect(min2(x+1,XDIM-1),j,k)]-
	dataset[IndexVect(max2(x-1,0),j,k)];
      if (j==y || k==z)
	grad_x += dataset[IndexVect(min2(x+1,XDIM-1),j,k)]-
	  dataset[IndexVect(max2(x-1,0),j,k)];
      if (j==y && k==z)
	grad_x += 2.0f*(dataset[IndexVect(min2(x+1,XDIM-1),j,k)]-
		       dataset[IndexVect(max2(x-1,0),j,k)]);
    }
  
  grad_y=0.0;
  for (i=max2(x-1,0); i<=min2(x+1,XDIM-1); i++) 
    for (k=max2(z-1,0); k<=min2(z+1,ZDIM-1); k++) {
      grad_y += dataset[IndexVect(i,min2(y+1,YDIM-1),k)]-
	dataset[IndexVect(i,max2(y-1,0),k)];
      if (i==x || k==z)
	grad_y += dataset[IndexVect(i,min2(y+1,YDIM-1),k)]-
	  dataset[IndexVect(i,max2(y-1,0),k)];
      if (i==x && k==z)
	grad_y += 2.0f*(dataset[IndexVect(i,min2(y+1,YDIM-1),k)]-
		       dataset[IndexVect(i,max2(y-1,0),k)]);
    }
  
  grad_z=0.0;
  for (i=max2(x-1,0); i<=min2(x+1,XDIM-1); i++) 
    for (j=max2(y-1,0); j<=min2(y+1,YDIM-1); j++) { 
      grad_z += dataset[IndexVect(i,j,min2(z+1,ZDIM-1))]-
	dataset[IndexVect(i,j,max2(z-1,0))];
      if (i==x || j==y)
	grad_z += dataset[IndexVect(i,j,min2(z+1,ZDIM-1))]-
	  dataset[IndexVect(i,j,max2(z-1,0))];
      if (i==x && j==y)
	grad_z += 2.0f*(dataset[IndexVect(i,j,min2(z+1,ZDIM-1))]-
		       dataset[IndexVect(i,j,max2(z-1,0))]);
    }
 
  gradient=(float)sqrt(grad_x*grad_x+grad_y*grad_y+grad_z*grad_z);
  return(gradient/16.0f);
}

};

int main(int argc, char *argv[])
{

  if(argc==1) 
  {
  	std::cout<<"Usage: "<<argv[0]<<"  <inputVolume> [low threshold] [h num] [k num]" <<std::endl;
	return 0;
  }

  VolMagick::Volume inputVol;

  VolMagick::readVolumeFile(inputVol,argv[1]); 
      
  std::cout<<"Input voxel type: "<<inputVol.voxelType();
 
  uint dim[3];

  dim[0] = inputVol.XDim();
  
  dim[1] = inputVol.YDim();
  
  dim[2] = inputVol.ZDim();

  uint XDIM = dim[0]; 

  uint YDIM = dim[1];

  float* vol = new float[dim[0]*dim[1]*dim[2]];

  std::cout<<"\n"<<dim[0]<<"  "<<dim[1]<<" "<<dim[2]<<"\n";

  float max = -100000;

  float min = 100000;

  for(uint k = 0; k<dim[2]; k++)
    for(uint j = 0; j<dim[1]; j++)
      for(uint i = 0; i<dim[0]; i++) { 

	//vol[IndexVect(i,j,k)] = inputVol(i, j, k);

	  if(  inputVol(i, j, k)<min) 
	    min =   inputVol(i, j, k);

	  if(  inputVol(i, j, k)>max) 
	    max = inputVol(i, j, k);
      }

  std::cout<<"min "<<min<<" max "<<max<<"\n";
 
 float maxag = -100000;

 float minag = 100000;
  
  for(uint k = 0; k<dim[2]; k++)
    for(uint j = 0; j<dim[1]; j++)
      for(uint i = 0; i<dim[0]; i++) { 

	vol[IndexVect(i,j,k)] = ((inputVol(i, j, k) - min)/(-min+max))*255;//map from 0-255

	  if( vol[IndexVect(i,j,k)]<minag) 
	    minag =  vol[IndexVect(i,j,k)];

	  if( vol[IndexVect(i,j,k)]>maxag) 
	    maxag =  vol[IndexVect(i,j,k)];
      }
  
   std::cout<<"min "<<minag<<" max "<<maxag<<"\n";

 // std::ofstream outfile("volume.txt");
  /*
  for(uint i = 0; i<dim[0]; i++)
    for(uint j = 0; j<dim[1]; j++)
      for(uint k = 0; k<dim[2]; k++) 
	outfile<<vol[IndexVect(i,j,k)]<<"  ";
  */
 // outfile.close();

  SegSubunit::CPNT *critical;

  if(argc>2)
  {
  	minag = atof(argv[2]);
    SegSubunit::FindCriticalPoints(dim[0],dim[1],dim[2],vol,&critical,minag,atoi(argv[3]),atoi(argv[4]));
  }
  else if(argc == 2)
          SegSubunit::FindCriticalPoints(dim[0],dim[1],dim[2],vol,&critical,minag,2,1);
  //  std::cout<<(critical[1])->x<<"  "<<(critical[1])->y<<"  "<<(critical[2])->z<<"\n";


  delete [] vol;

  return 0;

}
