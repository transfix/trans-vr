/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

#include <dataCutterClient/message.h>
#include<string.h>
//#include<iostream>

//using namespace std;

DSDetails::DSDetails()
	{
	min[0]=min[1]=min[2]=0;
	max[0]=max[1]=max[2]=127;
	dim[0]=dim[1]=dim[2]=128;
	}

DSDetails::DSDetails(DSDetails &d)
	{
	setMin(d.min);
	setMax(d.max);
	setDim(d.dim);
	}


DSDetails::DSDetails(uchar *buffer)
	{
	deSerialize(buffer);
	}


void DSDetails::display()
	{
	//cout<<"\nDSDETAILS:\n";
	//cout<<min[0]<<","<<min[1]<<","<<min[2]<<endl;
	//cout<<max[0]<<","<<max[1]<<","<<max[2]<<endl;
	//cout<<dim[0]<<","<<dim[1]<<","<<dim[2]<<endl;
	}

/* set methods */

void DSDetails::setMin(uint *lmin)
	{
	min[0]=lmin[0];
	min[1]=lmin[1];
	min[2]=lmin[2];
	}

void DSDetails::setMax(uint *lmax)
	{
	max[0]=lmax[0];
	max[1]=lmax[1];
	max[2]=lmax[2];
	}

void DSDetails::setDim(uint *ldim)
	{
	dim[0]=ldim[0];
	dim[1]=ldim[1];
	dim[2]=ldim[2];
	}


/* get methods */

void DSDetails::getMin(uint *lmin)
	{
	lmin[0]=min[0];
	lmin[1]=min[1];
	lmin[2]=min[2];
	}

void DSDetails::getMax(uint *lmax)
	{
	lmax[0]=max[0];
	lmax[1]=max[1];
	lmax[2]=max[2];
	}

void DSDetails::getDim(uint *ldim)
	{
	ldim[0]=dim[0];
	ldim[1]=dim[1];
	ldim[2]=dim[2];
	}

int DSDetails::isValid(DSDetails *spec)
	{
	if ((spec->min[0]>=min[0])&&(spec->min[1]>=min[1])&&(spec->min[2]>=min[2])&&\
	 (spec->max[0]<=max[0])&&(spec->max[1]<=max[1])&&(spec->max[2]<=max[2])&& \
	 (spec->dim[0]<=dim[0])&&(spec->dim[1]<=dim[1])&&(spec->dim[2]<=dim[2]))
	return 1;
	else return 0;	
	}

uchar* DSDetails::serialize(uchar *buffer)
	{
	memcpy(buffer,min,3*sizeof(uint));
	buffer += 3 * sizeof(uint);
	memcpy(buffer,max,3*sizeof(uint));
	buffer += 3 * sizeof(uint);

	memcpy(buffer,dim,3*sizeof(uint));
	buffer += 3*sizeof(uint);

	return buffer;
	}

uchar* DSDetails::deSerialize(uchar *buffer)
	{
	memcpy(min,buffer,3*sizeof(uint));
	buffer += 3 * sizeof(uint);
	memcpy(max,buffer,3*sizeof(uint));
	buffer += 3 * sizeof(uint);

	memcpy(dim,buffer,3*sizeof(uint));
	buffer += 3*sizeof(uint);

	return buffer;
	}

