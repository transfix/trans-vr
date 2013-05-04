/*
  Copyright 2011 The University of Texas at Austin

	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of MolSurf.

  MolSurf is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.


  MolSurf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with MolSurf; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
// dataslc.C - class for a triangular slice of scalar data
// Copyright (c) 1997 Dan Schikore - updated by Emilio Camahort, 1999

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <Contour/dataslc.h>
#include <Contour/compute.h>
#include <Contour/endian_io.h>

#if ! defined (__APPLE__)
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#ifndef WIN32
#include <unistd.h>
#endif

#define SQR(x) ((x)*(x))
#define FSAMPLES 256

extern int verbose;

// Dataslc() - create a volume with the specified type and dimensions with
//             data taken from the given file
Dataslc::Dataslc(DataType t, int ndata, char* fn) : Data(t, ndata, fn)
{
	u_int i;
	float grad[3];
	float len;
	verts = (double(*)[2])malloc(sizeof(double[2])*getNVerts());
	vgrad = (float(*)[3])malloc(sizeof(float[3])*getNVerts());
	cells = (u_int(*)[3])malloc(sizeof(u_int[3])*getNCells());
	celladj = (int (*)[3])malloc(sizeof(int[3])*getNCells());
	if(verbose)
	{
		printf("reading verts\n");
	}
	//fread(verts, sizeof(double[2]), getNVerts(), fp);
	fread_double(verts, sizeof(double), 2 * getNVerts(), fp);
	if(verbose)
	{
		printf("reading cells\n");
	}
	for(i=0; i<getNCells(); i++)
	{
		//fread(cells[i], sizeof(u_int[3]), 1, fp);
		//fread(celladj[i], sizeof(int[3]), 1, fp);
		fread_int(cells[i],   sizeof(u_int), 3, fp);
		fread_int(celladj[i], sizeof(int),   3, fp);
	}
	for(i=0; i<getNCells(); i++)
	{
		for(u_int j=0; j<getNCellFaces(); j++)
		{
			int adj = celladj[i][j];
			int same = 0;
			if(adj != -1)
			{
				for(int k=0; k<3; k++)
					for(int l=0; l<3; l++)
						if(cells[i][k] == cells[adj][l])
						{
							same++;
						}
				if(verbose)
					if(same != 2)
						printf("cell %d (%d %d %d) not adj to %d (%d %d %d)\n",
							   i, cells[i][0], cells[i][1], cells[i][2],
							   adj, cells[adj][0], cells[adj][1], cells[adj][2]);
			}
		}
	}
	readData();
	for(i=0; i<getNCells(); i++)
	{
		getCellGrad3(i, grad);
		vgrad[getCellVert(i,0)][0] += grad[0];
		vgrad[getCellVert(i,0)][1] += grad[1];
		vgrad[getCellVert(i,0)][2] += grad[2];
		vgrad[getCellVert(i,1)][0] += grad[0];
		vgrad[getCellVert(i,1)][1] += grad[1];
		vgrad[getCellVert(i,1)][2] += grad[2];
		vgrad[getCellVert(i,2)][0] += grad[0];
		vgrad[getCellVert(i,2)][1] += grad[1];
		vgrad[getCellVert(i,2)][2] += grad[2];
	}
	for(i=0; i<getNVerts(); i++)
	{
		if(verbose > 1)
		{
			printf("scaling vgrad %d\n", i);
		}
		len = (float)sqrt(vgrad[i][0]*vgrad[i][0] + vgrad[i][1]*vgrad[i][1] +
						  vgrad[i][2]*vgrad[i][2]);
		if(len != 0.0)
		{
			vgrad[i][0] /= len;
			vgrad[i][1] /= len;
			vgrad[i][2] /= len;
		}
	}
}

// Dataslc() - alternative constructor for the libcontour library
Dataslc::Dataslc(Data::DataType t, int ndata, u_int nverts, u_int ncells, double* _verts, u_int* _cells, int* _celladj, u_char* data)
	: Data(t, ndata, data)
{
	u_int i;
	float grad[3];
	float len;
	Dataslc::nverts = nverts;			// initializations
	Dataslc::ncells = ncells;
	verts   = (double(*)[2])_verts;
	cells   = (u_int(*)[3])_cells;
	celladj = (int (*)[3])_celladj;
	if(verbose)
	{
		printf("computing extent\n");    // compute data extents
	}
	minext[0] = minext[1] =
					maxext[0] = maxext[1] =
									minext[2] = maxext[2] = 0.0;
	for(i = 0; i < nverts; i++)
	{
		if(verts[i][0] < minext[0])
		{
			minext[0] = (float)(verts[i][0]);
		}
		if(verts[i][0] > maxext[0])
		{
			maxext[0] = (float)(verts[i][0]);
		}
		if(verts[i][1] < minext[1])
		{
			minext[1] = (float)(verts[i][1]);
		}
		if(verts[i][1] > maxext[1])
		{
			maxext[1] = (float)(verts[i][1]);
		}
	}
	if(verbose)
		printf("  min = %f %f %f  max = %f %f %f\n",
			   minext[0], minext[1], minext[2],
			   maxext[0], maxext[1], maxext[2]);
	if(verbose)
	{
		printf("%d verts, %d cells\n", Dataslc::nverts, Dataslc::ncells);
	}
	// compute gradients
	vgrad = (float(*)[3])malloc(sizeof(float[3])*getNVerts());
	if(verbose)
	{
		printf("processing cells\n");
	}
	for(i=0; i<getNCells(); i++)
	{
		for(u_int j=0; j<getNCellFaces(); j++)
		{
			int adj = celladj[i][j];
			int same = 0;
			if(adj != -1)
			{
				for(int k=0; k<3; k++)
					for(int l=0; l<3; l++)
						if(cells[i][k] == cells[adj][l])
						{
							same++;
						}
				if(verbose)
					if(same != 2)
						printf("cell %d (%d %d %d) not adj to %d (%d %d %d)\n",
							   i, cells[i][0], cells[i][1], cells[i][2],
							   adj, cells[adj][0], cells[adj][1], cells[adj][2]);
			}
		}
	}
	preprocessData(data);
	for(i=0; i<getNCells(); i++)
	{
		getCellGrad3(i, grad);
		vgrad[getCellVert(i,0)][0] += grad[0];
		vgrad[getCellVert(i,0)][1] += grad[1];
		vgrad[getCellVert(i,0)][2] += grad[2];
		vgrad[getCellVert(i,1)][0] += grad[0];
		vgrad[getCellVert(i,1)][1] += grad[1];
		vgrad[getCellVert(i,1)][2] += grad[2];
		vgrad[getCellVert(i,2)][0] += grad[0];
		vgrad[getCellVert(i,2)][1] += grad[1];
		vgrad[getCellVert(i,2)][2] += grad[2];
	}
	for(i=0; i<getNVerts(); i++)
	{
		if(verbose > 1)
		{
			printf("scaling vgrad %d\n", i);
		}
		len = (float)sqrt(vgrad[i][0]*vgrad[i][0] + vgrad[i][1]*vgrad[i][1] +
						  vgrad[i][2]*vgrad[i][2]);
		if(len != 0.0)
		{
			vgrad[i][0] /= len;
			vgrad[i][1] /= len;
			vgrad[i][2] /= len;
		}
	}
}

// ~Dataslc() - destroy a volume
Dataslc::~Dataslc()
{
	if(filename)
	{
		free(verts);
		free(cells);
		free(celladj);
	}
}

//  compLength() -
float* Dataslc::compLength(u_int& len, float** funx)
{
	float* val = (float*)malloc(sizeof(float)*FSAMPLES);
	float* fx = (float*)malloc(sizeof(float)*FSAMPLES);
	u_int c;
	u_int* v;
	len = FSAMPLES;
	memset(val, 0, sizeof(float)*len);
	*funx = fx;
	for(c=0; c<len; c++)
	{
		fx[c] = getMin() + (c/(len-1.0f)) * (getMax()-getMin());
	}
	for(c=0; c<getNCells(); c++)
	{
		v = getCellVerts(c);
		triSurfIntegral(getVert(v[0]), getVert(v[1]), getVert(v[2]),
						getValue(v[0]), getValue(v[1]), getValue(v[2]), fx, val, len,
						getMin(), getMax(), 1.0);
	}
	return(val);
}

//  compArea() -
float* Dataslc::compArea(u_int& len, float** funx)
{
	float* val = (float*)malloc(sizeof(float)*FSAMPLES);
	float* cum = (float*)malloc(sizeof(float)*FSAMPLES);
	float* fx = (float*)malloc(sizeof(float)*FSAMPLES);
	u_int c;
	u_int* v;
	float sum;
	u_int b;
	len = FSAMPLES;
	memset(val, 0, sizeof(float)*len);
	memset(cum, 0, sizeof(float)*len);
	*funx = fx;
	for(c=0; c<len; c++)
	{
		fx[c] = getMin() + (c/(len-1.0f)) * (getMax()-getMin());
	}
	for(c=0; c<getNCells(); c++)
	{
		v = getCellVerts(c);
		triVolIntegral(getVert(v[0]), getVert(v[1]), getVert(v[2]),
					   getValue(v[0]), getValue(v[1]), getValue(v[2]), fx, val, cum,
					   len, getMin(), getMax(), 1.0);
	}
	// now we need to sum from the first to last
	sum = 0.0;
	for(b=0; b<len; b++)
	{
		val[b]+=sum;
		sum+=cum[b];
	}
	free(cum);
	return(val);
}

//  compMaxArea() -
float* Dataslc::compMaxArea(u_int& len, float** funx)
{
	float* f;
	float max;
	f = compArea(len, funx);
	max = f[len-1];
	for(int i=0; i<len; i++)
	{
		f[i] = max-f[i];
	}
	return(f);
}

//  compGradient() -
float* Dataslc::compGradient(u_int& len, float** funx)
{
	float* val = (float*)malloc(sizeof(float)*FSAMPLES);
	float* fx = (float*)malloc(sizeof(float)*FSAMPLES);
	u_int c;
	u_int* v;
	float cellgrad[3], scaling;
	len = FSAMPLES;
	memset(val, 0, sizeof(float)*len);
	*funx = fx;
	for(c=0; c<len; c++)
	{
		fx[c] = getMin() + (c/(len-1.0f)) * (getMax()-getMin());
	}
	for(c=0; c<getNCells(); c++)
	{
		v = getCellVerts(c);
		getCellGrad3(c, cellgrad);
		scaling = (float)sqrt(cellgrad[0]*cellgrad[0] + cellgrad[1]*cellgrad[1]) / (cellgrad[2]);
		triSurfIntegral(getVert(v[0]), getVert(v[1]), getVert(v[2]),
						getValue(v[0]), getValue(v[1]), getValue(v[2]), fx, val, len,
						getMin(), getMax(), (float)fabs(scaling));
	}
	return(val);
}

//  compFunction(), fName() -
float* Dataslc::compFunction(int n, u_int& len, float** fx)
{
	switch(n)
	{
		case 0:
			return(compLength(len, fx));
		case 1:
			return(compArea(len, fx));
		case 2:
			return(compMaxArea(len, fx));
		case 3:
			return(compGradient(len, fx));
	}
	return(NULL);
}

char* Dataslc::fName(int n)
{
	switch(n)
	{
		case 0:
			return((char*)"Length");
		case 1:
			return((char*)"Min Area");
		case 2:
			return((char*)"Max Area");
		case 3:
			return((char*)"Gradient");
	}
	return(NULL);
}
