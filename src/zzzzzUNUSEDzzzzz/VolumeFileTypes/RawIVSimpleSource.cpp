/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// RawIVSimpleSource.cpp: implementation of the RawIVSimpleSource class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/RawIVSimpleSource.h>
#include <ByteOrder/ByteSwapping.h>
#include <math.h>
#include <qfileinfo.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RawIVSimpleSource::RawIVSimpleSource()
{
	setDefaults();
}

RawIVSimpleSource::~RawIVSimpleSource()
{

}

bool RawIVSimpleSource::setFile(const QString& file)
{
	QFileInfo fileInfo(file);
	unsigned int size = fileInfo.size();


	FILE* fp = fopen(file, "rb");
	if (!fp) {
		setDefaults();
		return false;
	}

	if (!readHeader(fp)) {
		setDefaults();
		fclose(fp);
		return false;
	}

	// we only support character files for now, check file size
	if (size-68 != getNumVerts()) {
		// not a byte file or messed up header
		setDefaults();
		fclose(fp);
		return false;
	}

	// allocate space for data
	if (!allocateMemory(getNumVerts())) {
		setDefaults();
		fclose(fp);
		return false;
	}

	// load data
	if (getNumVerts() != fread(m_Data, sizeof(unsigned char), getNumVerts(), fp)) {
		destroyMemory();
		setDefaults();
		fclose(fp);
		return false;
	}

	fclose(fp);
	return true;;
}

double linear(double x) {
	double IxI = fabs(x);
	if (0 <= IxI && IxI < 1) {
		return 1.0-IxI;
	}
	else {
		return 0.0;
	}
}

void RawIVSimpleSource::fillData(char* data, double xMin, double yMin, double zMin,
	double xMax, double yMax, double zMax,
	unsigned int xDim, unsigned int yDim, unsigned int zDim)
{
	
	double f_x, i_x, spacingX = (xMax-xMin)/(double)xDim;
	double f_y, i_y, spacingY = (yMax-yMin)/(double)yDim;
	double f_z, i_z, spacingZ = (zMax-zMin)/(double)zDim;

	double sum, sample, b1, b2, b3;
	unsigned int xClamp, yClamp, zClamp;

	for(unsigned int k=0; k<zDim; k++) {
		f_z = zMin + spacingZ * (double)k;
		i_z = floor(f_z);
		for(unsigned int j=0; j<yDim; j++) {
			f_y = yMin + spacingY * (double)j;
			i_y = floor(f_y);
			for(unsigned int i=0; i<xDim; i++) {
				f_x = xMin + spacingX * (double)i;
				i_x = floor(f_x);
				sum = 0;
				for(int m=0; m<2; m++) {
					b1 = linear(f_x - (i_x+m));
					xClamp = (i_x+m<=0.0 ? 0 : (unsigned int)i_x+m);
					xClamp = (xClamp>=m_DimX ? m_DimX-1 : xClamp);
					for(int n=0; n<2; n++) {
						b2 = linear(f_y - (i_y+n));
						yClamp = (i_y+m<=0.0 ? 0 : (unsigned int)i_y+m);
						yClamp = (yClamp>=m_DimY ? m_DimY-1 : yClamp);
						for(int o=0; o<2; o++) {
							b3 = linear(f_z - (i_z+o));
							zClamp = (i_z+m<=0.0 ? 0 : (unsigned int)i_z+m);
							zClamp = (zClamp>=m_DimZ ? m_DimZ-1 : zClamp);

							sample = m_Data[zClamp*m_DimY*m_DimX + 
								yClamp*m_DimX + xClamp];

							sum += sample *b1*b2*b3;
						}
					}
				}
				data[k*yDim*xDim + j*xDim + i] = (unsigned char)sum;
				
			}
		}
	}
	
}

void RawIVSimpleSource::fillThumbnail(char* data, 
	unsigned int xDim, unsigned int yDim, unsigned int zDim)
{
	fillData(data, m_MinX, m_MinY, m_MinZ, m_MaxX, m_MaxY, m_MaxZ, xDim, yDim, zDim);
}

void RawIVSimpleSource::fillData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim)
{
	qDebug("xDim: %d, yDim: %d, zDim: %d", xDim, yDim, zDim);

	// the test prevents divide by zero
	uint strideX = (xDim==1 ? 1 : (xMax-xMin) / (xDim-1));
	uint strideY = (yDim==1 ? 1 : (yMax-yMin) / (yDim-1));
	uint strideZ = (zDim==1 ? 1 : (zMax-zMin) / (zDim-1));

	unsigned char sample;

	unsigned int i,j,k;
	unsigned int counter = 0;

	for (k=zMin; k<=zMax; k+=strideZ) {
		for (j=yMin; j<=yMax; j+=strideY) {
			for (i=xMin; i<=xMax; i+=strideX) {
				sample = m_Data[k*m_DimY*m_DimX + j*m_DimX + i];
				data[(k-zMin)*yDim*xDim + (j-yMin)*xDim + (i-xMin)] = sample;
				counter++;
			}
		}
	}
	if (counter != xDim * yDim * zDim) {
		qDebug("done resampling. Warning! Expected %d, got %d samples", xDim*yDim*zDim, counter);
	}
	else {
		qDebug("done resampling.");
	}
}

bool RawIVSimpleSource::readHeader(FILE* fp)
{
	float floatVals[3];
	unsigned int uintVals[3];

	// read in the mins
	fread(floatVals, sizeof(float), 3, fp);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	setMinX(floatVals[0]);
	setMinY(floatVals[1]);
	setMinZ(floatVals[2]);

	// read in the maxs
	fread(floatVals, sizeof(float), 3, fp);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);
	setMaxX(floatVals[0]);
	setMaxY(floatVals[1]);
	setMaxZ(floatVals[2]);

	// ignore num verts and num cells, redundant
	fread(uintVals, sizeof(unsigned int), 2, fp);

	// read in the dimensions
	fread(uintVals, sizeof(unsigned int), 3, fp);
	if (isLittleEndian()) swapByteOrder(uintVals, 3);
	setDimX(uintVals[0]);
	setDimY(uintVals[1]);
	setDimZ(uintVals[2]);

	// ignore the "origin"  not sure what it means...probably redundant
	fread(floatVals, sizeof(float), 3, fp);

	// read in the span...redundant but will use it to check min and max
	fread(floatVals, sizeof(float), 3, fp);
	if (isLittleEndian()) swapByteOrder(floatVals, 3);

	// check min + span*dim == max to make sure file is consistant
	if ( !(
		fabs(m_MinX + floatVals[0]*(double)(m_DimX-1) - m_MaxX) < 0.0001 &&
		fabs(m_MinY + floatVals[1]*(double)(m_DimY-1) - m_MaxY) < 0.0001 &&
		fabs(m_MinZ + floatVals[2]*(double)(m_DimZ-1) - m_MaxZ) < 0.0001)) {

		// inconsistant, junk it all and replace with simple defaults
		m_MinX = 0; m_MinY = 0; m_MinZ = 0;
		m_MaxX = m_MinX + (double)(m_DimX-1)*floatVals[0];
		m_MaxY = m_MinY + (double)(m_DimY-1)*floatVals[1];
		m_MaxZ = m_MinZ + (double)(m_DimZ-1)*floatVals[2];

	}
	
	return true;
}

void RawIVSimpleSource::setDefaults()
{
	VolumeSource::setDefaults();
	m_Data = 0;
}

bool RawIVSimpleSource::allocateMemory(unsigned int num)
{
	destroyMemory();
	m_Data = new unsigned char[num];
	if (!m_Data) {
		return false;
	}
	else {
		return true;
	}
}

void RawIVSimpleSource::destroyMemory()
{
	delete [] m_Data;
	m_Data = 0;
}

