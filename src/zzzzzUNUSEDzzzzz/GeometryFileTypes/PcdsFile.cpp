/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	         Jose Rivera <transfix@ices.utexas.edu>
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

// Pcds.cpp: implementation of the Pcds class.
//
//////////////////////////////////////////////////////////////////////

/* $Id: PcdsFile.cpp 1787 2010-06-01 18:51:21Z transfix $ */

#include <GeometryFileTypes/PcdsFile.h>
#include <cvcraw_geometry/Geometry.h>
#include <stdio.h>
//#include <qfileinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

//TODO: this will explode if we try to save a surface as a point cloud... fix that later!

PcdsFile PcdsFile::ms_PcdsFileRepresentative;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

PcdsFile::PcdsFile()
{

}

PcdsFile::~PcdsFile()
{

}

Geometry* PcdsFile::loadFile(const string& fileName)
{
	bool zeroSeen = false, maxSeen = false;
	//QFileInfo fileInfo(fileName);

	//if (!fileInfo.exists()) {
	//	qDebug("File does not exist");
	//	return 0;
	//}
	struct stat fst;
	if (stat(fileName.c_str(), &fst) == -1) {
		printf("File does not exist\n");
		return 0;
	}

	//QString absFilePath = fileInfo.absFilePath();

	//FILE* fp = fopen(absFilePath, "r");
	FILE* fp = fopen(fileName.c_str(), "r");
	if (!fp) {
		//qDebug("Error opening file");
		printf("Error opening file\n");
		return 0;
	}

	// get the number of verts and triangles
	int numverts;
	if (1!=fscanf(fp, "%d", &numverts)) {
	  //qDebug("Error reading in number of verts and tris");
	  printf("Error reading in number of verts and tris\n");
	  fclose(fp);
	  return 0;
	}
	// make sure the number of verts & tris are positive
	if (numverts<0) {
		//qDebug("Negative number of verts or tris");
		printf("Negative number of verts\n");
		fclose(fp);
		return 0;
	}

	// initialize the geometry
	Geometry* geometry = new Geometry();
	geometry->AllocatePoints(numverts);
	geometry->AllocatePointScalars();

	int c;
	// read in the verts
	for (c=0; c<numverts; c++) {
	  // read in a single vert
	  if (4!=fscanf(fp, "%f %f %f %f", 
			&(geometry->m_Points[c*3+0]),
			&(geometry->m_Points[c*3+1]),
			&(geometry->m_Points[c*3+2]),
			&(geometry->m_PointScalars[c]))) {
	    //qDebug("Error reading in vert # %d", c);
	    printf("Error reading in vert # %d\n", c);
	    delete geometry;
	    fclose(fp);
	    return 0;
	  }
	}
	
	fclose(fp);
	return geometry;
}

bool PcdsFile::checkType(const string& fileName)
{
	bool zeroSeen = false, maxSeen = false;
	float f1,f2,f3,f4;
	unsigned int u1,u2,u3;
	// go through the file and make sure we understand it
	//QFileInfo fileInfo(fileName);

	//if (!fileInfo.exists()) {
	//	qDebug("File does not exist");
	//	return false;
	//}
	struct stat fst;
	if (stat(fileName.c_str(), &fst) == -1) {
		printf("File does not exist\n");
		return false;
	}

	//QString absFilePath = fileInfo.absFilePath();

	//FILE* fp = fopen(absFilePath, "r");
	FILE* fp = fopen(fileName.c_str(), "r");
	if (!fp) {
		//qDebug("Error opening file");
		printf("Error opening file\n");
		return false;
	}

	// get the number of verts and triangles
	int numverts, numtris;
	if (1!=fscanf(fp, "%d", &numverts)) {
		//qDebug("Error reading in number of verts and tris");
		printf("Error reading in number of verts\n");
		fclose(fp);
		return false;
	}
	// make sure the number of verts & tris are positive
	if (numverts<0) {
		//qDebug("Negative number of verts or tris");
		printf("Negative number of verts\n");
		fclose(fp);
		return false;
	}


	int c;
	// read in the verts
	for (c=0; c<numverts; c++) {
		// read in a single vert
	  if (4!=fscanf(fp, "%f %f %f %f", 
			&(f1),
			&(f2),
			&(f3),
			&(f4))) {
			//qDebug("Error reading in vert # %d", c);
			printf("Error reading in vert # %d\n", c);
			fclose(fp);
			return false;
		}
	}
	
	fclose(fp);
	return true;
}

bool PcdsFile::saveFile(const Geometry* geometry, const string& fileName)
{
	FILE* fp;
	// open the file
	fp = fopen(fileName.c_str(), "w");
	if (!fp) {
		printf("Error opening the output file\n");
		return false;
	}

	// write the number of verts & tris
	if (0>=fprintf(fp, "%d\n", geometry->m_NumPoints)) {
		//qDebug("Error writing the number of verts and tris");
		printf("Error writing the number of points\n");
		fclose(fp);
		return false;
	}

	unsigned int c;
	// write out the verts
	for (c=0; c<geometry->m_NumPoints; c++) {
	  if(geometry->m_PointScalars)
	    {
	      if (0>=fprintf(fp, "%f %f %f %f\n", 
			     (geometry->m_Points[c*3+0]),
			     (geometry->m_Points[c*3+1]),
			     (geometry->m_Points[c*3+2]),
			     (geometry->m_PointScalars[c]))) {
		//qDebug("Error writing out vert # %d", c);
		printf("Error writing out point # %d\n", c);
		fclose(fp);
		return false;
	      }
	    }
	  else //no point scalars, so just use 0.0
	    {
	      if (0>=fprintf(fp, "%f %f %f %f\n", 
			     (geometry->m_Points[c*3+0]),
			     (geometry->m_Points[c*3+1]),
			     (geometry->m_Points[c*3+2]),
			     0.0)) {
		//qDebug("Error writing out vert # %d", c);
		printf("Error writing out point # %d\n", c);
		fclose(fp);
		return false;
	      }
	    }
	}

	fclose(fp);
	return true;
}

GeometryFileType* PcdsFile::getRepresentative()
{
	return &ms_PcdsFileRepresentative;
}

