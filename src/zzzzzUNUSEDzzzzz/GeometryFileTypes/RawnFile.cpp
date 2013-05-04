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

// RawnFile.cpp: implementation of the RawnFile class.
//
//////////////////////////////////////////////////////////////////////

#include <GeometryFileTypes/RawnFile.h>
#include <cvcraw_geometry/Geometry.h>
#include <stdio.h>
//#include <qfileinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

RawnFile RawnFile::ms_RawnFileRepresentative;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RawnFile::RawnFile()
{

}

RawnFile::~RawnFile()
{

}

Geometry* RawnFile::loadFile(const string& fileName)
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
	int numverts, numtris;
	if (2!=fscanf(fp, "%d %d", &numverts, &numtris)) {
		//qDebug("Error reading in number of verts and tris");
		printf("Error reading in number of verts and tris\n");
		fclose(fp);
		return 0;
	}
	// make sure the number of verts & tris are positive
	if (numverts<0 || numtris<0) {
		//qDebug("Negative number of verts or tris");
		printf("Negative number of verts or tris\n");
		fclose(fp);
		return 0;
	}

	// initialize the geometry
	Geometry* geometry = new Geometry();
	geometry->AllocateTris(numverts, numtris);


	int c;
	// read in the verts
	for (c=0; c<numverts; c++) {
		// read in a single vert, which includes 
		// position and normal
		if (6!=fscanf(fp, "%f %f %f %f %f %f", 
			&(geometry->m_TriVerts[c*3+0]),
			&(geometry->m_TriVerts[c*3+1]),
			&(geometry->m_TriVerts[c*3+2]),
			&(geometry->m_TriVertNormals[c*3+0]),
			&(geometry->m_TriVertNormals[c*3+1]),
			&(geometry->m_TriVertNormals[c*3+2]))) {
			//qDebug("Error reading in vert # %d", c);
			printf("Error reading in vert # %d\n", c);
			delete geometry;
			fclose(fp);
			return 0;
		}
	}
	// read in the triangles
	for (c=0; c<numtris; c++) {
		// read in 3 integers for each triangle
		if (3!=fscanf(fp, "%u %u %u", 
			&(geometry->m_Tris[c*3+0]),
			&(geometry->m_Tris[c*3+1]),
			&(geometry->m_Tris[c*3+2]))) {
			//qDebug("Error reading in tri # %d", c);
			printf("Error reading in tri # %d\n", c);
			delete geometry;
			fclose(fp);
			return 0;
		}
		// the file might start indexing verts from 1 or 0
		// check if indexes go up to the num of verts or if they
		// start from 0
		if (geometry->m_Tris[c*3+0]==0 || geometry->m_Tris[c*3+1]==0 || geometry->m_Tris[c*3+2]==0 ) {
			zeroSeen = true;
		}
		if (geometry->m_Tris[c*3+0]==(unsigned int)numverts || geometry->m_Tris[c*3+1]==(unsigned int)numverts || geometry->m_Tris[c*3+2]==(unsigned int)numverts ) {
			maxSeen = true;
		}
		// cant have both!
		if (maxSeen && zeroSeen) {
			//qDebug("Found 0 & max in tri # %d", c);
			printf("Found 0 & max in tri # %d\n", c);
			delete geometry;
			fclose(fp);
			return 0;
		}

		// check the bounds on each vert
		if (geometry->m_Tris[c*3+0]>(unsigned int)numverts || geometry->m_Tris[c*3+1]>(unsigned int)numverts || geometry->m_Tris[c*3+2]>(unsigned int)numverts ) {
			//qDebug("Bounds error reading in tri # %d", c);
			printf("Bounds error reading in tri # %d\n", c);
			delete geometry;
			fclose(fp);
			return 0;
		}
	}

	// if the file starts from 1, we have to subtract 1 from each vert index
	if (maxSeen) {
		for (c=0; c<numtris; c++) {
			geometry->m_Tris[c*3+0]--;
			geometry->m_Tris[c*3+1]--;
			geometry->m_Tris[c*3+2]--;
		}
	}

	fclose(fp);
	geometry->SetTriNormalsReady();
	return geometry;
}

bool RawnFile::checkType(const string& fileName)
{
	bool zeroSeen = false, maxSeen = false;
	float f1,f2,f3,f4,f5,f6;
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
		return 0;
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
	if (2!=fscanf(fp, "%d %d", &numverts, &numtris)) {
		//qDebug("Error reading in number of verts and tris");
		printf("Error reading in number of verts and tris\n");
		fclose(fp);
		return false;
	}
	// make sure the number of verts & tris are positive
	if (numverts<0 || numtris<0) {
		//qDebug("Negative number of verts or tris");
		printf("Negative number of verts or tris\n");
		fclose(fp);
		return false;
	}


	int c;
	// read in the verts
	for (c=0; c<numverts; c++) {
		// read in a single vert, which includes 
		// position and normal
		if (6!=fscanf(fp, "%f %f %f %f %f %f", 
			&(f1),
			&(f2),
			&(f3),
			&(f4),
			&(f5),
			&(f6))) {
			//qDebug("Error reading in vert # %d", c);
			printf("Error reading in vert # %d\n", c);
			fclose(fp);
			return false;
		}
	}
	// read in the triangles
	for (c=0; c<numtris; c++) {
		// read in 3 integers for each triangle
		if (3!=fscanf(fp, "%u %u %u", 
			&(u1),
			&(u2),
			&(u3))) {
			//qDebug("Error reading in tri # %d", c);
			printf("Error reading in tri # %d\n", c);
			fclose(fp);
		}
		// the file might start indexing verts from 1 or 0
		// check if indexes go up to the num of verts or if they
		// start from 0
		if (u1==0 || u2==0 || u3==0 ) {
			zeroSeen = true;
		}
		if (u1==(unsigned int)numverts || u2==(unsigned int)numverts || u3==(unsigned int)numverts ) {
			maxSeen = true;
		}
		// cant have both!
		if (maxSeen && zeroSeen) {
			//qDebug("Found 0 & max in tri # %d", c);
			printf("Found 0 & max in tri # %d\n", c);
			fclose(fp);
			return false;
		}
		// check the bounds on each vert
		if (u1>(unsigned int)numverts || u2>(unsigned int)numverts || u3>(unsigned int)numverts ) {
			//qDebug("Bounds error reading in tri # %d", c);
			printf("Bounds error reading in tri # %d\n", c);
			fclose(fp);
			return false;
		}
	}

	fclose(fp);
	return true;
}

bool RawnFile::saveFile(const Geometry* geometry, const string& fileName)
{
	FILE* fp;
	// open the file
	fp = fopen(fileName.c_str(), "w");
	if (!fp) {
		printf("Error opening the output file\n");
		return false;
	}

	// write the number of verts & tris
	if (0>=fprintf(fp, "%d %d\n", geometry->m_NumTriVerts, geometry->m_NumTris)) {
		//qDebug("Error writing the number of verts and tris");
		printf("Error writing the number of verts and tris\n");
		return false;
	}

	unsigned int c;
	// write out the verts
	for (c=0; c<geometry->m_NumTriVerts; c++) {
		if (0>=fprintf(fp, "%f %f %f %f %f %f\n", 
			(geometry->m_TriVerts[c*3+0]),
			(geometry->m_TriVerts[c*3+1]),
			(geometry->m_TriVerts[c*3+2]),
			(geometry->m_TriVertNormals[c*3+0]),
			(geometry->m_TriVertNormals[c*3+1]),
			(geometry->m_TriVertNormals[c*3+2]))) {
			//qDebug("Error writing out vert # %d", c);
			printf("Error writing out vert # %d\n", c);
			fclose(fp);
			return false;
		}
	}
	// write out the tris
	for (c=0; c<geometry->m_NumTris; c++) {
		if (0>=fprintf(fp, "%d %d %d\n", 
			(geometry->m_Tris[c*3+0]),
			(geometry->m_Tris[c*3+1]),
			(geometry->m_Tris[c*3+2]))) {
			//qDebug("Error writing out tri # %d", c);
			printf("Error writing out tri # %d\n", c);
			fclose(fp);
			return false;
		}
	}
	fclose(fp);
	return true;
}


GeometryFileType* RawnFile::getRepresentative()
{
	return &ms_RawnFileRepresentative;
}

