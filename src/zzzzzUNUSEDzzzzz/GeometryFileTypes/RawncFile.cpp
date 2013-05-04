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

// RawncFile.cpp: implementation of the RawncFile class.
//
//////////////////////////////////////////////////////////////////////

#include <GeometryFileTypes/RawncFile.h>
#include <cvcraw_geometry/Geometry.h>
#include <stdio.h>
//#include <qfileinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

RawncFile RawncFile::ms_RawncFileRepresentative;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RawncFile::RawncFile()
{

}

RawncFile::~RawncFile()
{

}

Geometry* RawncFile::loadFile(const string& fileName)
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
	geometry->AllocateTriVertColors();


	int c;
	// read in the verts
	for (c=0; c<numverts; c++) {
		// read in a single vert, which includes 
		// position and normal
		if (9!=fscanf(fp, "%f %f %f %f %f %f %f %f %f", 
			&(geometry->m_TriVerts[c*3+0]),
			&(geometry->m_TriVerts[c*3+1]),
			&(geometry->m_TriVerts[c*3+2]),
			&(geometry->m_TriVertNormals[c*3+0]),
			&(geometry->m_TriVertNormals[c*3+1]),
			&(geometry->m_TriVertNormals[c*3+2]),
			&(geometry->m_TriVertColors[c*3+0]),
			&(geometry->m_TriVertColors[c*3+1]),
			&(geometry->m_TriVertColors[c*3+2]))) {
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

bool RawncFile::checkType(const string& fileName)
{
	bool zeroSeen = false, maxSeen = false;
	float f1,f2,f3,f4,f5,f6,f7,f8,f9;
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
		if (9!=fscanf(fp, "%f %f %f %f %f %f %f %f %f", 
			&(f1),
			&(f2),
			&(f3),
			&(f4),
			&(f5),
			&(f6),
			&(f7),
			&(f8),
			&(f9))) {
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

bool RawncFile::saveFile(const Geometry* geometry, const string& fileName)
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
		if (geometry->m_TriVertColors) {
			if (0>=fprintf(fp, "%f %f %f %f %f %f %f %f %f\n", 
				(geometry->m_TriVerts[c*3+0]),
				(geometry->m_TriVerts[c*3+1]),
				(geometry->m_TriVerts[c*3+2]),
				(geometry->m_TriVertNormals[c*3+0]),
				(geometry->m_TriVertNormals[c*3+1]),
				(geometry->m_TriVertNormals[c*3+2]),
				(geometry->m_TriVertColors[c*3+0]),
				(geometry->m_TriVertColors[c*3+1]),
				(geometry->m_TriVertColors[c*3+2]))) {
				//qDebug("Error writing out vert # %d", c);
				printf("Error writing out vert # %d\n", c);
				fclose(fp);
				return false;
			}
		}
		else {
			if (0>=fprintf(fp, "%f %f %f %f %f %f %f %f %f\n", 
				(geometry->m_TriVerts[c*3+0]),
				(geometry->m_TriVerts[c*3+1]),
				(geometry->m_TriVerts[c*3+2]),
				(geometry->m_TriVertNormals[c*3+0]),
				(geometry->m_TriVertNormals[c*3+1]),
				(geometry->m_TriVertNormals[c*3+2]),
				(1.0),
				(1.0),
				(1.0))) {
				//qDebug("Error writing out vert # %d", c);
				printf("Error writing out vert # %d\n", c);
				fclose(fp);
				return false;
			}
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


GeometryFileType* RawncFile::getRepresentative()
{
	return &ms_RawncFileRepresentative;
}

