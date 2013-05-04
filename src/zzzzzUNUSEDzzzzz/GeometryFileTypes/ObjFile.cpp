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

/* $Id: ObjFile.cpp 1787 2010-06-01 18:51:21Z transfix $ */

// ObjFile.cpp: implementation of the ObjFile class.
//
//////////////////////////////////////////////////////////////////////

#include <vector>
#include <GeometryFileTypes/ObjFile.h>
#include <cvcraw_geometry/Geometry.h>
#include <stdio.h>
#include <string.h>
			 //#include <qfileinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

ObjFile ObjFile::ms_ObjFileRepresentative;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ObjFile::ObjFile()
{

}

ObjFile::~ObjFile()
{

}

Geometry* ObjFile::loadFile(const string& fileName)
{
  // 2 loops, once get num to allocate, second, read actual values;
  int numVerts = 0; // probably redundant
  int numTris = 0;

  std::vector<double> vertices; // ordered list of x, y, z
  std::vector<double> texcoords; // ordered list of u, v
  std::vector<double> normals; // ordered list of nx, ny, nz
  std::vector<int> vindices; // indices into vertex list
  std::vector<int> tindices; // indices into texture list
  std::vector<int> nindices; // indices into normal list

  {
    FILE* fp = fopen( fileName.c_str(), "r" );
    if( !fp )
      {
	fprintf(stderr,"Could not open file %s\n",fileName.c_str());
	return 0;
      }
    char line[1024];

    bool vPresent = false;
    bool nPresent = false;
    bool tPresent = false;

    while( fgets( line, 1023, fp ) ) 
      {
	if( strlen(line) > 3 )
	  {
	    if( line[0] == 'v' && line[1] == ' ' ) 
	      {
		double x, y, z;
		if( sscanf( line, "v %lf %lf %lf\n", &x, &y, &z ) == 3 ) 
		  {
		    vertices.push_back(x);
		    vertices.push_back(y);
		    vertices.push_back(z);
		    numVerts ++;
		    vPresent = true;
		  }
	      }
	  }
	if( strlen(line) > 4 )
	  {
	    if( line[0] == 'v' && line[1] == 'n' && line[2] == ' ' ) 
	      {
		double nx, ny, nz;
		if( sscanf( line, "vn %lf %lf %lf\n", &nx, &ny, &nz ) == 3 ) 
		  {
		    nPresent = true;
		  }
	      }
	  }
	if( strlen(line) > 4 )
	  {
	    if( line[0] == 'v' && line[1] == 't' && line[2] == ' ' ) 
	      {
		double u, v;
		if( sscanf( line, "vt %lf %lf\n", &u, &v ) == 2 ) 
		  {
		    texcoords.push_back(u);
		    texcoords.push_back(v);
		    tPresent = true;
		  }
	      }
	  }
	if( strlen(line) > 3 )
	  {
	    if( line[0] == 'f' && line[1] == ' ' ) 
	      {
		int idx_1, idx_2, idx_3;
		int t_idx_1, t_idx_2, t_idx_3;
		int n_idx_1, n_idx_2, n_idx_3;

		if( vPresent && !nPresent && !tPresent )
		  {
		    if( sscanf( line, "f %d %d %d", &idx_1, &idx_2, &idx_3 ) == 3 )
		      {
			vindices.push_back(idx_1-1);
			vindices.push_back(idx_2-1);
			vindices.push_back(idx_3-1);
			numTris ++;
		      }
		  }
		else if( vPresent && !nPresent && tPresent )
		  {
		    if( sscanf( line, "f %d/%d %d/%d %d/%d", &idx_1, &t_idx_1, &idx_2, &t_idx_2, &idx_3, &t_idx_3 ) == 6 )
		      {
			vindices.push_back(idx_1-1);
			vindices.push_back(idx_2-1);
			vindices.push_back(idx_3-1);

			tindices.push_back(t_idx_1-1);
			tindices.push_back(t_idx_2-1);
			tindices.push_back(t_idx_3-1);
			numTris ++;
		      }
		  }
		else if( vPresent && nPresent && !tPresent )
		  {
		    if( sscanf( line, "f %d//%d %d//%d %d//%d", &idx_1, &n_idx_1, &idx_2, &n_idx_2, &idx_3, &n_idx_3 ) == 6 )
		      {
			vindices.push_back(idx_1-1);
			vindices.push_back(idx_2-1);
			vindices.push_back(idx_3-1);
			numTris ++;
		      }
		  }
		else if( vPresent && nPresent && tPresent )
		  {
		    if( sscanf( line, "f %d/%d/%d %d/%d/%d %d/%d/%d", 
				&idx_1, &t_idx_1, &n_idx_1, 
				&idx_2, &t_idx_2, &n_idx_2, 
				&idx_3, &t_idx_3, &t_idx_3 ) == 9 )
		      {
			vindices.push_back(idx_1-1);
			vindices.push_back(idx_2-1);
			vindices.push_back(idx_3-1);

			tindices.push_back(t_idx_1-1);
			tindices.push_back(t_idx_2-1);
			tindices.push_back(t_idx_3-1);
			numTris ++;
		      }
		  }
	      }
	  }
      }
    fclose( fp );
  }
	
  if( !numVerts || !numTris )
    {
      vertices.clear();
      texcoords.clear();
      vindices.clear();
      tindices.clear();
      fprintf(stderr,"Invalid numVerts or numTris (%d,%d)\n",numVerts,numTris);
      return 0;
    }

  Geometry* geometry = new Geometry();
  geometry->AllocateTris( numVerts, numTris );

  {
    int i;
    for( i=0; i<numVerts*3; i++ )
      {
	geometry->m_TriVerts[i] = vertices[i];
      }
    for( i=0; i<numTris*3; i++ )
      {
	geometry->m_Tris[i] = vindices[i];
      }
    if( tindices.size() )
      {
	geometry->AllocateTriTexCoords();
	for( i=0; i<tindices.size()/3; i++ )
	  {
	    int tex_index_0 = tindices.at(i*3+0);
	    int tex_index_1 = tindices.at(i*3+1);
	    int tex_index_2 = tindices.at(i*3+2);

	    double u0 = texcoords.at(tex_index_0*2+0);
	    double v0 = texcoords.at(tex_index_0*2+1);

	    double u1 = texcoords.at(tex_index_1*2+0);
	    double v1 = texcoords.at(tex_index_1*2+1);

	    double u2 = texcoords.at(tex_index_2*2+0);
	    double v2 = texcoords.at(tex_index_2*2+1);

	    int vert0 = geometry->m_Tris[i*3+0];
	    int vert1 = geometry->m_Tris[i*3+1];
	    int vert2 = geometry->m_Tris[i*3+2];

	    geometry->m_TriVertTexCoords[vert0*2+0] = v0;
	    geometry->m_TriVertTexCoords[vert0*2+1] = u0;

	    geometry->m_TriVertTexCoords[vert1*2+0] = v1;
	    geometry->m_TriVertTexCoords[vert1*2+1] = u1;

	    geometry->m_TriVertTexCoords[vert2*2+0] = v2;
	    geometry->m_TriVertTexCoords[vert2*2+1] = u2;
	  }
      }
  }

  vertices.clear();
  vindices.clear();
  tindices.clear();
  texcoords.clear();

  return geometry;
}

bool ObjFile::checkType(const string& fileName)
{
  Geometry *g = loadFile(fileName);
  if(g)
    {
      delete g; return true;
    }
  else
    return false;
}

bool ObjFile::saveFile(const Geometry* geometry, const string& fileName)
{
  if( !geometry ) return false;
  if( (geometry->m_NumTriVerts <= 0) || !geometry->m_TriVerts || (geometry->m_NumTris <= 0) || !geometry->m_Tris) return false;

  FILE* fp;
  // open the file
  fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    printf("Error opening the output file\n");
    return false;
  }

  fprintf( fp, "# OBJ file exported from VolumeRover.\n");
  fprintf( fp, "# CVC, The University of Texas at Austin.\n");
  // write the number of verts & tris
  if (0>=fprintf(fp, "# Number of vertices, triangles: %d %d\n", geometry->m_NumTriVerts, geometry->m_NumTris)) {
    //qDebug("Error writing the number of verts and tris");
    printf("Error writing the number of verts and tris\n");
    return false;
  }

  unsigned int c;
  // write out the verts
  for (c=0; c<geometry->m_NumTriVerts; c++) 
    {	
      if (0>=fprintf(fp, "v %f %f %f\n", 
		     (geometry->m_TriVerts[c*3+0]),
		     (geometry->m_TriVerts[c*3+1]),
		     (geometry->m_TriVerts[c*3+2]))) {
	//qDebug("Error writing out vert # %d", c);
	printf("Error writing out vert # %d\n", c);
	fclose(fp);
	return false;
      }
    }
  if( geometry->m_TriVertNormals )
    {
      for (c=0; c<geometry->m_NumTriVerts; c++) 
	{	
	  if (0>=fprintf(fp, "vn %f %f %f\n", 
			 (geometry->m_TriVertNormals[c*3+0]),
			 (geometry->m_TriVertNormals[c*3+1]),
			 (geometry->m_TriVertNormals[c*3+2]))) {
	    //qDebug("Error writing out vert # %d", c);
	    printf("Error writing out vert normal # %d\n", c);
	    fclose(fp);
	    return false;
	  }
	}
    }
  // write out the tris
  if( geometry->m_TriVertNormals )
    {
      for (c=0; c<geometry->m_NumTris; c++) {
	if (0>=fprintf(fp, "f %d//%d %d//%d %d//%d\n", 
		       (geometry->m_Tris[c*3+0]+1),(geometry->m_Tris[c*3+0]+1),
		       (geometry->m_Tris[c*3+1]+1),(geometry->m_Tris[c*3+1]+1),
		       (geometry->m_Tris[c*3+2]+1),(geometry->m_Tris[c*3+2]+1)
		       )) {
	  printf("Error writing out tri # %d\n", c);
	  fclose(fp);
	  return false;
	}
      }
    }
  else
    {
      for (c=0; c<geometry->m_NumTris; c++) {
	if (0>=fprintf(fp, "f %d %d %d\n", 
		       (geometry->m_Tris[c*3+0]+1),
		       (geometry->m_Tris[c*3+1]+1),
		       (geometry->m_Tris[c*3+2]+1)
		       )) {
	  printf("Error writing out tri # %d\n", c);
	  fclose(fp);
	  return false;
	}
      }
    }
  fclose(fp);
  return true;
}


GeometryFileType* ObjFile::getRepresentative()
{
  return &ms_ObjFileRepresentative;
}

