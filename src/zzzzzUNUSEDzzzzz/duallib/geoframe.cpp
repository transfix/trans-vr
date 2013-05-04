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

#include <duallib/geoframe.h>
#include <stdio.h>
#include <math.h>
//#include <qfiledialog.h>

//#include <qfile.h>
//#include <qfileinfo.h>
//#include <qdir.h>

//#include <qtextstream.h>

geoframe::geoframe()
{
	normals = 0;
	numverts = 0;
	numtris = 0;
	numquads=0;
	
	vsize=10000;
	tsize=10000;
	qsize=10000;
	qsize=10000;
	
	verts  = (float (*)[3])malloc(sizeof(float[3]) * vsize);
	normals  = (float (*)[3])malloc(sizeof(float[3]) * tsize);
	triangles   = (unsigned int (*)[3])malloc(sizeof(unsigned int[3]) * tsize);
	quads = (unsigned int (*)[4])malloc(sizeof(unsigned int[4]) * qsize);
}

geoframe::~geoframe()
{
	free(triangles);
	free(quads);
	free(verts);
  	free(normals);
}

/*
void geoframe::loadgeoframe(const char * name, int num)
{
  char buffer[256];
  
  sprintf(buffer, "%s%05d.raw", name, num );
  //sprintf(buffer, "%s", name);
  QFile file(buffer);

  file.open(IO_ReadOnly);
  QTextStream stream(&file);

  stream >> numverts >> numtris;
  verts = new double[numverts*3];
  triangles = new int[numtris*3];
  int c;
  //int min = 1<<30;
  for (c=0; c<numverts; c++) {
    stream >> verts[c*3+0];
    stream >> verts[c*3+1];
    stream >> verts[c*3+2];
  }
  for (c=0; c<numtris; c++) {
    stream >> triangles[c*3+0];
    //min = obj.faceArray[c*3+0]<min?obj.faceArray[c*3+0]:min;
    stream >> triangles[c*3+1];
    //min = obj.faceArray[c*3+0]<min?obj.faceArray[c*3+1]:min;
    stream >> triangles[c*3+2];
    //min = obj.faceArray[c*3+0]<min?obj.faceArray[c*3+2]:min;
  }
  calculatenormals();
  calculateExtents();
}
*/

void cross(float* dest, const float* v1, const float* v2)
{
	dest[0] = v1[1]*v2[2] - v1[2]*v2[1];
	dest[1] = v1[2]*v2[0] - v1[0]*v2[2];
	dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


void geoframe::calculateTriangleNormal(float* norm, unsigned int c)
{
	
  float v1[3], v2[3];
  int vert;
  vert = triangles[c][0];
  v1[0] = v2[0] = -verts[vert][0];
  v1[1] = v2[1] = -verts[vert][1];
  v1[2] = v2[2] = -verts[vert][2];

  vert = triangles[c][1];
  v1[0] += verts[vert][0];
  v1[1] += verts[vert][1];
  v1[2] += verts[vert][2];

  vert = triangles[c][2];
  v2[0] += verts[vert][0];
  v2[1] += verts[vert][1];
  v2[2] += verts[vert][2];

  cross(norm, v1, v2);
  
}

void geoframe::calculatenormals()
{
	
	
	int c, vert;
	float normal[3];
	float len;
	
	// for each triangle
	for (c=0; c<numtris; c++) {
		calculateTriangleNormal(normal, c);
		normals[c][0] = normal[0];
		normals[c][1] = normal[1];
		normals[c][2] = normal[2];
	}
	
	// normalize the vectors
	for (vert=0; vert<numtris; vert++) {
		len = (float) sqrt(
			normals[vert][0] * normals[vert][0] +
			normals[vert][1] * normals[vert][1] +
			normals[vert][2] * normals[vert][2]);
		normals[vert][0]/=len;
		normals[vert][1]/=len;
		normals[vert][2]/=len;
	}
	
}

void geoframe::calculateExtents()
{
  int c;
  float max_x, min_x;
  float max_y, min_y;
  float max_z, min_z;
  float value;

  for (c=0; c<numverts; c++) {
    if (c==0) {
      max_x = min_x = verts[c][0];
      max_y = min_y = verts[c][1];
      max_z = min_z = verts[c][2];
    }
    else {
      value = verts[c][0];
      max_x = (value>max_x?value:max_x);
      min_x = (value<min_x?value:min_x);

      value = verts[c][1];
      max_y = (value>max_y?value:max_y);
      min_y = (value<min_y?value:min_y);

      value = verts[c][2];
      max_z = (value>max_z?value:max_z);
      min_z = (value<min_z?value:min_z);
    }
  }

    biggestDim = (max_y-min_y>max_x-min_x?max_y-min_y:max_x-min_x);
    biggestDim = (max_z-min_z>biggestDim?max_z-min_z:biggestDim);
    centerx = (max_x+min_x)/2.0;
    centery = (max_y+min_y)/2.0;
    centerz = (max_z+min_z)/2.0;
}

/*void geoframe::display()
{
  int vert;
  glBegin(GL_TRIANGLES);
  for (int c=0; c<numtris; c++) {
    vert = triangles[c*3+0];
    glNormal3d( 
      normals[vert*3+0],
      normals[vert*3+1],
      normals[vert*3+2]);
    glVertex3d(
      verts[vert*3+0],
      verts[vert*3+1],
      verts[vert*3+2]);
    vert = triangles[c*3+1];
    glNormal3d( 
      normals[vert*3+0],
      normals[vert*3+1],
      normals[vert*3+2]);
    glVertex3d(
      verts[vert*3+0],
      verts[vert*3+1],
      verts[vert*3+2]);
    vert = triangles[c*3+2];
    glNormal3d( 
      normals[vert*3+0],
      normals[vert*3+1],
      normals[vert*3+2]);
    glVertex3d(
      verts[vert*3+0],
      verts[vert*3+1],
      verts[vert*3+2]);
  }
  glEnd();
}*/
