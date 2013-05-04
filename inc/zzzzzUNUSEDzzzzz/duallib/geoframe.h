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

#ifndef __GEOFRAME_H__

#define __GEOFRAME_H__



#include<stdio.h>
#include<stdlib.h>

//#include<malloc.h>

#include<assert.h>



#define GRAD



class geoframe {

public:

	geoframe();

	~geoframe();

	

	void Clear() { numtris=0; numverts=0; numquads=0; }

	int getNTri(void) { return numtris; }

	int getNVert(void) { return numverts; }



	int AddQuad(unsigned int* v , int num)

	{

		assert (num==3 || num==4);



		if (numquads >= qsize) {

			qsize<<=1;

			quads = (unsigned int (*)[4])realloc(quads, sizeof(unsigned int[4]) * qsize);

			// grad addtion

//#ifndef GRAD

//			normals = (float(*)[3])realloc(normals,sizeof(float[3])*tsize);

//#endif

			

		}





		if (num==4) {

			quads[numquads][0] = v[0];

			quads[numquads][1] = v[1];

			quads[numquads][2] = v[2];

			quads[numquads][3] = v[3];

			return numquads++;

						

		} else if (num==3) {

			triangles[numtris][0] = v[0];

			triangles[numtris][1] = v[1];

			triangles[numtris][2] = v[2];

			return numtris++;

		}

		
		return -1;

	}



	int AddTri(unsigned int v1,unsigned int v2,unsigned int v3)

	{

		if (numtris+1 >= tsize) {

			tsize<<=1;

			triangles = (unsigned int (*)[3])realloc(triangles, sizeof(unsigned int[3]) * tsize);

			// grad addtion

//#ifndef GRAD

//			normals = (float(*)[3])realloc(normals,sizeof(float[3])*tsize);

//#endif

			

		}

		

		triangles[numtris][0] = v1;

		triangles[numtris][1] = v2;

		triangles[numtris][2] = v3;

		

		return numtris++;

	}



	int AddVert(float v_pos[3], float norm[3])

	{

		int i;

		if (numverts+1 > vsize) {

			vsize<<=1;

			verts = (float (*)[3])realloc(verts,sizeof(float[3])*vsize);

//#ifdef GRAD

			// grad addtion

			 normals = (float(*)[3])realloc(normals,sizeof(float[3])*vsize);

//#endif GRAD



		}

		for (i=0;i<3;i++)

			verts[numverts][i]=v_pos[i];

//#ifdef GRAD

		for (i=0;i<3;i++)

			normals[numverts][i]=norm[i];

//#endif 

		return numverts++;

	}



	int center_vtx(int v1,int v2,int v3)

	{

		float center_vtx[3],norm[3];

		for (int i=0; i<3; i++) {

			center_vtx[i]=(verts[v1][i] + verts[v2][i] + verts[v3][i])/3.f;

			// grad addtion

			 norm[i]=(normals[v1][i]+normals[v2][i]+normals[v3][i])/3.f;

		}

		

		return AddVert(center_vtx,norm);

	}

	

	//void loadgeoframe(const char * name, int num);

	void calculatenormals();

	void calculateTriangleNormal(float* norm, unsigned int c);

	void calculateExtents();

	

	//void display();

	

	

	int numverts;

	int numtris;

	int numquads;

	int tsize,vsize,qsize;

	float (*verts)[3];

	float (*normals)[3];

	unsigned int (*triangles)[3];

	unsigned int (*quads)[4];

	

	double biggestDim;

	double centerx, centery, centerz;

};



#endif //__GEOFRAME_H__
