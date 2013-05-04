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

#include <duallib/cellQueue.h>

#include <duallib/duallib.h>

#include <duallib/cubes.h>

#include<stdio.h>

#include <duallib/geoframe.h>

#include<assert.h>

#include <duallib/pcio.h>

#include<stdlib.h>

#include <duallib/bishoulder.h>



Octree::Octree()

{

	vol_min=10000; vol_max=-10000;

	vtx_idx_arr=NULL;

	refine_flag=NULL;

	ebit=NULL;



}



Octree::~Octree()

{

	free(refine_flag);

	//free(orig_vol);

	free(vtx_idx_arr);



}



void Octree::Octree_init(unsigned char* char_data, int Dim[3], float Orig[3], float Span[3])

{

	int i;

	
	leaf_num=0;



//	read_header();   // read dimension, span, orig and so on

	for (i=0;i<3;i++) {

		dim[i]=Dim[i];

		orig[i]=Orig[i];

		span[i]=Span[i];

	}

	nverts=dim[0]*dim[1]*dim[2];

	ncells=(dim[0]-1)*(dim[1]-1)*(dim[2]-1);



	

	// octree information

	refine_flag = (char*)realloc(refine_flag, sizeof(char)*ncells);

	memset(refine_flag,0,sizeof(char)*ncells);

	orig_vol    = char_data;

	ebit		= (char*)realloc(ebit, sizeof(char)*(ncells));

	memset(ebit,0,ncells);



	if (vtx_idx_arr) free(vtx_idx_arr);

	vtx_idx_arr = (int*)malloc(sizeof(int)*ncells);

	for (int k=0;k<ncells;k++) vtx_idx_arr[k]=-1;

}





// given the path of a volume,

// load the volume data and construct octree from that

// allocate associate arrays

void Octree::Octree_init(const char* rawiv_fname)

{

//	int i;	

	vol_fp=fopen(rawiv_fname,"rb");

	

	if (vol_fp==NULL) {

		printf("wrong name : %s\n",rawiv_fname);

		return;

	}



	read_header();   // read dimension, span, orig and so on



	orig_vol    = (unsigned char*)malloc(sizeof(unsigned char)*nverts);

	read_data();

	Octree_init(orig_vol,dim,orig,span);

}









int Octree::is_min_edge(int oc_id, int e_id, unsigned int* vtx, int& vtx_num, int intersect_id, geoframe& geofrm)

{

	int x,y,z, i;
//	int level;

	unsigned int temp_vtx[4];

	

	cell2xyz(oc_id,x,y,z);

	

	vtx_num=4;



	switch (e_id) {

		case 0 : 

			if (is_refined(x,y,z-1) || is_refined(x,y-1,z-1) || is_refined(x,y-1,z)) return 0;

			temp_vtx[1]=min_vtx(x,y,z-1,geofrm);

			temp_vtx[2]=min_vtx(x,y-1,z-1,geofrm);

			temp_vtx[3]=min_vtx(x,y-1,z,geofrm);

			break;



		case 1 : 

			if (is_refined(x,y-1,z) || is_refined(x+1,y-1,z) || is_refined(x+1,y,z)) return 0;

			temp_vtx[1]=min_vtx(x+1,y,z,geofrm);

			temp_vtx[2]=min_vtx(x+1,y-1,z,geofrm);

			temp_vtx[3]=min_vtx(x,y-1,z,geofrm);

			break;



		case 2 : 

			if (is_refined(x,y,z+1) || is_refined(x,y-1,z+1) || is_refined(x,y-1,z)) return 0;

			temp_vtx[1]=min_vtx(x,y,z+1,geofrm);

			temp_vtx[2]=min_vtx(x,y-1,z+1,geofrm);

			temp_vtx[3]=min_vtx(x,y-1,z,geofrm);

			break;



		case 3 : 

			if (is_refined(x,y-1,z) || is_refined(x-1,y-1,z) || is_refined(x-1,y,z)) return 0;

			temp_vtx[1]=min_vtx(x,y-1,z,geofrm);

			temp_vtx[2]=min_vtx(x-1,y-1,z,geofrm);

			temp_vtx[3]=min_vtx(x-1,y,z,geofrm);

			break;



		case 4 : 

			if (is_refined(x,y,z-1) || is_refined(x,y+1,z-1) || is_refined(x,y+1,z)) return 0;

			temp_vtx[1]=min_vtx(x,y+1,z,geofrm);

			temp_vtx[2]=min_vtx(x,y+1,z-1,geofrm);

			temp_vtx[3]=min_vtx(x,y,z-1,geofrm);

			break;



		case 5 :

			if (is_refined(x,y+1,z) || is_refined(x+1,y,z) || is_refined(x+1,y+1,z)) return 0;

			temp_vtx[1]=min_vtx(x,y+1,z,geofrm);

			temp_vtx[2]=min_vtx(x+1,y+1,z,geofrm);

			temp_vtx[3]=min_vtx(x+1,y,z,geofrm);

			break;



		case 6 :

			if (is_refined(x,y+1,z) || is_refined(x,y+1,z+1) || is_refined(x,y,z+1)) return 0;

			temp_vtx[1]=min_vtx(x,y+1,z,geofrm);

			temp_vtx[2]=min_vtx(x,y+1,z+1,geofrm);

			temp_vtx[3]=min_vtx(x,y,z+1,geofrm);

			break;



		case 7 :

			if (is_refined(x-1,y,z) || is_refined(x-1,y+1,z) || is_refined(x,y+1,z)) return 0;

			temp_vtx[1]=min_vtx(x-1,y,z,geofrm);

			temp_vtx[2]=min_vtx(x-1,y+1,z,geofrm);

			temp_vtx[3]=min_vtx(x,y+1,z,geofrm);

			break;



		case 8 :

			if (is_refined(x,y,z-1) || is_refined(x-1,y,z-1) || is_refined(x-1,y,z)) return 0;

			temp_vtx[1]=min_vtx(x-1,y,z,geofrm);

			temp_vtx[2]=min_vtx(x-1,y,z-1,geofrm);

			temp_vtx[3]=min_vtx(x,y,z-1,geofrm);

			break;



		case 9 :

			if (is_refined(x,y,z-1) || is_refined(x+1,y,z-1) || is_refined(x+1,y,z)) return 0;

			temp_vtx[1]=min_vtx(x,y,z-1,geofrm);

			temp_vtx[2]=min_vtx(x+1,y,z-1,geofrm);

			temp_vtx[3]=min_vtx(x+1,y,z,geofrm);

			break;



		case 10 :

			if (is_refined(x,y,z+1) || is_refined(x-1,y,z+1) || is_refined(x-1,y,z)) return 0;

			temp_vtx[1]=min_vtx(x,y,z+1,geofrm);

			temp_vtx[2]=min_vtx(x-1,y,z+1,geofrm);

			temp_vtx[3]=min_vtx(x-1,y,z,geofrm);

			break;



		case 11 :

			if (is_refined(x,y,z+1) || is_refined(x+1,y,z+1) || is_refined(x+1,y,z)) return 0;

			temp_vtx[1]=min_vtx(x+1,y,z,geofrm);

			temp_vtx[2]=min_vtx(x+1,y,z+1,geofrm);

			temp_vtx[3]=min_vtx(x,y,z+1,geofrm);

			break;

	}

	temp_vtx[0]=min_vtx(x,y,z,geofrm);



	assert(intersect_id==1 || intersect_id==-1);



	if (intersect_id==1) 

		for (i=0;i<4;i++) vtx[i]=temp_vtx[i];

	else if (intersect_id==-1)

		for (i=0;i<4;i++) vtx[i]=temp_vtx[3-i];





	return 1;



}



void Octree::get_solution(int oc_id, float* pos)

{

	int x,y,z;

	float val[8];

	Vtx Bishoulder;

	

	cell2xyz(oc_id,x,y,z);

	getCellValues(oc_id, val);



	GetBishoulder(val , Bishoulder ,iso_val) ;



	if (Bishoulder.x>=1 || Bishoulder.x<=0) 

	{

		Bishoulder.x=0.5;

	}

	if (Bishoulder.y>=1 || Bishoulder.y<=0) 

	{

		Bishoulder.y=0.5;

	}

	if (Bishoulder.z>=1 || Bishoulder.z<=0) 

	{

		Bishoulder.z=0.5;

	}



	pos[0]=orig[0]+span[0]*(x+Bishoulder.x);

	pos[1]=orig[1]+span[1]*(y+Bishoulder.y);

	pos[2]=orig[2]+span[2]*(z+Bishoulder.z);



	//printf("2 qef pos : %f %f %f\n",pos[0],pos[1],pos[2]);



}



void Octree::get_vtx(int x, int y, int z, float* pos)

{

	int oc_id;

	oc_id = xyz2cell(x,y,z);

	get_solution(oc_id,pos);

}





int Octree::min_vtx(int x, int y, int z, geoframe& geofrm)

{

	int tx,ty,tz;

	tx=x; ty=y; tz=z;

	float vtx[3], norm[3];

	int vi;



	assert( tx>=0 && ty>=0 && tz>=0 );

	assert( !is_refined(tx,ty,tz) );



	



	if ((vi=vtx_idx_arr[xyz2cell(tx,ty,tz)])==-1) {

		get_vtx(tx,ty,tz,vtx);

		vi = geofrm.AddVert(vtx,norm);

		vtx_idx_arr[xyz2cell(tx,ty,tz)]=vi;

		return vi;

	} else {

		return vi;

	}

	//return geofrm.AddVert(vtx,norm);

}







void Octree::eflag_clear()

{

	memset(ebit,0,ncells);

}





int Octree::is_eflag_on(int x, int y, int z, int e)

{

	int idx;

	switch (e) {

	case 0 : 

		idx = 3*xyz2cell(x,y,z) + 0;

		break;

	case 1 :

		idx = 3*xyz2cell(x+1,y,z) + 2;

		break;

	case 2 :

		idx = 3*xyz2cell(x,y,z+1) + 0;

		break;

	case 3 :

		idx = 3*xyz2cell(x,y,z) + 2;

		break;

	case 4 :

		idx = 3*xyz2cell(x,y+1,z) + 0;

		break;

	case 5 :

		idx = 3*xyz2cell(x+1,y+1,z) + 2;

		break;

	case 6 :

		idx = 3*xyz2cell(x,y+1,z+1) + 0;

		break;

	case 7 :

		idx = 3*xyz2cell(x,y+1,z) + 2;

		break;

	case 8 :

		idx = 3*xyz2cell(x,y,z) + 1;

		break;

	case 9 :

		idx = 3*xyz2cell(x+1,y,z) + 1;

		break;

	case 10 :

		idx = 3*xyz2cell(x,y,z+1) + 1;

		break;

	case 11 :

		idx = 3*xyz2cell(x+1,y,z+1) + 1;

		break;

	}

	

	if (ebit[idx/8]&(1<<(idx%8))) return 1;

	else return 0;

	

}





void Octree::eflag_on(int x, int y, int z, int e)

{

	int idx;

	switch (e) {

	case 0 : 

		idx = 3*xyz2cell(x,y,z) + 0;

		break;

	case 1 :

		idx = 3*xyz2cell(x+1,y,z) + 2;

		break;

	case 2 :

		idx = 3*xyz2cell(x,y,z+1) + 0;

		break;

	case 3 :

		idx = 3*xyz2cell(x,y,z) + 2;

		break;

	case 4 :

		idx = 3*xyz2cell(x,y+1,z) + 0;

		break;

	case 5 :

		idx = 3*xyz2cell(x+1,y+1,z) + 2;

		break;

	case 6 :

		idx = 3*xyz2cell(x,y+1,z+1) + 0;

		break;

	case 7 :

		idx = 3*xyz2cell(x,y+1,z) + 2;

		break;

	case 8 :

		idx = 3*xyz2cell(x,y,z) + 1;

		break;

	case 9 :

		idx = 3*xyz2cell(x+1,y,z) + 1;

		break;

	case 10 :

		idx = 3*xyz2cell(x,y,z+1) + 1;

		break;

	case 11 :

		idx = 3*xyz2cell(x+1,y,z+1) + 1;

		break;

	}

	

	ebit[idx/8]|=(1<<(idx%8));

	

}









void Octree::polygonize(geoframe& geofrm)

{



	float val[8];

	unsigned int vtx[4];

	int vtx_num;

	int intersect_id;

	int x,y,z;

	int i,j ;



	for (int k=0;k<ncells;k++) vtx_idx_arr[k]=-1;



	geofrm.Clear();

	eflag_clear();



	for (i = 0 ; i < ncells ; i++) {

		cell2xyz(i, x, y, z);

		getCellValues(i, val);

		for (j = 0 ; j < 12 ; j++) {

			if (is_eflag_on(x, y, z, j)) continue;

			intersect_id =  is_intersect(val, j);

			if (intersect_id != 0) {

				if (is_min_edge(i, j, vtx, vtx_num, intersect_id, geofrm)) {

					eflag_on(x, y, z, j);

					geofrm.AddQuad(vtx, vtx_num);

				}

			}



		}



	}



}





int Octree::is_intersect(float* val, int e_id)

{

	float f1,f2;



	f1=val[cube_eid[e_id][0]];

	f2=val[cube_eid[e_id][1]];



	if (iso_val<=f1 && iso_val>=f2)

		return -1;

	else if (iso_val<=f2 && iso_val>=f1)

		return 1;

	else return 0;

}





void Octree::read_header()

{

	getFloat(minext,3,vol_fp);

	getFloat(maxext,3,vol_fp);

	

	getInt(&nverts,1,vol_fp);

	getInt(&ncells,1,vol_fp);

	

	getInt(dim,3,vol_fp);

	getFloat(orig,3,vol_fp);

	getFloat(span,3,vol_fp);

	

}



void Octree::read_data()

{

	// currently support only float data

	//getFloat(orig_vol,dim[0]*dim[1]*dim[2], vol_fp);

	

}





int Octree::is_refined(int x, int y, int z)

{

	int idx=0;



	idx=xyz2cell(x,y,z);

	

	if (x<0 || y<0 || z<0) return 1;

	if (x>=dim[0]-1 || y>=dim[1]-1 || z>=dim[2]-1 ) return 1;

	

	if (refine_flag[idx]==0) return 0;

	else return 1;

	//return oct_array[idx].refine_flag;

}





void Octree::cell2xyz(int oc_id,int& x,int& y,int& z)

{	

	x = oc_id%(dim[0]-1);

	y = (oc_id/(dim[0]-1))%(dim[1]-1);

	z = oc_id/((dim[0]-1)*(dim[1]-1));

}





int Octree::xyz2cell(int x,int y,int z)

{

	return x+y*(dim[0]-1)+z*(dim[0]-1)*(dim[1]-1);

}





int Octree::xyz2vtx(int x, int y, int z)

{

	return x+y*dim[0]+z*dim[0]*dim[1];

}





void Octree::idx2vtx(int oc_id, int* vtx)

{

	int x,y,z;

	

	cell2xyz(oc_id,x,y,z);

	

	vtx[0]=xyz2vtx(x,y,z);

	vtx[1]=xyz2vtx(x+1,y,z);

	vtx[2]=xyz2vtx(x+1,y,z+1);

	vtx[3]=xyz2vtx(x,y,z+1);

	vtx[4]=xyz2vtx(x,y+1,z);

	vtx[5]=xyz2vtx(x+1,y+1,z);

	vtx[6]=xyz2vtx(x+1,y+1,z+1);

	vtx[7]=xyz2vtx(x,y+1,z+1);

	

}





void Octree::getCellValues(int oc_id, float* val)

{

	int vtx[8];

	int i;



	idx2vtx(oc_id, vtx);	



	for (i = 0 ; i < 8 ; i++) {

		val[i] = (float) orig_vol[vtx[i]];

	}

}







float Octree::getValue(int i, int j, int k)

{

	/*

	int cell_size=(dim[0]-1)/(1<<level);

	return orig_vol[i*cell_size + j*cell_size*dim[0] + k*cell_size*dim[0]*dim[1]];

	*/



	return (float) orig_vol[i + j*dim[0] + k*dim[0]*dim[1]];

}



void Octree::getVertGrad(int i, int j, int k, float g[3]) 

{

	

	if (i==0) {

		

		g[0] = getValue(i+1, j, k) - getValue(i, j, k);

	}

	else if (i>=dim[0]-1) {

		

		g[0] = getValue(i, j, k) - getValue(i-1, j, k);

	}

	else {

		

		g[0] = (getValue(i+1, j, k) - getValue(i-1, j, k)) * 0.5f;

	}

	

	if (j==0) {

		g[1] = getValue(i, j+1, k) - getValue(i, j, k);

	}

	else if (j>=dim[1]-1) {

		g[1] = getValue(i, j, k) - getValue(i, j-1, k);

	}

	else {

		g[1] = (getValue(i, j+1, k) - getValue(i, j-1, k)) * 0.5f;

	}

	

	if (k==0) {

		g[2] = getValue(i, j, k+1) - getValue(i, j, k);

	}

	else if (k>=dim[2]-1) {

		g[2] = getValue(i, j, k) - getValue(i, j, k-1);

	}

	else {

		g[2] = (getValue(i, j, k+1) - getValue(i, j, k-1)) * 0.5f;

	}

	

}











void getDualContour(unsigned char* char_data,int dim[3],float orig[3],float span[3], float isoval , float err_tol , geoframe& geofrm)

{

	Octree oc;

	oc.set_isovalue(isoval);

	oc.Octree_init(char_data,dim,orig,span);

	oc.polygonize(geofrm);

}



void getDualContour(char* rawiv_fn , float isoval , float err_tol , geoframe& geofrm)

{



	Octree oc;

	oc.set_isovalue(isoval);

	oc.Octree_init(rawiv_fn);

	oc.polygonize(geofrm);

}



void writeDualContourRaw(char* mesh_fn , geoframe& geofrm)

{

	int i,j;



	FILE* fp = fopen(mesh_fn , "w");

	if (fp==NULL) {

		fprintf(stderr , "Wrong file name!\n");

		exit(0);

	}



	fprintf(fp,"%d %d\n",geofrm.getNVert(),2*geofrm.numquads); 



	for (i=0 ; i<geofrm.getNVert() ; i++) {

		for (j=0 ; j<3 ; j++) {

			fprintf(fp, "%f ",geofrm.verts[i][j]);

		} 

		fprintf(fp, "\n");

	}



		;

	for (i=0 ; i<geofrm.numquads ; i++) {

		

		fprintf(fp, "%d  ",geofrm.quads[i][0]) ;

		fprintf(fp, "%d  ",geofrm.quads[i][1]) ;

		fprintf(fp, "%d  ",geofrm.quads[i][2]) ;

		 

		fprintf(fp,"\n");

		fprintf(fp, "%d ",geofrm.quads[i][0]);

		fprintf(fp, "%d ",geofrm.quads[i][2]);

		fprintf(fp, "%d ",geofrm.quads[i][3]);

		fprintf(fp,"\n");

	}

}



void writeDualContour(char* mesh_fn , geoframe& geofrm)

{

	int i,j;



	FILE* fp = fopen(mesh_fn , "w");

	if (fp==NULL) {

		fprintf(stderr , "Wrong file name!\n");

		exit(0);

	}

	fprintf(fp, "#Inventor V2.1 ascii\n\n" );

	fprintf(fp, "Separator {\n");

	fprintf(fp, "Coordinate3 {\n");

	fprintf(fp, "point[\n");

	for (i=0 ; i<geofrm.getNVert()-1 ; i++) {

		for (j=0 ; j<3 ; j++) {

			fprintf(fp, "%f ",geofrm.verts[i][j]);

		} 

		fprintf(fp, ",\n");

	}

	for (j=0 ; j<3 ; j++) {

		fprintf(fp, "%f ",geofrm.verts[i][j]);

	}

	fprintf(fp,"]\n");



	fprintf(fp, "}\n");





	fprintf(fp, "IndexedFaceSet {\n");

    fprintf(fp, "coordIndex      [\n");



	if (geofrm.numtris==0 && geofrm.numquads==0) {

		;

	} else if (geofrm.numtris>0 && geofrm.numquads==0) {

		for (i=0 ; i<geofrm.numtris-1 ; i++) {

			for (j=0 ; j<3 ; j++) {

				fprintf(fp, "%d , ",geofrm.triangles[i][2-j]);

			}	 

			fprintf(fp, "-1 , \n");

		}

		for (j=0 ; j<3 ; j++) {

				fprintf(fp, "%d , ",geofrm.triangles[i][2-j]);

		}

		fprintf(fp, "-1 ");

	} else if (geofrm.numtris==0 && geofrm.numquads>0) {

		for (i=0 ; i<geofrm.numquads-1 ; i++) {

			for (j=0 ; j<4 ; j++) {

				fprintf(fp, "%d , ",geofrm.quads[i][3-j]) ;

			} 

			fprintf(fp, "-1 , \n");

		}

		for (j=0;j<4;j++) fprintf(fp, "%d , ",geofrm.quads[i][3-j]);

		fprintf(fp, "-1 ");

 

	} else {

		for (i=0 ; i<geofrm.numtris ; i++) {

			for (j=0 ; j<3 ; j++) {

				fprintf(fp, "%d , ",geofrm.triangles[i][2-j]);

			}	 

			fprintf(fp, "-1 , \n");

		}

		for (i=0 ; i<geofrm.numquads-1 ; i++) {

			for (j=0 ; j<4 ; j++) {

				fprintf(fp, "%d , ",geofrm.quads[i][3-j]) ;

			} 

			fprintf(fp, "-1 , \n");

		}

		for (j=0;j<4;j++) fprintf(fp, "%d , ",geofrm.quads[i][3-j]);

		fprintf(fp, "-1 ");



	}

 

	fprintf(fp,"]\n");



	fprintf(fp, "}\n");

	fprintf(fp, "}\n");



}

