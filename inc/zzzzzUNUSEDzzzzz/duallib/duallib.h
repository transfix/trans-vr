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

#ifndef __OCTREE_H__
#define __OCTREE_H__

#include<stdio.h>
#include <duallib/geoframe.h>

typedef struct _octcell {
	char refine_flag;
} octcell;

typedef struct _MinMax {
	float min;
	float max;
} MinMax;

#define NUM_PARENT_CELL 6
#define FLOAT_MINIMUM -10000000
#define FLOAT_MAXIMUM  10000000

typedef struct { 
	int dir;
	int di,dj,dk;
	int d1,d2;
} EdgeInfo;


static int cube_eid[12][2] = { {0,1}, {1,2}, {2,3}, {0,3}, {4,5}, {5,6}, {6,7}, {4,7}, {0,4}, {1,5}, {3,7} , {2,6} };

static EdgeInfo edgeinfo[12] = {
	{ 0, 0, 0, 0, 0, 1 },
	{ 2, 1, 0, 0, 1, 2 },
	{ 0, 0, 0, 1, 3, 2 },
	{ 2, 0, 0, 0, 0, 3 },
	{ 0, 0, 1, 0, 4, 5 },
	{ 2, 1, 1, 0, 5, 6 },
	{ 0, 0, 1, 1, 7, 6 },
	{ 2, 0, 1, 0, 4, 7 },
	{ 1, 0, 0, 0, 0, 4 },
	{ 1, 1, 0, 0, 1, 5 },
	{ 1, 0, 0, 1, 3, 7 },
	{ 1, 1, 0, 1, 2, 6 }
};

static int po[8][6][3] = {
	{{0,-1,-1},{-1,0,-1},{0,0,-1},{-1,-1,0},{0,-1,0},{-1,0,0}},
	{{0,-1,-1},{0,0,-1},{1,0,-1},{0,-1,0},{1,-1,0},{1,0,0}},
	{{0,1,-1},{-1,0,-1},{0,0,-1},{-1,1,0},{0,1,0},{-1,0,0}},
	{{0,1,-1},{0,0,-1},{1,0,-1},{0,1,0},{1,1,0},{1,0,0}},
	{{0,-1,1},{-1,0,1},{0,0,1},{-1,-1,0},{0,-1,0},{-1,0,0}},
	{{0,-1,1},{0,0,1},{1,0,1},{0,-1,0},{1,-1,0},{1,0,0}},
	{{0,1,1},{-1,0,1},{0,0,1},{-1,1,0},{0,1,0},{-1,0,0}},
	{{0,1,1},{0,0,1},{1,0,1},{0,1,0},{1,1,0},{1,0,0}}
};
class Octree 
{

private :

//	Contour3d curcon;

	//char fname[256];
	FILE*	vol_fp;
	float	iso_val;
	int		leaf_num;
	char*	refine_flag;
	int		octcell_num;
	int		cell_num;
	int		oct_depth;
	int		level_res[10];
	int*	cut_array;
	int* vtx_idx_arr;


	//int* interpol_bit;

	double** qef_array;

	//float* orig_vol;
	unsigned char* orig_vol;
	char * ebit;

	MinMax* minmax;

	int max_align_res;



	//char data_type; // 0 : uchar , 1 : ushort , 2 : uint , 3 : float

	float minext[3], maxext[3];
	int nverts, ncells;
	int dim[3];
	float orig[3];
	float span[3];

	void read_header();
	void read_data();
	int is_refined(int x, int y, int z);

	void idx2vtx(int oc_id,int* vtx);
	void cell2xyz(int oc_id,int& x,int& y,int& z);

	int xyz2cell(int x,int y,int z);
	int xyz2vtx(int x, int y, int z);
	void getCellValues(int oc_id,float* val);
	int is_intersect(float* val, int e_id);

	float getValue(int i, int j, int k);

	void getVertGrad(int i, int j, int k, float g[3]);

	void get_solution(int oc_id, float* pos);
	void get_vtx(int x, int y, int z, float* pos);

	
	
	
	int is_min_edge(int oc_id, int e_id, unsigned int* vtx, int& vtx_num,int intersect_id,geoframe& geofrm);
	int min_vtx(int x, int y, int z, geoframe& geofrm);

	int is_eflag_on(int x, int y, int z,int e);
	void eflag_on(int x, int y, int z, int e);
	void eflag_clear();



public :
	Octree();  
	~Octree();


	float vol_min, vol_max;

	// isosurface simplification operations
	void Octree_init(const char* rawiv_fname);
	void Octree_init(unsigned char* char_data, int Dim[3], float Orig[3], float Span[3]);
	void set_isovalue(float val) { iso_val=val; }

	// dual contouring operations
	void polygonize(geoframe& geofrm);  // extract contour from the found cells, 
		
};

void getDualContour(char* rawiv_fn , float isoval , float err_tol , geoframe& geofrm);
void getDualContour(unsigned char* char_data,int dim[3],float orig[3],float span[3], float isoval , float err_tol , geoframe& geofrm);
void writeDualContour(char* mesh_fn , geoframe& geofrm);
void writeDualContourRaw(char* mesh_fn , geoframe& geofrm);

#endif __OCTREE_H__
