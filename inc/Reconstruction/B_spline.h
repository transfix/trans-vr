#ifndef _B_SPLINE_H
#define _B_SPLINE_H
//#include <stdio.h>
//#include <string.h>


struct BGrids
{
  long double  scale;
  int   dim[3];
  int   size;
  float StartXYZ[3], FinishXYZ[3];
};




class Bspline{

private:
int N;           // N+1 is the total Bspline Base Functions along each volume axis x,y,z.
int M;           // M+1 is the total Gauss nodes number along each volume axis x,y,z. 0,...,M.  
int N3 ;
int M1 ;
int N1 ;
int nx, ny, nz;
float *BernBase;
long double delx;
 int ImgNx, ImgNy, ImgNz;
 int StartX,FinishX;

public:
float Kot[4];        // The four Gauss nodes in [0,1].
float Weit[4];       // The four weights corresponding to above four Gauss nodes.

 float *OrthoBBaseGrid, *OrthoBBaseGrid2;
 float *BBaseGN, *OrthoBBaseGN,  *BernBaseGrid;
 float *OrthoBBImgGrid, *BBaseImgGrid, *subBBaseImgGrid, *BBaseImgGrid3;
 
 long double *SchmidtMat, *InvSchmidtMat, *SchmidtMatG;

public:
Bspline();
virtual ~Bspline();
int          Reorder_BaseIndex_Volume(unsigned i,unsigned j, unsigned k, unsigned N);
 long double* GramSchmidtBsplineBaseAtGrids(int bN, int nx, int gM);
long double* GramSchmidtofBsplineBaseFunctions(int bN,int gM);
 long double* GramSchmidtofBsplineBaseFunctions2();

long double  EvaluateIntegralInnerProducts(long double *u1, long double *u2, int M);
 long double EvaluateIntegralInnerProducts2(long double *u1, long double *u2, int gM, int gM2);


 void         Evaluate_BSpline_Basis_AtImgGrid();
 void         Evaluate_BSpline_Basis_AtImgGrid_sub();

 void         Evaluate_BSpline_Basis_AtGaussNodes();
 void         Evaluate_OrthoBSpline_Basis_AtVolGrid2();
 void         Evaluate_OrthoBSpline_Basis_AtVolGrid();
 void         Evaluate_OrthoBSpline_Basis_AtGaussNodes();

 

void Evaluate_B_Spline_Basis();

 void Phi_ijk_Partials_ImgGrid(int qx, int qy, int qz, int i, int j, int k, float *partials);
 void Phi_ijk_Partials_ImgGrid_3(int qx, int qy, int qz, int i, int j, int k, float *partials);
void  Phi_ijk_Partials_ImgGrid_9(int qx, int qy, int qz, int i, int j, int k, float *partials);

 void Ortho_Phi_ijk_Partials_ImgGrid(int qx, int qy, int qz, int i, int j, int k, float *partials);

void Phi_ijk_PartialsGN(int qx, int qy, int qz, int i, int j, int k, float *partials);

void Phi_ijk_Partials_at_Grids(int ix, int iy, int iz, int i,
							   int j, int k, float *partials);


int Reorder_BaseIndex_Volume(int i,int j, int k, int nx, int ny, int nz);

void Support_Index(int i,int *min_index, int *max_index);



float ComputeVolImgBsplineInpolatAtAnyPoint(float *coeff, int nx,int ny,int nz,float u, float v, float w);
float Cubic_Bspline_Interpo_kernel(float u, int shift);
void convertToInterpolationCoefficients_3D(float *s, int nx, int ny,int nz,   float EPSILON);
void convertToInterpolationCoefficients_2D(float *s, int nx, int ny, float EPSILON);
void convertToInterpolationCoefficients_1D(float *s, int DataLength, float EPSILON);


float SplineBasePoly(float u,int MM,int i,int k,float *U);
float Dx_SplineBasePoly(float u,int orderR,int MM,int i,int k,float *U);


void Spline_N_0(long double u, int m, long double *values);
void Spline_N_1(long double u, int m, long double *values);
void Spline_N_2(long double u, int m, long double *values);
void Spline_N_Base(long double u, int m, long double *values);
void Spline_N_Base_3(float u, float *values);
void Spline_N_Base_1(float u, float *values);
void Spline_N_Base_2(float u, float *values);

void Spline_N_Base(float u, float *value);

void Spline_N_i(long double u, int i, int m, long double *values);

 void BsplineSetting(int ImgDim[3], BGrids *bgrids);

 void ConvertToInterpolationCoefficients_1D(float *s, int DataLength, float EPSILON) ;

 void ConvertToInterpolationCoefficients_2D(float *s, int nx, int ny, float EPSILON) ;

void ConvertToInterpolationCoefficients_3D(float *s, int nx, int ny, 
										   int nz,   float EPSILON);

bool Bspline_Projection(int p, int q, int r, int sub, float rotmat[9], int sample_num, float translate, float *prjimg, int *start_point);

bool  Bspline_GridProjection(int p, int q, int r, float rotmat[9], int sample_num, float translate, float *prjimg, int *start_point);

void  ObtainObjectFromCoeffs(float *Rcoef,float *f);
void  ObtainObjectFromNonOrthoCoeffs(float *Rcoef,float *f);
void  ObtainObjectFromNonOrthoCoeffs_sub(float *Rcoef,float *f, int sub);


 void ObtainObjectFromNonOrthoCoeffs_FA(float *Rcoef,float *f);
};


#endif
