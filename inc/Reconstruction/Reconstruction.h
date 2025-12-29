#ifndef _RECONSTRUCTION_H
#define _RECONSTRUCTION_H
#include <Reconstruction/B_spline.h>
#include <Reconstruction/utilities.h>
#include <VolMagick/VolMagick.h>
#include <boost/tuple/tuple.hpp>
#include <fftw3.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
// View structure
// #ifndef _View_
// #define _View_

using namespace std;
class Bspline;

// arand: line below gives a warning
// typedef struct CUBE
struct CUBE {
  float ox, oy, oz,
      l; // ox oy oz coordinate center of cube. l is the edge length.
  float cubef8[8]; // 8 vertex values.
};

struct Views {
  struct Views *next; // Pointer for linked lists
  float x, y, z;      // The view vector - line of sight
  float a; // The view rotation angle - rotation perpendicular to view vector
};

// #endif

// Euler structure
// #ifndef _Euler_
// #define _Euler_
struct EulerAngles {

  struct EulerAngles *next;
  float rot;  // 0 -- 2PI;    view.angle - phi
  float tilt; // 0 -- PI/2;   arccos(view.z)
  float psi;  // 0 -- 2PI;    arctan(view.y/view.x)
};

// #endif

struct PrjImg {
  int nx, ny;     // nx * ny size projection image.
  float *centers; // Origin (the center of 2D pixel image).
  // float  *Eulers;    // Euler Angles.

  float *data;

  float *background;
};

/*
struct Oimage
{
   int     x, y, z, c; // Dimensions ,x y z and channels.
   int     n;          // Numbers of images.
   float   avg, std;   // Average and standard deviation.
   float   ux, uy, uz; // Voxel units (angstrom/pixel edge)
   float*  data;                // pixel values at each grid.
};
*/

struct Boundingbox // The volume which enclose the reconstructed 3D images.
{
  float ox, oy,
      oz; // The Cartesian coordinate centers of reconstructed 3D iamges.
  int nx, ny, nz;             // The volume dimensions dimx, dimy,dimz;
  int size;                   // The Volume data size;
  float minExt[3], maxExt[3]; // The start point and end point cartesian
                              // coordinates of volume data.
};

class Reconstruction {

private:
  float *slicemat, *ReBFT;
  int BStartX, BFinishX;
  int Gnx, Gny;

  int N;     // N+1 is the bspline grid dimension along each axis.
  int usedN; // usedN + 1 the used bspline number along each axis. // N + 1 is
             // the total bspline number.
  int tolbsp; // The total number of Cubic Vol Bspline defined in the grids.
  int usedtolbsp; // usedtolbsp = ( usedN+1)^3;
  int VolImgSize;
  float pixelscale;

  float *tempf;
  float *O_Kcoef; // the kth step ortho coefficients for k+1 step.

  // int      numb_2dimg;       //The number of original Cryo-EM 2D image
  // data.
  int img2dsize; // img2dsize = nx * ny, the size of each Cryo-EM image data.
  int IMG2DSIZE; // sampling image dimension in fourier space.
  int ImgNx, ImgNy, ImgNz; // projection and volume image dimensions.

  int ox,
      oy; // The Cartesian coordinate center of original aligned 2D iamges.
  int M;  // M+1 is the number of Gauss nodes along arbitray bounding box axis
          // x,y,z.
  float Ox, Oy;

  // bounding box axis x,y,z.

  float delx; // the interval length between two neighbour grids.

  int nv; // the number of view vectors.
  float alpha,
      fac; // Delta Function alpha.  fac*alpha.   fac = 1/3 is defalut value.
  int flows;   // flows =1,2,3.    2-mean curvature flow. 3-willmore flow.
  float V0;    // Molecular Volume.
  float Vcube; // the volume of the voxcell.
  float reconj1, al, be, ga, la;

  float StartXYZ[3], FinishXYZ[3];

  BGrids *bgrids;

  Bspline *bspline;
  float Bscale;
  float gox, goy, goz; // The Volume coordinate center.
  fftw_complex *BsplineFT;

  float *Matrix, *coefficients; //

  /* variables for accumulating iters.***************/
  // Initial compute data before inters.
  int Iters;
  PrjImg *gd, *prjimg; // The initial Cryo-EM aligned projection images.
  float *gd_coefs;     // convert gd to bspline.
  // struct Oimage* VolF;

  float *Rmat, *InvRmat; // all rotation matrice and inverse rotation matrice.

  fftw_complex *gd_FFT;
  float *proj_VolB;
  int *startp;
  float *gdxdphi;
  float *xdgdphi; // for J1.
  float *Mat;
  int newnv, bandwidth; // for speed up the algorithm.
  float *EJ1_k, *EJ1_k1;
  // variables should be saved for iters.
  int *OrthoOrdered, CurrentId, OrderManner;
  int Recon_Method;

  const char *volfilename;

public:
  struct Oimage *VolImg;
  float FW; // [-FW , FW ] is the cuted interval of FFT bspline for integral.
  int FN;   // the interval number of [-FW , FW ].
  Reconstruction();
  virtual ~Reconstruction();

  void setThick(int thickness);

  struct Oimage *Reconstruction3D_Ordered(PrjImg *gd, int iter, float tau,
                                          EulerAngles *eulers, Views *nview,
                                          int m_Itercounts, int object);

  Oimage *Reconstruction3D(int ReconManner, int iter, float tau,
                           EulerAngles *eulers, int m_Itercounts, int object);

  Oimage *Reconstruction3D_FBP(int iter, float tau, EulerAngles *eulers,
                               int m_Itercounts, int object);

  Oimage *ReconstructionIters(int iter, float tau, float *coef, float *Rcoef,
                              float *ocoef, float *f);

  void Converttoijk(vector<int> &suppi, int *ijk);

  void GetSubNarrowBand(vector<CUBE> &cubes, vector<CUBE> &cubes1,
                        float *IsoC, float *coefs, float *f,
                        float param); //, vector<float> &DeltafIsoC,
                                      //vector<float> &PdeltafIsoC);

  float ConvertCoeffsToNewCoeffs(float *coef, float *ocoef,
                                 long double *TransformMat);

  void ComputeDeltafIsoC(float *f, float *IsoC, int n,
                         vector<vector<int>> &ijk, float *DeltafIsoC,
                         float *PdeltafIsoC);

  void ComputeHfGradientf(float *coefs, float *IsoC, int n,
                          vector<vector<int>> &ijk, vector<CUBE> &cubes1,
                          int subfactor, float *DeltafIsoC,
                          float *PdeltafIsoC, float *Hf, float *Gradientf,
                          float *Ngradf, float *HGf, float *GHGf, float *XtN,
                          float *NNt, float *XtP, float *f);

  void ComputeG_HfGf(int n, vector<vector<int>> &ijk, float *coef, float *Hf,
                     float *Gradientf, float *Hfx, float *Hfy, float *Hfz,
                     float *G_HfGf);

  void ComputeHGfGHGf(int n, vector<vector<int>> &ijk, float *Hf,
                      float *Gradientf, float *HGf, float *GHGf);

  float ComputeJ2(int i1, int j1, int k1, int n, vector<vector<int>> &ijk,
                  float *Gradientf, float *Hf, float *HGf, float *GHGf,
                  float *Ngradf, float *G_HfGf, float *DeltafIsoC,
                  float *PdeltafIsoC);

  float ComputeJ3(int i1, int j1, int k1, int n, vector<vector<int>> &ijk,
                  float *Gradientf, float *Ngradf, float *DeltafIsoC,
                  float *PdeltafIsoC);
  void ComputeJ3_Xu(float *coefs, float taube, float *J3phi);
  void ComputeJ3_HTF(float *coefs, float taube, float *J234phi);
  void ComputeJ3_MCF(float *coefs, float taube, float *J3phi);
  void ComputeJ3_WMF(float *coefs, float taube, float *J234phi);

  void ComputeHc_f(float *f, float *IsoC, vector<int> &outliers);

  float ComputeJ4(int i1, int j1, int k1, int n, vector<vector<int>> &ijk,
                  float *f, float *DeltafIsoC, vector<int> *outliers);

  void ComputeJ4outliers(float *J4phi, float *f, vector<int> &outliers,
                         float tauga);

  float ComputeJ5(int i1, int j1, int k1, int n, vector<vector<int>> &ijk,
                  vector<float> &DeltafIsoC, float *HGf, float *Ngradf,
                  float *Gradientf, float *Hf, float V0);

  void ComputeJ5_Xu(float *J5phi, float taula, int n,
                    vector<vector<int>> &subijk, vector<CUBE> &cubes1,
                    int subfactor, float *DeltafIsoC, float *HGf,
                    float *Ngradf, float *Gradientf, float *Hf, float *XtN,
                    float *NNt, float *XtP, float V0);

  void ComputeJ2J4J5(float *coefs, float *IsoC, int n, vector<CUBE> &cubes1,
                     float V0, float taual, float tauga, float taula,
                     float *J2phi, float *J5phi);

  float EvaluateCubeCenterValue(float *J2f, int i, int j, int k);

  float EvaluateGdXdPhiMatrix(Oimage *gd, Views *nview, float *InvRmat);

  void EvaluateCubicBsplineBaseProjectionIntegralMatrix(float *ocoef);

  void EvaluateFFT_GdXdPhiMatrix();

  void EvaluateFFT_GdXdOrthoPhiMatrix();

  float EvaluateFFT_GdXdPhiIntegral(int i1, int j1, int k1,
                                    fftw_complex *slice, fftw_complex *B_k,
                                    Oimage *image);

  // float   PickfromMat(unsigned long  ii,unsigned long jj, float* Mat, char
  // ch);

  // void    InserttoMatrix(unsigned long i, unsigned long j,float res, float
  // *Mat, char ch);

  void writetofile(float *Mat, int size, char *filename);

  float EvaluateGdXdPhiIntegral(Oimage *gd, int i1, int j1, int k1,
                                Views *view, float *InvRmat);

  Oimage *GdiFromGd(Oimage *gd, int i);

  void FFT_gdMatrix();

  void FFT_gdi(int i, fftw_complex *in, fftw_complex *out);

  void Convertgd2BsplineCoefs(float *gddata);

  float EvaluateCubicBsplineBaseProjectionIntegral(int i, int j, int k,
                                                   int i1, int j1, int k1,
                                                   Views *view);
  float *EvaluateBsplineBaseFT();

  void Evaluate_BsplineFT_ijk_pqr();

  float Evaluate_XdPhi_XdPhi_FFT();

  void BsplineCentralSlice(int i, int j, int k, int iv, float rotmat[9],
                           fftw_complex *slice, fftw_complex *B_k, int index);

  // new.
  void GetCentralSlice(const fftw_complex *Vol, float *COEFS, int sub);

  // old.
  void GetCentralSlice(fftw_complex *Vol, float *COEFS);

  float BsplineCentralSlice1(int i, int j, int k, float rotmat[9]);

  void VolBsplineBaseFT2(int i, int j, int k, float X, float Y, float Z,
                         int iv, int jj, fftw_complex *B_k,
                         int index); // scale = 1.0;

  void VolBsplineBaseFT(int i, int j, int k, float X, float Y, float Z,
                        float scale, fftw_complex *B_k);

  void BsplineBaseFT2D(int i, int j, float X, float Y, float scale,
                       fftw_complex *B_k);

  // float    testMatrix();

  void BsplineFT1D(int k, float omega, float scale, fftw_complex *B_k);

  float BilinearInterpolation(float *data, int ix, int iy, float xd,
                              float yd);

  float TrilinearInterpolation(float coordi[3], int i, int j, int k);

  float TrilinearInterpolationVolumeData(float coordi[3], float *Voldata);

  Oimage *InitImageParameters(int c, int x, int y, int z, int n);

  Oimage *InitImageHeader(int c, int x, int y, int z, int n);

  Oimage *InitImage();

  boost::tuple<bool, VolMagick::Volume> ConvertToVolume(Oimage *p);
  VolMagick::Volume *GetVolume(Oimage *p);

  void SaveVolume(Oimage *p);

  boost::tuple<bool, VolMagick::Volume> gd2Volume(float *data);

  Oimage *Phantoms(int dimx, int dimy, int dimz, int object);

  Oimage *Phantoms_Volume(int object);

  void Phantoms_gd(Oimage *Volume, EulerAngles *eulers);

  Oimage *InitialFunction(int function, const char *filename,
                          const char *path);

  void DesignInitFunction(int function, Oimage *p);

  void ConvertToOrthoCoeffs();

  float SimpsonIntegrationOn2DImage(Oimage *p);

  Oimage *ImageForTestSimpsonIntegration();

  int CountList(char *list);

  bool Initialize(int dimx, int dimy, int dimz, int m, int NewNv,
                  int BandWidth, float Alpha, float Fac, float Volume,
                  int Flow, int Recon_method);

  void SetJ12345Coeffs(float ReconJ1, float Al, float Be, float Ga, float La);

  void setTolNvRmatGdSize(const int TolNv);

  int kill_all_but_main_img(Oimage *p);

  float Deltafunc(float x);

  float Deltafunc3d(float x, float y, float z);

  float DeltafuncPartials(float x);

  void SortDecreasing(float *arr, int *oldindex, int n);

  void SortDecreasing(float *arr, int n);

  void EestimateIsoValueC(float *f, float *IsoC, int n, vector<int> &suppi);

  void EestimateIsoValueC(float *f, float *IsoC, int n, vector<CUBE> &cubes,
                          vector<CUBE> &cubes1,
                          float param); //, vector<float> &DeltafIsoC,
                                        //vector<float> &PdeltafIsoC);

  // void  ObtainObjectFromCoeffs(float *Rcoef,float *f);

  float Test1DSchmidtMatrix();

  void TestXdF_Gd();

  void testxdf_gd(Views *nview);
  void compare_Xdphi_Phi_pd(Views *nview);

  void ObtainObjectFromCoeffs(float *Rcoef, float *f);

  void simple_backprojection(float *oneprj, float *invrotmat, Oimage *vol,
                             int diameter);

  void backprojection(Oimage *vol);

  float MeanError(float *data1, float *data0);

  float GlobalMeanError(Oimage *object);

  void VolBsplineProjections();

  void VolBsplineProjections_FA();

  void ComputeXdf(float *coef);

  void ComputeFFT_gd();

  void subdivision_gd();

  void testFFT_sub();

  void ComputeXdf_gd(float *f, float *coefs);
  void ComputeXdf_gd(int newnv, int *index, float *f, float *coefs);
  void ComputeEJ1(float *f, float *coefs, int *index);
  void ComputeEJ1_Acurate(float *f, float *coefs);
  void ComputeTau(float *coefs, float *Rcoef, float *tau, float reconj1,
                  float taube);
  void ComputeTau(int newnv, int *index, float *coefs, float *Rcoef,
                  float *tau, float reconj1, float taube);
  void ComputeTau(float *coefs, float *diff_coefs, float *tau, float reconj1,
                  float taube, int flows);
  void J3_Tau(float tau0, float *coefs, float *diff_coefs, float taube,
              int flows, float *numerator1, float *denominator1);
  void ComputeTau(int newnv, int *index, float *coefs, float *diff_coefs,
                  float *tau, float reconj1, float taube, int flows);

  // new.
  void ComputeIFSFf_gd(const float *f, float *padf, int padfactor, int nx,
                       int padnx, int sub, float fill_value, float *ocoef,
                       fftw_complex *in, fftw_complex *out);

  // old.
  void ComputeIFSFf_gd(float *f, float *padf, int padfactor, int nx,
                       int padnx, float fill_value, float *ocoef,
                       fftw_complex *in, fftw_complex *out);

  // new.
  void ImgPad(const float *f, float *padf, int size, int padsize,
              float fill_value);

  // old.
  void ImgPad(float *f, float *padf, int size, int padsize, float fill_value);

  void FFT_padf(const float *padf, int padsize);

  void ImgPhaseShift2Origin(fftw_complex *ftdata, int padsize);

  void ImgPhaseShift2Origin2D(fftw_complex *ftdata, int padsize);

  void Test_GridProjectioin(EulerAngles *eulers);

  void ComputePhi_ijk_Bilinear(int i, int j, int k, float e1[3], float e2[3],
                               int start_point0[2], int start_point[2],
                               int sub, float *prjimg, float *prjimg_sub);
  void ComputePhi_ijk_Constant(int i, int j, int k, float e1[3], float e2[3],
                               int start_point0[2], int start_point[2],
                               int sub, float *prjimg, float *prjimg_sub);
  void ComputePhi_ijk_Constant_Simplify(int i, int j, int k, float e1[3],
                                        float e2[3], int start_point0[2],
                                        int start_point[2], int sub,
                                        float *prjimg, float *prjimg_sub);

  void Volume_Projection_FromBasis(int v, int sample_num, float *prjimg,
                                   float *coefs);
  void Volume_Projection_FromBasis_Simplify(int v, int sample_num,
                                            float *prjimg, float *coefs);
  void Volume_Projection_FromBasis_1(int v, int sample_num, float *prjimg,
                                     float *prjimg_IJK, int *start_Point,
                                     float *coefs);
  void Volume_Projection_FromBasis_2(int v, int sample_num, float *prjimg,
                                     float *coefs, float *prjimg1,
                                     float *coefs1);

  void B_spline_Function_Gradient_Grid(float *coefs, float *gradient, int ix,
                                       int iy, int iz);
  void B_spline_Function_Gradient_Grid_All(float *coefs, float *gradient,
                                           int ix, int iy, int iz);

  void B_spline_Function_Hessian_Grid(float *coefs, float *gradient, int ix,
                                      int iy, int iz);
  void readFiles(const char *filename, const char *path, int dimN);
  void imageInterpolation();
  void SetNewNv(int NewNv);
  void SetBandWidth(int BandWidth);
  void SetFlow(int Flow);
  void setOrders(const int tolnv);
  void getSubSetFromOrthoSets(int *subset);
  void setOrderManner(int ordermanner);
};

#endif
