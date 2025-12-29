/*
  Copyright 2006-2007 The University of Texas at Austin

        Authors: Dr. Xu Guo Liang <xuguo@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of HLevelSet.

  HLevelSet is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  HLevelSet is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

//////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

float InitialAntiCausalCoefficient(float *, int, float);
float InitialCausalCoefficient(float *, int, float, float);
void ConvertToInterpolationCoefficients_Qu(float *, int, float *, int, float);

void ConvertToInterpolationCoefficients_1D(float *, int, float);
void ConvertToInterpolationCoefficients_2D(float *, int, int, float);
void ConvertToInterpolationCoefficients_3D(float *, int, int, int, float);

void Tensor_333(float *, float *, float *, float *);

void EvaluateCubicSplineOrder1PartialsAtGridPoint(float *, float, float,
                                                  float, int, int, int, int,
                                                  int, int, float *);
void EvaluateCubicSplineOrder2PartialsAtGridPoint(float *, float, float,
                                                  float, int, int, int, int,
                                                  int, int, float *);
void Divided_DifferenceOrder2PartialsAtGridPoint(float *, float, float, float,
                                                 int, int, int, int, int, int,
                                                 float *);

void InitialData_Sphere(float *, float, float, float, float, float, float,
                        int, int, int);
void InitialData_Box(float *, float, float, float, float, float, float, int,
                     int, int);
void MeanCurvatureFlow(float *, float *, float *, float, float, float, float,
                       int, int, int);
void ComputeTensorXYZ();

void ReInitilazation(float *, float *, float, float, float, int, int, int);
void ReInitilazation_Upwind_Eno_Engquist(float *, float *, float *, float,
                                         float, float, int, int, int);
void ReInitilazation_Upwind_Eno_Godunov(float *, float *, float *, float,
                                        float, float, int, int, int, int);

void DiviededDifferencing_3j(float *, float *, float *, float *, float);
void DiviededDifferencing_2j(float *, float *, float *, float);
void DiviededDifferencing_1j(float *, float *, float);

void Cubic_Eno_Interpolation(float *f0, float *u_plus, float *u_minus,
                             float dx);
void Quadr_Eno_Interpolation(float *f0, float *u_plus, float *u_minus,
                             float dx);
void Linear_Eno_Interpolation(float *f0, float *u_plus, float *u_minus,
                              float dx);

void Get_Seven_Function_Values_X(float *coeff, int nx, int ny, int nz, int i,
                                 int j, int k, float *fx);
void Get_Seven_Function_Values_Y(float *coeff, int nx, int ny, int nz, int i,
                                 int j, int k, float *fy);
void Get_Seven_Function_Values_Z(float *coeff, int nx, int ny, int nz, int i,
                                 int j, int k, float *fz);
void computeFunction_Xu(float *funcvalue, int *dim, float *minExtent,
                        float *maxExtent);

void HLevel_set();

float TakeACoefficient_Fast(float *c, int nx, int ny, int nz, int u, int v,
                            int w);
float TakeACoefficient_Slow(float *c, int nx, int ny, int nz, int u, int v,
                            int w);
float EvaluateCubicSplineAtGridPoint(float *, int, int, int, int, int, int);
float EvaluateCubicSplineAtAnyGivenPoint(float *, int, int, int, float, float,
                                         float);
float Extreme_Positive(float a, float b);
float Extreme_Negative(float a, float b);
float Gradient_2(float *fun, float dx, float dy, float dz, int i, int j,
                 int k, int nx, int ny, int nz);

static float TensorF[27], TensorFx[27], TensorFy[27], TensorFz[27],
    TensorFxx[27], TensorFxy[27], TensorFxz[27], TensorFyy[27], TensorFyz[27],
    TensorFzz[27];

static float Height0, Height, Height1;

#define CVC_DBL_EPSILON 1.0e-9f
#define XUGUO_Z1 sqrt(3.0) - 2.0
#define OneSix3 1.0 / 216.0
#define OneSix2 2.0 / 108.0
#define OneSix1 4.0 / 54.0
#define OneSix0 8.0 / 27.0

main() {
  // get the atom list from the molecule
  int dim[3];
  float minExtent[3];
  float maxExtent[3];
  float data[274625];
  minExtent[0] = -5.0;
  minExtent[1] = -5.0;
  minExtent[2] = -5.0;

  maxExtent[0] = 5.0;
  maxExtent[1] = 5.0;
  maxExtent[2] = 5.0;

  dim[0] = 65;
  dim[1] = 65;
  dim[2] = 65;

  // initialize data and extent.
  printf("Xu Test 4\n");

  // compute the function
  computeFunction_Xu(data, dim, minExtent, maxExtent);
}

void computeFunction_Xu(float *funcvalue, int *dim, float *minExtent,
                        float *maxExtent) {
  float funcvalue_bak[274625], coefficent[274625], dx, dy, dz;
  float minx, maxx, miny, maxy, minz, maxz;
  float dt;
  int i, k, nx, ny, nz, t, end;

  // data preparaion
  nx = dim[0];
  ny = dim[1];
  nz = dim[2];

  minx = minExtent[0];
  maxx = maxExtent[0];

  miny = minExtent[1];
  maxy = maxExtent[1];

  minz = minExtent[2];
  maxz = maxExtent[2];

  ComputeTensorXYZ();

  dx = (maxx - minx) / (nx - 1.0);
  dy = (maxy - miny) / (ny - 1.0);
  dz = (maxz - minz) / (nz - 1.0);

  k = log(dx * dx * 0.005 / 1.732) / log(0.26795);
  // k = 8;
  k = k + 2;
  printf("k = %d\n", k);

  Height0 = (k - 4) * dx;
  Height = k * dx;
  Height1 = k * dx;
  // Height1 = 7.0*dx;

  printf("Height0, Height, Height1 = %f, %f, %f, dx = %f, %f\n", Height0,
         Height, Height1, dx, pow(88.0, 3));

  // Compute initial data
  // InitialData_Sphere(funcvalue, minx, maxx, miny, maxy, minz, maxz, nx, ny,
  // nz);
  InitialData_Box(funcvalue, minx, maxx, miny, maxy, minz, maxz, nx, ny, nz);

  // Convert to spline function
  for (i = 0; i < nx * ny * nz; i++) {
    coefficent[i] = funcvalue[i];
  }
  ConvertToInterpolationCoefficients_3D(coefficent, nx, ny, nz,
                                        CVC_DBL_EPSILON);

  // Evolution
  dt = 0.01;
  end = 1;
  for (t = 0; t < end; t++) {

    printf("Interation for time = %d\n", t);
    MeanCurvatureFlow(funcvalue, funcvalue_bak, coefficent, dt, dx, dy, dz,
                      nx, ny, nz);

    for (i = 0; i < nx * ny * nz; i++) {
      coefficent[i] = funcvalue[i];
    }
    ConvertToInterpolationCoefficients_3D(coefficent, nx, ny, nz,
                                          CVC_DBL_EPSILON);

    // Reinitilization
    // ReInitilazation(funcvalue, coefficent, dx, dy, dz, nx, ny, nz);
    // ReInitilazation_Upwind_Eno_Engquist(funcvalue, funcvalue_bak,
    // coefficent, dx, dy, dz, nx, ny, nz); if (t < end - 1)
    // ReInitilazation_Upwind_Eno_Godunov(funcvalue, funcvalue_bak,
    // coefficent, dx, dy, dz, nx, ny, nz, t);
  }

  float minf, maxf;

  minf = 1000.0;
  maxf = -1000.0;
  for (i = 0; i < nx * ny * nz; i++) {
    if (funcvalue[i] > maxf)
      maxf = funcvalue[i];
    if (funcvalue[i] < minf)
      minf = funcvalue[i];
  }

  printf("minf, maxf = %f, %f\n", minf, maxf);

  maxf = 1.0 / (maxf - minf);
  for (i = 0; i < nx * ny * nz; i++) {
    funcvalue[i] = (funcvalue[i] - minf) * maxf;
  }
}

// New conversion code by Xuguo
/*-----------------------------------------------------------------------------*/
void ConvertToInterpolationCoefficients_1D(float *s, int DataLength,
                                           float EPSILON)
//		float	*s,		/* input samples --> output
//coefficients */ 		int	DataLength,	/* number of samples or coefficients
//*/ 		float	EPSILON		/* admissible relative error             */

{
  int i, n, ni, ni1, K;
  float sum, z1, w1, w2;

  n = DataLength + 1;
  z1 = sqrt(3.0) - 2.0;
  K = log(EPSILON) / log(fabs(z1));
  // printf("K = %i\n", K);

  // compute initial value s(0)
  sum = 0.0;
  w2 = pow(z1, 2 * n);
  if (n < K) {
    for (i = 1; i < n; i++) {
      w1 = pow(z1, i);
      sum = sum + s[i - 1] * (w1 - w2 / w1);
    }
  } else {
    for (i = 1; i < n; i++) {
      sum = (sum + s[n - i - 1]) * z1;
    }
  }
  sum = -sum / (1.0 - w2);

  // compute c^{+}
  n = DataLength;
  s[0] = s[0] + z1 * sum;
  for (i = 1; i < n; i++) {
    s[i] = s[i] + z1 * s[i - 1];
    // printf("cp[%i] = %e, %f \n", i, cp[i], z1);
  }

  // compute c^-
  s[n - 1] = -z1 * s[n - 1];
  for (i = 1; i < n; i++) {
    ni = n - i;
    ni1 = ni - 1;
    s[ni1] = z1 * (s[ni] - s[ni1]);
  }

  for (i = 0; i < n; i++) {
    s[i] = 6.0 * s[i];
  }
}

/*-----------------------------------------------------------------------------*/
void ConvertToInterpolationCoefficients_2D(float *s, int nx, int ny,
                                           float EPSILON)
//		float	*s,     /* input samples --> output coefficients */
//		int	nx,	/* number of samples or coefficients in x
//direction    */ 		int	ny,	/* number of samples or coefficients in y
//direction    */ 		float	EPSILON	/* admissible relative error */
{
  float *d, z1;
  int i, l;

  d = (float *)malloc(nx * sizeof(float));

  z1 = sqrt(3.0) - 2.0;
  // x-direction interpolation
  for (l = 0; l < ny; l++) {
    for (i = 0; i < nx; i++) {
      d[i] = s[i * ny + l];
    }
    // ConvertToInterpolationCoefficients_1D(d, nx,CVC_DBL_EPSILON);
    ConvertToInterpolationCoefficients_Qu(d, nx, &z1, 1, CVC_DBL_EPSILON);

    for (i = 0; i < nx; i++) {
      s[i * ny + l] = d[i];
    }
  }

  // y-direction interpolation
  for (i = 0; i < nx; i++) {
    // ConvertToInterpolationCoefficients_1D(s+i*ny, ny,CVC_DBL_EPSILON);
    ConvertToInterpolationCoefficients_Qu(s + i * ny, ny, &z1, 1,
                                          CVC_DBL_EPSILON);
  }

  free(d);
}

/*-----------------------------------------------------------------------------*/
void ConvertToInterpolationCoefficients_3D(float *s, int nx, int ny, int nz,
                                           float EPSILON)
//              float   *s,    /* input samples --> output coefficients */ int
//              nx,    /* number of samples or coefficients in x direction */
//              int     ny,    /* number of samples or coefficients in y
//              direction    */ int     nz,    /* number of samples or
//              coefficients in z direction    */ float   EPSILON/* admissible
//              relative error                           */
{
  float *d, z1;
  int u, v, w, k, kk;

  d = (float *)malloc(nx * sizeof(float));

  k = ny * nz;
  z1 = sqrt(3.0) - 2.0;

  // x-direction interpolation
  for (v = 0; v < ny; v++) {
    for (w = 0; w < nz; w++) {
      kk = v * nz + w;
      for (u = 0; u < nx; u++) {
        d[u] = s[u * k + kk];
      }
      // ConvertToInterpolationCoefficients_1D(d, nx,CVC_DBL_EPSILON);
      ConvertToInterpolationCoefficients_Qu(d, nx, &z1, 1, CVC_DBL_EPSILON);

      for (u = 0; u < nx; u++) {
        s[u * k + kk] = d[u];
      }
    }
  }

  for (u = 0; u < nx; u++) {
    ConvertToInterpolationCoefficients_2D(s + u * k, ny, nz, CVC_DBL_EPSILON);
  }

  free(d);
}

/*-----------------------------------------------------------------------------*/
float EvaluateCubicSplineAtGridPoint(float *c, int nx, int ny, int nz, int u,
                                     int v, int w)
//              float   *c,    /* the spline  coefficients */ int     nx, /*
//              number of samples or coefficients in x direction    */ int ny,
//              /* number of samples or coefficients in y direction    */ int
//              nz,    /* number of samples or coefficients in z direction */
{
  float result;

  if ((u > 0 && u < nx - 1) && (v > 0 && v < ny - 1) &&
      (w > 0 && w < nz - 1)) {

    result =
        OneSix3 * (TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v - 1, w - 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v - 1, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v + 1, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v + 1, w - 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v - 1, w - 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v - 1, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v + 1, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v + 1, w - 1));

    result =
        OneSix2 * (TakeACoefficient_Fast(c, nx, ny, nz, u, v - 1, w - 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u, v - 1, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u, v + 1, w - 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u, v + 1, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v, w - 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v, w - 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v, w + 1) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v - 1, w) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v + 1, w) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v - 1, w) +
                   TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v + 1, w)) +
        result;

    result = OneSix1 * (TakeACoefficient_Fast(c, nx, ny, nz, u, v, w - 1) +
                        TakeACoefficient_Fast(c, nx, ny, nz, u, v, w + 1) +
                        TakeACoefficient_Fast(c, nx, ny, nz, u, v - 1, w) +
                        TakeACoefficient_Fast(c, nx, ny, nz, u, v + 1, w) +
                        TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v, w) +
                        TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v, w)) +
             result;

    result = OneSix0 * TakeACoefficient_Fast(c, nx, ny, nz, u, v, w) + result;
    return (result);
  }

  result =
      OneSix3 * (TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v - 1, w - 1) +
                 TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v - 1, w + 1) +
                 TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v + 1, w + 1) +
                 TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v + 1, w - 1) +
                 TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v - 1, w - 1) +
                 TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v - 1, w + 1) +
                 TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v + 1, w + 1) +
                 TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v + 1, w - 1));

  result = OneSix2 * (TakeACoefficient_Slow(c, nx, ny, nz, u, v - 1, w - 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u, v - 1, w + 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u, v + 1, w - 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u, v + 1, w + 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v, w - 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v, w + 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v, w - 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v, w + 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v - 1, w) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v + 1, w) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v - 1, w) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v + 1, w)) +
           result;

  result = OneSix1 * (TakeACoefficient_Slow(c, nx, ny, nz, u, v, w - 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u, v, w + 1) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u, v - 1, w) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u, v + 1, w) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v, w) +
                      TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v, w)) +
           result;

  result = OneSix0 * TakeACoefficient_Slow(c, nx, ny, nz, u, v, w) + result;
  return (result);
}

/*-----------------------------------------------------------------------------*/
float TakeACoefficient_Fast(float *c, int nx, int ny, int nz, int u, int v,
                            int w) {
  return (c[(u * ny + v) * nz + w]);
}

/*-----------------------------------------------------------------------------*/
float TakeACoefficient_Slow(float *c, int nx, int ny, int nz, int u, int v,
                            int w) {
  float result;

  result = 0.0;
  if ((u >= 0 && u < nx) && (v >= 0 && v < ny) && (w >= 0 && w < nz)) {

    result = c[(u * ny + v) * nz + w];
  }

  return (result);
}

/*-----------------------------------------------------------------------------*/
void Tensor_333(float *xx, float *yy, float *zz, float *result) {
  int i, j, k, l;

  l = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        result[l] = xx[i] * yy[j] * zz[k];
        l = l + 1;
      }
    }
  }
}

/*-----------------------------------------------------------------------------*/
void Take_27_Coefficients(float *c, int nx, int ny, int nz, int u, int v,
                          int w, float *c27)
//              float   *c,    /* the spline  coefficients */ int     nx, /*
//              number of samples or coefficients in x direction    */ int ny,
//              /* number of samples or coefficients in y direction    */ int
//              nz,    /* number of samples or coefficients in z direction */
//              float   *c27   /* 27 coefficients */
{
  if ((u > 0 && u < nx - 1) && (v > 0 && v < ny - 1) &&
      (w > 0 && w < nz - 1)) {

    c27[0] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v - 1, w - 1);
    c27[1] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v - 1, w);
    c27[2] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v - 1, w + 1);

    c27[3] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v, w - 1);
    c27[4] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v, w);
    c27[5] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v, w + 1);

    c27[6] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v + 1, w - 1);
    c27[7] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v + 1, w);
    c27[8] = TakeACoefficient_Fast(c, nx, ny, nz, u - 1, v + 1, w + 1);

    c27[9] = TakeACoefficient_Fast(c, nx, ny, nz, u, v - 1, w - 1);
    c27[10] = TakeACoefficient_Fast(c, nx, ny, nz, u, v - 1, w);
    c27[11] = TakeACoefficient_Fast(c, nx, ny, nz, u, v - 1, w + 1);

    c27[12] = TakeACoefficient_Fast(c, nx, ny, nz, u, v, w - 1);
    c27[13] = TakeACoefficient_Fast(c, nx, ny, nz, u, v, w);
    c27[14] = TakeACoefficient_Fast(c, nx, ny, nz, u, v, w + 1);

    c27[15] = TakeACoefficient_Fast(c, nx, ny, nz, u, v + 1, w - 1);
    c27[16] = TakeACoefficient_Fast(c, nx, ny, nz, u, v + 1, w);
    c27[17] = TakeACoefficient_Fast(c, nx, ny, nz, u, v + 1, w + 1);

    c27[18] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v - 1, w - 1);
    c27[19] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v - 1, w);
    c27[20] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v - 1, w + 1);

    c27[21] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v, w - 1);
    c27[22] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v, w);
    c27[23] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v, w + 1);

    c27[24] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v + 1, w - 1);
    c27[25] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v + 1, w);
    c27[26] = TakeACoefficient_Fast(c, nx, ny, nz, u + 1, v + 1, w + 1);

    return;
  }

  c27[0] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v - 1, w - 1);
  c27[1] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v - 1, w);
  c27[2] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v - 1, w + 1);

  c27[3] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v, w - 1);
  c27[4] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v, w);
  c27[5] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v, w + 1);

  c27[6] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v + 1, w - 1);
  c27[7] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v + 1, w);
  c27[8] = TakeACoefficient_Slow(c, nx, ny, nz, u - 1, v + 1, w + 1);

  c27[9] = TakeACoefficient_Slow(c, nx, ny, nz, u, v - 1, w - 1);
  c27[10] = TakeACoefficient_Slow(c, nx, ny, nz, u, v - 1, w);
  c27[11] = TakeACoefficient_Slow(c, nx, ny, nz, u, v - 1, w + 1);

  c27[12] = TakeACoefficient_Slow(c, nx, ny, nz, u, v, w - 1);
  c27[13] = TakeACoefficient_Slow(c, nx, ny, nz, u, v, w);
  c27[14] = TakeACoefficient_Slow(c, nx, ny, nz, u, v, w + 1);

  c27[15] = TakeACoefficient_Slow(c, nx, ny, nz, u, v + 1, w - 1);
  c27[16] = TakeACoefficient_Slow(c, nx, ny, nz, u, v + 1, w);
  c27[17] = TakeACoefficient_Slow(c, nx, ny, nz, u, v + 1, w + 1);

  c27[18] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v - 1, w - 1);
  c27[19] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v - 1, w);
  c27[20] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v - 1, w + 1);

  c27[21] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v, w - 1);
  c27[22] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v, w);
  c27[23] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v, w + 1);

  c27[24] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v + 1, w - 1);
  c27[25] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v + 1, w);
  c27[26] = TakeACoefficient_Slow(c, nx, ny, nz, u + 1, v + 1, w + 1);

  return;
}

/*-----------------------------------------------------------------------------*/
void EvaluateCubicSplineOrder2PartialsAtGridPoint(float *c, float dx,
                                                  float dy, float dz, int nx,
                                                  int ny, int nz, int u,
                                                  int v, int w,
                                                  float *partials)
//              float   *c,    /* the spline  coefficients */ float   dx /*
//              spacing in x direction                              */ float
//              dy     /* spacing in y direction */ float   dz     /* spacing
//              in z direction                              */ int     nx, /*
//              number of samples or coefficients in x direction    */ int ny,
//              /* number of samples or coefficients in y direction    */ int
//              nz,    /* number of samples or coefficients in z direction */
//              float   *partial /* partial derivatives */
{
  float c27[27], indx, indy, indz;
  int i;

  Take_27_Coefficients(c, nx, ny, nz, u, v, w, c27);

  for (i = 0; i < 10; i++) {
    partials[i] = 0.0;
  }

  for (i = 0; i < 27; i++) {
    partials[0] = partials[0] + c27[i] * TensorF[i];

    partials[1] = partials[1] + c27[i] * TensorFx[i];
    partials[2] = partials[2] + c27[i] * TensorFy[i];
    partials[3] = partials[3] + c27[i] * TensorFz[i];

    partials[4] = partials[4] + c27[i] * TensorFxx[i];
    partials[5] = partials[5] + c27[i] * TensorFxy[i];
    partials[6] = partials[6] + c27[i] * TensorFxz[i];

    partials[7] = partials[7] + c27[i] * TensorFyy[i];
    partials[8] = partials[8] + c27[i] * TensorFyz[i];

    partials[9] = partials[9] + c27[i] * TensorFzz[i];

    // printf("i = %d   Coeff = %f,     Fxx = %f\n", i, c27[i], TensorFxx[i]);
  }

  indx = 1.0 / dx;
  indy = 1.0 / dy;
  indz = 1.0 / dz;

  // printf("indx = %f\n", partials[4]);

  partials[1] = partials[1] * indx;
  partials[2] = partials[2] * indy;
  partials[3] = partials[3] * indz;

  partials[4] = partials[4] * indx * indx;
  partials[5] = partials[5] * indx * indy;
  partials[6] = partials[6] * indx * indz;

  partials[7] = partials[7] * indy * indy;
  partials[8] = partials[8] * indy * indz;

  partials[9] = partials[9] * indz * indz;

  // printf("indx = %f\n", partials[4]);
}

/*-----------------------------------------------------------------------------*/
void EvaluateCubicSplineOrder1PartialsAtGridPoint(float *c, float dx,
                                                  float dy, float dz, int nx,
                                                  int ny, int nz, int u,
                                                  int v, int w,
                                                  float *partials)
//              float   *c,    /* the spline  coefficients */ float   dx /*
//              spacing in x direction                              */ float
//              dy     /* spacing in y direction */ float   dz     /* spacing
//              in z direction                              */ int     nx, /*
//              number of samples or coefficients in x direction    */ int ny,
//              /* number of samples or coefficients in y direction    */ int
//              nz,    /* number of samples or coefficients in z direction */
//              float   *partial /* partial derivatives */
{
  float c27[27], indx, indy, indz;
  int i;

  Take_27_Coefficients(c, nx, ny, nz, u, v, w, c27);

  for (i = 0; i < 4; i++) {
    partials[i] = 0.0;
  }

  for (i = 0; i < 27; i++) {
    partials[0] = partials[0] + c27[i] * TensorF[i];

    partials[1] = partials[1] + c27[i] * TensorFx[i];
    partials[2] = partials[2] + c27[i] * TensorFy[i];
    partials[3] = partials[3] + c27[i] * TensorFz[i];
  }

  indx = 1.0 / dx;
  indy = 1.0 / dy;
  indz = 1.0 / dz;

  // printf("indx = %f, %f, %f\n", indx, indy, indz);

  partials[1] = partials[1] * indx;
  partials[2] = partials[2] * indy;
  partials[3] = partials[3] * indz;
}

/*-----------------------------------------------------------------------------*/
void InitialData_Sphere(float *v, float minx, float maxx, float miny,
                        float maxy, float minz, float maxz, int nx, int ny,
                        int nz)
//              float   *v,    /* the volume data of initial signed distance
//              function */ float   minx   /* left end-point of x diection */
//              float   maxx   /* right end-point of x diection */ float miny
//              /* left end-point of y diection                        */
//              float   maxy   /* right end-point of y diection */ float minz
//              /* left end-point of z diection                        */
//              float   maxz   /* right end-point of z diection */ int     nx,
//              /* number of points in x direction                     */ int
//              ny,    /* number of points in y direction */ int     nz,    /*
//              number of points in z direction                     */
{
  int i, j, k, l;
  float dx, dy, dz, x, y, z;
  float cx, cy, cz, radius, dis;

  dx = (maxx - minx) / (nx - 1.0);
  dy = (maxy - miny) / (ny - 1.0);
  dz = (maxz - minz) / (nz - 1.0);

  // determine the center and radius
  cx = (maxx + minx) / 2.0;
  cy = (maxy + miny) / 2.0;
  cz = (maxz + minz) / 2.0;

  radius = (maxx - minx) / 2.0;

  if (radius > (maxy - miny) / 2.0)
    radius = (maxy - miny) / 2.0;
  if (radius > (maxz - minz) / 2.0)
    radius = (maxz - minz) / 2.0;

  radius = radius - 3 * (dx + dy + dz);
  // radius = radius - dx;
  printf("radius = %f\n", radius);

  // compute function value at grid points
  l = 0;
  for (i = 0; i < nx; i++) {
    x = minx + i * dx;
    for (j = 0; j < ny; j++) {
      y = miny + j * dy;
      for (k = 0; k < nz; k++) {
        z = minz + k * dz;
        dis = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) +
                   (z - cz) * (z - cz));
        dis = dis - radius;

        // cut
        if (dis < -Height)
          dis = -Height1;
        if (dis > Height)
          dis = Height1;

        v[l] = dis;
        // v[l] = dis;
        // printf("volume[%d, %d, %d] = %f\n", i,j,k , v[l]);
        l = l + 1;
      }
    }
  }
}

/*-----------------------------------------------------------------------------*/
void InitialData_Box(float *v, float minx, float maxx, float miny, float maxy,
                     float minz, float maxz, int nx, int ny, int nz)
//              float   *v,    /* the volume data of initial signed distance
//              function */ float   minx   /* left end-point of x diection */
//              float   maxx   /* right end-point of x diection */ float miny
//              /* left end-point of y diection                        */
//              float   maxy   /* right end-point of y diection */ float minz
//              /* left end-point of z diection                        */
//              float   maxz   /* right end-point of z diection */ int     nx,
//              /* number of points in x direction                     */ int
//              ny,    /* number of points in y direction */ int     nz,    /*
//              number of points in z direction                     */
{
  int i, j, k, l;
  float dx, dy, dz, x, y, z;
  float cx, cy, cz, radius, dis;

  dx = (maxx - minx) / (nx - 1.0);
  dy = (maxy - miny) / (ny - 1.0);
  dz = (maxz - minz) / (nz - 1.0);

  // determine the center and radius
  cx = (maxx + minx) / 2.0;
  cy = (maxy + miny) / 2.0;
  cz = (maxz + minz) / 2.0;

  radius = (maxx - minx) / 2.0;

  if (radius > (maxy - miny) / 2.0)
    radius = (maxy - miny) / 2.0;
  if (radius > (maxz - minz) / 2.0)
    radius = (maxz - minz) / 2.0;

  radius = radius - 3 * (dx + dy + dz);
  // radius = radius - dx;
  printf("radius = %f\n", radius);

  // compute function value at grid points
  l = 0;
  for (i = 0; i < nx; i++) {
    x = fabs(minx + i * dx);
    for (j = 0; j < ny; j++) {
      y = fabs(miny + j * dy);
      for (k = 0; k < nz; k++) {
        z = fabs(minz + k * dz);

        if (x >= radius && y >= radius && z >= radius) {
          dis =
              sqrt((x - radius) * (x - radius) + (y - radius) * (y - radius) +
                   (z - radius) * (z - radius));
        } else {

          dis = 0.0;
          if (x >= radius)
            dis = (x - radius) * (x - radius);
          if (y >= radius)
            dis = (y - radius) * (y - radius) + dis;
          if (z >= radius)
            dis = (z - radius) * (z - radius) + dis;
          dis = sqrt(dis);

          if (x >= radius && y <= radius && z <= radius)
            dis = x - radius;
          if (x <= radius && y >= radius && z <= radius)
            dis = y - radius;
          if (x <= radius && y <= radius && z >= radius)
            dis = z - radius;

          /*
                      if (x >= radius && y >= radius && z <= radius)
                         dis = sqrt((x - radius)*(x - radius) + (y -
             radius)*(y - radius)); if (x >= radius && y <= radius && z >=
             radius) dis = y - radius; dis = sqrt((x - radius)*(x - radius) +
             (z - radius)*(z - radius)); if (x <= radius && y >= radius && z
             >= radius) dis = z - radius; dis = sqrt((z - radius)*(z - radius)
             + (y - radius)*(y - radius));
          */

          if (x <= radius && y <= radius && z <= radius) {
            dis = radius - x;
            if (dis > radius - y)
              dis = radius - y;
            if (dis > radius - z)
              dis = radius - z;
            dis = -dis;
          }
        }

        // cut
        if (dis < -Height)
          dis = -Height1;
        if (dis > Height)
          dis = Height1;

        v[l] = dis;
        // v[l] = dis;
        // printf("volume[%d, %d, %d] = %f\n", i,j,k , v[l]);
        l = l + 1;
      }
    }
  }
}

/*-----------------------------------------------------------------------------*/
void MeanCurvatureFlow(float *fun, float *fun_bak, float *coeff, float dt,
                       float dx, float dy, float dz, int nx, int ny, int nz)
//              float   *fun   /* the volume data of initial signed distance
//              function */ float   *coeff /* spline coefficients */ float dt
//              /* time step size                                      */
//              float   dx     /* x direction spacing */ float   dy     /* y
//              direction spacing                                 */ float dz
//              /* z direction spacing                                 */ int
//              nx,    /* number of points in x direction */ int     ny,    /*
//              number of points in y direction                     */ int nz,
//              /* number of points in z direction                     */
{
  int i, j, k, index;
  float partials[10], grad1, grad2, H;
  float x, y, z, r, fvalue, maxf, maxt;
  float radius;

  radius = 5.0 - 6 * dx;

  maxf = -1000.0;

  printf("333 = %f\n", TakeACoefficient_Fast(fun, nx, ny, nz, 10, 10, 10));
  printf("444 = %f\n",
         TakeACoefficient_Fast(fun, nx, ny, nz, nx - 11, ny - 11, nz - 11));
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      for (k = 0; k < nz; k++) {

        if ((i == 10 && j == 10 && k == 10) ||
            (i == nx - 11 && j == ny - 11 && k == nz - 11)) {
          index = (i * ny + j) * nz + k;
          fvalue = fun[index];
          fun_bak[index] = 0.0;

          // if (fabs(fvalue) < Height0) {
          if (fabs(fvalue) < Height) {
            // if (i == 10 & j == 25 & k == 25) {
            // EvaluateCubicSplineOrder2PartialsAtGridPoint(coeff, dx, dy, dz,
            // nx, ny, nz, i, j, k, partials);
            Divided_DifferenceOrder2PartialsAtGridPoint(
                fun, dx, dy, dz, nx, ny, nz, i, j, k, partials);

            grad2 = partials[1] * partials[1] + partials[2] * partials[2] +
                    partials[3] * partials[3];
            if (grad2 < 0.0001)
              grad2 = 0.0001;

            grad1 = sqrt(grad2);

            H = -partials[1] *
                    (partials[4] * partials[1] + partials[5] * partials[2] +
                     partials[6] * partials[3]) -
                partials[2] *
                    (partials[5] * partials[1] + partials[7] * partials[2] +
                     partials[8] * partials[3]) -
                partials[3] *
                    (partials[6] * partials[1] + partials[8] * partials[2] +
                     partials[9] * partials[3]);
            // H = (H/grad2 + partials[4] + partials[7] + partials[9])/grad1;
            H = H / grad2 + partials[4] + partials[7] + partials[9];

            x = -5.0 + i * dx;
            y = -5.0 + j * dy;
            z = -5.0 + k * dz;
            r = sqrt(x * x + y * y + z * z);

            // if (fabs(fvalue) < 0.01) {
            printf("i, j, k = %d, %d, %d H = %f, Exact H = %f, %e, %f, %e\n",
                   i, j, k, H / 2.0, 1.0 / r, partials[0], grad1,
                   r - radius - partials[0]);
            printf("Dxyz = %f, %f, %f, %f,%f, %f, %f, %f,%f,%f\n",
                   partials[0], partials[1], partials[2], partials[3],
                   partials[4], partials[5], partials[6], partials[7],
                   partials[8], partials[9]);
            //}

            fun_bak[index] = H * grad1;
            if (fabs(fvalue) < Height0) {
              if (fabs(fun_bak[index]) > maxf)
                maxf = fabs(fun_bak[index]);
            }

            // fun_bak[index] = fvalue + dt*H*grad1;

            // fvalue = fvalue + dt*fun_bak[index];

            // if (fvalue < -Height) fvalue = -Height1;
            // if (fvalue > Height) fvalue = Height1;

            // fun[index] = fvalue;
          }
        }
      }
    }
  }

  maxt = dx / maxf;
  printf("maxf = %f, dx = %f, maxt = %f\n", maxf, dx, maxt);

  for (i = 0; i < nx * ny * nz; i++) {
    fun[i] = fun[i] + maxt * fun_bak[i];

    // cut the function by Height
    if (fun[i] < -Height)
      fun[i] = -Height1;
    if (fun[i] > Height)
      fun[i] = Height1;
  }
}

/*-----------------------------------------------------------------------------*/
void ComputeTensorXYZ() {
  float f[3], fx[3], fxx[3];

  f[0] = 1.0 / 6.0;
  f[1] = 2.0 / 3.0;
  f[2] = f[0];

  fx[0] = -0.5;
  fx[1] = 0.0;
  fx[2] = 0.5;

  fxx[0] = 1.0;
  fxx[1] = -2.0;
  fxx[2] = 1.0;

  Tensor_333(f, f, f, TensorF);

  Tensor_333(fx, f, f, TensorFx);
  Tensor_333(f, fx, f, TensorFy);
  Tensor_333(f, f, fx, TensorFz);

  Tensor_333(fxx, f, f, TensorFxx);
  Tensor_333(fx, fx, f, TensorFxy);
  Tensor_333(fx, f, fx, TensorFxz);

  Tensor_333(f, fxx, f, TensorFyy);
  Tensor_333(f, fx, fx, TensorFyz);
  Tensor_333(f, f, fxx, TensorFzz);
}

/*-----------------------------------------------------------------------------*/
void ReInitilazation(float *fun, float *coeff, float dx, float dy, float dz,
                     int nx, int ny, int nz)
//              float   *fun   /* the volume data of initial signed distance
//              function */ float   *coeff /* spline coefficients */ float dx
//              /* x direction spacing                                 */
//              float   dy     /* y direction spacing */ float   dz     /* z
//              direction spacing                                 */ int nx,
//              /* number of points in x direction                     */ int
//              ny,    /* number of points in y direction */ int     nz,    /*
//              number of points in z direction                     */
{
  int i, j, k, index, t;
  float partials[4], grad1, grad2;
  float x, y, z, r;
  float dt, sd;

  dt = 0.01;

  for (t = 0; t < 1000; t++) {

    printf("Initialization t = %d\n", t);
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {
          EvaluateCubicSplineOrder1PartialsAtGridPoint(
              coeff, dx, dy, dz, nx, ny, nz, i, j, k, partials);
          // printf("i , j, k = %d, %d, %d Dxyz = %f, %f, %f, %f\n", i, j,
          // k,partials[0], partials[1],partials[2], partials[3]);

          grad2 = partials[1] * partials[1] + partials[2] * partials[2] +
                  partials[3] * partials[3];
          grad1 = sqrt(grad2);
          sd =
              partials[0] / sqrt(partials[0] * partials[0] + grad2 * dx * dx);

          x = -5.0 + i * dx;
          y = -5.0 + j * dy;
          z = -5.0 + k * dz;
          r = sqrt(x * x + y * y + z * z);

          // if (r < 4.0)
          if (i == 44 && j == 31 && k == 26)
            printf("t = %d, i , j, k = %d, %d, %d, grad1 =  %f  S(d) = %f\n",
                   t, i, j, k, grad1, sd);
          index = (i * ny + j) * nz + k;
          fun[index] = fun[index] - dt * sd * (grad1 - 1.0);
        }
      }
    }

    for (i = 0; i < nx * ny * nz; i++) {
      coeff[i] = fun[i];
    }
    ConvertToInterpolationCoefficients_3D(coeff, nx, ny, nz, CVC_DBL_EPSILON);
  }
}

/*-----------------------------------------------------------------------------*/
void ReInitilazation_Upwind_Eno_Engquist(float *fun, float *fun_bak,
                                         float *coeff, float dx, float dy,
                                         float dz, int nx, int ny, int nz)
//              float   *fun   /* the volume data of initial signed distance
//              function */ float   *coeff /* spline coefficients */ float dx
//              /* x direction spacing                                 */
//              float   dy     /* y direction spacing */ float   dz     /* z
//              direction spacing                                 */ int nx,
//              /* number of points in x direction                     */ int
//              ny,    /* number of points in y direction */ int     nz,    /*
//              number of points in z direction                     */
{
  int i, j, k, index, t;
  float fvalue, grad1, grad2, ww, ww1;
  float x, y, z, r, fx[7], fy[7], fz[7];
  float dt, sd, sdp, sdm;
  float a, b, c, d, e, f;
  float ap, bp, cp, dp, ep, fp;
  float am, bm, cm, dm, em, fm;

  dt = 0.5 * dx;
  dt = 0.05;

  for (t = 0; t < 100; t++) {
    printf("Initialization t = %d\n", t);
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        // for (i = 10; i < 11; i++) {
        //    for (j = 10; j < 11; j++) {
        for (k = 0; k < nz; k++) {

          Get_Seven_Function_Values_X(fun, nx, ny, nz, i, j, k, fx);
          // Cubic_Eno_Interpolation(fx, &b, &a, dx);
          Quadr_Eno_Interpolation(fx, &b, &a, dx);
          // Linear_Eno_Interpolation(fx, &b, &a, dx);

          ap = a;
          if (a < 0.0)
            ap = 0.0;
          bp = b;
          if (b < 0.0)
            bp = 0.0;

          am = a;
          if (a > 0.0)
            am = 0.0;
          bm = b;
          if (b > 0.0)
            bm = 0.0;

          Get_Seven_Function_Values_Y(fun, nx, ny, nz, i, j, k, fy);
          // Cubic_Eno_Interpolation(fy, &d, &c, dy);
          Quadr_Eno_Interpolation(fy, &d, &c, dy);
          // Linear_Eno_Interpolation(fy, &d, &c, dy);

          cp = c;
          if (c < 0.0)
            cp = 0.0;
          dp = d;
          if (d < 0.0)
            dp = 0.0;

          cm = c;
          if (c > 0.0)
            cm = 0.0;
          dm = d;
          if (d > 0.0)
            dm = 0.0;

          Get_Seven_Function_Values_Z(fun, nx, ny, nz, i, j, k, fz);
          // printf("%f,%f,%f,%f,%f,%f,%f\n",
          // fz[0]-fz[1],fz[1]-fz[2],fz[2]-fz[3],fz[3]-fz[4],fz[4]-fz[5],fz[5]-fz[6],dx);
          // Cubic_Eno_Interpolation(fz, &f, &e, dz);
          Quadr_Eno_Interpolation(fz, &f, &e, dz);
          // Linear_Eno_Interpolation(fz, &f, &e, dz);

          ep = e;
          if (e < 0.0)
            ep = 0.0;
          fp = f;
          if (f < 0.0)
            fp = 0.0;

          em = e;
          if (e > 0.0)
            em = 0.0;
          fm = f;
          if (f > 0.0)
            fm = 0.0;

          // printf("i,j,k = %d, %d, %d a--f = %f, %f, %f, %f, %f, %f, grad =
          // %f, %f \n", i, j, k, a, b, c, d, e, f,
          //                            a*a + c*c + e*e, b*b + d*d + f*f);

          grad2 = 0.5 * (a * a + c * c + e * e + b * b + d * d + f * f);
          grad1 = sqrt(grad2);
          index = (i * ny + j) * nz + k;

          fvalue = fun[index];
          sd = fvalue / sqrt(fvalue * fvalue + grad2 * dx * dx);
          sdp = sd;
          if (sd < 0.0)
            sdp = 0.0;

          sdm = sd;
          if (sd > 0.0)
            sdm = 0.0;

          x = -5.0 + i * dx;
          y = -5.0 + j * dy;
          z = -5.0 + k * dz;
          r = sqrt(x * x + y * y + z * z);

          // if (r < 0.4)
          // if (r  > 1.0)
          // if (i == 44 && j== 31 && k ==  26)
          // printf("t = %d, i , j, k = %d, %d, %d, grad1 =  %f  S(d) = %f\n",
          // t, i, j, k, grad1, sd);
          if (sdp > 0.0) {
            ap = ap * ap;
            cp = cp * cp;
            ep = ep * ep;
            bm = bm * bm;
            dm = dm * dm;
            fm = fm * fm;

            // Engquist-Osher scheme
            ww = sqrt(ap + cp + ep + bm + dm + fm);
            // end of Engquist-Osher scheme

            /*
                        // Godunov scheme
                        ww = ap;
                        if (ww < bm) ww = bm;
                        ww1 = cp;
                        if (ww1 < dm) ww1 = dm;
                        ww = ww + ww1;
                        ww1 = ep;
                        if (ww1 < fm) ww1 = fm;
                        ww = ww + ww1;
                        ww = sqrt(ww);
                        // end Godunov scheme
            */

            fun_bak[index] = fvalue - dt * sdp * (ww - 1.0);
          }
          if (sdm < 0.0) {
            am = am * am;
            cm = cm * cm;
            em = em * em;
            bp = bp * bp;
            dp = dp * dp;
            fp = fp * fp;

            // Engquist-Osher scheme
            ww = sqrt(am + cm + em + bp + dp + fp);
            // end Engquist-Osher scheme

            /*
                        // Godunov scheme
                        ww = am;
                        if (ww < bp) ww = bp;
                        ww1 = cm;
                        if (ww1 < dp) ww1 = dp;
                        ww = ww + ww1;
                        ww1 = em;
                        if (ww1 < fp) ww1 = fp;
                        ww = ww + ww1;
                        ww = sqrt(ww);
                        // end Godunov scheme
            */

            fun_bak[index] = fvalue - dt * sdm * (ww - 1.0);
          }

          // if (fabs(fvalue)  < 0.3)
          // if (r  < 4.01 && r > 3.9)
          if (r < 0.4)
            // if (i == nx/2 && j == ny/2 && k == nz/2)
            // if (i == 0 && j == 0 && k == 0)
            printf("t = %d, i , j, k = %d, %d, %d, grad1 =  %f  S(d) = %f, "
                   "fun = %f,  ww = %f\n",
                   t, i, j, k, grad1, sd, fvalue, ww);
        }
      }
    }

    for (i = 0; i < nx * ny * nz; i++) {
      fun[i] = fun_bak[i];
    }
  }

  for (i = 0; i < nx * ny * nz; i++) {
    coeff[i] = fun[i];
  }
  ConvertToInterpolationCoefficients_3D(coeff, nx, ny, nz, CVC_DBL_EPSILON);
}

/*-----------------------------------------------------------------------------*/
void ReInitilazation_Upwind_Eno_Godunov(float *fun, float *fun_bak,
                                        float *coeff, float dx, float dy,
                                        float dz, int nx, int ny, int nz,
                                        int tt)
//              float   *fun   /* the volume data of initial signed distance
//              function */ float   *coeff /* spline coefficients */ float dx
//              /* x direction spacing                                 */
//              float   dy     /* y direction spacing */ float   dz     /* z
//              direction spacing                                 */ int nx,
//              /* number of points in x direction                     */ int
//              ny,    /* number of points in y direction */ int     nz,    /*
//              number of points in z direction                     */
{
  int i, j, k, index, t;
  float fvalue, fvalue_new, grad1, grad2, ww, ww1;
  float x, y, z, r, fx[7], fy[7], fz[7];
  float dt, sd, sdp, sdm, maxt, maxf;
  float a, b, c, d, e, f;

  dt = 0.5 * dx;
  // dt = 0.05;

  // for (t = 0; t < 10; t++) {
  for (t = 0; t < 20; t++) {

    maxf = -100000.0;
    printf("Initialization t = %d\n", t);
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        // for (i = 10; i < 11; i++) {
        //    for (j = 10; j < 11; j++) {
        for (k = 0; k < nz; k++) {

          index = (i * ny + j) * nz + k;
          fvalue = fun[index];
          fun_bak[index] = 0.0;

          Get_Seven_Function_Values_X(fun, nx, ny, nz, i, j, k, fx);
          Get_Seven_Function_Values_Y(fun, nx, ny, nz, i, j, k, fy);
          Get_Seven_Function_Values_Z(fun, nx, ny, nz, i, j, k, fz);

          if (fabs(fx[2]) < Height || fabs(fx[4]) < Height ||
              fabs(fy[2]) < Height || fabs(fy[4]) < Height ||
              fabs(fz[2]) < Height || fabs(fz[4]) < Height ||
              fabs(fvalue) < Height) {

            // Cubic_Eno_Interpolation(fx, &b, &a, dx);
            Quadr_Eno_Interpolation(fx, &b, &a, dx);
            // Linear_Eno_Interpolation(fx, &b, &a, dx);

            // Cubic_Eno_Interpolation(fy, &d, &c, dy);
            Quadr_Eno_Interpolation(fy, &d, &c, dy);
            // Linear_Eno_Interpolation(fy, &d, &c, dy);

            // Cubic_Eno_Interpolation(fz, &f, &e, dz);
            Quadr_Eno_Interpolation(fz, &f, &e, dz);
            // Linear_Eno_Interpolation(fz, &f, &e, dz);

            /*
            if (i == nx/2 && j == ny/2 && k == nz/2)
            printf("i,j,k = %d, %d, %d a--f = %f, %f, %f, %f, %f, %f, grad =
            %f, %f \n", i, j, k, a, b, c, d, e, f, a*a + c*c + e*e, b*b + d*d
            + f*f);
            */

            // grad2 =  Gradient_2(fun, dx, dy, dz, i, j, k, nx, ny, nz);

            grad2 = 0.5 * (a * a + c * c + e * e + b * b + d * d + f * f);
            grad1 = sqrt(grad2);

            sd = fvalue / sqrt(fvalue * fvalue + grad2 * dx * dx);
            /*
            sd = 1.0;
            if (fvalue < 0.0)
            sd = -1.0;
            */
            sdp = sd;
            if (sd < 0.0)
              sdp = 0.0;

            sdm = sd;
            if (sd > 0.0)
              sdm = 0.0;

            x = -5.0 + i * dx;
            y = -5.0 + j * dy;
            z = -5.0 + k * dz;
            r = sqrt(x * x + y * y + z * z);

            // if (r < 0.4)
            // if (r  > 1.0)
            // if (i == 44 && j== 31 && k ==  26)
            // printf("t = %d, i , j, k = %d, %d, %d, grad1 =  %f  S(d) =
            // %f\n", t, i, j, k, grad1, sd);

            if (sdp >= 0.0) {

              ww = sqrt(Extreme_Positive(a, b) + Extreme_Positive(c, d) +
                        Extreme_Positive(e, f));
              fvalue_new = -dt * sdp * (ww - 1.0); // Zhao's method
            }
            if (sdm < 0.0) {

              ww = sqrt(Extreme_Negative(a, b) + Extreme_Negative(c, d) +
                        Extreme_Negative(e, f));
              fvalue_new = -dt * sdm * (ww - 1.0); // Zhao's method
            }

            // if (fvalue_new < -Height) fvalue_new = -Height1;
            // if (fvalue_new > Height)  fvalue_new =  Height1;

            fun_bak[index] = fvalue_new;
            if (fabs(fun_bak[index]) > maxf)
              maxf = fabs(fun_bak[index]);

            // fun_bak[index] = fvalue - dt*fvalue * (grad1 - 1.0)/grad1; //
            // Xu's method

            // if (fabs(fvalue)  < 0.001*Height)
            // if (r  < 4.01 && r > 3.9)
            // if (r  < 0.4)
            // if (i == nx/2 && j == ny/2 && k == nz/2)
            // if (i == 0 && j == 0 && k == 0)
            // if (tt == 9 && t == 99)
            // printf("t = %d, i , j, k = %d, %d, %d, grad1 =  %f  S(d) = %f,
            // fun = %f,  ww = %f, r=%f\n",
            //         t, i, j, k, grad1, sd, fvalue, ww, r);
          }
        }
      }
    }

    maxt = dx / maxf;
    printf("maxf = %f, dx = %f, maxt = %f\n", maxf, dx, maxt);
    if (maxt > t)
      maxt = t;

    for (i = 0; i < nx * ny * nz; i++) {
      fun[i] = fun[i] + maxt * fun_bak[i];

      // cut the function by Height
      if (fun[i] < -Height)
        fun[i] = -Height1;
      if (fun[i] > Height)
        fun[i] = Height1;
    }
  }

  for (i = 0; i < nx * ny * nz; i++) {
    coeff[i] = fun[i];
  }
  ConvertToInterpolationCoefficients_3D(coeff, nx, ny, nz, CVC_DBL_EPSILON);
}

/*-----------------------------------------------------------------------------*/
void DiviededDifferencing_3j(float *f0, float *f1, float *f2, float *f3,
                             float dx) {
  int i;
  float indx1, indx2, indx3;

  indx1 = 1.0 / dx;

  // The first order divided difference
  for (i = 0; i < 6; i++) {
    f1[i] = (f0[i + 1] - f0[i]) * indx1;
  }

  // The second order divided difference
  indx2 = indx1 / 2.0;
  for (i = 0; i < 5; i++) {
    f2[i] = (f1[i + 1] - f1[i]) * indx2;
  }

  // The third order divided difference
  indx3 = indx1 / 3.0;
  for (i = 0; i < 4; i++) {
    f3[i] = (f2[i + 1] - f2[i]) * indx3;
  }
}

/*-----------------------------------------------------------------------------*/
void DiviededDifferencing_2j(float *f0, float *f1, float *f2, float dx) {
  int i;
  float indx1, indx2, indx3;

  indx1 = 1.0 / dx;

  // The first order divided difference
  for (i = 1; i < 5; i++) {
    f1[i] = (f0[i + 1] - f0[i]) * indx1;
  }

  // The second order divided difference
  indx2 = indx1 / 2.0;
  for (i = 1; i < 4; i++) {
    f2[i] = (f1[i + 1] - f1[i]) * indx2;
  }
}

/*-----------------------------------------------------------------------------*/
void DiviededDifferencing_1j(float *f0, float *f1, float dx) {
  int i;
  float indx1, indx2, indx3;

  indx1 = 1.0 / dx;

  // The first order divided difference
  for (i = 2; i < 4; i++) {
    f1[i] = (f0[i + 1] - f0[i]) * indx1;
  }
}

/*-----------------------------------------------------------------------------*/
void Cubic_Eno_Interpolation(float *f0, float *u_plus, float *u_minus,
                             float dx) {
  int k;
  float f1[6], f2[5], f3[4];
  float x1, x2, x3;

  DiviededDifferencing_3j(f0, f1, f2, f3, dx);
  // printf("%f,%f,%f,%f,%f,%f,%f\n",
  // f0[0],f0[1],f0[2],f0[3],f0[4],f0[5],f0[6]);

  // Compute u^-
  k = 2;
  if (fabs(f2[k - 1]) < fabs(f2[k]))
    k = k - 1;
  if (fabs(f3[k - 1]) < fabs(f3[k]))
    k = k - 1;

  x1 = (3 - k) * dx;
  x2 = (2 - k) * dx;
  x3 = (1 - k) * dx;

  *u_minus =
      f1[k] + f2[k] * (x1 + x2) + f3[k] * (x1 * x2 + x1 * x3 + x2 * x3);
  // printf("k- = %d, u_minus = %f\n", k, *u_minus);

  // Compute u^+
  k = 3;
  if (fabs(f2[k - 1]) < fabs(f2[k]))
    k = k - 1;
  if (fabs(f3[k - 1]) < fabs(f3[k]))
    k = k - 1;

  x1 = (3 - k) * dx;
  x2 = (2 - k) * dx;
  x3 = (1 - k) * dx;

  *u_plus = f1[k] + f2[k] * (x1 + x2) + f3[k] * (x1 * x2 + x1 * x3 + x2 * x3);
  // printf("k+ = %d, *u_plus = %f\n", k, *u_plus);
}

/*-----------------------------------------------------------------------------*/
void Quadr_Eno_Interpolation(float *f0, float *u_plus, float *u_minus,
                             float dx) {
  int k;
  float f1[6], f2[5];
  float x1, x2, x3;

  DiviededDifferencing_2j(f0, f1, f2, dx);
  // printf("%f,%f,%f,%f,%f,%f,%f\n",
  // f0[0],f0[1],f0[2],f0[3],f0[4],f0[5],f0[6]);

  // Compute u^-
  k = 2;
  if (fabs(f2[k - 1]) < fabs(f2[k]))
    k = k - 1;

  x1 = (3 - k) * dx;
  x2 = (2 - k) * dx;

  *u_minus = f1[k] + f2[k] * (x1 + x2);
  // printf("k- = %d, u_minus = %f\n", k, *u_minus);

  // Compute u^+
  k = 3;
  if (fabs(f2[k - 1]) < fabs(f2[k]))
    k = k - 1;

  x1 = (3 - k) * dx;
  x2 = (2 - k) * dx;

  *u_plus = f1[k] + f2[k] * (x1 + x2);
  // printf("k+ = %d, *u_plus = %f\n", k, *u_plus);
}

/*-----------------------------------------------------------------------------*/
void Linear_Eno_Interpolation(float *f0, float *u_plus, float *u_minus,
                              float dx) {
  int k;
  float f1[6];
  float x1, x2, x3;

  DiviededDifferencing_1j(f0, f1, dx);
  // printf("%f,%f,%f,%f,%f,%f,%f\n",
  // f0[0],f0[1],f0[2],f0[3],f0[4],f0[5],f0[6]);

  // Compute u^-
  k = 2;
  *u_minus = f1[k];

  // printf("k- = %d, u_minus = %f\n", k, *u_minus);

  // Compute u^+
  k = 3;
  *u_plus = f1[k];
  // printf("k+ = %d, *u_plus = %f\n", k, *u_plus);
}

/*-----------------------------------------------------------------------------*/
void Get_Seven_Function_Values_X(float *coeff, int nx, int ny, int nz, int u,
                                 int v, int w, float *fx) {
  int j, l, uj;

  if (u >= 3 && u <= nx - 4) {
    for (j = 0; j < 7; j++) {
      l = u - 3 + j;
      fx[j] = coeff[(l * ny + v) * nz + w];
    }
    return;
  }

  if (u == 2) {
    for (j = 1; j < 7; j++) {
      l = j - 1;
      fx[j] = coeff[(l * ny + v) * nz + w];
    }
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  if (u == 1) {
    for (j = 2; j < 7; j++) {
      l = j - 2;
      fx[j] = coeff[(l * ny + v) * nz + w];
    }

    fx[1] = 2 * fx[2] - fx[3];
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  if (u == 0) {
    for (j = 3; j < 7; j++) {
      l = j - 3;
      fx[j] = coeff[(l * ny + v) * nz + w];
    }

    fx[2] = 2 * fx[3] - fx[4];
    fx[1] = 2 * fx[2] - fx[3];
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  /*
  if (u <= 2) {
     for (j = 3 - u; j < 7; j++) {
        l = u - 3 + j;
        fx[j] = coeff[(l*ny + v)*nz + w];
     }

     for (j = 0; j < 3 - u; j++) {
        uj = u - j;
        fx[2 - uj] = 2*fx[3 -  uj] - fx[4 - uj];
     }
     return;
  }
  */

  if (u == nx - 3) {
    for (j = 0; j < 6; j++) {
      l = u - 3 + j;
      fx[j] = coeff[(l * ny + v) * nz + w];
    }
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }

  if (u == nx - 2) {
    for (j = 0; j < 5; j++) {
      l = u - 3 + j;
      fx[j] = coeff[(l * ny + v) * nz + w];
    }
    fx[5] = 2 * fx[4] - fx[3];
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }

  if (u == nx - 1) {
    for (j = 0; j < 4; j++) {
      l = u - 3 + j;
      fx[j] = coeff[(l * ny + v) * nz + w];
    }
    fx[4] = 2 * fx[3] - fx[2];
    fx[5] = 2 * fx[4] - fx[3];
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }
}

/*-----------------------------------------------------------------------------*/
void Get_Seven_Function_Values_Y(float *coeff, int nx, int ny, int nz, int u,
                                 int v, int w, float *fx) {
  int j, l, uj;

  if (v >= 3 && v <= ny - 4) {
    for (j = 0; j < 7; j++) {
      l = v - 3 + j;
      fx[j] = coeff[(u * ny + l) * nz + w];
    }
    return;
  }

  if (v == 2) {
    for (j = 1; j < 7; j++) {
      l = j - 1;
      fx[j] = coeff[(u * ny + l) * nz + w];
    }
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  if (v == 1) {
    for (j = 2; j < 7; j++) {
      l = j - 2;
      fx[j] = coeff[(u * ny + l) * nz + w];
    }

    fx[1] = 2 * fx[2] - fx[3];
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  if (v == 0) {
    for (j = 3; j < 7; j++) {
      l = j - 3;
      fx[j] = coeff[(u * ny + l) * nz + w];
    }

    fx[2] = 2 * fx[3] - fx[4];
    fx[1] = 2 * fx[2] - fx[3];
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  /*
  if (v <= 2) {
     for (j = 3 - v; j < 7; j++) {
        l = v - 3 + j;
        fx[j] = coeff[(u*ny + l)*nz + w];
     }

     for (j = 0; j < 3 - v; j++) {
        uj = v - j;
        fx[2 - uj] = 2*fx[3 -  uj] - fx[4 - uj];
     }
     return;
  }
  */

  if (v == ny - 3) {
    for (j = 0; j < 6; j++) {
      l = v - 3 + j;
      fx[j] = coeff[(u * ny + l) * nz + w];
    }
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }

  if (v == ny - 2) {
    for (j = 0; j < 5; j++) {
      l = v - 3 + j;
      fx[j] = coeff[(u * ny + l) * nz + w];
    }
    fx[5] = 2 * fx[4] - fx[3];
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }

  if (v == ny - 1) {
    for (j = 0; j < 4; j++) {
      l = v - 3 + j;
      fx[j] = coeff[(u * ny + l) * nz + w];
    }
    fx[4] = 2 * fx[3] - fx[2];
    fx[5] = 2 * fx[4] - fx[3];
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }
}

/*-----------------------------------------------------------------------------*/
void Get_Seven_Function_Values_Z(float *coeff, int nx, int ny, int nz, int u,
                                 int v, int w, float *fx) {
  int j, l, uj;

  if (w >= 3 && w <= nz - 4) {
    for (j = 0; j < 7; j++) {
      l = w - 3 + j;
      fx[j] = coeff[(u * ny + v) * nz + l];
    }
    return;
  }

  if (w == 2) {
    for (j = 1; j < 7; j++) {
      l = j - 1;
      fx[j] = coeff[(u * ny + v) * nz + l];
    }
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  if (w == 1) {
    for (j = 2; j < 7; j++) {
      l = j - 2;
      fx[j] = coeff[(u * ny + v) * nz + l];
    }

    fx[1] = 2 * fx[2] - fx[3];
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  if (w == 0) {
    for (j = 3; j < 7; j++) {
      l = j - 3;
      fx[j] = coeff[(u * ny + v) * nz + l];
    }

    fx[2] = 2 * fx[3] - fx[4];
    fx[1] = 2 * fx[2] - fx[3];
    fx[0] = 2 * fx[1] - fx[2];
    return;
  }

  /*
  if (w <= 2) {
     for (j = 3 - w; j < 7; j++) {
        l = w - 3 + j;
        fx[j] = coeff[(u*ny + v)*nz + l];
     }

     for (j = 0; j < 3 - w; j++) {
        uj = w - j;
        fx[2 - uj] = 2*fx[3 -  uj] - fx[4 - uj];
     }
     return;
  }
  */

  if (w == nz - 3) {
    for (j = 0; j < 6; j++) {
      l = w - 3 + j;
      fx[j] = coeff[(u * ny + v) * nz + l];
    }
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }

  if (w == nz - 2) {
    for (j = 0; j < 5; j++) {
      l = w - 3 + j;
      fx[j] = coeff[(u * ny + v) * nz + l];
    }
    fx[5] = 2 * fx[4] - fx[3];
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }

  if (w == nz - 1) {
    for (j = 0; j < 4; j++) {
      l = w - 3 + j;
      fx[j] = coeff[(u * ny + v) * nz + l];
    }
    fx[4] = 2 * fx[3] - fx[2];
    fx[5] = 2 * fx[4] - fx[3];
    fx[6] = 2 * fx[5] - fx[4];
    return;
  }
}

/*-----------------------------------------------------------------------------*/
float Extreme_Positive(float a, float b) {
  float aa, bb;

  aa = a * a;
  bb = b * b;

  if (a <= b) {
    if (b < 0.0)
      return (bb);
    if (a > 0.0)
      return (aa);
    return (0.0);
  }

  if (aa > bb)
    return (aa);
  return (bb);
}

/*-----------------------------------------------------------------------------*/
float Extreme_Negative(float a, float b) {
  float aa, bb;

  aa = a * a;
  bb = b * b;

  if (a <= b) {
    if (aa > bb)
      return (aa);
    return (bb);
  }

  if (a < 0.0)
    return (aa);
  if (b > 0.0)
    return (bb);
  return (0.0);
}

/*-----------------------------------------------------------------------------*/
float Gradient_2(float *fun, float dx, float dy, float dz, int i, int j,
                 int k, int nx, int ny, int nz) {
  float fx, fy, fz;

  // f_ijk = fun[(i*ny + j)*nz + k];

  if (i > 0 && i < nx - 1)
    fx = (fun[((i + 1) * ny + j) * nz + k] -
          fun[((i - 1) * ny + j) * nz + k]) /
         (dx + dx);
  if (i == 0)
    fx = (fun[((1) * ny + j) * nz + k] - fun[j * nz + k]) / dx;
  if (i == nx - 1)
    fx = (fun[((nx - 1) * ny + j) * nz + k] -
          fun[((nx - 2) * ny + j) * nz + k]) /
         dx;

  if (j > 0 && j < ny - 1)
    fy = (fun[(i * ny + j + 1) * nz + k] - fun[(i * ny + j - 1) * nz + k]) /
         (dy + dy);
  if (j == 0)
    fy = (fun[(i * ny + 1) * nz + k] - fun[(i * ny) * nz + k]) / dy;
  if (j == ny - 1)
    fy = (fun[(i * ny + ny - 1) * nz + k] - fun[(i * ny + ny - 2) * nz + k]) /
         dy;

  if (k > 0 && k < nz - 1)
    fz = (fun[(i * ny + j) * nz + k + 1] - fun[(i * ny + j) * nz + k - 1]) /
         (dz + dz);
  if (k == 0)
    fz = (fun[(i * ny + j) * nz + 1] - fun[(i * ny + j) * nz]) / dz;
  if (k == nz - 1)
    fz = (fun[(i * ny + j) * nz + nz - 1] - fun[(i * ny + j) * nz + nz - 2]) /
         dz;

  return (fx * fx + fy * fy + fz * fz);
}

/*-----------------------------------------------------------------------------*/
void Divided_DifferenceOrder2PartialsAtGridPoint(float *c, float dx, float dy,
                                                 float dz, int nx, int ny,
                                                 int nz, int u, int v, int w,
                                                 float *partials)
//              float   *c,    /* the spline  coefficients */ float   dx /*
//              spacing in x direction                              */ float
//              dy     /* spacing in y direction */ float   dz     /* spacing
//              in z direction                              */ int     nx, /*
//              number of samples or coefficients in x direction    */ int ny,
//              /* number of samples or coefficients in y direction    */ int
//              nz,    /* number of samples or coefficients in z direction */
//              float   *partial /* partial derivatives */
{
  float c27[27], indx, indy, indz;
  int i;

  Take_27_Coefficients(c, nx, ny, nz, u, v, w, c27);

  indx = 1.0 / dx;
  indy = 1.0 / dy;
  indz = 1.0 / dz;

  partials[0] = c27[13];

  // fx, fy, fz
  partials[1] = (c27[22] - c27[4]) * indx * 0.5;
  partials[2] = (c27[16] - c27[10]) * indy * 0.5;
  partials[3] = (c27[14] - c27[12]) * indz * 0.5;

  // fxx, fxy, fxz
  partials[4] = (c27[22] + c27[4] - c27[13] - c27[13]) * indx * indx;
  partials[5] = (c27[25] + c27[1] - c27[19] - c27[7]) * indx * indy * 0.25;
  partials[6] = (c27[23] + c27[3] - c27[21] - c27[5]) * indx * indz * 0.25;

  // fyy, fyz
  partials[7] = (c27[16] + c27[10] - c27[13] - c27[13]) * indy * indy;
  partials[8] = (c27[17] + c27[9] - c27[15] - c27[11]) * indy * indz * 0.25;

  // fzz
  partials[9] = (c27[12] + c27[14] - c27[13] - c27[13]) * indz * indz;
}

/*--------------------------------------------------------------------------*/
void ConvertToInterpolationCoefficients_Qu(
    float *c,       /* input samples --> output coefficients */
    int DataLength, /* number of samples or coefficients */
    float *z,       /* poles */
    int NbPoles,    /* number of poles */
    float Tolerance /* admissible relative error */
)

{ /* begin ConvertToInterpolationCoefficients */

  float Lambda = 1.0;
  int n, k;

  /* special case required by mirror boundaries */
  if (DataLength == 1) {
    return;
  }
  /* compute the overall gain */
  for (k = 0; k < NbPoles; k++) {
    Lambda = Lambda * (1.0f - z[k]) * (1.0f - 1.0f / z[k]);
  }
  /* apply the gain */
  for (n = 0; n < DataLength; n++) {
    c[n] *= Lambda;
  }
  /* loop over all poles */
  for (k = 0; k < NbPoles; k++) {
    /* causal initialization */
    c[0] = InitialCausalCoefficient(c, DataLength, z[k], Tolerance);
    /* causal recursion */
    for (n = 1; n < DataLength; n++) {
      c[n] += z[k] * c[n - 1];
      // printf("c^+ Qin = %f\n", c[n]);
    }
    /* anticausal initialization */
    c[DataLength - 1] = InitialAntiCausalCoefficient(c, DataLength, z[k]);
    /* anticausal recursion */
    for (n = DataLength - 2; 0 <= n; n--) {
      c[n] = z[k] * (c[n + 1] - c[n]);
    }
  }
} /* end ConvertToInterpolationCoefficients */

/*--------------------------------------------------------------------------*/
float InitialCausalCoefficient(float *c,       /* coefficients */
                               int DataLength, /* number of coefficients */
                               float z,        /* actual pole */
                               float Tolerance /* admissible relative error */
)

{ /* begin InitialCausalCoefficient */

  float Sum, zn, z2n, iz;
  int n, Horizon;

  /* this initialization corresponds to mirror boundaries */
  Horizon = DataLength;
  if (Tolerance > 0.0) {
    Horizon = (int)ceil(log(Tolerance) / log(fabs(z)));
  }
  if (Horizon < DataLength) {
    /* accelerated loop */
    zn = z;
    Sum = c[0];
    for (n = 1; n < Horizon; n++) {
      Sum += zn * c[n];
      zn *= z;
    }
    return (Sum);
  } else {
    /* full loop */
    zn = z;
    iz = 1.0f / z;
    z2n = (float)pow(z, (DataLength - 1));
    Sum = c[0] + z2n * c[DataLength - 1];
    z2n *= z2n * iz;
    for (n = 1; n <= DataLength - 2; n++) {
      Sum += (zn + z2n) * c[n];
      zn *= z;
      z2n *= iz;
    }
    return (Sum / (1.0f - zn * zn));
  }
} /* end InitialCausalCoefficient */

/*--------------------------------------------------------------------------*/
float InitialAntiCausalCoefficient(
    float *c,       /* coefficients */
    int DataLength, /* number of samples or coefficients */
    float z         /* actual pole */
)

{ /* begin InitialAntiCausalCoefficient */

  /* this initialization corresponds to mirror boundaries */
  return ((z / (z * z - 1.0f)) * (z * c[DataLength - 2] + c[DataLength - 1]));
} /* end InitialAntiCausalCoefficient */
