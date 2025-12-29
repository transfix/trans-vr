/******************************************************************************
                                Copyright

This code is developed within the Computational Visualization Center at The
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser
General Public License (LGPL)
(http://www.ices.utexas.edu/cvc/software/license.html) and terms that you have
agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of
the code that results in any published work, including scientific papers,
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular
Imaging Journal of Structural Biology, Volume 144, Issues 1-2, October 2003,
Pages 132-143.

If you desire to use this code for a profit venture, or if you do not wish to
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The
University of Texas at Austin for a different license.
******************************************************************************/

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define max2(x, y) ((x > y) ? (x) : (y))
#define min2(x, y) ((x < y) ? (x) : (y))
#define PIE 3.1415926f
#define ANGL1 1.107149f
#define ANGL2 2.034444f

#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))

namespace SegCapsid {

typedef struct {
  float x;
  float y;
  float z;
} VECTOR;

typedef struct {
  float sx;
  float sy;
  float sz;
  float ex;
  float ey;
  float ez;
} DB_VECTOR;

typedef struct {
  unsigned short x;
  unsigned short y;
  unsigned short z;
} SVECTOR;

typedef struct CriticalPoint CPNT;
struct CriticalPoint {
  unsigned short x;
  unsigned short y;
  unsigned short z;
  CPNT *next;
};

typedef struct {
  float trans;
  float rotat;
  float angle;
} CVM;

static int XDIM, YDIM, ZDIM;
static float *dataset;
static VECTOR *FiveFold;
static DB_VECTOR *LocalFold;

VECTOR Rotate(float, float, float, float, float, float, int, int, int);
float GetScore(float, float, float, int, int);
VECTOR CoVarRotate(float, float, float, float, float, float);
VECTOR FindCopy(float, float, float, int, int, CVM *, int);

void CapsidSegmentScore(int xd, int yd, int zd, float *data, float *result,
                        VECTOR *fvfold, DB_VECTOR *lcfold, CVM *coVar,
                        int axisnum, int foldnum, float small_radius,
                        float large_radius) {
  int i, j, k;
  float x, y, z;
  float cx, cy, cz;
  float xx, yy, zz;
  int m, n, min_index = 0, start_index;
  float min_dist, dist, tmp;
  float a1, b1, c1;
  VECTOR sv;
  float theta, phi;
  float sx, sy, sz;
  float ex, ey, ez;
  float nx, ny, nz;
  float *score;
  float radius, max_score;

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  FiveFold = fvfold;
  LocalFold = lcfold;
  score = (float *)malloc(sizeof(float) * axisnum * 6);

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++)
        result[IndexVect(i, j, k)] = 0;

  max_score = -999.0f;
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        radius = (float)sqrt(double((i - XDIM / 2) * (i - XDIM / 2) +
                                    (j - YDIM / 2) * (j - YDIM / 2) +
                                    (k - ZDIM / 2) * (k - ZDIM / 2)));
        if (dataset[IndexVect(i, j, k)] > 0 && radius < large_radius + 3 &&
            radius > small_radius - 3) {
          min_dist = 99999.0f;
          for (m = 0; m < axisnum * 60; m++) {
            if (LocalFold[m].sx != -9999) {
              cx = 0.5f * (LocalFold[m].sx + LocalFold[m].ex);
              cy = 0.5f * (LocalFold[m].sy + LocalFold[m].ey);
              cz = 0.5f * (LocalFold[m].sz + LocalFold[m].ez);
              if ((i - XDIM / 2) * (cx - XDIM / 2) +
                      (j - YDIM / 2) * (cy - YDIM / 2) +
                      (k - ZDIM / 2) * (cz - ZDIM / 2) >
                  0) {
                a1 = LocalFold[m].sx - LocalFold[m].ex;
                b1 = LocalFold[m].sy - LocalFold[m].ey;
                c1 = LocalFold[m].sz - LocalFold[m].ez;
                tmp = (float)sqrt(a1 * a1 + b1 * b1 + c1 * c1);
                x = (float)i - LocalFold[m].ex;
                y = (float)j - LocalFold[m].ey;
                z = (float)k - LocalFold[m].ez;
                xx = b1 * z - c1 * y;
                yy = x * c1 - a1 * z;
                zz = a1 * y - x * b1;
                dist = (float)(sqrt(xx * xx + yy * yy + zz * zz) / tmp);
                if (dist < min_dist) {
                  min_dist = dist;
                  min_index = m;
                }
              }
            }
          }

          start_index = (min_index / 60) * 60;
          cx = FiveFold[0].x - XDIM / 2;
          cy = FiveFold[0].y - YDIM / 2;
          cz = FiveFold[0].z - ZDIM / 2;
          m = (min_index - start_index) / 5;
          n = min_index - start_index - 5 * m;
          if (m == 0) {
            if (n == 0) {
              sv.x = (float)i;
              sv.y = (float)j;
              sv.z = (float)k;
            } else {
              theta = (float)atan2(cy, cx);
              phi = (float)atan2(cz, sqrt(cx * cx + cy * cy));
              sv = Rotate((float)i, (float)j, (float)k, theta, phi,
                          -n * 2.0f * PIE / 5.0f, XDIM, YDIM, ZDIM);
            }
          } else if (m < 11) {
            nx = FiveFold[m].x - XDIM / 2;
            ny = FiveFold[m].y - YDIM / 2;
            nz = FiveFold[m].z - ZDIM / 2;
            theta = (float)atan2(ny, nx);
            phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
            sv =
                Rotate((float)i, (float)j, (float)k, theta, phi,
                       -n * 2.0f * PIE / 5.0f - PIE / 5.0f, XDIM, YDIM, ZDIM);
            sx = sv.x;
            sy = sv.y;
            sz = sv.z;

            ex = ny * cz - nz * cy;
            ey = nz * cx - nx * cz;
            ez = nx * cy - ny * cx;
            theta = (float)atan2(ey, ex);
            phi = (float)atan2(ez, sqrt(ex * ex + ey * ey));
            if (m < 6)
              sv = Rotate(sx, sy, sz, theta, phi, ANGL1, XDIM, YDIM, ZDIM);
            else
              sv = Rotate(sx, sy, sz, theta, phi, ANGL2, XDIM, YDIM, ZDIM);
          } else if (m == 11) {
            if (n == 0) {
              sx = (float)i;
              sy = (float)j;
              sz = (float)k;
            } else {
              nx = FiveFold[11].x - XDIM / 2;
              ny = FiveFold[11].y - YDIM / 2;
              nz = FiveFold[11].z - ZDIM / 2;
              theta = (float)atan2(ny, nx);
              phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
              sv = Rotate((float)i, (float)j, (float)k, theta, phi,
                          -n * 2.0f * PIE / 5.0f, XDIM, YDIM, ZDIM);
              sx = sv.x;
              sy = sv.y;
              sz = sv.z;
            }
            nx = FiveFold[1].x - XDIM / 2;
            ny = FiveFold[1].y - YDIM / 2;
            nz = FiveFold[1].z - ZDIM / 2;
            ex = ny * cz - nz * cy;
            ey = nz * cx - nx * cz;
            ez = nx * cy - ny * cx;
            theta = (float)atan2(ey, ex);
            phi = (float)atan2(ez, sqrt(ex * ex + ey * ey));
            sv = Rotate(sx, sy, sz, theta, phi, PIE, XDIM, YDIM, ZDIM);
          }
          xx = sv.x;
          yy = sv.y;
          zz = sv.z;

          m = (min_index / 60);
          for (n = 0; n < axisnum; n++) {
            if (n != m) {
              sv = FindCopy(xx, yy, zz, m, n, coVar, axisnum);
              x = sv.x;
              y = sv.y;
              z = sv.z;
            } else {
              x = xx;
              y = yy;
              z = zz;
            }
            score[n] = GetScore(x, y, z, n * 60, foldnum);
          }

          for (n = 0; n < axisnum; n++) {
            tmp = 999999.0;
            for (m = n; m < axisnum; m++) {
              if (score[m] < tmp) {
                tmp = score[m];
                min_index = m;
              }
            }
            tmp = score[n];
            score[n] = score[min_index];
            score[min_index] = tmp;
          }

          if (axisnum % 2 == 0)
            result[IndexVect(i, j, k)] =
                0.5f * (score[axisnum / 2 - 1] + score[axisnum / 2]);
          else
            result[IndexVect(i, j, k)] = score[axisnum / 2];

          if (result[IndexVect(i, j, k)] > max_score)
            max_score = result[IndexVect(i, j, k)];
        }
      }

  printf("max_score = %f \n", max_score);

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++)
        result[IndexVect(i, j, k)] =
            255.0f * result[IndexVect(i, j, k)] / max_score;
}

float GetScore(float ax, float ay, float az, int index, int foldnum) {
  float sx, sy, sz;
  float ex, ey, ez;
  float min_den, max_den;
  float a[3][3], b[3][3];
  float x00, x01, x10, x11, y0, y1;
  float dx, dy, dz;
  float tmp, theta, phi;
  float x, y, z;
  float xx, yy, zz;
  int i, j, k, m;

  sx = LocalFold[index].sx;
  sy = LocalFold[index].sy;
  sz = LocalFold[index].sz;
  ex = LocalFold[index].ex;
  ey = LocalFold[index].ey;
  ez = LocalFold[index].ez;
  i = (int)(ax + 0.5);
  j = (int)(ay + 0.5);
  k = (int)(az + 0.5);
  min_den = dataset[IndexVect(i, j, k)];
  max_den = dataset[IndexVect(i, j, k)];

  theta = (float)atan2(sy - ey, sx - ex);
  phi = (float)atan2(sz - ez,
                     sqrt((sx - ex) * (sx - ex) + (sy - ey) * (sy - ey)));

  a[0][0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  a[0][1] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  a[0][2] = (float)-sin(0.5 * PIE - phi);
  a[1][0] = (float)-sin(theta);
  a[1][1] = (float)cos(theta);
  a[1][2] = 0.f;
  a[2][0] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  a[2][1] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  a[2][2] = (float)cos(0.5 * PIE - phi);

  b[0][0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  b[0][1] = (float)-sin(theta);
  b[0][2] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  b[1][0] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  b[1][1] = (float)cos(theta);
  b[1][2] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  b[2][0] = (float)-sin(0.5 * PIE - phi);
  b[2][1] = 0.f;
  b[2][2] = (float)cos(0.5 * PIE - phi);

  x = ax - ex;
  y = ay - ey;
  z = az - ez;
  xx = a[0][0] * x + a[0][1] * y + a[0][2] * z;
  yy = a[1][0] * x + a[1][1] * y + a[1][2] * z;
  zz = a[2][0] * x + a[2][1] * y + a[2][2] * z;

  for (m = 1; m < foldnum; m++) {
    x = (float)(cos(2 * PIE * (float)(m) / (float)(foldnum)) * xx -
                sin(2 * PIE * (float)(m) / (float)(foldnum)) * yy);
    y = (float)(sin(2 * PIE * (float)(m) / (float)(foldnum)) * xx +
                cos(2 * PIE * (float)(m) / (float)(foldnum)) * yy);
    z = zz;

    dx = b[0][0] * x + b[0][1] * y + b[0][2] * z + ex;
    dy = b[1][0] * x + b[1][1] * y + b[1][2] * z + ey;
    dz = b[2][0] * x + b[2][1] * y + b[2][2] * z + ez;
    x00 = dataset[IndexVect((int)dx, (int)dy, (int)dz)] * ((int)dx + 1 - dx) +
          dataset[IndexVect((int)dx + 1, (int)dy, (int)dz)] * (dx - (int)dx);
    x01 = dataset[IndexVect((int)dx, (int)dy, (int)dz + 1)] *
              ((int)dx + 1 - dx) +
          dataset[IndexVect((int)dx + 1, (int)dy, (int)dz + 1)] *
              (dx - (int)dx);
    x10 = dataset[IndexVect((int)dx, (int)dy + 1, (int)dz)] *
              ((int)dx + 1 - dx) +
          dataset[IndexVect((int)dx + 1, (int)dy + 1, (int)dz)] *
              (dx - (int)dx);
    x11 = dataset[IndexVect((int)dx, (int)dy + 1, (int)dz + 1)] *
              ((int)dx + 1 - dx) +
          dataset[IndexVect((int)dx + 1, (int)dy + 1, (int)dz + 1)] *
              (dx - (int)dx);
    y0 = x00 * ((int)dy + 1 - dy) + x10 * (dy - (int)dy);
    y1 = x01 * ((int)dy + 1 - dy) + x11 * (dy - (int)dy);
    tmp = y0 * ((int)dz + 1 - dz) + y1 * (dz - (int)dz);

    if (tmp > max_den)
      max_den = tmp;
    if (tmp < min_den)
      min_den = tmp;
  }

  return (max_den - min_den);
}

VECTOR FindCopy(float ax, float ay, float az, int index1, int index2,
                CVM *coVar, int axisnum) {
  DB_VECTOR temp1, temp2;
  float alpha, tmp;
  float x, y, z;
  float rotat, trans;
  float theta, phi;
  float t_theta, t_phi;
  float fx, fy, fz;
  float gx, gy, gz;
  float px, py, pz;
  float dx, dy, dz;
  VECTOR sv;

  temp1.sx = LocalFold[index1 * 60].sx;
  temp1.sy = LocalFold[index1 * 60].sy;
  temp1.sz = LocalFold[index1 * 60].sz;
  temp1.ex = LocalFold[index1 * 60].ex;
  temp1.ey = LocalFold[index1 * 60].ey;
  temp1.ez = LocalFold[index1 * 60].ez;
  temp2.sx = LocalFold[index2 * 60].sx;
  temp2.sy = LocalFold[index2 * 60].sy;
  temp2.sz = LocalFold[index2 * 60].sz;
  temp2.ex = LocalFold[index2 * 60].ex;
  temp2.ey = LocalFold[index2 * 60].ey;
  temp2.ez = LocalFold[index2 * 60].ez;

  alpha = coVar[index1 * axisnum + index2].angle;
  rotat = coVar[index1 * axisnum + index2].rotat;
  trans = coVar[index1 * axisnum + index2].trans;

  gx = temp2.sx - temp2.ex;
  gy = temp2.sy - temp2.ey;
  gz = temp2.sz - temp2.ez;
  fx = temp1.sx - temp1.ex;
  fy = temp1.sy - temp1.ey;
  fz = temp1.sz - temp1.ez;
  px = fy * gz - fz * gy;
  py = fz * gx - fx * gz;
  pz = fx * gy - fy * gx;
  t_theta = (float)atan2(py, px);
  t_phi = (float)atan2(pz, sqrt(px * px + py * py));
  theta = (float)atan2(gy, gx);
  phi = (float)atan2(gz, sqrt(gx * gx + gy * gy));
  tmp = (float)sqrt(gx * gx + gy * gy + gz * gz);
  dx = gx / tmp;
  dy = gy / tmp;
  dz = gz / tmp;

  x = ax - temp1.ex;
  y = ay - temp1.ey;
  z = az - temp1.ez;

  sv = CoVarRotate(x, y, z, t_theta, t_phi, alpha);
  x = sv.x;
  y = sv.y;
  z = sv.z;
  sv = CoVarRotate(x, y, z, theta, phi, rotat);
  x = sv.x + temp2.ex + dx * trans;
  y = sv.y + temp2.ey + dy * trans;
  z = sv.z + temp2.ez + dz * trans;

  sv.x = x;
  sv.y = y;
  sv.z = z;

  return (sv);
}

VECTOR CoVarRotate(float sx, float sy, float sz, float theta, float phi,
                   float angle) {
  float x, y, z;
  float xx, yy, zz;
  float a[3][3], b[3][3];
  VECTOR tmp;

  a[0][0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  a[0][1] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  a[0][2] = (float)-sin(0.5 * PIE - phi);
  a[1][0] = (float)-sin(theta);
  a[1][1] = (float)cos(theta);
  a[1][2] = 0.f;
  a[2][0] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  a[2][1] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  a[2][2] = (float)cos(0.5 * PIE - phi);

  b[0][0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  b[0][1] = (float)-sin(theta);
  b[0][2] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  b[1][0] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  b[1][1] = (float)cos(theta);
  b[1][2] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  b[2][0] = (float)-sin(0.5 * PIE - phi);
  b[2][1] = 0.f;
  b[2][2] = (float)cos(0.5 * PIE - phi);

  x = a[0][0] * sx + a[0][1] * sy + a[0][2] * sz;
  y = a[1][0] * sx + a[1][1] * sy + a[1][2] * sz;
  z = a[2][0] * sx + a[2][1] * sy + a[2][2] * sz;

  xx = (float)(cos(angle) * x - sin(angle) * y);
  yy = (float)(sin(angle) * x + cos(angle) * y);
  zz = z;

  tmp.x = b[0][0] * xx + b[0][1] * yy + b[0][2] * zz;
  tmp.y = b[1][0] * xx + b[1][1] * yy + b[1][2] * zz;
  tmp.z = b[2][0] * xx + b[2][1] * yy + b[2][2] * zz;

  return (tmp);
}

}; // namespace SegCapsid
