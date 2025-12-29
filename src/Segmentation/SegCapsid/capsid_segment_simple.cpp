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

static int XDIM, YDIM, ZDIM;
static float *dataset;
static VECTOR *FiveFold;

VECTOR Rotate(float, float, float, float, float, float, int, int, int);
void SetSeedIndexSimple(int, int, int);

void SimpleCapsidSegment(int xd, int yd, int zd, float *data, float tlow,
                         int sx, int sy, int sz,
                         /*CPNT *critical,*/ VECTOR *five_fold) {
  int i, j, k;
  int x, y, z;
  int InitSeed_x, InitSeed_y, InitSeed_z;
  unsigned char index;
  //  float max_inten;
  // CPNT *critical_tmp;
  SVECTOR *stack;
  int stack_size;

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  FiveFold = five_fold;

  /*
  max_inten = 0;
  critical_tmp = critical;
  while (critical_tmp != NULL) {
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;
    if (dataset[IndexVect(i,j,k)] > max_inten) {
      InitSeed_x = i;
      InitSeed_y = j;
      InitSeed_z = k;
      max_inten = dataset[IndexVect(i,j,k)];
    }
    critical_tmp = critical_tmp->next;
  }*/

  InitSeed_x = sx;
  InitSeed_y = sy;
  InitSeed_z = sz;
  printf("Capsid seed point: %d %d %d\n", InitSeed_x, InitSeed_y, InitSeed_z);

  stack = (SVECTOR *)malloc(sizeof(SVECTOR) * XDIM * YDIM * ZDIM / 3);
  stack[0].x = InitSeed_x;
  stack[0].y = InitSeed_y;
  stack[0].z = InitSeed_z;
  stack_size = 1;

  while (stack_size > 0) {
    if (stack_size >= XDIM * YDIM * ZDIM / 5) {
      printf("too small tlow...\n");
      exit(0);
    }
    stack_size--;
    x = stack[stack_size].x;
    y = stack[stack_size].y;
    z = stack[stack_size].z;
    SetSeedIndexSimple(x, y, z);

    for (k = max2(z - 1, 0); k <= min2(z + 1, ZDIM - 1); k++)
      for (j = max2(y - 1, 0); j <= min2(y + 1, YDIM - 1); j++)
        for (i = max2(x - 1, 0); i <= min2(x + 1, XDIM - 1); i++) {
          if (dataset[IndexVect(i, j, k)] > tlow) {
            stack[stack_size].x = i;
            stack[stack_size].y = j;
            stack[stack_size].z = k;
            stack_size++;
          }
        }
  }

  for (z = 0; z < ZDIM; z++)
    for (y = 0; y < YDIM; y++)
      for (x = 0; x < XDIM; x++) {
        if (dataset[IndexVect(x, y, z)] > tlow) {
          index = 0;
          for (k = max2(z - 1, 0); k <= min2(z + 1, ZDIM - 1); k++)
            for (j = max2(y - 1, 0); j <= min2(y + 1, YDIM - 1); j++)
              for (i = max2(x - 1, 0); i <= min2(x + 1, XDIM - 1); i++) {
                if (dataset[IndexVect(i, j, k)] < 0)
                  index = 1;
              }
          if (index)
            dataset[IndexVect(x, y, z)] = -dataset[IndexVect(x, y, z)];
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        if (dataset[IndexVect(i, j, k)] > 0)
          dataset[IndexVect(i, j, k)] = 0;
        else
          dataset[IndexVect(i, j, k)] = -dataset[IndexVect(i, j, k)];
      }

  /*
  num = 0;
  critical_tmp = critical;
  critical_prv = critical;
  while (critical_tmp != NULL) {
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;
    if (dataset[IndexVect(i,j,k)] == 0) {
      num++;
      if (critical_tmp == critical) {
        critical = critical->next;
        critical_prv = critical;
      }
      else
        critical_prv->next = critical_tmp->next;
    }
    else {
      critical_prv = critical_tmp;
    }
    critical_tmp = critical_tmp->next;
  }
  printf("number of seeds removed: %d \n",num);
  */

  free(stack);
}

VECTOR Rotate(float sx, float sy, float sz, float theta, float phi,
              float angle, int xd, int yd, int zd) {
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

  sx = sx - xd / 2;
  sy = sy - yd / 2;
  sz = sz - zd / 2;

  x = a[0][0] * sx + a[0][1] * sy + a[0][2] * sz;
  y = a[1][0] * sx + a[1][1] * sy + a[1][2] * sz;
  z = a[2][0] * sx + a[2][1] * sy + a[2][2] * sz;

  xx = (float)(cos(angle) * x - sin(angle) * y);
  yy = (float)(sin(angle) * x + cos(angle) * y);
  zz = z;

  tmp.x = b[0][0] * xx + b[0][1] * yy + b[0][2] * zz + xd / 2;
  tmp.y = b[1][0] * xx + b[1][1] * yy + b[1][2] * zz + yd / 2;
  tmp.z = b[2][0] * xx + b[2][1] * yy + b[2][2] * zz + zd / 2;

  return (tmp);
}

void SetSeedIndexSimple(int ax, int ay, int az) {
  int i, j, k;
  int m, n;
  float cx, cy, cz;
  float nx, ny, nz;
  float sx, sy, sz;
  float theta, phi;
  VECTOR sv;

  cx = FiveFold[0].x - XDIM / 2;
  cy = FiveFold[0].y - YDIM / 2;
  cz = FiveFold[0].z - ZDIM / 2;
  i = ax;
  j = ay;
  k = az;
  if (i < 0 || i >= XDIM || j < 0 || j >= YDIM || k < 0 || k >= ZDIM) {
    printf("check the correctness of global symmetry ...\n");
    exit(0);
  }

  if (dataset[IndexVect(i, j, k)] > 0)
    dataset[IndexVect(i, j, k)] = -dataset[IndexVect(i, j, k)];
  theta = (float)atan2(cy, cx);
  phi = (float)atan2(cz, sqrt(cx * cx + cy * cy));
  for (n = 1; n < 5; n++) {
    sv = Rotate((float)ax, (float)ay, (float)az, theta, phi,
                n * 2.0f * PIE / 5.0f, XDIM, YDIM, ZDIM);
    i = (int)(sv.x + 0.5);
    j = (int)(sv.y + 0.5);
    k = (int)(sv.z + 0.5);
    if (i < 0 || i >= XDIM || j < 0 || j >= YDIM || k < 0 || k >= ZDIM) {
      printf("check the correctness of global symmetry ...\n");
      exit(0);
    }

    if (dataset[IndexVect(i, j, k)] > 0)
      dataset[IndexVect(i, j, k)] = -dataset[IndexVect(i, j, k)];
  }

  for (m = 1; m < 11; m++) {
    nx = FiveFold[m].x - XDIM / 2;
    ny = FiveFold[m].y - YDIM / 2;
    nz = FiveFold[m].z - ZDIM / 2;
    sx = nz * cy - ny * cz;
    sy = nx * cz - nz * cx;
    sz = ny * cx - nx * cy;
    theta = (float)atan2(sy, sx);
    phi = (float)atan2(sz, sqrt(sx * sx + sy * sy));
    if (m < 6)
      sv = Rotate((float)ax, (float)ay, (float)az, theta, phi, ANGL1, XDIM,
                  YDIM, ZDIM);
    else
      sv = Rotate((float)ax, (float)ay, (float)az, theta, phi, ANGL2, XDIM,
                  YDIM, ZDIM);
    sx = sv.x;
    sy = sv.y;
    sz = sv.z;

    theta = (float)atan2(ny, nx);
    phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
    for (n = 0; n < 5; n++) {
      sv = Rotate(sx, sy, sz, theta, phi, n * 2.0f * PIE / 5.0f + PIE / 5.0f,
                  XDIM, YDIM, ZDIM);
      i = (int)(sv.x + 0.5);
      j = (int)(sv.y + 0.5);
      k = (int)(sv.z + 0.5);
      if (i < 0 || i >= XDIM || j < 0 || j >= YDIM || k < 0 || k >= ZDIM) {
        printf("check the correctness of global symmetry ...\n");
        exit(0);
      }

      if (dataset[IndexVect(i, j, k)] > 0)
        dataset[IndexVect(i, j, k)] = -dataset[IndexVect(i, j, k)];
    }
  }

  nx = FiveFold[1].x - XDIM / 2;
  ny = FiveFold[1].y - YDIM / 2;
  nz = FiveFold[1].z - ZDIM / 2;
  sx = nz * cy - ny * cz;
  sy = nx * cz - nz * cx;
  sz = ny * cx - nx * cy;
  theta = (float)atan2(sy, sx);
  phi = (float)atan2(sz, sqrt(sx * sx + sy * sy));
  sv = Rotate((float)ax, (float)ay, (float)az, theta, phi, PIE, XDIM, YDIM,
              ZDIM);
  sx = sv.x;
  sy = sv.y;
  sz = sv.z;
  i = (int)(sx + 0.5);
  j = (int)(sy + 0.5);
  k = (int)(sz + 0.5);
  if (i < 0 || i >= XDIM || j < 0 || j >= YDIM || k < 0 || k >= ZDIM) {
    printf("check the correctness of global symmetry ...\n");
    exit(0);
  }

  if (dataset[IndexVect(i, j, k)] > 0)
    dataset[IndexVect(i, j, k)] = -dataset[IndexVect(i, j, k)];
  nx = FiveFold[11].x - XDIM / 2;
  ny = FiveFold[11].y - YDIM / 2;
  nz = FiveFold[11].z - ZDIM / 2;
  theta = (float)atan2(ny, nx);
  phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
  for (n = 1; n < 5; n++) {
    sv = Rotate(sx, sy, sz, theta, phi, n * 2.0f * PIE / 5.0f, XDIM, YDIM,
                ZDIM);
    i = (int)(sv.x + 0.5);
    j = (int)(sv.y + 0.5);
    k = (int)(sv.z + 0.5);
    if (i < 0 || i >= XDIM || j < 0 || j >= YDIM || k < 0 || k >= ZDIM) {
      printf("check the correctness of global symmetry ...\n");
      exit(0);
    }

    if (dataset[IndexVect(i, j, k)] > 0)
      dataset[IndexVect(i, j, k)] = -dataset[IndexVect(i, j, k)];
  }
}

}; // namespace SegCapsid
