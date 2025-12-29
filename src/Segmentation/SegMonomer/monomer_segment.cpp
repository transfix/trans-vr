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

#define max(x, y) ((x > y) ? (x) : (y))
#define min(x, y) ((x < y) ? (x) : (y))
#define PIE 3.1415926f

#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))
#define MAX_TIME 9999999.0f
#define ALPHA 0.1f

namespace SegMonomer {

typedef struct {
  unsigned short *x;
  unsigned short *y;
  unsigned short *z;
  float *t;
  int size;
} MinHeapS;

typedef struct {
  float x;
  float y;
  float z;
} VECTOR;

typedef struct CriticalPoint CPNT;
struct CriticalPoint {
  unsigned short x;
  unsigned short y;
  unsigned short z;
  CPNT *next;
};

typedef struct {
  unsigned short x;
  unsigned short y;
  unsigned short z;
} INTVEC;

typedef struct {
  float sx;
  float sy;
  float sz;
  float ex;
  float ey;
  float ez;
} DB_VECTOR;

static int min_x, min_y, min_z;
static float min_t;
static MinHeapS *min_heap;
static MinHeapS *T_min_heap;
static unsigned char T_index;
static unsigned char *seed_index;
static float t_low, t_high;
static int XDIM, YDIM, ZDIM;
static float *dataset;
static float *tdata;
static float *T_tdata;
static float *ImgGrad;
static CPNT *critical_start;
static int SeedNum;
int PhaseNum;
static int start_ptr, end_ptr;
static INTVEC *IndexArray;
static DB_VECTOR *symmetry;
static int fold_num;

void GetMinimum(void);
void InsertHeap(int, int, int);
void GetTime(float *, float *);
void T_GetMinimum(void);
void T_InsertHeap(int, int, int);
void T_GetTime(void);
unsigned char SearchIndex(int *, int *, int *);
void SetSeedIndex(int, int, int, float *, float *, unsigned char);
void SetSeedIndexTime(int, int, int, float *, float *, unsigned char, float);
VECTOR MonomerRotate(float, float, float, float, float, float);
void InitialTime(int, int, int);

void get_average(int xd, int yd, int zd, float *data, float *orig_tmp,
                 float *span_tmp, DB_VECTOR *symmetry_axis, int foldnum) {
  int i, j, k;
  float *tmpmap, avg;
  int d;
  float theta, phi;
  float sx, sy, sz;
  float dx, dy, dz;
  VECTOR sv;
  float x00, x01, x10, x11, y0, y1;

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;

  tmpmap = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

  dx = symmetry_axis->sx - symmetry_axis->ex;
  dy = symmetry_axis->sy - symmetry_axis->ey;
  dz = symmetry_axis->sz - symmetry_axis->ez;
  theta = (float)atan2(dy, dx);
  phi = (float)atan2(dz, sqrt(dx * dx + dy * dy));

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        if (data[IndexVect(i, j, k)] == 0)
          tmpmap[IndexVect(i, j, k)] = 0;
        else {
          sx = orig_tmp[0] + i * span_tmp[0] - symmetry_axis->ex;
          sy = orig_tmp[1] + j * span_tmp[1] - symmetry_axis->ey;
          sz = orig_tmp[2] + k * span_tmp[2] - symmetry_axis->ez;
          //	  sx = (float)i-symmetry_axis->ex;
          //	  sy = (float)j-symmetry_axis->ey;
          //	  sz = (float)k-symmetry_axis->ez;

          avg = data[IndexVect(i, j, k)];
          for (d = 1; d < foldnum; d++) {
            sv = MonomerRotate((float)sx, (float)sy, (float)sz, theta, phi,
                               2.0f * d * PIE / (float)(foldnum));
            //	    dx = sv.x+symmetry_axis->ex;
            //	    dy = sv.y+symmetry_axis->ey;
            //	    dz = sv.z+symmetry_axis->ez;

            dx = (sv.x + symmetry_axis->ex - orig_tmp[0]) / span_tmp[0];
            dy = (sv.y + symmetry_axis->ey - orig_tmp[1]) / span_tmp[1];
            dz = (sv.z + symmetry_axis->ez - orig_tmp[2]) / span_tmp[2];

            x00 = data[IndexVect((int)dx, (int)dy, (int)dz)] *
                      ((int)dx + 1 - dx) +
                  data[IndexVect((int)dx + 1, (int)dy, (int)dz)] *
                      (dx - (int)dx);
            x01 = data[IndexVect((int)dx, (int)dy, (int)dz + 1)] *
                      ((int)dx + 1 - dx) +
                  data[IndexVect((int)dx + 1, (int)dy, (int)dz + 1)] *
                      (dx - (int)dx);
            x10 = data[IndexVect((int)dx, (int)dy + 1, (int)dz)] *
                      ((int)dx + 1 - dx) +
                  data[IndexVect((int)dx + 1, (int)dy + 1, (int)dz)] *
                      (dx - (int)dx);
            x11 = data[IndexVect((int)dx, (int)dy + 1, (int)dz + 1)] *
                      ((int)dx + 1 - dx) +
                  data[IndexVect((int)dx + 1, (int)dy + 1, (int)dz + 1)] *
                      (dx - (int)dx);
            y0 = x00 * ((int)dy + 1 - dy) + x10 * (dy - (int)dy);
            y1 = x01 * ((int)dy + 1 - dy) + x11 * (dy - (int)dy);
            avg += y0 * ((int)dz + 1 - dz) + y1 * (dz - (int)dz);
          }
          tmpmap[IndexVect(i, j, k)] = avg / (float)(foldnum);
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++)
        data[IndexVect(i, j, k)] = tmpmap[IndexVect(i, j, k)];

  free(tmpmap);
}

void MonomerSegment(int xd, int yd, int zd, float *orig_tmp, float *span_tmp,
                    float *data, float *edge_mag, CPNT *critical,
                    unsigned char *result, DB_VECTOR *symmetry_axis,
                    int foldnum) {
  int i, j, k;
  float MaxSeed;
  int MaxSeed_x = 0, MaxSeed_y = 0, MaxSeed_z = 0;
  unsigned char index;
  CPNT *critical_tmp;
  float tlow;

  tlow = 19.0f;
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  ImgGrad = edge_mag;
  critical_start = critical;
  seed_index = result;
  symmetry = symmetry_axis;
  fold_num = foldnum;

  min_heap = (MinHeapS *)malloc(sizeof(MinHeapS));
  min_heap->x =
      (unsigned short *)malloc(sizeof(unsigned short) * XDIM * YDIM * ZDIM);
  min_heap->y =
      (unsigned short *)malloc(sizeof(unsigned short) * XDIM * YDIM * ZDIM);
  min_heap->z =
      (unsigned short *)malloc(sizeof(unsigned short) * XDIM * YDIM * ZDIM);
  min_heap->t = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  tdata = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  T_min_heap = (MinHeapS *)malloc(sizeof(MinHeapS));
  T_min_heap->x =
      (unsigned short *)malloc(sizeof(unsigned short) * XDIM * YDIM * ZDIM);
  T_min_heap->y =
      (unsigned short *)malloc(sizeof(unsigned short) * XDIM * YDIM * ZDIM);
  T_min_heap->z =
      (unsigned short *)malloc(sizeof(unsigned short) * XDIM * YDIM * ZDIM);
  T_min_heap->t = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  T_tdata = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        seed_index[IndexVect(i, j, k)] = 255;
        tdata[IndexVect(i, j, k)] = MAX_TIME;
        T_tdata[IndexVect(i, j, k)] = MAX_TIME;
      }

  MaxSeed = -999;
  SeedNum = 0;
  critical_tmp = critical_start;
  while (critical_tmp != NULL) {
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;
    seed_index[IndexVect(i, j, k)] = 254;

    if (dataset[IndexVect(i, j, k)] > MaxSeed) {
      MaxSeed = dataset[IndexVect(i, j, k)];
      MaxSeed_x = i;
      MaxSeed_y = j;
      MaxSeed_z = k;
    }
    SeedNum++;
    critical_tmp = critical_tmp->next;
  }

  PhaseNum = 1;

  min_heap->size = 0;
  tdata[IndexVect(MaxSeed_x, MaxSeed_y, MaxSeed_z)] = 0.0;
  InsertHeap(MaxSeed_x, MaxSeed_y, MaxSeed_z);
  SetSeedIndex(MaxSeed_x, MaxSeed_y, MaxSeed_z, orig_tmp, span_tmp, 1);

  IndexArray = (INTVEC *)malloc(sizeof(INTVEC) * XDIM * YDIM * ZDIM);

  t_low = 255.0f;
  while (SeedNum > 0) {

    index = SearchIndex(&i, &j, &k);

    if (index == 255) {

      t_low = t_low - 1.0f;
      if (t_low < tlow)
        break;
      else
        continue;
    }

    t_high = t_low;
    t_low = 255.0f;

    tdata[IndexVect(i, j, k)] = 0.0;
    InsertHeap(i, j, k);
    SetSeedIndex(i, j, k, orig_tmp, span_tmp, index);
    while (1) {
      GetMinimum();
      if (min_t >= MAX_TIME)
        break;
      GetTime(orig_tmp, span_tmp);
    }
  }

  min_heap->size = 0;
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        tdata[IndexVect(i, j, k)] = MAX_TIME;
        if (seed_index[IndexVect(i, j, k)] < 254) {
          min_heap->x[min_heap->size] = i;
          min_heap->y[min_heap->size] = j;
          min_heap->z[min_heap->size] = k;
          min_heap->t[min_heap->size] = 0.0;
          min_heap->size++;
          tdata[IndexVect(i, j, k)] = 0.0;
        }
      }

  /* Fast Marching Method */
  PhaseNum = 2;
  t_high = tlow;
  while (1) {

    GetMinimum();

    if (min_t >= MAX_TIME)
      break;
    GetTime(orig_tmp, span_tmp);
  }

  free(min_heap->x);
  free(min_heap->y);
  free(min_heap->z);
  free(min_heap->t);
  free(min_heap);
  free(tdata);
  free(T_min_heap->x);
  free(T_min_heap->y);
  free(T_min_heap->z);
  free(T_min_heap->t);
  free(T_min_heap);
  free(T_tdata);
  free(IndexArray);
}

void InitialTime(int xx, int yy, int zz) {
  int i, j, k;
  int x, y, z;

  T_tdata[IndexVect(xx, yy, zz)] = MAX_TIME;
  start_ptr = 0;
  end_ptr = 1;
  IndexArray[0].x = xx;
  IndexArray[0].y = yy;
  IndexArray[0].z = zz;

  while (start_ptr < end_ptr) {
    x = IndexArray[start_ptr].x;
    y = IndexArray[start_ptr].y;
    z = IndexArray[start_ptr].z;
    start_ptr++;

    for (k = max(z - 1, 0); k <= min(z + 1, ZDIM - 1); k++)
      for (j = max(y - 1, 0); j <= min(y + 1, YDIM - 1); j++)
        for (i = max(x - 1, 0); i <= min(x + 1, XDIM - 1); i++) {
          if (T_tdata[IndexVect(i, j, k)] != MAX_TIME) {
            T_tdata[IndexVect(i, j, k)] = MAX_TIME;
            IndexArray[end_ptr].x = i;
            IndexArray[end_ptr].y = j;
            IndexArray[end_ptr].z = k;
            end_ptr++;
          }
        }
  }
}

unsigned char SearchIndex(int *m, int *n, int *l) {
  int i, j, k;
  int x0, y0, z0;
  int x1, y1, z1;
  float t;
  CPNT *critical_tmp;
  CPNT *critical_prv;

  critical_tmp = critical_start;
  while (critical_tmp != NULL) {
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;
    InitialTime(i, j, k);
    critical_tmp = critical_tmp->next;
  }

  T_min_heap->size = 0;
  critical_tmp = critical_start;
  critical_prv = critical_start;
  while (critical_tmp != NULL) {
    i = critical_tmp->x;
    j = critical_tmp->y;
    k = critical_tmp->z;

    if (seed_index[IndexVect(i, j, k)] == 254) {
      T_min_heap->x[T_min_heap->size] = i;
      T_min_heap->y[T_min_heap->size] = j;
      T_min_heap->z[T_min_heap->size] = k;
      T_min_heap->t[T_min_heap->size] = 0.0;
      T_tdata[IndexVect(i, j, k)] = 0.0;
      T_min_heap->size++;

      if (critical_tmp != critical_start)
        critical_prv = critical_prv->next;
    } else {
      if (critical_tmp == critical_start) {
        critical_start = critical_start->next;
        critical_prv = critical_start;
      } else
        critical_prv->next = critical_tmp->next;
    }

    critical_tmp = critical_tmp->next;
  }

  if (T_min_heap->size == 0) {
    printf("error !!!! \n");
    return (255);
  }

  T_index = 255;
  while (1) {

    T_GetMinimum();

    if (min_t >= MAX_TIME)
      return (255);

    T_GetTime();

    if (T_index < 254)
      break;
  }

  x0 = min_x;
  y0 = min_y;
  z0 = min_z;
  x1 = x0;
  y1 = y0;
  z1 = z0;
  t = T_tdata[IndexVect(x0, y0, z0)];

  while (t > 0) {

    for (k = max(0, z0 - 1); k <= min(ZDIM - 1, z0 + 1); k++)
      for (j = max(0, y0 - 1); j <= min(YDIM - 1, y0 + 1); j++)
        for (i = max(0, x0 - 1); i <= min(XDIM - 1, x0 + 1); i++) {
          if (T_tdata[IndexVect(i, j, k)] < t) {
            t = T_tdata[IndexVect(i, j, k)];
            x1 = i;
            y1 = j;
            z1 = k;
          }
        }
    if (x0 == x1 && y0 == y1 && z0 == z1) {
      printf("some seeds were ignored here ... \n");
      return (255);
    } else {
      x0 = x1;
      y0 = y1;
      z0 = z1;
      t = T_tdata[IndexVect(x0, y0, z0)];
    }
  }
  (*m) = x0;
  (*n) = y0;
  (*l) = z0;
  return (T_index);
}

void SetSeedIndex(int ax, int ay, int az, float *orig_tmp, float *span_tmp,
                  unsigned char index) {
  int i, j, k;
  int d;
  float theta, phi;
  float sx, sy, sz;
  float nx, ny, nz;
  VECTOR sv;

  if (seed_index[IndexVect(ax, ay, az)] == 254)
    SeedNum--;
  seed_index[IndexVect(ax, ay, az)] = index;

  nx = symmetry->sx - symmetry->ex;
  ny = symmetry->sy - symmetry->ey;
  nz = symmetry->sz - symmetry->ez;
  sx = orig_tmp[0] + span_tmp[0] * ax - symmetry->ex;
  sy = orig_tmp[1] + span_tmp[1] * ay - symmetry->ey;
  sz = orig_tmp[2] + span_tmp[2] * az - symmetry->ez;
  // sx = ax-symmetry->ex;
  // sy = ay-symmetry->ey;
  // sz = az-symmetry->ez;
  theta = (float)atan2(ny, nx);
  phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
  for (d = 1; d < fold_num; d++) {
    sv = MonomerRotate(sx, sy, sz, theta, phi,
                       2.0f * d * PIE / (float)(fold_num));
    // sv =
    // MonomerRotate((float)sx,(float)sy,(float)sz,theta,phi,2.0f*d*PIE/(float)(fold_num));
    i = (int)((sv.x + symmetry->ex - orig_tmp[0]) / span_tmp[0] + 0.5);
    j = (int)((sv.y + symmetry->ey - orig_tmp[1]) / span_tmp[1] + 0.5);
    k = (int)((sv.z + symmetry->ez - orig_tmp[2]) / span_tmp[2] + 0.5);
    //    i = (int)(sv.x+symmetry->ex+0.5);
    //   j = (int)(sv.y+symmetry->ey+0.5);
    //  k = (int)(sv.z+symmetry->ez+0.5);
    if (seed_index[IndexVect(i, j, k)] == 254)
      SeedNum--;
    seed_index[IndexVect(i, j, k)] = (index + d - 1) % fold_num + 1;
  }
}

void SetSeedIndexTime(int ax, int ay, int az, float *orig_tmp,
                      float *span_tmp, unsigned char index, float t) {
  int i, j, k;
  int d;
  float theta, phi;
  float sx, sy, sz;
  float nx, ny, nz;
  VECTOR sv;

  seed_index[IndexVect(ax, ay, az)] = index;

  nx = symmetry->sx - symmetry->ex;
  ny = symmetry->sy - symmetry->ey;
  nz = symmetry->sz - symmetry->ez;
  sx = orig_tmp[0] + span_tmp[0] * ax - symmetry->ex;
  sy = orig_tmp[1] + span_tmp[1] * ay - symmetry->ey;
  sz = orig_tmp[2] + span_tmp[2] * az - symmetry->ez;
  //  sx = ax-symmetry->ex;
  //  sy = ay-symmetry->ey;
  //  sz = az-symmetry->ez;
  theta = (float)atan2(ny, nx);
  phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
  for (d = 1; d < fold_num; d++) {
    sv = MonomerRotate(sx, sy, sz, theta, phi,
                       2.0f * d * PIE / (float)(fold_num));
    // sv =
    // MonomerRotate((float)sx,(float)sy,(float)sz,theta,phi,2.0f*d*PIE/(float)(fold_num));
    i = (int)((sv.x + symmetry->ex - orig_tmp[0]) / span_tmp[0] + 0.5);
    j = (int)((sv.y + symmetry->ey - orig_tmp[1]) / span_tmp[1] + 0.5);
    k = (int)((sv.z + symmetry->ez - orig_tmp[2]) / span_tmp[2] + 0.5);
    //    i = (int)(sv.x+symmetry->ex+0.5);
    //   j = (int)(sv.y+symmetry->ey+0.5);
    //  k = (int)(sv.z+symmetry->ez+0.5);
    seed_index[IndexVect(i, j, k)] = (index + d - 1) % fold_num + 1;
    if (tdata[IndexVect(i, j, k)] == MAX_TIME) {
      tdata[IndexVect(i, j, k)] = t;
      InsertHeap(i, j, k);
    }
  }
}

void GetMinimum(void) {
  int pointer, left, right;
  int x, y, z;
  float t;

  min_x = min_heap->x[0];
  min_y = min_heap->y[0];
  min_z = min_heap->z[0];
  min_t = min_heap->t[0];

  x = min_heap->x[min_heap->size - 1];
  y = min_heap->y[min_heap->size - 1];
  z = min_heap->z[min_heap->size - 1];
  t = min_heap->t[min_heap->size - 1];

  min_heap->size--;

  pointer = 1;
  while (pointer <= min_heap->size / 2) {
    left = 2 * pointer;
    right = 2 * pointer + 1;
    if ((min_heap->t[left - 1] <= min_heap->t[right - 1]) &&
        (min_heap->t[left - 1] < t)) {
      min_heap->x[pointer - 1] = min_heap->x[left - 1];
      min_heap->y[pointer - 1] = min_heap->y[left - 1];
      min_heap->z[pointer - 1] = min_heap->z[left - 1];
      min_heap->t[pointer - 1] = min_heap->t[left - 1];
      pointer = left;
    } else if ((min_heap->t[left - 1] > min_heap->t[right - 1]) &&
               (min_heap->t[right - 1] < t)) {
      min_heap->x[pointer - 1] = min_heap->x[right - 1];
      min_heap->y[pointer - 1] = min_heap->y[right - 1];
      min_heap->z[pointer - 1] = min_heap->z[right - 1];
      min_heap->t[pointer - 1] = min_heap->t[right - 1];
      pointer = right;
    } else
      break;
  }

  min_heap->x[pointer - 1] = x;
  min_heap->y[pointer - 1] = y;
  min_heap->z[pointer - 1] = z;
  min_heap->t[pointer - 1] = t;
}

void InsertHeap(int x, int y, int z) {
  int pointer, parent;
  float t;

  t = tdata[IndexVect(x, y, z)];
  min_heap->size++;
  pointer = min_heap->size;

  while (pointer > 1) {
    if (pointer % 2 == 0) {
      parent = pointer / 2;
      if (t < min_heap->t[parent - 1]) {
        min_heap->x[pointer - 1] = min_heap->x[parent - 1];
        min_heap->y[pointer - 1] = min_heap->y[parent - 1];
        min_heap->z[pointer - 1] = min_heap->z[parent - 1];
        min_heap->t[pointer - 1] = min_heap->t[parent - 1];
        pointer = parent;
      } else
        break;
    } else if (pointer % 2 == 1) {
      parent = (pointer - 1) / 2;
      if (t < min_heap->t[parent - 1]) {
        min_heap->x[pointer - 1] = min_heap->x[parent - 1];
        min_heap->y[pointer - 1] = min_heap->y[parent - 1];
        min_heap->z[pointer - 1] = min_heap->z[parent - 1];
        min_heap->t[pointer - 1] = min_heap->t[parent - 1];
        pointer = parent;
      } else
        break;
    }
  }

  min_heap->x[pointer - 1] = x;
  min_heap->y[pointer - 1] = y;
  min_heap->z[pointer - 1] = z;
  min_heap->t[pointer - 1] = t;
}

void GetTime(float *orig_tmp, float *span_tmp) {
  int tempt_x, tempt_y, tempt_z;
  int tx, ty, tz;
  float intens, value;
  char boundary;
  char index;

  index = seed_index[IndexVect(min_x, min_y, min_z)];

  tempt_x = max(min_x - 1, 0);
  tempt_y = min_y;
  tempt_z = min_z;

  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME &&
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] >= 254) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1.0f;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);

    if (PhaseNum == 1)
      SetSeedIndex(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index);
    else
      SetSeedIndexTime(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index,
                       tdata[IndexVect(tempt_x, tempt_y, tempt_z)]);
  }

  tempt_x = min(min_x + 1, XDIM - 1);
  tempt_y = min_y;
  tempt_z = min_z;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME &&
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] >= 254) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1.0f;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);

    if (PhaseNum == 1)
      SetSeedIndex(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index);
    else
      SetSeedIndexTime(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index,
                       tdata[IndexVect(tempt_x, tempt_y, tempt_z)]);
  }

  tempt_y = max(min_y - 1, 0);
  tempt_x = min_x;
  tempt_z = min_z;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME &&
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] >= 254) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1.0f;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);

    if (PhaseNum == 1)
      SetSeedIndex(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index);
    else
      SetSeedIndexTime(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index,
                       tdata[IndexVect(tempt_x, tempt_y, tempt_z)]);
  }

  tempt_y = min(min_y + 1, YDIM - 1);
  tempt_x = min_x;
  tempt_z = min_z;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME &&
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] >= 254) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1.0f;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);

    if (PhaseNum == 1)
      SetSeedIndex(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index);
    else
      SetSeedIndexTime(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index,
                       tdata[IndexVect(tempt_x, tempt_y, tempt_z)]);
  }

  tempt_z = max(min_z - 1, 0);
  tempt_x = min_x;
  tempt_y = min_y;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME &&
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] >= 254) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1.0f;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);

    if (PhaseNum == 1)
      SetSeedIndex(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index);
    else
      SetSeedIndexTime(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index,
                       tdata[IndexVect(tempt_x, tempt_y, tempt_z)]);
  }

  tempt_z = min(min_z + 1, ZDIM - 1);
  tempt_x = min_x;
  tempt_y = min_y;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME &&
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] >= 254) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1.0f;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);

    if (PhaseNum == 1)
      SetSeedIndex(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index);
    else
      SetSeedIndexTime(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp, index,
                       tdata[IndexVect(tempt_x, tempt_y, tempt_z)]);
  }
}

void T_GetMinimum(void) {
  int pointer, left, right;
  int x, y, z;
  float t;

  min_x = T_min_heap->x[0];
  min_y = T_min_heap->y[0];
  min_z = T_min_heap->z[0];
  min_t = T_min_heap->t[0];

  x = T_min_heap->x[T_min_heap->size - 1];
  y = T_min_heap->y[T_min_heap->size - 1];
  z = T_min_heap->z[T_min_heap->size - 1];
  t = T_min_heap->t[T_min_heap->size - 1];

  T_min_heap->size--;

  pointer = 1;
  while (pointer <= T_min_heap->size / 2) {
    left = 2 * pointer;
    right = 2 * pointer + 1;
    if ((T_min_heap->t[left - 1] <= T_min_heap->t[right - 1]) &&
        (T_min_heap->t[left - 1] < t)) {
      T_min_heap->x[pointer - 1] = T_min_heap->x[left - 1];
      T_min_heap->y[pointer - 1] = T_min_heap->y[left - 1];
      T_min_heap->z[pointer - 1] = T_min_heap->z[left - 1];
      T_min_heap->t[pointer - 1] = T_min_heap->t[left - 1];
      pointer = left;
    } else if ((T_min_heap->t[left - 1] > T_min_heap->t[right - 1]) &&
               (T_min_heap->t[right - 1] < t)) {
      T_min_heap->x[pointer - 1] = T_min_heap->x[right - 1];
      T_min_heap->y[pointer - 1] = T_min_heap->y[right - 1];
      T_min_heap->z[pointer - 1] = T_min_heap->z[right - 1];
      T_min_heap->t[pointer - 1] = T_min_heap->t[right - 1];
      pointer = right;
    } else
      break;
  }

  T_min_heap->x[pointer - 1] = x;
  T_min_heap->y[pointer - 1] = y;
  T_min_heap->z[pointer - 1] = z;
  T_min_heap->t[pointer - 1] = t;
}

void T_InsertHeap(int x, int y, int z) {
  int pointer, parent;
  float t;

  t = T_tdata[IndexVect(x, y, z)];
  T_min_heap->size++;
  pointer = T_min_heap->size;

  while (pointer > 1) {
    if (pointer % 2 == 0) {
      parent = pointer / 2;
      if (t < T_min_heap->t[parent - 1]) {
        T_min_heap->x[pointer - 1] = T_min_heap->x[parent - 1];
        T_min_heap->y[pointer - 1] = T_min_heap->y[parent - 1];
        T_min_heap->z[pointer - 1] = T_min_heap->z[parent - 1];
        T_min_heap->t[pointer - 1] = T_min_heap->t[parent - 1];
        pointer = parent;
      } else
        break;
    } else if (pointer % 2 == 1) {
      parent = (pointer - 1) / 2;
      if (t < T_min_heap->t[parent - 1]) {
        T_min_heap->x[pointer - 1] = T_min_heap->x[parent - 1];
        T_min_heap->y[pointer - 1] = T_min_heap->y[parent - 1];
        T_min_heap->z[pointer - 1] = T_min_heap->z[parent - 1];
        T_min_heap->t[pointer - 1] = T_min_heap->t[parent - 1];
        pointer = parent;
      } else
        break;
    }
  }

  T_min_heap->x[pointer - 1] = x;
  T_min_heap->y[pointer - 1] = y;
  T_min_heap->z[pointer - 1] = z;
  T_min_heap->t[pointer - 1] = t;
}

void T_GetTime(void) {
  int tempt_x, tempt_y, tempt_z;
  int tx, ty, tz;
  float intens, value;
  char boundary;

  tempt_x = max(min_x - 1, 0);
  tempt_y = min_y;
  tempt_z = min_z;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    value = MAX_TIME;
    boundary = 0;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_x = min(min_x + 1, XDIM - 1);
  tempt_y = min_y;
  tempt_z = min_z;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    value = MAX_TIME;
    boundary = 0;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = max(min_y - 1, 0);
  tempt_x = min_x;
  tempt_z = min_z;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    value = MAX_TIME;
    boundary = 0;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = min(min_y + 1, YDIM - 1);
  tempt_x = min_x;
  tempt_z = min_z;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    value = MAX_TIME;
    boundary = 0;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = max(min_z - 1, 0);
  tempt_x = min_x;
  tempt_y = min_y;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    value = MAX_TIME;
    boundary = 0;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = min(min_z + 1, ZDIM - 1);
  tempt_x = min_x;
  tempt_y = min_y;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    value = MAX_TIME;
    boundary = 0;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = (float)(0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]));
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap(tempt_x, tempt_y, tempt_z);
  }
}

VECTOR MonomerRotate(float sx, float sy, float sz, float theta, float phi,
                     float angle) {
  float x, y, z;
  float xx, yy, zz;
  float a[3][3], b[3][3];
  VECTOR tmp;

  a[0][0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  a[0][1] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  a[0][2] = (float)(-sin(0.5 * PIE - phi));
  a[1][0] = (float)(-sin(theta));
  a[1][1] = (float)cos(theta);
  a[1][2] = 0.f;
  a[2][0] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  a[2][1] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  a[2][2] = (float)cos(0.5 * PIE - phi);

  b[0][0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  b[0][1] = (float)(-sin(theta));
  b[0][2] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  b[1][0] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  b[1][1] = (float)cos(theta);
  b[1][2] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  b[2][0] = (float)(-sin(0.5 * PIE - phi));
  b[2][1] = 0;
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

}; // namespace SegMonomer
