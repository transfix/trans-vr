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
#define ANGL1 1.107149f
#define ANGL2 2.034444f

#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))
#define MAX_TIME 9999999.0f
#define ALPHA 0.1f

namespace SegSubunit {

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
  int x;
  int y;
  int z;
} INTVEC;

static int min_x, min_y, min_z;
static float min_t;
static MinHeapS *min_heap;
static MinHeapS *T_min_heap;
static unsigned short T_index;
static unsigned short *seed_index;
static float t_low, t_high;
static int XDIM, YDIM, ZDIM;
static float *dataset;
static float *tdata;
static float *T_tdata;
// static float *ImgGrad;
static VECTOR *FiveFold;
static CPNT *critical_start;
static int SeedNum;
static int PhaseNum;
static long start_ptr, end_ptr;
static INTVEC *IndexArray;

void GetMinimum3(void);
void InsertHeap3(int, int, int);
void GetTime3(float *, float *);
void T_GetMinimum3(void);
void T_InsertHeap3(int, int, int);
void T_GetTime3(void);
unsigned short SearchIndex(int *, int *, int *);
void SetSeedIndex3(int, int, int, float *, float *);
VECTOR Rotate(float, float, float, float, float, float, float, float,
              float); // int, int, int);
void InitialTime(int, int, int);
float GetImgGradient(int, int, int);

void AsymSubunitSegment(int xd, int yd, int zd, float *orig_tmp,
                        float *span_tmp, float *data, unsigned short *result,
                        float tlow, float thigh, CPNT *critical,
                        VECTOR *five_fold) {
  int i, j, k;
  int m, n;
  float MaxSeed;
  int MaxSeed_x = 0, MaxSeed_y = 0, MaxSeed_z = 0;
  unsigned short index;
  float cx, cy, cz;
  float nx, ny, nz;
  float ax, ay, az;
  float sx = 0.f, sy = 0.f, sz = 0.f;
  float theta, phi;
  VECTOR sv;
  CPNT *critical_tmp;
  float max_tmp[3];
  float positx, posity, positz;

  t_low = tlow;
  t_high = thigh;
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  FiveFold = five_fold;
  critical_start = critical;
  seed_index = result;

  max_tmp[0] = orig_tmp[0] + span_tmp[0] * (XDIM - 1);
  max_tmp[1] = orig_tmp[1] + span_tmp[1] * (YDIM - 1);
  max_tmp[2] = orig_tmp[2] + span_tmp[2] * (ZDIM - 1);

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

  MaxSeed = -999.0f;
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
  printf("SeedNum = %d....\n", SeedNum);

  PhaseNum = 1;
  SetSeedIndex3(MaxSeed_x, MaxSeed_y, MaxSeed_z, orig_tmp, span_tmp);

  min_heap->size = 0;
  tdata[IndexVect(MaxSeed_x, MaxSeed_y, MaxSeed_z)] = 0.0;
  InsertHeap3(MaxSeed_x, MaxSeed_y, MaxSeed_z);

  while (1) {

    GetMinimum3();

    if (min_t >= MAX_TIME)
      break;
    GetTime3(orig_tmp, span_tmp);
  }
  printf("SeedNum = %d \n", SeedNum);

  IndexArray = (INTVEC *)malloc(sizeof(INTVEC) * XDIM * YDIM * ZDIM);

  while (SeedNum > 0) {

    index = SearchIndex(&i, &j, &k);
    if (index == 255)
      break;

    if (seed_index[IndexVect(i, j, k)] == 254)
      SeedNum--;
    seed_index[IndexVect(i, j, k)] = 255;

    cx = FiveFold[0].x - max_tmp[0] / 2.0; // XDIM/2;
    cy = FiveFold[0].y - max_tmp[1] / 2.0; // YDIM/2;
    cz = FiveFold[0].z - max_tmp[2] / 2.0; // ZDIM/2;

    positx = orig_tmp[0] + i * span_tmp[0];
    posity = orig_tmp[1] + j * span_tmp[1];
    positz = orig_tmp[2] + k * span_tmp[2];

    m = index / 5;
    n = index - 5 * m;
    if (m == 0) {
      if (n == 0) {
        //	sx = (float)i;
        //	sy = (float)j;
        //	sz = (float)k;
        sx = positx;
        sy = posity;
        sz = positz;
      } else {
        theta = (float)atan2(cy, cx);
        phi = (float)atan2(cz, sqrt(cx * cx + cy * cy));
        // sv = Rotate((float)i,(float)j,(float)k,theta,phi,-n*2.0f*PIE/5.0f,
        // max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
        sv =
            Rotate(positx, posity, positz, theta, phi, -n * 2.0f * PIE / 5.0f,
                   max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
        sx = sv.x;
        sy = sv.y;
        sz = sv.z;
      }
    } else if (m < 11) {
      nx = FiveFold[m].x - max_tmp[0] / 2.0; //-XDIM/2;
      ny = FiveFold[m].y - max_tmp[1] / 2.0; //-YDIM/2;
      nz = FiveFold[m].z - max_tmp[2] / 2.0; //-ZDIM/2;
      theta = (float)atan2(ny, nx);
      phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
      //   sv =
      //   Rotate((float)i,(float)j,(float)k,theta,phi,-n*2.0f*PIE/5.0f-PIE/5.0f,
      //   max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
      sv = Rotate(positx, posity, positz, theta, phi,
                  -n * 2.0f * PIE / 5.0f - PIE / 5.0f, max_tmp[0], max_tmp[1],
                  max_tmp[2]); // XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;

      ax = ny * cz - nz * cy;
      ay = nz * cx - nx * cz;
      az = nx * cy - ny * cx;
      theta = (float)atan2(ay, ax);
      phi = (float)atan2(az, sqrt(ax * ax + ay * ay));
      if (m < 6)
        sv = Rotate(sx, sy, sz, theta, phi, ANGL1, max_tmp[0], max_tmp[1],
                    max_tmp[2]); // XDIM,YDIM,ZDIM);
      else
        sv = Rotate(sx, sy, sz, theta, phi, ANGL2, max_tmp[0], max_tmp[1],
                    max_tmp[2]); // XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;
    } else if (m == 11) {
      if (n == 0) {
        //	sx = (float)i;
        //	sy = (float)j;
        //	sz = (float)k;
        sx = positx;
        sy = posity;
        sz = positz;
      } else {
        nx = FiveFold[11].x - max_tmp[0] / 2.0; //-XDIM/2;
        ny = FiveFold[11].y - max_tmp[1] / 2.0; //-YDIM/2;
        nz = FiveFold[11].z - max_tmp[2] / 2.0; //-ZDIM/2;
        theta = (float)atan2(ny, nx);
        phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
        // sv = Rotate((float)i,(float)j,(float)k,theta,phi,-n*2.0f*PIE/5.0f,
        // max_tmp[0], max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
        sv =
            Rotate(positx, posity, positz, theta, phi, -n * 2.0f * PIE / 5.0f,
                   max_tmp[0], max_tmp[1], max_tmp[2]); // XDIM,YDIM,ZDIM);
        sx = sv.x;
        sy = sv.y;
        sz = sv.z;
      }
      nx = FiveFold[1].x - max_tmp[0] / 2.0; //-XDIM/2;
      ny = FiveFold[1].y - max_tmp[1] / 2.0; //-YDIM/2;
      nz = FiveFold[1].z - max_tmp[2] / 2.0; //-ZDIM/2;
      ax = ny * cz - nz * cy;
      ay = nz * cx - nx * cz;
      az = nx * cy - ny * cx;
      theta = (float)atan2(ay, ax);
      phi = (float)atan2(az, sqrt(ax * ax + ay * ay));
      sv = Rotate(sx, sy, sz, theta, phi, PIE, max_tmp[0], max_tmp[1],
                  max_tmp[2]); // XDIM,YDIM,ZDIM);
      sx = sv.x;
      sy = sv.y;
      sz = sv.z;
    } else
      printf("wrong: classification of critical points ... \n");

    // i = (int)(sx+0.5);
    // j = (int)(sy+0.5);
    // k = (int)(sz+0.5);
    i = (int)((sx - orig_tmp[0]) / span_tmp[0] + 0.5);
    j = (int)((sy - orig_tmp[1]) / span_tmp[1] + 0.5);
    k = (int)((sz - orig_tmp[2]) / span_tmp[2] + 0.5);

    T_index = seed_index[IndexVect(i, j, k)];

    if (T_index == 0)
      continue;
    else if (T_index < 60) {
      printf("there is an error: ignored ...\n");
      continue;
    } else {
      SetSeedIndex3(i, j, k, orig_tmp, span_tmp);
      min_heap->size = 0;
      tdata[IndexVect(i, j, k)] = 0.0;
      InsertHeap3(i, j, k);
      while (1) {
        GetMinimum3();
        /* Joe R: 9/21/2006 - somehow the min_x value was sometimes calculated
        to be much greater than XDIM, causing segfaults in GetTime3(). */
        if (min_x > (XDIM - 1))
          min_x = XDIM - 1;
        if (min_y > (YDIM - 1))
          min_y = YDIM - 1;
        if (min_z > (ZDIM - 1))
          min_z = ZDIM - 1;
        // printf("minimum: %d %d %d\n",min_x,min_y,min_z);
        if (min_t >= MAX_TIME)
          break;
        GetTime3(orig_tmp, span_tmp);
      }

      if (SeedNum % 500 == 0)
        printf("Index = %d    SeedNum = %d \n", index, SeedNum);
    }
  }

  printf("\nBegin the fast marching ....\n");
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

  PhaseNum = 2;
  t_high = t_low;
  while (1) {

    GetMinimum3();

    if (min_t == MAX_TIME + 1)
      break;
    GetTime3(orig_tmp, span_tmp);
  }

  /*Jesse-4/17/07-Some of the following calls to free() are causing a
  segmentation fault. commenting them out is a cheap fix that causes a
  significant memory leak.*/
  // free(min_heap->x);
  // free(min_heap->y);
  // free(min_heap->z);
  // free(min_heap->t);
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

unsigned short SearchIndex(int *m, int *n, int *l) {
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

    T_GetMinimum3();

    if (min_t == MAX_TIME + 1) {
      printf("too large t_low: some seeds were ignored ....\n");
      return (255);
    }

    T_GetTime3();

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

void SetSeedIndex3(int ax, int ay, int az, float *orig_tmp, float *span_tmp) {
  int i, j, k;
  int m, n;
  float cx, cy, cz;
  float nx, ny, nz;
  float sx, sy, sz;
  float theta, phi;
  VECTOR sv;
  unsigned short index;

  float max_tmpp[3];
  float positx, posity, positz;

  max_tmpp[0] = orig_tmp[0] + span_tmp[0] * (XDIM - 1);
  max_tmpp[1] = orig_tmp[1] + span_tmp[1] * (YDIM - 1);
  max_tmpp[2] = orig_tmp[2] + span_tmp[2] * (ZDIM - 1);

  index = 0;

  cx = FiveFold[0].x - max_tmpp[0] / 2.0; // XDIM/2;
  cy = FiveFold[0].y - max_tmpp[1] / 2.0; // YDIM/2;
  cz = FiveFold[0].z - max_tmpp[2] / 2.0; // ZDIM/2;
  i = ax;
  j = ay;
  k = az;
  positx = orig_tmp[0] + ax * span_tmp[0];
  posity = orig_tmp[1] + ay * span_tmp[1];
  positz = orig_tmp[2] + az * span_tmp[2];

  if (seed_index[IndexVect(i, j, k)] == 254)
    SeedNum--;
  seed_index[IndexVect(i, j, k)] = index;
  index++;
  theta = (float)atan2(cy, cx);
  phi = (float)atan2(cz, sqrt(cx * cx + cy * cy));
  for (n = 1; n < 5; n++) {
    //    sv =
    //    Rotate((float)ax,(float)ay,(float)az,theta,phi,n*2.0f*PIE/5.0f,max_tmp[0],
    //    max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    sv = Rotate(positx, posity, positz, theta, phi, n * 2.0f * PIE / 5.0f,
                max_tmpp[0], max_tmpp[1], max_tmpp[2]); // XDIM,YDIM,ZDIM);
    i = (int)((sv.x - orig_tmp[0]) / span_tmp[0] + 0.5);
    j = (int)((sv.y - orig_tmp[1]) / span_tmp[1] + 0.5);
    k = (int)((sv.z - orig_tmp[2]) / span_tmp[2] + 0.5);
    if (seed_index[IndexVect(i, j, k)] == 254)
      SeedNum--;
    seed_index[IndexVect(i, j, k)] = index;
    index++;
  }

  for (m = 1; m < 11; m++) {
    nx = FiveFold[m].x - max_tmpp[0] / 2.0; // XDIM/2;
    ny = FiveFold[m].y - max_tmpp[1] / 2.0; // YDIM/2;
    nz = FiveFold[m].z - max_tmpp[2] / 2.0; // ZDIM/2;
    sx = nz * cy - ny * cz;
    sy = nx * cz - nz * cx;
    sz = ny * cx - nx * cy;
    theta = (float)atan2(sy, sx);
    phi = (float)atan2(sz, sqrt(sx * sx + sy * sy));
    if (m < 6)
      sv = Rotate(positx, posity, positz, theta, phi, ANGL1, max_tmpp[0],
                  max_tmpp[1], max_tmpp[2]); // XDIM,YDIM,ZDIM);
    // sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL1,max_tmp[0],
    // max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    else
      sv = Rotate(positx, posity, positz, theta, phi, ANGL2, max_tmpp[0],
                  max_tmpp[1], max_tmpp[2]); // XDIM,YDIM,ZDIM);
    // sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,ANGL2,max_tmp[0],
    // max_tmpp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
    sx = sv.x;
    sy = sv.y;
    sz = sv.z;

    theta = (float)atan2(ny, nx);
    phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
    for (n = 0; n < 5; n++) {
      sv = Rotate(sx, sy, sz, theta, phi, n * 2.0f * PIE / 5.0f + PIE / 5.0f,
                  max_tmpp[0], max_tmpp[1], max_tmpp[2]); // XDIM,YDIM,ZDIM);
      i = (int)((sv.x - orig_tmp[0]) / span_tmp[0] + 0.5);
      j = (int)((sv.y - orig_tmp[1]) / span_tmp[1] + 0.5);
      k = (int)((sv.z - orig_tmp[2]) / span_tmp[2] + 0.5);
      //     i = (int)(sv.x+0.5);
      //     j = (int)(sv.y+0.5);
      //     k = (int)(sv.z+0.5);
      if (seed_index[IndexVect(i, j, k)] == 254)
        SeedNum--;
      seed_index[IndexVect(i, j, k)] = index;
      index++;
    }
  }

  nx = FiveFold[1].x - max_tmpp[0] / 2.0; // XDIM/2;
  ny = FiveFold[1].y - max_tmpp[1] / 2.0; // YDIM/2;
  nz = FiveFold[1].z - max_tmpp[2] / 2.0; // ZDIM/2;
  sx = nz * cy - ny * cz;
  sy = nx * cz - nz * cx;
  sz = ny * cx - nx * cy;
  theta = (float)atan2(sy, sx);
  phi = (float)atan2(sz, sqrt(sx * sx + sy * sy));
  sv = Rotate(positx, posity, positz, theta, phi, PIE, max_tmpp[0],
              max_tmpp[1], max_tmpp[2]); // XDIM,YDIM,ZDIM);
  // sv = Rotate((float)ax,(float)ay,(float)az,theta,phi,PIE,max_tmp[0],
  // max_tmp[1], max_tmp[2]); //XDIM,YDIM,ZDIM);
  sx = sv.x;
  sy = sv.y;
  sz = sv.z;
  i = (int)((sx - orig_tmp[0]) / span_tmp[0] + 0.5);
  j = (int)((sy - orig_tmp[1]) / span_tmp[1] + 0.5);
  k = (int)((sz - orig_tmp[2]) / span_tmp[2] + 0.5);
  //  i = (int)(sx+0.5);
  //  j = (int)(sy+0.5);
  //  k = (int)(sz+0.5);
  if (seed_index[IndexVect(i, j, k)] == 254)
    SeedNum--;
  seed_index[IndexVect(i, j, k)] = index;
  index++;
  nx = FiveFold[11].x - max_tmpp[0] / 2.0; // XDIM/2;
  ny = FiveFold[11].y - max_tmpp[1] / 2.0; // YDIM/2;
  nz = FiveFold[11].z - max_tmpp[2] / 2.0; // ZDIM/2;
  theta = (float)atan2(ny, nx);
  phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
  for (n = 1; n < 5; n++) {
    sv = Rotate(sx, sy, sz, theta, phi, n * 2.0f * PIE / 5.0f, max_tmpp[0],
                max_tmpp[1], max_tmpp[2]); // XDIM,YDIM,ZDIM);
    i = (int)((sv.x - orig_tmp[0]) / span_tmp[0] + 0.5);
    j = (int)((sv.y - orig_tmp[1]) / span_tmp[1] + 0.5);
    k = (int)((sv.z - orig_tmp[2]) / span_tmp[2] + 0.5);

    //    i = (int)(sv.x+0.5);
    //    j = (int)(sv.y+0.5);
    //    k = (int)(sv.z+0.5);
    if (seed_index[IndexVect(i, j, k)] == 254)
      SeedNum--;
    seed_index[IndexVect(i, j, k)] = index;
    index++;
  }
}

void GetMinimum3(void) {
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

void InsertHeap3(int x, int y, int z) {
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

void GetTime3(float *orig_tmp, float *span_tmp) {
  int tempt_x, tempt_y, tempt_z;
  int tx, ty, tz;
  float intens, value;
  char boundary;
  char index;

  index = (char)seed_index[IndexVect(min_x, min_y, min_z)];

  tempt_x = max(min_x - 1, 0);
  tempt_y = min_y;
  tempt_z = min_z;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
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
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (PhaseNum == 1)
      SetSeedIndex3(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp);
    else
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;

    InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_x = min(min_x + 1, XDIM - 1);
  tempt_y = min_y;
  tempt_z = min_z;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
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
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (PhaseNum == 1)
      SetSeedIndex3(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp);
    else
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;

    InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = max(min_y - 1, 0);
  tempt_x = min_x;
  tempt_z = min_z;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
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
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (PhaseNum == 1)
      SetSeedIndex3(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp);
    else
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;

    InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = min(min_y + 1, YDIM - 1);
  tempt_x = min_x;
  tempt_z = min_z;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
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
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (PhaseNum == 1)
      SetSeedIndex3(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp);
    else
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;

    InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = max(min_z - 1, 0);
  tempt_x = min_x;
  tempt_y = min_y;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
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
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (PhaseNum == 1)
      SetSeedIndex3(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp);
    else
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;

    InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = min(min_z + 1, ZDIM - 1);
  tempt_x = min_x;
  tempt_y = min_y;
  if (tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        (seed_index[IndexVect(tx, ty, tz)] < 254 &&
         seed_index[IndexVect(tx, ty, tz)] != index))
      boundary = 1;
    else {
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
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (PhaseNum == 1)
      SetSeedIndex3(tempt_x, tempt_y, tempt_z, orig_tmp, span_tmp);
    else
      seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;

    InsertHeap3(tempt_x, tempt_y, tempt_z);
  }
}

void T_GetMinimum3(void) {
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

void T_InsertHeap3(int x, int y, int z) {
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

void T_GetTime3(void) {
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
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_x = min(min_x + 1, XDIM - 1);
  tempt_y = min_y;
  tempt_z = min_z;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = max(min_y - 1, 0);
  tempt_x = min_x;
  tempt_z = min_z;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = min(min_y + 1, YDIM - 1);
  tempt_x = min_x;
  tempt_z = min_z;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = max(min_z - 1, 0);
  tempt_x = min_x;
  tempt_y = min_y;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap3(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = min(min_z + 1, ZDIM - 1);
  tempt_x = min_x;
  tempt_y = min_y;
  if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] < 254) {
    T_index = seed_index[IndexVect(tempt_x, tempt_y, tempt_z)];
    return;
  } else if (T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {
    value = (float)(MAX_TIME + 1.0);
    boundary = 0;
    intens = GetImgGradient(tempt_x, tempt_y, tempt_z);
    intens = (float)(exp(ALPHA * intens));

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      T_tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    T_InsertHeap3(tempt_x, tempt_y, tempt_z);
  }
}

}; // namespace SegSubunit
