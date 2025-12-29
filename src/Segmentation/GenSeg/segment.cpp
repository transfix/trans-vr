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

#include <VolMagick/VolMagick.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace GenSeg {

typedef struct {
  int *x;
  int *y;
  int *z;
  float *t;
  int size;
} MinHeapS;

typedef struct {
  float x;
  float y;
  float z;
} VECTOR;

typedef struct SeedPoint SDPNT;
struct SeedPoint {
  int x;
  int y;
  int z;
  SDPNT *next;
};

typedef struct {
  int x;
  int y;
  int z;
} INTVEC;

#define max(x, y) ((x > y) ? (x) : (y))
#define min(x, y) ((x < y) ? (x) : (y))
#define PIE 3.1415927

#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))
#define MAX_TIME 9999999.0f
#define ALPHA 0.12f

int min_x, min_y, min_z;
float min_t;
MinHeapS *min_heap;
MinHeapS *T_min_heap;
unsigned char T_index;
unsigned char *seed_index;
float t_low, t_high;
int XDIM, YDIM, ZDIM;
float *dataset;
float *tdata;
float *T_tdata;
float *ImgGrad;
int SeedNum;
int PhaseNum;
int start_ptr, end_ptr;
INTVEC *IndexArray;

float minext[3], maxext[3];
int nverts, ncells;
unsigned int dim[3];
float orig[3], span[3];

extern VolMagick::VolumeFileInfo loadedVolumeInfo;

void GetMinimum(void);
void InsertHeap(int, int, int);
void GetTime(void);
void T_GetMinimum(void);
void T_InsertHeap(int, int, int);
void T_GetTime(void);
unsigned char SearchIndex(int *, int *, int *, SDPNT *);
void InitialTime(int, int, int);
void GetGradient(float *);
void GetSeeds(unsigned char *);
void write_rawiv_float(float *, FILE *);

void Segment(int xd, int yd, int zd, float *data, float tlow, float thigh,
             SDPNT **seed_list, int classnum, const char *output_filename) {
  int i, j, k;
  int num;
  int Seed_x, Seed_y, Seed_z;
  unsigned char index;
  SDPNT *seed_tmp;
  char file_name[1024];
  SDPNT *critical_end;
  SDPNT *critical_start;
  FILE *fp;
  long step;

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  t_high = thigh;

  ImgGrad = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  GetGradient(ImgGrad);

  seed_index =
      (unsigned char *)malloc(sizeof(unsigned char) * XDIM * YDIM * ZDIM);
  GetSeeds(seed_index);

  min_heap = (MinHeapS *)malloc(sizeof(MinHeapS));
  min_heap->x = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  min_heap->y = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  min_heap->z = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  min_heap->t = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  tdata = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  T_min_heap = (MinHeapS *)malloc(sizeof(MinHeapS));
  T_min_heap->x = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  T_min_heap->y = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  T_min_heap->z = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  T_min_heap->t = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  T_tdata = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  IndexArray = (INTVEC *)malloc(sizeof(INTVEC) * XDIM * YDIM * ZDIM);

  SeedNum = 0;
  critical_start = NULL;
  critical_end = NULL;
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        tdata[IndexVect(i, j, k)] = MAX_TIME;
        T_tdata[IndexVect(i, j, k)] = MAX_TIME;

        if (seed_index[IndexVect(i, j, k)] == 254) {
          SeedNum++;
          if (critical_start == NULL) {
            critical_end = (SDPNT *)malloc(sizeof(SDPNT));
            critical_start = critical_end;
          } else {
            critical_end->next = (SDPNT *)malloc(sizeof(SDPNT));
            critical_end = critical_end->next;
          }

          critical_end->x = i;
          critical_end->y = j;
          critical_end->z = k;
        }
      }
  if (critical_end != NULL)
    critical_end->next = NULL;
  printf("SeedNum = %d....\n", SeedNum);

  /* if the intensity matters
          for (k=0; k<ZDIM; k++)
          for (j=0; j<YDIM; j++)
          for (i=0; i<XDIM; i++) {
                  if (seed_index[IndexVect(i,j,k)] == 254) {
                          if (dataset[IndexVect(i,j,k)] > thigh)
                                  seed_index[IndexVect(i,j,k)] = 1;
                          else if (dataset[IndexVect(i,j,k)] > tlow)
                                  seed_index[IndexVect(i,j,k)] = 2;
                  }
          }
  */

  PhaseNum = 1;
  min_heap->size = 0;

  for (num = 0; num < classnum; num++) {
    seed_tmp = seed_list[num];
    while (seed_tmp != NULL) {
      Seed_x = seed_tmp->x;
      Seed_y = seed_tmp->y;
      Seed_z = seed_tmp->z;
      if (seed_index[IndexVect(Seed_x, Seed_y, Seed_z)] == 254)
        SeedNum--;
      seed_index[IndexVect(Seed_x, Seed_y, Seed_z)] = num;
      tdata[IndexVect(Seed_x, Seed_y, Seed_z)] = 0.0;
      InsertHeap(Seed_x, Seed_y, Seed_z);
      seed_tmp = seed_tmp->next;
    }
  }

  t_low = 255.0f;
  while (SeedNum > 0) {

    index = SearchIndex(&i, &j, &k, critical_start);
    if (index == 255) {
      t_low = t_low - 1.0f;
      if (t_low < tlow)
        break;
      else
        continue;
    }
    t_high = t_low;
    t_low = 255.0f;

    min_heap->size = 0;
    tdata[IndexVect(i, j, k)] = 0.0;
    if (seed_index[IndexVect(i, j, k)] == 254)
      SeedNum--;
    seed_index[IndexVect(i, j, k)] = index;
    InsertHeap(i, j, k);

    while (1) {
      GetMinimum();
      if (min_t >= MAX_TIME)
        break;
      GetTime();
    }

    printf("Index = %d    SeedNum = %d \n", index, SeedNum);
  }

  printf("\n\nBegin the fast marching ....\n");
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

  printf("begin fast marching ...\n");
  PhaseNum = 2;
  t_high = tlow;
  step = 0;
  while (1) {

    GetMinimum();
    if (step % 500 == 0)
      printf("step: %ld  min_t: %f \n", step, min_t);

    if (min_t >= MAX_TIME)
      break;
    GetTime();
    step++;
  }
  printf("end fast marching  %d...\n", classnum);

  /* write the border indices */
  unsigned int index_count = 0;
  printf("writing border indices ...\n");
  sprintf(file_name, "%s_indices.txt", output_filename);
  fprintf(stderr, "filename: %s\n", file_name);
  if ((fp = fopen(file_name, "w")) == NULL) {
    printf("write error...\n");
    exit(0);
  }
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        if (tdata[IndexVect(i, j, k)] == MAX_TIME + 1) {
          index_count++;
          fprintf(fp, "%d %d %d\n", i, j, k);
        }
      }
  fclose(fp);

  /* write a point cloud corresponding to border indices */
  {
    printf("writing border indices ...\n");
    sprintf(file_name, "%s_indices.pcd", output_filename);
    if ((fp = fopen(file_name, "w")) == NULL) {
      printf("write error...\n");
      exit(0);
    }
    fprintf(fp, "%d\n", index_count);
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          if (tdata[IndexVect(i, j, k)] == MAX_TIME + 1)
            fprintf(fp, "%f %f %f\n",
                    loadedVolumeInfo.XMin() + loadedVolumeInfo.XSpan() * i,
                    loadedVolumeInfo.YMin() + loadedVolumeInfo.YSpan() * j,
                    loadedVolumeInfo.ZMin() + loadedVolumeInfo.ZSpan() * k);
        }
    fclose(fp);
  }

  /* write the results */
  for (num = 0; num < classnum; num++) {
    printf("writing %d ...\n", num);
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          if (seed_index[IndexVect(i, j, k)] == (unsigned char)num)
            tdata[IndexVect(i, j, k)] = dataset[IndexVect(i, j, k)];
          else
            tdata[IndexVect(i, j, k)] = 0;
        }
    sprintf(file_name, "%s_subunit_%.2d.rawiv", output_filename, num);
    if ((fp = fopen(file_name, "wb")) == NULL) {
      printf("write error...\n");
      exit(0);
    }
    write_rawiv_float(tdata, fp);
  }
  printf("end writing ...\n");

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

  free(seed_index);
  free(ImgGrad);
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

unsigned char SearchIndex(int *m, int *n, int *l, SDPNT *critical_start) {
  int i, j, k;
  int x0, y0, z0;
  int x1, y1, z1;
  float t;
  SDPNT *critical_tmp;
  SDPNT *critical_prv;

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
    if (x0 == x1 && y0 == y1 && z0 == z1)
      return (255);
    else {
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

void GetTime(void) {
  int tempt_x, tempt_y, tempt_z;
  int tx, ty, tz;
  float intens, value;
  char boundary;
  unsigned char index;

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
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
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
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
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
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
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
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
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
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
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
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_high ||
        seed_index[IndexVect(tx, ty, tz)] < 254 &&
            seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + tdata[IndexVect(tx, ty, tz)])
        value = intens + tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    if (seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] == 254)
      SeedNum--;
    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
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
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
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
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
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
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
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
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
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
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
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
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + T_tdata[IndexVect(tx, ty, tz)])
        value = intens + T_tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (dataset[IndexVect(tx, ty, tz)] < t_low)
      boundary = 1;
    else {
      intens = 0.01f * (float)exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
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

void GetGradient(float *edge_mag) {
  int x, y, z;
  int i, j, k;
  float gradient, grad_x, grad_y, grad_z;
  float maxgrad, mingrad;

  maxgrad = 0;
  mingrad = 999999.0;
  for (z = 0; z < ZDIM; z++)
    for (y = 0; y < YDIM; y++)
      for (x = 0; x < XDIM; x++) {
        grad_x = 0.0;
        for (j = max(y - 1, 0); j <= min(y + 1, YDIM - 1); j++)
          for (k = max(z - 1, 0); k <= min(z + 1, ZDIM - 1); k++) {
            grad_x += dataset[IndexVect(min(x + 1, XDIM - 1), j, k)] -
                      dataset[IndexVect(max(x - 1, 0), j, k)];
            if (j == y || k == z)
              grad_x += dataset[IndexVect(min(x + 1, XDIM - 1), j, k)] -
                        dataset[IndexVect(max(x - 1, 0), j, k)];
            if (j == y && k == z)
              grad_x +=
                  2.0f * (dataset[IndexVect(min(x + 1, XDIM - 1), j, k)] -
                          dataset[IndexVect(max(x - 1, 0), j, k)]);
          }

        grad_y = 0.0;
        for (i = max(x - 1, 0); i <= min(x + 1, XDIM - 1); i++)
          for (k = max(z - 1, 0); k <= min(z + 1, ZDIM - 1); k++) {
            grad_y += dataset[IndexVect(i, min(y + 1, YDIM - 1), k)] -
                      dataset[IndexVect(i, max(y - 1, 0), k)];
            if (i == x || k == z)
              grad_y += dataset[IndexVect(i, min(y + 1, YDIM - 1), k)] -
                        dataset[IndexVect(i, max(y - 1, 0), k)];
            if (i == x && k == z)
              grad_y +=
                  2.0f * (dataset[IndexVect(i, min(y + 1, YDIM - 1), k)] -
                          dataset[IndexVect(i, max(y - 1, 0), k)]);
          }

        grad_z = 0.0;
        for (i = max(x - 1, 0); i <= min(x + 1, XDIM - 1); i++)
          for (j = max(y - 1, 0); j <= min(y + 1, YDIM - 1); j++) {
            grad_z += dataset[IndexVect(i, j, min(z + 1, ZDIM - 1))] -
                      dataset[IndexVect(i, j, max(z - 1, 0))];
            if (i == x || j == y)
              grad_z += dataset[IndexVect(i, j, min(z + 1, ZDIM - 1))] -
                        dataset[IndexVect(i, j, max(z - 1, 0))];
            if (i == x && j == y)
              grad_z +=
                  2.0f * (dataset[IndexVect(i, j, min(z + 1, ZDIM - 1))] -
                          dataset[IndexVect(i, j, max(z - 1, 0))]);
          }

        gradient =
            (float)sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
        edge_mag[IndexVect(x, y, z)] = gradient;

        if (gradient < mingrad)
          mingrad = gradient;
        if (gradient > maxgrad)
          maxgrad = gradient;
      }

  if (mingrad < maxgrad) {
    for (z = 0; z < ZDIM; z++)
      for (y = 0; y < YDIM; y++)
        for (x = 0; x < XDIM; x++) {

          edge_mag[IndexVect(x, y, z)] =
              255.0f * (edge_mag[IndexVect(x, y, z)] - mingrad) /
              (maxgrad - mingrad);
        }
  }
}

void GetSeeds(unsigned char *result) {
  int i, j, k;
  int m, n, l, u;
  float tmp;
  float *smth_img, *temp;

  smth_img = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  temp = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++)
        smth_img[IndexVect(i, j, k)] = dataset[IndexVect(i, j, k)];

  for (u = 0; u < 2; u++) {
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          temp[IndexVect(i, j, k)] =
              0.4f * smth_img[IndexVect(i, j, k)] +
              0.1f * (smth_img[IndexVect(max(0, i - 1), j, k)] +
                      smth_img[IndexVect(min(XDIM - 1, i + 1), j, k)] +
                      smth_img[IndexVect(i, max(0, j - 1), k)] +
                      smth_img[IndexVect(i, min(YDIM - 1, j + 1), k)] +
                      smth_img[IndexVect(i, j, max(0, k - 1))] +
                      smth_img[IndexVect(i, j, min(ZDIM - 1, k + 1))]);
        }
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++)
          smth_img[IndexVect(i, j, k)] = temp[IndexVect(i, j, k)];
  }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {

        result[IndexVect(i, j, k)] = 255;
        tmp = smth_img[IndexVect(i, j, k)];
        if (tmp > t_high) {
          u = 0;
          for (l = max(k - 1, 0); l <= min(k + 1, ZDIM - 1); l++)
            for (n = max(j - 1, 0); n <= min(j + 1, YDIM - 1); n++)
              for (m = max(i - 1, 0); m <= min(i + 1, XDIM - 1); m++) {
                if (tmp < smth_img[IndexVect(m, n, l)])
                  u = 1;
              }

          if (u == 0)
            result[IndexVect(i, j, k)] = 254;
        }
      }

  free(temp);
  free(smth_img);
}

}; // namespace GenSeg
