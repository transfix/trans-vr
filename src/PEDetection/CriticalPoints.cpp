/*
  Copyright 2006 The University of Texas at Austin

        Authors: Sangmin Park <smpark@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PEDetection.

  PEDetection is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PEDetection is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/**************************************
 * Zeyun Yu (zeyun.yu@gmail.com)    *
 * Department of Computer Science    *
 * University of Texas at Austin     *
 **************************************/

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define max2(x, y) ((x > y) ? (x) : (y))
#define min2(x, y) ((x < y) ? (x) : (y))
#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))

#include <PEDetection/CriticalPoints.h>

// namespace SegCapsid {
/*
typedef struct CriticalPoint CPNT;
struct CriticalPoint{
  unsigned short x;
  unsigned short y;
  unsigned short z;
  CPNT *next;
};
*/

static int XDIM, YDIM, ZDIM;
static float *dataset;
float *temp_gfloat;

float GetImgGradient(int, int, int);

CPNT *FindCriticalPoints(int xd, int yd, int zd, float *data, float tlow,
                         int h_num, int k_num) {
  int i, j, k;
  int x, y, z;
  int u, v;
  CPNT *critical_end = NULL;
  CPNT *critical_start = NULL;
  float tmp;
  unsigned char *temp = NULL;
  float *temp_float = NULL;
  float max_grad, min_grad;
  unsigned long histogram[256];
  unsigned long number, tri_num;

  dataset = data;
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;

  printf("Find Critical Points ... \n");
  printf("XYZ Dim = %3d %3d %3d \n", xd, yd, zd);
  fflush(stdout);

  temp = (unsigned char *)malloc(sizeof(unsigned char) * XDIM * YDIM * ZDIM);
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++)
        temp[IndexVect(i, j, k)] = 0;
  for (k = 0; k < 256; k++)
    histogram[k] = 0;

  number = 0;
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        tmp = dataset[IndexVect(i, j, k)];
        if (tmp > tlow) {
          u = 0;
          for (z = max2(k - 1, 0); z <= min2(k + 1, ZDIM - 1); z++)
            for (y = max2(j - 1, 0); y <= min2(j + 1, YDIM - 1); y++)
              for (x = max2(i - 1, 0); x <= min2(i + 1, XDIM - 1); x++) {
                if (tmp < dataset[IndexVect(x, y, z)])
                  u = 1;
              }
          if (u == 0) {
            temp[IndexVect(i, j, k)] = 1; // bigger than 26 neighbors
            v = (int)(tmp + 0.5);
            histogram[v] += 1;
            number++;
          }
        }
      }

  tri_num = 3000 * (h_num * h_num + k_num * k_num + h_num * k_num);
  number = 0;
  for (k = 255; k >= 0; k--) {
    number += histogram[k];
    if (number > tri_num)
      break;
  }
  tmp = (float)k;
  if (tmp < tlow + 20.0)
    tmp = tlow + 20.0;
  //	tmp = 282;
  printf("thigh = %f \n", tmp);
  fflush(stdout);

  critical_start = NULL;
  critical_end = NULL;
  number = 0;
  for (k = 0; k < ZDIM; k++) {
    for (j = 0; j < YDIM; j++) {
      for (i = 0; i < XDIM; i++) {
        if (temp[IndexVect(i, j, k)] == 1 &&
            dataset[IndexVect(i, j, k)] > tmp) {
          number++;
          if (critical_start == NULL) {
            critical_end = (CPNT *)malloc(sizeof(CPNT));
            critical_end->next = NULL;
            critical_start = critical_end;
          } else {
            critical_end->next = (CPNT *)malloc(sizeof(CPNT));
            critical_end = critical_end->next;
            critical_end->next = NULL;
          }
          critical_end->x = (unsigned short)i;
          critical_end->y = (unsigned short)j;
          critical_end->z = (unsigned short)k;
        }
      }
    }
  }
  printf("number of maximal critical points: %ld \n", number);
  printf("%ld < min2(50000, tri_num/5) = %ld\n", number,
         min2(50000, tri_num / 5));
  fflush(stdout);

  if (number < min2(50000, tri_num / 5)) {
    temp_float = NULL;
    temp_float = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

    if (temp_float == NULL) {
      printf("temp_float is NULL)\n");
      fflush(stdout);
      exit(1);
    }

    min_grad = 999999.0f;
    max_grad = 0.0f;
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          if (dataset[IndexVect(i, j, k)] == 0)
            temp_float[IndexVect(i, j, k)] = 0.0f;
          else {
            u = 0;
            for (z = max2(k - 1, 0); z <= min2(k + 1, ZDIM - 1); z++)
              for (y = max2(j - 1, 0); y <= min2(j + 1, YDIM - 1); y++)
                for (x = max2(i - 1, 0); x <= min2(i + 1, XDIM - 1); x++) {
                  if (dataset[IndexVect(x, y, z)] == 0)
                    u = 1;
                }
            if (u == 0)
              temp_float[IndexVect(i, j, k)] = GetImgGradient(i, j, k);
            else
              temp_float[IndexVect(i, j, k)] = 0.0f;
          }
          if (temp_float[IndexVect(i, j, k)] > max_grad)
            max_grad = temp_float[IndexVect(i, j, k)];
          if (temp_float[IndexVect(i, j, k)] < min_grad)
            min_grad = temp_float[IndexVect(i, j, k)];
        }

    printf("Min & Max Gradient = %.4f %.4f\n", min_grad, max_grad);
    fflush(stdout);

    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++)
          temp_float[IndexVect(i, j, k)] =
              255.0f * (temp_float[IndexVect(i, j, k)] - min_grad) /
              (max_grad - min_grad);

    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++)
          temp[IndexVect(i, j, k)] = 0;
    for (k = 0; k < 256; k++)
      histogram[k] = 0;

    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          if (dataset[IndexVect(i, j, k)] > tlow) {
            u = 0;
            tmp = temp_float[IndexVect(i, j, k)];
            for (z = max2(k - 1, 0); z <= min2(k + 1, ZDIM - 1); z++)
              for (y = max2(j - 1, 0); y <= min2(j + 1, YDIM - 1); y++)
                for (x = max2(i - 1, 0); x <= min2(i + 1, XDIM - 1); x++) {
                  if (tmp < temp_float[IndexVect(x, y, z)])
                    u = 1;
                }

            if (u == 0) {
              temp[IndexVect(i, j, k)] = 1; // bigger than 26 neighbors
              v = (int)(tmp + 0.5);
              histogram[v] += 1;
            }
          }
        }

    tri_num = number;
    number = 0;
    for (k = 255; k >= 0; k--) {
      number += histogram[k];
      if (number > tri_num)
        break;
    }
    tmp = (float)k;
    if (tmp < tlow + 20.0)
      tmp = tlow + 20.0;
    printf("thigh = %f \n", tmp);

    printf("here -- 1\n");
    fflush(stdout);

    //		do {
    printf("tmp = %.4f\n", tmp);
    fflush(stdout);
    number = 0;
    for (k = 0; k < ZDIM; k++) {
      for (j = 0; j < YDIM; j++) {
        for (i = 0; i < XDIM; i++) {
          if (temp[IndexVect(i, j, k)] == 1 &&
              temp_float[IndexVect(i, j, k)] > tmp) {
            number++;
            if (critical_start == NULL) {
              critical_end = (CPNT *)malloc(sizeof(CPNT));
              critical_end->next = NULL;
              critical_start = critical_end;
            } else {
              critical_end->next = (CPNT *)malloc(sizeof(CPNT));
              critical_end = critical_end->next;
              critical_end->next = NULL;
            }

            critical_end->x = (unsigned short)i;
            critical_end->y = (unsigned short)j;
            critical_end->z = (unsigned short)k;
          }
        }
      }
    }
    tmp -= 10;
    //		} while (number<=0 && tmp>0);

    printf("number of saddle critical points: %ld \n", number);
    fflush(stdout);
    free(temp_float);
    printf("here -- 2\n");
    fflush(stdout);
  }

  printf("here -- 3\n");
  fflush(stdout);

  free(temp);
  return critical_start;
}

float GetImgGradient(int x, int y, int z) {
  int i, j, k;
  float grad_x, grad_y, grad_z;
  float gradient;

  grad_x = 0.0;
  for (j = max2(y - 1, 0); j <= min2(y + 1, YDIM - 1); j++)
    for (k = max2(z - 1, 0); k <= min2(z + 1, ZDIM - 1); k++) {
      grad_x += dataset[IndexVect(min2(x + 1, XDIM - 1), j, k)] -
                dataset[IndexVect(max2(x - 1, 0), j, k)];
      if (j == y || k == z)
        grad_x += dataset[IndexVect(min2(x + 1, XDIM - 1), j, k)] -
                  dataset[IndexVect(max2(x - 1, 0), j, k)];
      if (j == y && k == z)
        grad_x += 2.0f * (dataset[IndexVect(min2(x + 1, XDIM - 1), j, k)] -
                          dataset[IndexVect(max2(x - 1, 0), j, k)]);
    }

  grad_y = 0.0;
  for (i = max2(x - 1, 0); i <= min2(x + 1, XDIM - 1); i++)
    for (k = max2(z - 1, 0); k <= min2(z + 1, ZDIM - 1); k++) {
      grad_y += dataset[IndexVect(i, min2(y + 1, YDIM - 1), k)] -
                dataset[IndexVect(i, max2(y - 1, 0), k)];
      if (i == x || k == z)
        grad_y += dataset[IndexVect(i, min2(y + 1, YDIM - 1), k)] -
                  dataset[IndexVect(i, max2(y - 1, 0), k)];
      if (i == x && k == z)
        grad_y += 2.0f * (dataset[IndexVect(i, min2(y + 1, YDIM - 1), k)] -
                          dataset[IndexVect(i, max2(y - 1, 0), k)]);
    }

  grad_z = 0.0;
  for (i = max2(x - 1, 0); i <= min2(x + 1, XDIM - 1); i++)
    for (j = max2(y - 1, 0); j <= min2(y + 1, YDIM - 1); j++) {
      grad_z += dataset[IndexVect(i, j, min2(z + 1, ZDIM - 1))] -
                dataset[IndexVect(i, j, max2(z - 1, 0))];
      if (i == x || j == y)
        grad_z += dataset[IndexVect(i, j, min2(z + 1, ZDIM - 1))] -
                  dataset[IndexVect(i, j, max2(z - 1, 0))];
      if (i == x && j == y)
        grad_z += 2.0f * (dataset[IndexVect(i, j, min2(z + 1, ZDIM - 1))] -
                          dataset[IndexVect(i, j, max2(z - 1, 0))]);
    }

  gradient = (float)sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
  return (gradient / 16.0f);
}

//};
