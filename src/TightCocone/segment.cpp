/*
  Copyright 2007-2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/**************************************
 * Zeyun Yu (zeyun@cs.utexas.edu)    *
 * Department of Computer Science    *
 * University of Texas at Austin     *
 **************************************/

#include <TightCocone/segment.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define min/max macros for this file (removed from header to avoid conflicts)
#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

#define ALPHA 50
#define MINVAL 255
#define MAXVAL 0

namespace TightCocone {

int min_x, min_y, min_z;
float min_t;
MinHeapS *min_heap;
char *seed_index;
float seed_avg_mag;

void GetMinimum(void);
void InsertHeap(int x, int y, int z);
void GetTime(void);
char CheckMinimum(int x, int y, int z);
char CheckMaximum(int x, int y, int z);

void segment(float tlow, float thigh) {
  int i, j, k;
  long step, u;

  printf("entering segment \n");

  min_heap = (MinHeapS *)malloc(sizeof(MinHeapS));
  min_heap->x = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  min_heap->y = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  min_heap->z = (int *)malloc(sizeof(int) * XDIM * YDIM * ZDIM);
  min_heap->t = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  seed_index = (char *)malloc(sizeof(char) * XDIM * YDIM * ZDIM);

  printf("allocated memory \n");

  /* Initialize */

  min_heap->size = 0;
  printf("heap size\n");
  printf("hello");

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {

        bin_img[IndexVect(i, j, k)] = 200;

        seed_index[IndexVect(i, j, k)] = 0;

        if (CheckMinimum(i, j, k) &&
            dataset->data[IndexVect(i, j, k)] <= tlow) {
          min_heap->x[min_heap->size] = i;
          min_heap->y[min_heap->size] = j;
          min_heap->z[min_heap->size] = k;
          min_heap->t[min_heap->size] = 0.0;
          dataset->tdata[IndexVect(i, j, k)] = 0.0;
          min_heap->size++;
          seed_index[IndexVect(i, j, k)] = 1;

        } else if (CheckMaximum(i, j, k) &&
                   dataset->data[IndexVect(i, j, k)] >= thigh) {
          min_heap->x[min_heap->size] = i;
          min_heap->y[min_heap->size] = j;
          min_heap->z[min_heap->size] = k;
          min_heap->t[min_heap->size] = 0.0;
          dataset->tdata[IndexVect(i, j, k)] = 0.0;
          min_heap->size++;
          seed_index[IndexVect(i, j, k)] = 2;
        }
      }

  //  printf("entering Fast Marching method\n");

  // Fast Marching Method
  step = 0;
  while (TRUE) {

    GetMinimum();

    if (step % 20000 == 0)
      printf("step = %ld size = %d, min-time = %f  \n", step, min_heap->size,
             min_t);

    if (min_t == MAX_TIME + 1)
      break;
    GetTime();

    step++;
  }

  printf("minsize: %d \n", min_heap->size);

  for (u = 0; u < min_heap->size; u++) {
    i = min_heap->x[u];
    j = min_heap->y[u];
    k = min_heap->z[u];
    bin_img[IndexVect(i, j, k)] = 0;
  }

  free(min_heap->x);
  free(min_heap->y);
  free(min_heap->z);
  free(min_heap->t);
  free(min_heap);
  free(seed_index);
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

  t = dataset->tdata[IndexVect(x, y, z)];
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
  char boundary; /* on the isosurface */
  char index;

  tempt_x = max(min_x - 1, 0);
  tempt_y = min_y;
  tempt_z = min_z;
  index = seed_index[IndexVect(min_x, min_y, min_z)];

  if (dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_x = min(min_x + 1, XDIM - 1);
  tempt_y = min_y;
  tempt_z = min_z;
  if (dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = max(min_y - 1, 0);
  tempt_x = min_x;
  tempt_z = min_z;
  if (dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_y = min(min_y + 1, YDIM - 1);
  tempt_x = min_x;
  tempt_z = min_z;
  if (dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = max(min_z - 1, 0);
  tempt_x = min_x;
  tempt_y = min_y;
  if (dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
  }

  tempt_z = min(min_z + 1, ZDIM - 1);
  tempt_x = min_x;
  tempt_y = min_y;
  if (dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] == MAX_TIME) {

    boundary = 0;
    value = MAX_TIME;

    tx = max(tempt_x - 1, 0);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tx = min(tempt_x + 1, XDIM - 1);
    ty = tempt_y;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = max(tempt_y - 1, 0);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    ty = min(tempt_y + 1, YDIM - 1);
    tx = tempt_x;
    tz = tempt_z;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = max(tempt_z - 1, 0);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    tz = min(tempt_z + 1, ZDIM - 1);
    tx = tempt_x;
    ty = tempt_y;
    if (seed_index[IndexVect(tx, ty, tz)] > 0 &&
        seed_index[IndexVect(tx, ty, tz)] != index)
      boundary = 1;
    else {
      intens = 0.01 * exp(ALPHA * ImgGrad[IndexVect(tx, ty, tz)]);
      if (value > intens + dataset->tdata[IndexVect(tx, ty, tz)])
        value = intens + dataset->tdata[IndexVect(tx, ty, tz)];
    }

    if (boundary == 1)
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = MAX_TIME + 1;
    else
      dataset->tdata[IndexVect(tempt_x, tempt_y, tempt_z)] = value;

    seed_index[IndexVect(tempt_x, tempt_y, tempt_z)] = index;
    InsertHeap(tempt_x, tempt_y, tempt_z);
  }
}

char CheckMaximum(int x, int y, int z) {
  int i, j, k;
  float weight, temp;
  char zero_flag;

  weight = 0.0;
  zero_flag = 1;

  i = x;
  j = min(y + 1, YDIM - 1);
  k = z;
  temp = velocity[IndexVect(i, j, k)].y;
  if (temp > 0.0 && j != y)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  i = x;
  j = max(y - 1, 0);
  k = z;
  temp = -velocity[IndexVect(i, j, k)].y;
  if (temp > 0.0 && j != y)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = min(x + 1, XDIM - 1);
  k = z;
  temp = velocity[IndexVect(i, j, k)].x;
  if (temp > 0.0 && i != x)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = max(x - 1, 0);
  k = z;
  temp = -velocity[IndexVect(i, j, k)].x;
  if (temp > 0.0 && i != x)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = x;
  k = min(z + 1, ZDIM - 1);
  temp = velocity[IndexVect(i, j, k)].z;
  if (temp > 0.0 && k != z)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = x;
  k = max(z - 1, 0);
  temp = -velocity[IndexVect(i, j, k)].z;
  if (temp > 0.0 && k != z)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  if (velocity[IndexVect(x, y, z)].x == 0 &&
      velocity[IndexVect(x, y, z)].y == 0 &&
      velocity[IndexVect(x, y, z)].z == 0 && zero_flag == 1)
    return (FALSE);
  else {
    if (weight == 0.0) {
      seed_avg_mag = sqrt(
          velocity[IndexVect(x, y, z)].x * velocity[IndexVect(x, y, z)].x +
          velocity[IndexVect(x, y, z)].y * velocity[IndexVect(x, y, z)].y +
          velocity[IndexVect(x, y, z)].z * velocity[IndexVect(x, y, z)].z);
      i = x;
      j = min(y + 1, YDIM - 1);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      i = x;
      j = max(y - 1, 0);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = min(x + 1, XDIM - 1);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = max(x - 1, 0);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = x;
      k = min(z + 1, ZDIM - 1);
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = x;
      k = max(z - 1, 0);
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      seed_avg_mag /= 7;
      return (TRUE);
    } else
      return (FALSE);
  }
}

char CheckMinimum(int x, int y, int z) {
  int i, j, k;
  float weight, temp;
  char zero_flag;

  weight = 0.0;
  zero_flag = 1;

  i = x;
  j = min(y + 1, YDIM - 1);
  k = z;
  temp = -velocity[IndexVect(i, j, k)].y;
  if (temp > 0.0 && j != y)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  i = x;
  j = max(y - 1, 0);
  k = z;
  temp = velocity[IndexVect(i, j, k)].y;
  if (temp > 0.0 && j != y)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = min(x + 1, XDIM - 1);
  k = z;
  temp = -velocity[IndexVect(i, j, k)].x;
  if (temp > 0.0 && i != x)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = max(x - 1, 0);
  k = z;
  temp = velocity[IndexVect(i, j, k)].x;
  if (temp > 0.0 && i != x)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = x;
  k = min(z + 1, ZDIM - 1);
  temp = -velocity[IndexVect(i, j, k)].z;
  if (temp > 0.0 && k != z)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  j = y;
  i = x;
  k = max(z - 1, 0);
  temp = velocity[IndexVect(i, j, k)].z;
  if (temp > 0.0 && k != z)
    weight += temp;
  if (velocity[IndexVect(i, j, k)].x != 0 ||
      velocity[IndexVect(i, j, k)].y != 0 ||
      velocity[IndexVect(i, j, k)].z != 0)
    zero_flag = 0;

  if (velocity[IndexVect(x, y, z)].x == 0 &&
      velocity[IndexVect(x, y, z)].y == 0 &&
      velocity[IndexVect(x, y, z)].z == 0 && zero_flag == 1)
    return (FALSE);
  else {
    if (weight == 0.0) {
      seed_avg_mag = sqrt(
          velocity[IndexVect(x, y, z)].x * velocity[IndexVect(x, y, z)].x +
          velocity[IndexVect(x, y, z)].y * velocity[IndexVect(x, y, z)].y +
          velocity[IndexVect(x, y, z)].z * velocity[IndexVect(x, y, z)].z);
      i = x;
      j = min(y + 1, YDIM - 1);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      i = x;
      j = max(y - 1, 0);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = min(x + 1, XDIM - 1);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = max(x - 1, 0);
      k = z;
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = x;
      k = min(z + 1, ZDIM - 1);
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      j = y;
      i = x;
      k = max(z - 1, 0);
      seed_avg_mag += sqrt(
          velocity[IndexVect(i, j, k)].x * velocity[IndexVect(i, j, k)].x +
          velocity[IndexVect(i, j, k)].y * velocity[IndexVect(i, j, k)].y +
          velocity[IndexVect(i, j, k)].z * velocity[IndexVect(i, j, k)].z);

      seed_avg_mag /= 7;
      return (TRUE);
    } else
      return (FALSE);
  }
}

} // namespace TightCocone
