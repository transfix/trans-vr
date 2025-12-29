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
#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))

namespace SegCapsid {

typedef struct CriticalPoint CPNT;
struct CriticalPoint {
  unsigned short x;
  unsigned short y;
  unsigned short z;
  CPNT *next;
};

static int XDIM, YDIM, ZDIM;
static float *dataset;

float GetImgGradient(int, int, int);

void FindCriticalPoints(int xd, int yd, int zd, float *data, CPNT **critical,
                        float tlow, int h_num, int k_num) {
  int i, j, k;
  int x, y, z;
  int u, v;
  CPNT *critical_end;
  CPNT *critical_start;
  float tmp;
  unsigned char *temp;
  float *temp_float;
  float max_grad, min_grad;
  unsigned long histogram[256];
  unsigned long number, tri_num;

  dataset = data;
  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;

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
            temp[IndexVect(i, j, k)] = 1;
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
    tmp = tlow + 20.0f;
  printf("thigh = %f \n", tmp);

  critical_start = NULL;
  critical_end = NULL;
  number = 0;
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        if (temp[IndexVect(i, j, k)] == 1 &&
            dataset[IndexVect(i, j, k)] > tmp) {
          number++;
          if (critical_start == NULL) {
            critical_end = (CPNT *)malloc(sizeof(CPNT));
            critical_start = critical_end;
          } else {
            critical_end->next = (CPNT *)malloc(sizeof(CPNT));
            critical_end = critical_end->next;
          }

          critical_end->x = (unsigned short)i;
          critical_end->y = (unsigned short)j;
          critical_end->z = (unsigned short)k;
        }
      }

  printf("number of maximal critical points: %ld \n", number);

  if (number < min2(50000, tri_num / 5)) {
    temp_float = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

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
              temp[IndexVect(i, j, k)] = 1;
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
      tmp = tlow + 20.0f;
    printf("thigh = %f \n", tmp);

    number = 0;
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          if (temp[IndexVect(i, j, k)] == 1 &&
              temp_float[IndexVect(i, j, k)] > tmp) {
            number++;
            if (critical_start == NULL) {
              critical_end = (CPNT *)malloc(sizeof(CPNT));
              critical_start = critical_end;
            } else {
              critical_end->next = (CPNT *)malloc(sizeof(CPNT));
              critical_end = critical_end->next;
            }

            critical_end->x = (unsigned short)i;
            critical_end->y = (unsigned short)j;
            critical_end->z = (unsigned short)k;
          }
        }

    printf("number of saddle critical points: %ld \n", number);
    free(temp_float);
  }

  if (critical_end != NULL)
    critical_end->next = NULL;
  *critical = critical_start;

  free(temp);
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

}; // namespace SegCapsid
