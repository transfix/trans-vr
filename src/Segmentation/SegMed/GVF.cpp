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
#define WINDOW 2
#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))
#define GVF_SAMPLE 20

namespace SegMed {

typedef struct {
  float x;
  float y;
  float z;
} VECTOR;

typedef struct CriticalPoint CPNT;
struct CriticalPoint {
  int x;
  int y;
  int z;
  CPNT *next;
};

static VECTOR *velocity;
static int XDIM, YDIM, ZDIM;
static float *dataset;
static unsigned char *temp_class;
static unsigned char *result;
static unsigned long sample_num;
static float t_low;

void GVF_Cluster(int, int, int);
void gvfflow();
void GetGradient(float *);
void InitGVF(float *);

void GVF_Compute(int xd, int yd, int zd, float *data, float *edge_mag,
                 CPNT **critical, float tlow) {
  float temp;
  //	float max_val,min_val;
  int i, j, k;
  int x, y, z;
  int weight, zero_flag;
  //    FILE *fp;
  //    unsigned char c;
  //    int number,num;
  CPNT *critical_end;
  CPNT *critical_start;

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  t_low = tlow;

  GetGradient(edge_mag);

  velocity = (VECTOR *)malloc(sizeof(VECTOR) * XDIM * YDIM * ZDIM);

  InitGVF(edge_mag);

  return;

  result =
      (unsigned char *)malloc(sizeof(unsigned char) * XDIM * YDIM * ZDIM);

  printf("begin gvfflow...\n");
  gvfflow();

  /* Find the critical points */
  for (z = 0; z < ZDIM; z++)
    for (y = 0; y < YDIM; y++)
      for (x = 0; x < XDIM; x++) {
        weight = 1;
        zero_flag = 1;

        for (k = max(z - 1, 0); k <= min(z + 1, ZDIM - 1); k++)
          for (j = max(y - 1, 0); j <= min(y + 1, YDIM - 1); j++)
            for (i = max(x - 1, 0); i <= min(x + 1, XDIM - 1); i++) {
              if (i != x || j != y || k != z) {
                temp = (i - x) * velocity[IndexVect(i, j, k)].x +
                       (j - y) * velocity[IndexVect(i, j, k)].y +
                       (k - z) * velocity[IndexVect(i, j, k)].z;
                if (temp != 0)
                  zero_flag = 0;
                if (temp < 0)
                  weight = 0;
              }
            }

        if (velocity[IndexVect(x, y, z)].x == 0 &&
            velocity[IndexVect(x, y, z)].y == 0 &&
            velocity[IndexVect(x, y, z)].z == 0 && zero_flag == 1)
          result[IndexVect(x, y, z)] = 0;
        else {
          if (weight == 1 && dataset[IndexVect(x, y, z)] > tlow)
            result[IndexVect(x, y, z)] = 1;
          else
            result[IndexVect(x, y, z)] = 0;
        }
      }

  free(velocity);

  temp_class =
      (unsigned char *)malloc(sizeof(unsigned char) * XDIM * YDIM * ZDIM);
  sample_num = 0;
  for (z = 0; z < ZDIM; z++)
    for (y = 0; y < YDIM; y++)
      for (x = 0; x < XDIM; x++) {
        temp_class[IndexVect(x, y, z)] = 0;
        if (result[IndexVect(x, y, z)] > 0)
          sample_num++;
      }
  printf("before sampling: sample_num = %lu \n", sample_num);

  for (z = 0; z < ZDIM; z++)
    for (y = 0; y < YDIM; y++)
      for (x = 0; x < XDIM; x++) {
        if (result[IndexVect(x, y, z)] > 0) {
          temp_class[IndexVect(x, y, z)] = 1;
          result[IndexVect(x, y, z)] = 0;
          sample_num = 0;
          GVF_Cluster(x, y, z);
        }
      }

  sample_num = 0;
  for (z = 0; z < ZDIM; z++)
    for (y = 0; y < YDIM; y++)
      for (x = 0; x < XDIM; x++) {
        if (temp_class[IndexVect(x, y, z)] > 0)
          sample_num++;
        result[IndexVect(x, y, z)] = temp_class[IndexVect(x, y, z)];
      }
  printf("after sampling: sample_num = %lu \n", sample_num);

  critical_start = NULL;
  critical_end = NULL;
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        if (result[IndexVect(i, j, k)] > 0) {
          if (critical_start == NULL) {
            critical_end = (CPNT *)malloc(sizeof(CPNT));
            critical_start = critical_end;
          } else {
            critical_end->next = (CPNT *)malloc(sizeof(CPNT));
            critical_end = critical_end->next;
          }

          critical_end->x = i;
          critical_end->y = j;
          critical_end->z = k;
        }
      }
  if (critical_end != NULL)
    critical_end->next = NULL;
  *critical = critical_start;

  free(temp_class);
  free(result);
}

void GVF_Cluster(int x, int y, int z) {
  int m, n, l;

  for (l = max(z - 1, 0); l <= min(z + 1, ZDIM - 1); l++)
    for (n = max(y - 1, 0); n <= min(y + 1, YDIM - 1); n++)
      for (m = max(x - 1, 0); m <= min(x + 1, XDIM - 1); m++) {
        if (result[IndexVect(m, n, l)] > 0) {
          result[IndexVect(m, n, l)] = 0;
          if (sample_num < GVF_SAMPLE) {
            sample_num++;
            GVF_Cluster(m, n, l);
          } else {
            temp_class[IndexVect(m, n, l)] = 1;
            sample_num = 0;
            GVF_Cluster(m, n, l);
          }
        }
      }
}

void gvfflow() {
  float *u, *v, *w;
  float *tempx, *tempy, *tempz;
  int m, i, j, k;
  float up, down, left, right, front, back;
  float tmp, temp;
  float maxgrad, gradient;
  float Kapa = 2;

  //    float gf,hf;
  //    float cx,cy,cz;
  // float dt = 0.1666f;
  // float Kama = 1.0;

  if ((u = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM)) == NULL) {
    printf("GVF: not enough memory 111...\n");
    exit(0);
  }
  if ((v = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM)) == NULL) {
    printf("GVF: not enough memory 222...\n");
    exit(0);
  }
  if ((w = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM)) == NULL) {
    printf("GVF: not enough memory 333...\n");
    exit(0);
  }

  if ((tempx = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM)) == NULL) {
    printf("GVF: not enough memory 444...\n");
    exit(0);
  }
  if ((tempy = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM)) == NULL) {
    printf("GVF: not enough memory 555...\n");
    exit(0);
  }
  if ((tempz = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM)) == NULL) {
    printf("GVF: not enough memory 666...\n");
    exit(0);
  }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        u[IndexVect(i, j, k)] = velocity[IndexVect(i, j, k)].x;
        v[IndexVect(i, j, k)] = velocity[IndexVect(i, j, k)].y;
        w[IndexVect(i, j, k)] = velocity[IndexVect(i, j, k)].z;
      }

  for (m = 0; m < 10; m++) {

    printf("Iteration = %d \n", m);

    /* New GVD */
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          tmp = (float)sqrt(velocity[IndexVect(i, j, k)].x *
                                velocity[IndexVect(i, j, k)].x +
                            velocity[IndexVect(i, j, k)].y *
                                velocity[IndexVect(i, j, k)].y +
                            velocity[IndexVect(i, j, k)].z *
                                velocity[IndexVect(i, j, k)].z);

          if (tmp == 0) {
            tempx[IndexVect(i, j, k)] =
                (u[IndexVect(min(i + 1, XDIM - 1), j, k)] +
                 u[IndexVect(max(i - 1, 0), j, k)] +
                 u[IndexVect(i, min(j + 1, YDIM - 1), k)] +
                 u[IndexVect(i, max(j - 1, 0), k)] +
                 u[IndexVect(i, j, min(k + 1, ZDIM - 1))] +
                 u[IndexVect(i, j, max(k - 1, 0))]) /
                6.0f;
            tempy[IndexVect(i, j, k)] =
                (v[IndexVect(min(i + 1, XDIM - 1), j, k)] +
                 v[IndexVect(max(i - 1, 0), j, k)] +
                 v[IndexVect(i, min(j + 1, YDIM - 1), k)] +
                 v[IndexVect(i, max(j - 1, 0), k)] +
                 v[IndexVect(i, j, min(k + 1, ZDIM - 1))] +
                 v[IndexVect(i, j, max(k - 1, 0))]) /
                6.0f;
            tempz[IndexVect(i, j, k)] =
                (w[IndexVect(min(i + 1, XDIM - 1), j, k)] +
                 w[IndexVect(max(i - 1, 0), j, k)] +
                 w[IndexVect(i, min(j + 1, YDIM - 1), k)] +
                 w[IndexVect(i, max(j - 1, 0), k)] +
                 w[IndexVect(i, j, min(k + 1, ZDIM - 1))] +
                 w[IndexVect(i, j, max(k - 1, 0))]) /
                6.0f;
          } else {

            temp = (float)sqrt(
                velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].x *
                    velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].x +
                velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].y *
                    velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].y +
                velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].z *
                    velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].z);
            if (temp == 0)
              up = 0;
            else
              up = (float)exp(
                  Kapa *
                  ((velocity[IndexVect(i, j, k)].x *
                        velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].x +
                    velocity[IndexVect(i, j, k)].y *
                        velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].y +
                    velocity[IndexVect(i, j, k)].z *
                        velocity[IndexVect(i, j, min(k + 1, ZDIM - 1))].z) /
                       (tmp * temp) -
                   1));

            temp =
                (float)sqrt(velocity[IndexVect(i, j, max(k - 1, 0))].x *
                                velocity[IndexVect(i, j, max(k - 1, 0))].x +
                            velocity[IndexVect(i, j, max(k - 1, 0))].y *
                                velocity[IndexVect(i, j, max(k - 1, 0))].y +
                            velocity[IndexVect(i, j, max(k - 1, 0))].z *
                                velocity[IndexVect(i, j, max(k - 1, 0))].z);
            if (temp == 0)
              down = 0;
            else
              down = (float)exp(
                  Kapa * ((velocity[IndexVect(i, j, k)].x *
                               velocity[IndexVect(i, j, max(k - 1, 0))].x +
                           velocity[IndexVect(i, j, k)].y *
                               velocity[IndexVect(i, j, max(k - 1, 0))].y +
                           velocity[IndexVect(i, j, k)].z *
                               velocity[IndexVect(i, j, max(k - 1, 0))].z) /
                              (tmp * temp) -
                          1));

            temp =
                (float)sqrt(velocity[IndexVect(i, max(j - 1, 0), k)].x *
                                velocity[IndexVect(i, max(j - 1, 0), k)].x +
                            velocity[IndexVect(i, max(j - 1, 0), k)].y *
                                velocity[IndexVect(i, max(j - 1, 0), k)].y +
                            velocity[IndexVect(i, max(j - 1, 0), k)].z *
                                velocity[IndexVect(i, max(j - 1, 0), k)].z);
            if (temp == 0)
              left = 0;
            else
              left = (float)exp(
                  Kapa * ((velocity[IndexVect(i, j, k)].x *
                               velocity[IndexVect(i, max(j - 1, 0), k)].x +
                           velocity[IndexVect(i, j, k)].y *
                               velocity[IndexVect(i, max(j - 1, 0), k)].y +
                           velocity[IndexVect(i, j, k)].z *
                               velocity[IndexVect(i, max(j - 1, 0), k)].z) /
                              (tmp * temp) -
                          1));

            temp = (float)sqrt(
                velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].x *
                    velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].x +
                velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].y *
                    velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].y +
                velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].z *
                    velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].z);
            if (temp == 0)
              right = 0;
            else
              right = (float)exp(
                  Kapa *
                  ((velocity[IndexVect(i, j, k)].x *
                        velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].x +
                    velocity[IndexVect(i, j, k)].y *
                        velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].y +
                    velocity[IndexVect(i, j, k)].z *
                        velocity[IndexVect(i, min(j + 1, YDIM - 1), k)].z) /
                       (tmp * temp) -
                   1));

            temp =
                (float)sqrt(velocity[IndexVect(max(i - 1, 0), j, k)].x *
                                velocity[IndexVect(max(i - 1, 0), j, k)].x +
                            velocity[IndexVect(max(i - 1, 0), j, k)].y *
                                velocity[IndexVect(max(i - 1, 0), j, k)].y +
                            velocity[IndexVect(max(i - 1, 0), j, k)].z *
                                velocity[IndexVect(max(i - 1, 0), j, k)].z);
            if (temp == 0)
              back = 0;
            else
              back = (float)exp(
                  Kapa * ((velocity[IndexVect(i, j, k)].x *
                               velocity[IndexVect(max(i - 1, 0), j, k)].x +
                           velocity[IndexVect(i, j, k)].y *
                               velocity[IndexVect(max(i - 1, 0), j, k)].y +
                           velocity[IndexVect(i, j, k)].z *
                               velocity[IndexVect(max(i - 1, 0), j, k)].z) /
                              (tmp * temp) -
                          1));

            temp = (float)sqrt(
                velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].x *
                    velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].x +
                velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].y *
                    velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].y +
                velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].z *
                    velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].z);
            if (temp == 0)
              front = 0;
            else
              front = (float)exp(
                  Kapa *
                  ((velocity[IndexVect(i, j, k)].x *
                        velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].x +
                    velocity[IndexVect(i, j, k)].y *
                        velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].y +
                    velocity[IndexVect(i, j, k)].z *
                        velocity[IndexVect(min(i + 1, XDIM - 1), j, k)].z) /
                       (tmp * temp) -
                   1));

            temp = right + left + up + down + front + back;
            if (temp != 0) {
              right /= temp;
              left /= temp;
              up /= temp;
              down /= temp;
              front /= temp;
              back /= temp;
            }

            tempx[IndexVect(i, j, k)] =
                u[IndexVect(i, j, k)] +
                front * (u[IndexVect(min(i + 1, XDIM - 1), j, k)] -
                         u[IndexVect(i, j, k)]) +
                back * (u[IndexVect(max(i - 1, 0), j, k)] -
                        u[IndexVect(i, j, k)]) +
                right * (u[IndexVect(i, min(j + 1, YDIM - 1), k)] -
                         u[IndexVect(i, j, k)]) +
                left * (u[IndexVect(i, max(j - 1, 0), k)] -
                        u[IndexVect(i, j, k)]) +
                up * (u[IndexVect(i, j, min(k + 1, ZDIM - 1))] -
                      u[IndexVect(i, j, k)]) +
                down * (u[IndexVect(i, j, max(k - 1, 0))] -
                        u[IndexVect(i, j, k)]);

            tempy[IndexVect(i, j, k)] =
                v[IndexVect(i, j, k)] +
                front * (v[IndexVect(min(i + 1, XDIM - 1), j, k)] -
                         v[IndexVect(i, j, k)]) +
                back * (v[IndexVect(max(i - 1, 0), j, k)] -
                        v[IndexVect(i, j, k)]) +
                right * (v[IndexVect(i, min(j + 1, YDIM - 1), k)] -
                         v[IndexVect(i, j, k)]) +
                left * (v[IndexVect(i, max(j - 1, 0), k)] -
                        v[IndexVect(i, j, k)]) +
                up * (v[IndexVect(i, j, min(k + 1, ZDIM - 1))] -
                      v[IndexVect(i, j, k)]) +
                down * (v[IndexVect(i, j, max(k - 1, 0))] -
                        v[IndexVect(i, j, k)]);

            tempz[IndexVect(i, j, k)] =
                w[IndexVect(i, j, k)] +
                front * (w[IndexVect(min(i + 1, XDIM - 1), j, k)] -
                         w[IndexVect(i, j, k)]) +
                back * (w[IndexVect(max(i - 1, 0), j, k)] -
                        w[IndexVect(i, j, k)]) +
                right * (w[IndexVect(i, min(j + 1, YDIM - 1), k)] -
                         w[IndexVect(i, j, k)]) +
                left * (w[IndexVect(i, max(j - 1, 0), k)] -
                        w[IndexVect(i, j, k)]) +
                up * (w[IndexVect(i, j, min(k + 1, ZDIM - 1))] -
                      w[IndexVect(i, j, k)]) +
                down * (w[IndexVect(i, j, max(k - 1, 0))] -
                        w[IndexVect(i, j, k)]);
          }
        }

    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          u[IndexVect(i, j, k)] = tempx[IndexVect(i, j, k)];
          v[IndexVect(i, j, k)] = tempy[IndexVect(i, j, k)];
          w[IndexVect(i, j, k)] = tempz[IndexVect(i, j, k)];
        }
  }

  maxgrad = 0.0;
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        gradient = u[IndexVect(i, j, k)] * u[IndexVect(i, j, k)] +
                   v[IndexVect(i, j, k)] * v[IndexVect(i, j, k)] +
                   w[IndexVect(i, j, k)] * w[IndexVect(i, j, k)];
        if (gradient > maxgrad)
          maxgrad = gradient;
      }

  maxgrad = (float)sqrt(maxgrad);
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        velocity[IndexVect(i, j, k)].x = u[IndexVect(i, j, k)] / maxgrad;
        velocity[IndexVect(i, j, k)].y = v[IndexVect(i, j, k)] / maxgrad;
        velocity[IndexVect(i, j, k)].z = w[IndexVect(i, j, k)] / maxgrad;
      }

  free(u);
  free(v);
  free(w);
  free(tempx);
  free(tempy);
  free(tempz);
}

void GetGradient(float *edge_mag) {
  int x, y, z;
  int i, j, k;
  float gradient, grad_x, grad_y, grad_z;
  float maxgrad, mingrad;

  maxgrad = 0;
  mingrad = 999999;
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
              255 * (edge_mag[IndexVect(x, y, z)] - mingrad) /
              (maxgrad - mingrad);
        }
  }
}

void InitGVF(float *edge_mag) {
  int x, y, z;
  int i, j, k;
  float gradient, grad_x, grad_y, grad_z;
  float maxgrad, mingrad;

  maxgrad = 0;
  mingrad = 999999;
  for (z = 0; z < ZDIM; z++)
    for (y = 0; y < YDIM; y++)
      for (x = 0; x < XDIM; x++) {
        grad_x = 0.0;
        for (j = max(y - 1, 0); j <= min(y + 1, YDIM - 1); j++)
          for (k = max(z - 1, 0); k <= min(z + 1, ZDIM - 1); k++) {
            grad_x += edge_mag[IndexVect(min(x + 1, XDIM - 1), j, k)] -
                      edge_mag[IndexVect(max(x - 1, 0), j, k)];
            if (j == y || k == z)
              grad_x += edge_mag[IndexVect(min(x + 1, XDIM - 1), j, k)] -
                        edge_mag[IndexVect(max(x - 1, 0), j, k)];
            if (j == y && k == z)
              grad_x +=
                  2.0f * (edge_mag[IndexVect(min(x + 1, XDIM - 1), j, k)] -
                          edge_mag[IndexVect(max(x - 1, 0), j, k)]);
          }

        grad_y = 0.0;
        for (i = max(x - 1, 0); i <= min(x + 1, XDIM - 1); i++)
          for (k = max(z - 1, 0); k <= min(z + 1, ZDIM - 1); k++) {
            grad_y += edge_mag[IndexVect(i, min(y + 1, YDIM - 1), k)] -
                      edge_mag[IndexVect(i, max(y - 1, 0), k)];
            if (i == x || k == z)
              grad_y += edge_mag[IndexVect(i, min(y + 1, YDIM - 1), k)] -
                        edge_mag[IndexVect(i, max(y - 1, 0), k)];
            if (i == x && k == z)
              grad_y +=
                  2.0f * (edge_mag[IndexVect(i, min(y + 1, YDIM - 1), k)] -
                          edge_mag[IndexVect(i, max(y - 1, 0), k)]);
          }

        grad_z = 0.0;
        for (i = max(x - 1, 0); i <= min(x + 1, XDIM - 1); i++)
          for (j = max(y - 1, 0); j <= min(y + 1, YDIM - 1); j++) {
            grad_z += edge_mag[IndexVect(i, j, min(z + 1, ZDIM - 1))] -
                      edge_mag[IndexVect(i, j, max(z - 1, 0))];
            if (i == x || j == y)
              grad_z += edge_mag[IndexVect(i, j, min(z + 1, ZDIM - 1))] -
                        edge_mag[IndexVect(i, j, max(z - 1, 0))];
            if (i == x && j == y)
              grad_z +=
                  2.0f * (edge_mag[IndexVect(i, j, min(z + 1, ZDIM - 1))] -
                          edge_mag[IndexVect(i, j, max(z - 1, 0))]);
          }

        velocity[IndexVect(x, y, z)].x = grad_x;
        velocity[IndexVect(x, y, z)].y = grad_y;
        velocity[IndexVect(x, y, z)].z = grad_z;

        gradient =
            (float)sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        if (gradient < mingrad)
          mingrad = gradient;
        if (gradient > maxgrad)
          maxgrad = gradient;
      }

  if (mingrad < maxgrad) {
    for (z = 0; z < ZDIM; z++)
      for (y = 0; y < YDIM; y++)
        for (x = 0; x < XDIM; x++) {
          velocity[IndexVect(x, y, z)].x /= maxgrad;
          velocity[IndexVect(x, y, z)].y /= maxgrad;
          velocity[IndexVect(x, y, z)].z /= maxgrad;
        }
  }
}

}; // namespace SegMed
