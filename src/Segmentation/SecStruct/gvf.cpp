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
#define WINSIZE 1
#define WINDOW 3
#define SIGMA 0.8
#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))

namespace SecStruct {

typedef struct {
  float x;
  float y;
  float z;
} VECTOR;

VECTOR *velocity;
int XDIM, YDIM, ZDIM;
float *dataset;

void gvfflow();
void GetGradient();
void GetMinGradient();
void GetMaxGradient();

void GVF_Compute(int xd, int yd, int zd, float *data, VECTOR *vel) {

  XDIM = xd;
  YDIM = yd;
  ZDIM = zd;
  dataset = data;
  velocity = vel;

  GetGradient();
  gvfflow();
}

void gvfflow() {
  float *u, *v, *w;
  float *tempx, *tempy, *tempz;
  int m, i, j, k;
  float up, down, left, right, front, back;
  float tmp, temp;
  float Kapa = 2;

  float back_avg;

  u = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  v = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  w = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  tempx = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  tempy = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);
  tempz = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        u[IndexVect(i, j, k)] = velocity[IndexVect(i, j, k)].x;
        v[IndexVect(i, j, k)] = velocity[IndexVect(i, j, k)].y;
        w[IndexVect(i, j, k)] = velocity[IndexVect(i, j, k)].z;
      }

  back_avg =
      (dataset[IndexVect(1, 1, 1)] + dataset[IndexVect(XDIM - 2, 1, 1)] +
       dataset[IndexVect(1, YDIM - 2, 1)] +
       dataset[IndexVect(XDIM - 2, YDIM - 2, 1)] +
       dataset[IndexVect(1, 1, ZDIM - 2)] +
       dataset[IndexVect(XDIM - 2, 1, ZDIM - 2)] +
       dataset[IndexVect(1, YDIM - 2, ZDIM - 2)] +
       dataset[IndexVect(XDIM - 2, YDIM - 2, ZDIM - 2)]) /
      8.0f;

  for (m = 0; m < 3; m++) {

    printf("Iteration = %d \n", m);

    /* New GVD */
    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {
          if (dataset[IndexVect(i, j, k)] > back_avg + 2) {
            tmp = (float)sqrt(u[IndexVect(i, j, k)] * u[IndexVect(i, j, k)] +
                              v[IndexVect(i, j, k)] * v[IndexVect(i, j, k)] +
                              w[IndexVect(i, j, k)] * w[IndexVect(i, j, k)]);

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

              temp =
                  (float)sqrt(u[IndexVect(i, j, min(k + 1, ZDIM - 1))] *
                                  u[IndexVect(i, j, min(k + 1, ZDIM - 1))] +
                              v[IndexVect(i, j, min(k + 1, ZDIM - 1))] *
                                  v[IndexVect(i, j, min(k + 1, ZDIM - 1))] +
                              w[IndexVect(i, j, min(k + 1, ZDIM - 1))] *
                                  w[IndexVect(i, j, min(k + 1, ZDIM - 1))]);
              if (temp == 0)
                up = 0;
              else
                up = (float)exp(
                    Kapa * ((u[IndexVect(i, j, k)] *
                                 u[IndexVect(i, j, min(k + 1, ZDIM - 1))] +
                             v[IndexVect(i, j, k)] *
                                 v[IndexVect(i, j, min(k + 1, ZDIM - 1))] +
                             w[IndexVect(i, j, k)] *
                                 w[IndexVect(i, j, min(k + 1, ZDIM - 1))]) /
                                (tmp * temp) -
                            1));

              temp = (float)sqrt(u[IndexVect(i, j, max(k - 1, 0))] *
                                     u[IndexVect(i, j, max(k - 1, 0))] +
                                 v[IndexVect(i, j, max(k - 1, 0))] *
                                     v[IndexVect(i, j, max(k - 1, 0))] +
                                 w[IndexVect(i, j, max(k - 1, 0))] *
                                     w[IndexVect(i, j, max(k - 1, 0))]);
              if (temp == 0)
                down = 0;
              else
                down = (float)exp(Kapa *
                                  ((u[IndexVect(i, j, k)] *
                                        u[IndexVect(i, j, max(k - 1, 0))] +
                                    v[IndexVect(i, j, k)] *
                                        v[IndexVect(i, j, max(k - 1, 0))] +
                                    w[IndexVect(i, j, k)] *
                                        w[IndexVect(i, j, max(k - 1, 0))]) /
                                       (tmp * temp) -
                                   1));

              temp = (float)sqrt(u[IndexVect(i, max(j - 1, 0), k)] *
                                     u[IndexVect(i, max(j - 1, 0), k)] +
                                 v[IndexVect(i, max(j - 1, 0), k)] *
                                     v[IndexVect(i, max(j - 1, 0), k)] +
                                 w[IndexVect(i, max(j - 1, 0), k)] *
                                     w[IndexVect(i, max(j - 1, 0), k)]);
              if (temp == 0)
                left = 0;
              else
                left = (float)exp(Kapa *
                                  ((u[IndexVect(i, j, k)] *
                                        u[IndexVect(i, max(j - 1, 0), k)] +
                                    v[IndexVect(i, j, k)] *
                                        v[IndexVect(i, max(j - 1, 0), k)] +
                                    w[IndexVect(i, j, k)] *
                                        w[IndexVect(i, max(j - 1, 0), k)]) /
                                       (tmp * temp) -
                                   1));

              temp =
                  (float)sqrt(u[IndexVect(i, min(j + 1, YDIM - 1), k)] *
                                  u[IndexVect(i, min(j + 1, YDIM - 1), k)] +
                              v[IndexVect(i, min(j + 1, YDIM - 1), k)] *
                                  v[IndexVect(i, min(j + 1, YDIM - 1), k)] +
                              w[IndexVect(i, min(j + 1, YDIM - 1), k)] *
                                  w[IndexVect(i, min(j + 1, YDIM - 1), k)]);
              if (temp == 0)
                right = 0;
              else
                right = (float)exp(
                    Kapa * ((u[IndexVect(i, j, k)] *
                                 u[IndexVect(i, min(j + 1, YDIM - 1), k)] +
                             v[IndexVect(i, j, k)] *
                                 v[IndexVect(i, min(j + 1, YDIM - 1), k)] +
                             w[IndexVect(i, j, k)] *
                                 w[IndexVect(i, min(j + 1, YDIM - 1), k)]) /
                                (tmp * temp) -
                            1));

              temp = (float)sqrt(u[IndexVect(max(i - 1, 0), j, k)] *
                                     u[IndexVect(max(i - 1, 0), j, k)] +
                                 v[IndexVect(max(i - 1, 0), j, k)] *
                                     v[IndexVect(max(i - 1, 0), j, k)] +
                                 w[IndexVect(max(i - 1, 0), j, k)] *
                                     w[IndexVect(max(i - 1, 0), j, k)]);
              if (temp == 0)
                back = 0;
              else
                back = (float)exp(Kapa *
                                  ((u[IndexVect(i, j, k)] *
                                        u[IndexVect(max(i - 1, 0), j, k)] +
                                    v[IndexVect(i, j, k)] *
                                        v[IndexVect(max(i - 1, 0), j, k)] +
                                    w[IndexVect(i, j, k)] *
                                        w[IndexVect(max(i - 1, 0), j, k)]) /
                                       (tmp * temp) -
                                   1));

              temp =
                  (float)sqrt(u[IndexVect(min(i + 1, XDIM - 1), j, k)] *
                                  u[IndexVect(min(i + 1, XDIM - 1), j, k)] +
                              v[IndexVect(min(i + 1, XDIM - 1), j, k)] *
                                  v[IndexVect(min(i + 1, XDIM - 1), j, k)] +
                              w[IndexVect(min(i + 1, XDIM - 1), j, k)] *
                                  w[IndexVect(min(i + 1, XDIM - 1), j, k)]);
              if (temp == 0)
                front = 0;
              else
                front = (float)exp(
                    Kapa * ((u[IndexVect(i, j, k)] *
                                 u[IndexVect(min(i + 1, XDIM - 1), j, k)] +
                             v[IndexVect(i, j, k)] *
                                 v[IndexVect(min(i + 1, XDIM - 1), j, k)] +
                             w[IndexVect(i, j, k)] *
                                 w[IndexVect(min(i + 1, XDIM - 1), j, k)]) /
                                (tmp * temp) -
                            1));

              temp = front + back + right + left + up + down;
              if (temp != 0) {
                front /= temp;
                back /= temp;
                right /= temp;
                left /= temp;
                up /= temp;
                down /= temp;
              }

              tempx[IndexVect(i, j, k)] =
                  u[IndexVect(i, j, k)] +
                  (front * (u[IndexVect(min(i + 1, XDIM - 1), j, k)] -
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
                           u[IndexVect(i, j, k)]));

              tempy[IndexVect(i, j, k)] =
                  v[IndexVect(i, j, k)] +
                  (front * (v[IndexVect(min(i + 1, XDIM - 1), j, k)] -
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
                           v[IndexVect(i, j, k)]));

              tempz[IndexVect(i, j, k)] =
                  w[IndexVect(i, j, k)] +
                  (front * (w[IndexVect(min(i + 1, XDIM - 1), j, k)] -
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
                           w[IndexVect(i, j, k)]));
            }
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

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        velocity[IndexVect(i, j, k)].x = u[IndexVect(i, j, k)];
        velocity[IndexVect(i, j, k)].y = v[IndexVect(i, j, k)];
        velocity[IndexVect(i, j, k)].z = w[IndexVect(i, j, k)];
      }

  free(u);
  free(v);
  free(w);
  free(tempx);
  free(tempy);
  free(tempz);
}

/* minimum neighbor: for features brighter than background */
void GetMinGradient() {
  int i, j, k;
  int m, n, l;
  int min_x = 0, min_y = 0, min_z = 0;
  float min_value, tmp;
  //  float maxgrad;
  float gradient;
  float fx, fy, fz;

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        min_value = dataset[IndexVect(i, j, k)];
        for (l = max(0, k - WINSIZE); l <= min(ZDIM - 1, k + WINSIZE); l++)
          for (n = max(0, j - WINSIZE); n <= min(YDIM - 1, j + WINSIZE); n++)
            for (m = max(0, i - WINSIZE); m <= min(XDIM - 1, i + WINSIZE);
                 m++) {
              if (dataset[IndexVect(m, n, l)] < min_value)
                min_value = dataset[IndexVect(m, n, l)];
            }

        if (min_value == dataset[IndexVect(i, j, k)]) {
          velocity[IndexVect(i, j, k)].x = 0;
          velocity[IndexVect(i, j, k)].y = 0;
          velocity[IndexVect(i, j, k)].z = 0;
        } else {

          fx = 0;
          fy = 0;
          fz = 0;

          for (l = max(0, k - WINSIZE); l <= min(ZDIM - 1, k + WINSIZE); l++)
            for (n = max(0, j - WINSIZE); n <= min(YDIM - 1, j + WINSIZE);
                 n++)
              for (m = max(0, i - WINSIZE); m <= min(XDIM - 1, i + WINSIZE);
                   m++) {
                if (dataset[IndexVect(m, n, l)] == min_value) {
                  gradient = (float)sqrt(double((m - i) * (m - i) +
                                                (n - j) * (n - j) +
                                                (l - k) * (l - k)));
                  fx += (m - i) / gradient;
                  fy += (n - j) / gradient;
                  fz += (l - k) / gradient;
                }
              }

          gradient = (float)sqrt(fx * fx + fy * fy + fz * fz);
          if (gradient == 0) {
            velocity[IndexVect(i, j, k)].x = 0;
            velocity[IndexVect(i, j, k)].y = 0;
            velocity[IndexVect(i, j, k)].z = 0;
          } else {

            fx /= gradient;
            fy /= gradient;
            fz /= gradient;

            min_value = 0;
            for (l = max(0, k - WINSIZE); l <= min(ZDIM - 1, k + WINSIZE);
                 l++)
              for (n = max(0, j - WINSIZE); n <= min(YDIM - 1, j + WINSIZE);
                   n++)
                for (m = max(0, i - WINSIZE); m <= min(XDIM - 1, i + WINSIZE);
                     m++) {
                  if (m != i || n != j || l != k) {
                    gradient =
                        (float)((fx * (m - i) + fy * (n - j) + fz * (l - k)) /
                                sqrt(double((m - i) * (m - i) +
                                            (n - j) * (n - j) +
                                            (l - k) * (l - k))));
                    if (gradient > min_value) {
                      min_x = m;
                      min_y = n;
                      min_z = l;
                      min_value = gradient;
                    }
                  }
                }

            tmp = (float)(-(dataset[IndexVect(min_x, min_y, min_z)] -
                            dataset[IndexVect(i, j, k)]) /
                          sqrt(double((min_x - i) * (min_x - i) +
                                      (min_y - j) * (min_y - j) +
                                      (min_z - k) * (min_z - k))));
            velocity[IndexVect(i, j, k)].x = (min_x - i) * tmp;
            velocity[IndexVect(i, j, k)].y = (min_y - j) * tmp;
            velocity[IndexVect(i, j, k)].z = (min_z - k) * tmp;
          }
        }
      }

  /*
  maxgrad = -999;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
        gradient = velocity[IndexVect(i,j,k)].x * velocity[IndexVect(i,j,k)].x
  + velocity[IndexVect(i,j,k)].y * velocity[IndexVect(i,j,k)].y +
                   velocity[IndexVect(i,j,k)].z *
  velocity[IndexVect(i,j,k)].z; if (gradient > maxgrad) maxgrad = gradient;
      }

  maxgrad = sqrt(maxgrad);
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
        velocity[IndexVect(i,j,k)].x = velocity[IndexVect(i,j,k)].x / maxgrad;
        velocity[IndexVect(i,j,k)].y = velocity[IndexVect(i,j,k)].y / maxgrad;
        velocity[IndexVect(i,j,k)].z = velocity[IndexVect(i,j,k)].z / maxgrad;
      }
  */
}

/* maximum neighbor: for features darker than background */
void GetMaxGradient() {
  int i, j, k;
  int m, n, l;
  int max_x = 0, max_y = 0, max_z = 0;
  float max_value, tmp;
  //  float maxgrad;
  float gradient;
  float fx, fy, fz;

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        max_value = dataset[IndexVect(i, j, k)];
        for (l = max(0, k - WINSIZE); l <= min(ZDIM - 1, k + WINSIZE); l++)
          for (n = max(0, j - WINSIZE); n <= min(YDIM - 1, j + WINSIZE); n++)
            for (m = max(0, i - WINSIZE); m <= min(XDIM - 1, i + WINSIZE);
                 m++) {
              if (dataset[IndexVect(m, n, l)] > max_value)
                max_value = dataset[IndexVect(m, n, l)];
            }

        if (max_value == dataset[IndexVect(i, j, k)]) {
          velocity[IndexVect(i, j, k)].x = 0;
          velocity[IndexVect(i, j, k)].y = 0;
          velocity[IndexVect(i, j, k)].z = 0;
        } else {

          fx = 0;
          fy = 0;
          fz = 0;

          for (l = max(0, k - WINSIZE); l <= min(ZDIM - 1, k + WINSIZE); l++)
            for (n = max(0, j - WINSIZE); n <= min(YDIM - 1, j + WINSIZE);
                 n++)
              for (m = max(0, i - WINSIZE); m <= min(XDIM - 1, i + WINSIZE);
                   m++) {
                if (dataset[IndexVect(m, n, l)] == max_value) {
                  gradient = (float)sqrt(double((m - i) * (m - i) +
                                                (n - j) * (n - j) +
                                                (l - k) * (l - k)));
                  fx += (m - i) / gradient;
                  fy += (n - j) / gradient;
                  fz += (l - k) / gradient;
                }
              }

          gradient = (float)sqrt(fx * fx + fy * fy + fz * fz);
          if (gradient == 0) {
            velocity[IndexVect(i, j, k)].x = 0;
            velocity[IndexVect(i, j, k)].y = 0;
            velocity[IndexVect(i, j, k)].z = 0;
          } else {

            fx /= gradient;
            fy /= gradient;
            fz /= gradient;

            max_value = 0;
            for (l = max(0, k - WINSIZE); l <= min(ZDIM - 1, k + WINSIZE);
                 l++)
              for (n = max(0, j - WINSIZE); n <= min(YDIM - 1, j + WINSIZE);
                   n++)
                for (m = max(0, i - WINSIZE); m <= min(XDIM - 1, i + WINSIZE);
                     m++) {
                  if (m != i || n != j || l != k) {
                    gradient =
                        (float)((fx * (m - i) + fy * (n - j) + fz * (l - k)) /
                                sqrt(double((m - i) * (m - i) +
                                            (n - j) * (n - j) +
                                            (l - k) * (l - k))));
                    if (gradient > max_value) {
                      max_x = m;
                      max_y = n;
                      max_z = l;
                      max_value = gradient;
                    }
                  }
                }

            tmp = (float)((dataset[IndexVect(max_x, max_y, max_z)] -
                           dataset[IndexVect(i, j, k)]) /
                          sqrt(double((max_x - i) * (max_x - i) +
                                      (max_y - j) * (max_y - j) +
                                      (max_z - k) * (max_z - k))));
            velocity[IndexVect(i, j, k)].x = (max_x - i) * tmp;
            velocity[IndexVect(i, j, k)].y = (max_y - j) * tmp;
            velocity[IndexVect(i, j, k)].z = (max_z - k) * tmp;
          }
        }
      }

  /*
  maxgrad = -999;
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
        gradient = velocity[IndexVect(i,j,k)].x * velocity[IndexVect(i,j,k)].x
  + velocity[IndexVect(i,j,k)].y * velocity[IndexVect(i,j,k)].y +
                   velocity[IndexVect(i,j,k)].z *
  velocity[IndexVect(i,j,k)].z; if (gradient > maxgrad) maxgrad = gradient;
      }

  maxgrad = sqrt(maxgrad);
  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
        velocity[IndexVect(i,j,k)].x = velocity[IndexVect(i,j,k)].x / maxgrad;
        velocity[IndexVect(i,j,k)].y = velocity[IndexVect(i,j,k)].y / maxgrad;
        velocity[IndexVect(i,j,k)].z = velocity[IndexVect(i,j,k)].z / maxgrad;
      }
  */
}

/* Old method */
/* //Zeyun's code
void GetGradient()
{
  int i,j,k;
  int x,y,z;
  int m,n,l;
  double total,weight;
  double *template_;
  float *tempt;
  float  back_avg;


  back_avg = (dataset[IndexVect(1,1,1)]+
              dataset[IndexVect(XDIM-2,1,1)]+
              dataset[IndexVect(1,YDIM-2,1)]+
              dataset[IndexVect(XDIM-2,YDIM-2,1)]+
              dataset[IndexVect(1,1,ZDIM-2)]+
              dataset[IndexVect(XDIM-2,1,ZDIM-2)]+
              dataset[IndexVect(1,YDIM-2,ZDIM-2)]+
              dataset[IndexVect(XDIM-2,YDIM-2,ZDIM-2)])/8.0f;


  template_ =
(double*)malloc(sizeof(double)*(2*WINDOW+1)*(2*WINDOW+1)*(2*WINDOW+1)); total
= 0; for (k=0; k<2*WINDOW+1; k++) for (j=0; j<2*WINDOW+1; j++) for (i=0;
i<2*WINDOW+1; i++) { weight =
exp(-((i-WINDOW)*(i-WINDOW)+(j-WINDOW)*(j-WINDOW)+
                       (k-WINDOW)*(k-WINDOW))/(2.0*(float)SIGMA*(float)SIGMA));
        total += weight;
        template_[(2*WINDOW+1)*(2*WINDOW+1)*k+(2*WINDOW+1)*j+i] = weight;
      }
  for (k=0; k<2*WINDOW+1; k++)
    for (j=0; j<2*WINDOW+1; j++)
      for (i=0; i<2*WINDOW+1; i++)
        template_[(2*WINDOW+1)*(2*WINDOW+1)*k+(2*WINDOW+1)*j+i] /= total;

  tempt = (float*)malloc(sizeof(float)*XDIM*YDIM*ZDIM);


  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
        if (dataset[IndexVect(i,j,k)] > back_avg+2) {
          total = 0;
          for (l=0; l<2*WINDOW+1; l++)
            for (n=0; n<2*WINDOW+1; n++)
              for (m=0; m<2*WINDOW+1; m++) {
                x = i-WINDOW+m;
                y = j-WINDOW+n;
                z = k-WINDOW+l;
                if (x < 0)
                  x = 0;
                if (y < 0)
                  y = 0;
                if (z < 0)
                  z = 0;
                if (x > XDIM-1)
                  x = XDIM-1;
                if (y > YDIM-1)
                  y = YDIM-1;
                if (z > ZDIM-1)
                  z = ZDIM-1;

                total +=
template_[(2*WINDOW+1)*(2*WINDOW+1)*l+(2*WINDOW+1)*n+m]*
                  dataset[IndexVect(x,y,z)];
              }
          tempt[IndexVect(i,j,k)] = (float)total;
        }
        else
          tempt[IndexVect(i,j,k)] = dataset[IndexVect(i,j,k)];
      }

  for (k=0; k<ZDIM; k++)
    for (j=0; j<YDIM; j++)
      for (i=0; i<XDIM; i++) {
        velocity[IndexVect(i,j,k)].x =
-0.05f*(tempt[IndexVect(min(i+1,XDIM-1),j,k)]-
                                            tempt[IndexVect(max(i-1,0),j,k)]);
        velocity[IndexVect(i,j,k)].y =
-0.05f*(tempt[IndexVect(i,min(j+1,YDIM-1),k)]-
                                            tempt[IndexVect(i,max(j-1,0),k)]);
        velocity[IndexVect(i,j,k)].z =
-0.05f*(tempt[IndexVect(i,j,min(k+1,ZDIM-1))]-
                                            tempt[IndexVect(i,j,max(k-1,0))]);
      }

   free(tempt);
   free(template_);

}
*/

/* Old method */
void GetGradient() {
  int i, j, k;
  int x, y, z;
  int m, n, l;
  double total, weight;
  double *template_;
  float *tempt;
  float back_avg;

  /* Start new declarations RSS */
  int indx1, indx2, indx3;
  /* End new declarations RSS */

  back_avg =
      (dataset[IndexVect(1, 1, 1)] + dataset[IndexVect(XDIM - 2, 1, 1)] +
       dataset[IndexVect(1, YDIM - 2, 1)] +
       dataset[IndexVect(XDIM - 2, YDIM - 2, 1)] +
       dataset[IndexVect(1, 1, ZDIM - 2)] +
       dataset[IndexVect(XDIM - 2, 1, ZDIM - 2)] +
       dataset[IndexVect(1, YDIM - 2, ZDIM - 2)] +
       dataset[IndexVect(XDIM - 2, YDIM - 2, ZDIM - 2)]) /
      8.0;

  template_ = (double *)malloc(sizeof(double) * (2 * WINDOW + 1) *
                               (2 * WINDOW + 1) * (2 * WINDOW + 1));
  total = 0;
  for (k = 0; k < 2 * WINDOW + 1; k++)
    for (j = 0; j < 2 * WINDOW + 1; j++)
      for (i = 0; i < 2 * WINDOW + 1; i++) {
        weight =
            exp(-((i - WINDOW) * (i - WINDOW) + (j - WINDOW) * (j - WINDOW) +
                  (k - WINDOW) * (k - WINDOW)) /
                (2.0 * (float)SIGMA * (float)SIGMA));
        total += weight;
        template_[(2 * WINDOW + 1) * (2 * WINDOW + 1) * k +
                  (2 * WINDOW + 1) * j + i] = weight;
      }
  for (k = 0; k < 2 * WINDOW + 1; k++)
    for (j = 0; j < 2 * WINDOW + 1; j++)
      for (i = 0; i < 2 * WINDOW + 1; i++)
        template_[(2 * WINDOW + 1) * (2 * WINDOW + 1) * k +
                  (2 * WINDOW + 1) * j + i] /= total;

  tempt = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

  /* Start modified code RSS */
  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        if (dataset[IndexVect(i, j, k)] > back_avg + 2) {
          total = 0;
          for (l = 0; l < 2 * WINDOW + 1; l++) {
            z = k - WINDOW + l;
            if (z < 0)
              z = 0;
            if (z > ZDIM - 1)
              z = ZDIM - 1;
            indx1 = (2 * WINDOW + 1) * (2 * WINDOW + 1) * l;
            for (n = 0; n < 2 * WINDOW + 1; n++) {
              y = j - WINDOW + n;
              if (y < 0)
                y = 0;
              if (y > YDIM - 1)
                y = YDIM - 1;
              indx2 = (2 * WINDOW + 1) * n;
              for (m = 0; m < 2 * WINDOW + 1; m++) {
                x = i - WINDOW + m;
                if (x < 0)
                  x = 0;
                if (x > XDIM - 1)
                  x = XDIM - 1;
                indx3 = m;

                total += template_[indx1 + indx2 + indx3] *
                         dataset[IndexVect(x, y, z)];
              }
            }
          }
          tempt[IndexVect(i, j, k)] = total;
        } else
          tempt[IndexVect(i, j, k)] = dataset[IndexVect(i, j, k)];
      }
  /* End modified code RSS */

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        velocity[IndexVect(i, j, k)].x =
            -0.05 * (tempt[IndexVect(min(i + 1, XDIM - 1), j, k)] -
                     tempt[IndexVect(max(i - 1, 0), j, k)]);
        velocity[IndexVect(i, j, k)].y =
            -0.05 * (tempt[IndexVect(i, min(j + 1, YDIM - 1), k)] -
                     tempt[IndexVect(i, max(j - 1, 0), k)]);
        velocity[IndexVect(i, j, k)].z =
            -0.05 * (tempt[IndexVect(i, j, min(k + 1, ZDIM - 1))] -
                     tempt[IndexVect(i, j, max(k - 1, 0))]);
      }

  free(tempt);
  free(template_);
}

}; // namespace SecStruct
