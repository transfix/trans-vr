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

#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))
#define DIFF_ITER 5

namespace SecStruct {

void Diffuse(float *dataset, int XDIM, int YDIM, int ZDIM) {

  float cn, cs, ce, cw, cu, cd;
  /* six neighbors: north, south, east, west, up, down */
  float delta_n, delta_s, delta_e, delta_w, delta_u, delta_d;
  float K_para = 5;
  float Lamda_para = 0.16f;
  float *tempt;
  int i, j, k, m;

  tempt = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

  for (m = 0; m < DIFF_ITER; m++) {

    if (m % 2 == 0)
      printf("Iteration = %d \n", m);

    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {

          if (j < YDIM - 1)
            delta_s =
                dataset[IndexVect(i, j + 1, k)] - dataset[IndexVect(i, j, k)];
          else
            delta_s = 0.0;
          if (j > 0)
            delta_n =
                dataset[IndexVect(i, j - 1, k)] - dataset[IndexVect(i, j, k)];
          else
            delta_n = 0.0;
          if (i < XDIM - 1)
            delta_e =
                dataset[IndexVect(i + 1, j, k)] - dataset[IndexVect(i, j, k)];
          else
            delta_e = 0.0;
          if (i > 0)
            delta_w =
                dataset[IndexVect(i - 1, j, k)] - dataset[IndexVect(i, j, k)];
          else
            delta_w = 0.0;
          if (k < ZDIM - 1)
            delta_u =
                dataset[IndexVect(i, j, k + 1)] - dataset[IndexVect(i, j, k)];
          else
            delta_u = 0.0;
          if (k > 0)
            delta_d =
                dataset[IndexVect(i, j, k - 1)] - dataset[IndexVect(i, j, k)];
          else
            delta_d = 0.0;

          cn = 1.0f / (1.0f + ((delta_n * delta_n) / (K_para * K_para)));
          cs = 1.0f / (1.0f + ((delta_s * delta_s) / (K_para * K_para)));
          ce = 1.0f / (1.0f + ((delta_e * delta_e) / (K_para * K_para)));
          cw = 1.0f / (1.0f + ((delta_w * delta_w) / (K_para * K_para)));
          cu = 1.0f / (1.0f + ((delta_u * delta_u) / (K_para * K_para)));
          cd = 1.0f / (1.0f + ((delta_d * delta_d) / (K_para * K_para)));

          tempt[IndexVect(i, j, k)] =
              dataset[IndexVect(i, j, k)] +
              Lamda_para * (cn * delta_n + cs * delta_s + ce * delta_e +
                            cw * delta_w + cu * delta_u + cd * delta_d);
        }

    for (k = 0; k < ZDIM; k++)
      for (j = 0; j < YDIM; j++)
        for (i = 0; i < XDIM; i++) {

          dataset[IndexVect(i, j, k)] = tempt[IndexVect(i, j, k)];
        }
  }

  free(tempt);
}

}; // namespace SecStruct
