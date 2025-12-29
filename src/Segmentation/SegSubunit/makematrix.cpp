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
#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))
#define ANGL1 1.1071487178f
#define ANGL2 2.0344439358f

namespace SegSubunit {

typedef struct {
  float x;
  float y;
  float z;
} VECTOR;

typedef struct {
  float sx;
  float sy;
  float sz;
  float ex;
  float ey;
  float ez;
} DB_VECTOR;

typedef struct {
  float trans;
  float rotat;
  float angle;
} CVM;

void GetMatrix(float, float, float, float *);
void MultiplyMatrix(float *, float *, float *);
void IndentityMatrix(float *);
void GetTransform(int, int, FILE *);
void WriteMatrix(float *, FILE *f, float *, float *);

void MakeTransformMatrix(int xdim, int ydim, int zdim, VECTOR *fvfold,
                         DB_VECTOR *lcfold, int lcfdnum, int lcaxisnum,
                         CVM *covmatrix, float *span_tmp, float *orig_tmp,
                         FILE *fp) {
  int index, tmp_index;
  float theta, phi;
  float t_theta, t_phi;
  int i, num;
  int m, n;
  float fx, fy, fz;
  float gx, gy, gz;
  float px, py, pz;
  float dx, dy, dz;
  float cx, cy, cz;
  float sx, sy, sz;
  float nx, ny, nz;
  float *matrix, distance;
  float *tmp1, *tmp2, *tmp3, *tmp4;

  matrix = (float *)malloc(sizeof(float) * 16);
  tmp1 = (float *)malloc(sizeof(float) * 16);
  tmp2 = (float *)malloc(sizeof(float) * 16);
  tmp3 = (float *)malloc(sizeof(float) * 16);
  tmp4 = (float *)malloc(sizeof(float) * 16);

  for (index = 0; index < lcaxisnum; index++) {
    IndentityMatrix(tmp2);
    tmp2[3] -= lcfold[0].ex;
    tmp2[7] -= lcfold[0].ey;
    tmp2[11] -= lcfold[0].ez;
    gx = lcfold[index * 60].sx - lcfold[index * 60].ex;
    gy = lcfold[index * 60].sy - lcfold[index * 60].ey;
    gz = lcfold[index * 60].sz - lcfold[index * 60].ez;

    if (index != 0) {
      fx = lcfold[0].sx - lcfold[0].ex;
      fy = lcfold[0].sy - lcfold[0].ey;
      fz = lcfold[0].sz - lcfold[0].ez;
      px = fy * gz - fz * gy;
      py = fz * gx - fx * gz;
      pz = fx * gy - fy * gx;
      theta = (float)atan2(py, px);
      phi = (float)atan2(pz, sqrt(px * px + py * py));
      GetMatrix(theta, phi, covmatrix[index].angle, tmp1);
      MultiplyMatrix(tmp1, tmp2, tmp3);
      for (i = 0; i < 16; i++)
        tmp2[i] = tmp3[i];
    }

    distance = (float)sqrt(gx * gx + gy * gy + gz * gz);
    dx = gx / distance;
    dy = gy / distance;
    dz = gz / distance;
    t_theta = (float)atan2(gy, gx);
    t_phi = (float)atan2(gz, sqrt(gx * gx + gy * gy));

    for (i = 0; i < lcfdnum; i++) {
      GetMatrix(t_theta, t_phi,
                covmatrix[index].rotat + i * 2.0f * PIE / (float)(lcfdnum),
                tmp1);
      MultiplyMatrix(tmp1, tmp2, tmp3);

      tmp3[3] +=
          dx * covmatrix[index].trans + lcfold[index * 60].ex - xdim / 2;
      tmp3[7] +=
          dy * covmatrix[index].trans + lcfold[index * 60].ey - ydim / 2;
      tmp3[11] +=
          dz * covmatrix[index].trans + lcfold[index * 60].ez - zdim / 2;

      tmp_index = 0;
      for (num = 0; num < 16; num++)
        matrix[num] = tmp3[num];
      matrix[3] += xdim / 2;
      matrix[7] += ydim / 2;
      matrix[11] += zdim / 2;
      if (lcfold[index * 60 + tmp_index].sx != -9999)
        WriteMatrix(matrix, fp, span_tmp, orig_tmp);
      tmp_index++;

      cx = fvfold[0].x - xdim / 2;
      cy = fvfold[0].y - ydim / 2;
      cz = fvfold[0].z - zdim / 2;
      theta = (float)atan2(cy, cx);
      phi = (float)atan2(cz, sqrt(cx * cx + cy * cy));
      for (n = 1; n < 5; n++) {
        GetMatrix(theta, phi, n * 2.0f * PIE / 5.0f, tmp4);
        MultiplyMatrix(tmp4, tmp3, matrix);
        matrix[3] += xdim / 2;
        matrix[7] += ydim / 2;
        matrix[11] += zdim / 2;
        if (lcfold[index * 60 + tmp_index].sx != -9999)
          WriteMatrix(matrix, fp, span_tmp, orig_tmp);
        tmp_index++;
      }

      for (m = 1; m < 11; m++) {
        nx = fvfold[m].x - xdim / 2;
        ny = fvfold[m].y - ydim / 2;
        nz = fvfold[m].z - zdim / 2;
        sx = nz * cy - ny * cz;
        sy = nx * cz - nz * cx;
        sz = ny * cx - nx * cy;
        theta = (float)atan2(sy, sx);
        phi = (float)atan2(sz, sqrt(sx * sx + sy * sy));
        if (m < 6) {
          GetMatrix(theta, phi, ANGL1, tmp1);
          MultiplyMatrix(tmp1, tmp3, tmp4);
        } else {
          GetMatrix(theta, phi, ANGL2, tmp1);
          MultiplyMatrix(tmp1, tmp3, tmp4);
        }

        theta = (float)atan2(ny, nx);
        phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
        for (n = 0; n < 5; n++) {
          GetMatrix(theta, phi, n * 2.0f * PIE / 5.0f + PIE / 5.0f, tmp1);
          MultiplyMatrix(tmp1, tmp4, matrix);
          matrix[3] += xdim / 2;
          matrix[7] += ydim / 2;
          matrix[11] += zdim / 2;
          if (lcfold[index * 60 + tmp_index].sx != -9999)
            WriteMatrix(matrix, fp, span_tmp, orig_tmp);
          tmp_index++;
        }
      }

      nx = fvfold[1].x - xdim / 2;
      ny = fvfold[1].y - ydim / 2;
      nz = fvfold[1].z - zdim / 2;
      sx = nz * cy - ny * cz;
      sy = nx * cz - nz * cx;
      sz = ny * cx - nx * cy;
      theta = (float)atan2(sy, sx);
      phi = (float)atan2(sz, sqrt(sx * sx + sy * sy));
      GetMatrix(theta, phi, PIE, tmp1);
      MultiplyMatrix(tmp1, tmp3, tmp4);
      for (num = 0; num < 16; num++)
        matrix[num] = tmp4[num];
      matrix[3] += xdim / 2;
      matrix[7] += ydim / 2;
      matrix[11] += zdim / 2;
      if (lcfold[index * 60 + tmp_index].sx != -9999)
        WriteMatrix(matrix, fp, span_tmp, orig_tmp);
      tmp_index++;

      nx = fvfold[11].x - xdim / 2;
      ny = fvfold[11].y - ydim / 2;
      nz = fvfold[11].z - zdim / 2;
      theta = (float)atan2(ny, nx);
      phi = (float)atan2(nz, sqrt(nx * nx + ny * ny));
      for (n = 1; n < 5; n++) {
        GetMatrix(theta, phi, n * 2.0f * PIE / 5.0f, tmp1);
        MultiplyMatrix(tmp1, tmp4, matrix);
        matrix[3] += xdim / 2;
        matrix[7] += ydim / 2;
        matrix[11] += zdim / 2;
        if (lcfold[index * 60 + tmp_index].sx != -9999)
          WriteMatrix(matrix, fp, span_tmp, orig_tmp);
        tmp_index++;
      }
    }
  }

  fclose(fp);
}

void IndentityMatrix(float *a) {
  a[0] = 1.0f;
  a[1] = 0.f;
  a[2] = 0.f;
  a[3] = 0.f;
  a[4] = 0.f;
  a[5] = 1.0f;
  a[6] = 0.f;
  a[7] = 0.f;
  a[8] = 0.f;
  a[9] = 0.f;
  a[10] = 1.0f;
  a[11] = 0.f;
  a[12] = 0.f;
  a[13] = 0.f;
  a[14] = 0.f;
  a[15] = 1.0f;
}

void GetMatrix(float theta, float phi, float alpha, float *matrix) {
  int i;
  float *a, *b;

  a = (float *)malloc(sizeof(float) * 16);
  b = (float *)malloc(sizeof(float) * 16);

  a[0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  a[1] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  a[2] = (float)-sin(0.5 * PIE - phi);
  a[3] = 0.f;
  a[4] = (float)-sin(theta);
  a[5] = (float)cos(theta);
  a[6] = 0.f;
  a[7] = 0.f;
  a[8] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  a[9] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  a[10] = (float)cos(0.5 * PIE - phi);
  a[11] = 0.f;
  a[12] = 0.f;
  a[13] = 0.f;
  a[14] = 0.f;
  a[15] = 1.0f;

  b[0] = (float)cos(alpha);
  b[1] = (float)-sin(alpha);
  b[2] = 0.f;
  b[3] = 0.f;
  b[4] = (float)sin(alpha);
  b[5] = (float)cos(alpha);
  b[6] = 0.f;
  b[7] = 0.f;
  b[8] = 0.f;
  b[9] = 0.f;
  b[10] = 1.0f;
  b[11] = 0.f;
  b[12] = 0.f;
  b[13] = 0.f;
  b[14] = 0.f;
  b[15] = 1.0f;

  MultiplyMatrix(b, a, matrix);

  for (i = 0; i < 16; i++)
    a[i] = matrix[i];

  b[0] = (float)(cos(0.5 * PIE - phi) * cos(theta));
  b[1] = (float)-sin(theta);
  b[2] = (float)(sin(0.5 * PIE - phi) * cos(theta));
  b[3] = 0.f;
  b[4] = (float)(cos(0.5 * PIE - phi) * sin(theta));
  b[5] = (float)cos(theta);
  b[6] = (float)(sin(0.5 * PIE - phi) * sin(theta));
  b[7] = 0.f;
  b[8] = (float)-sin(0.5 * PIE - phi);
  b[9] = 0.f;
  b[10] = (float)cos(0.5 * PIE - phi);
  b[11] = 0.f;
  b[12] = 0.f;
  b[13] = 0.f;
  b[14] = 0.f;
  b[15] = 1.0f;

  MultiplyMatrix(b, a, matrix);

  free(a);
  free(b);
}

void MultiplyMatrix(float *b, float *a, float *matrix) {
  matrix[0] = b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12];
  matrix[1] = b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13];
  matrix[2] = b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14];
  matrix[3] = b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15];

  matrix[4] = b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12];
  matrix[5] = b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13];
  matrix[6] = b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14];
  matrix[7] = b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15];

  matrix[8] = b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12];
  matrix[9] = b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13];
  matrix[10] = b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14];
  matrix[11] = b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15];

  matrix[12] = b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12];
  matrix[13] = b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13];
  matrix[14] = b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14];
  matrix[15] = b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15];
}

void WriteMatrix(float *matrix, FILE *fp, float *span_tmp, float *orig_tmp) {
  float x11, x12, x13, x14;
  float x21, x22, x23, x24;
  float x31, x32, x33, x34;
  float x41, x42, x43, x44;

  x11 = matrix[0] / span_tmp[0];
  x12 = matrix[1] / span_tmp[1];
  x13 = matrix[2] / span_tmp[2];
  x14 = -matrix[0] * orig_tmp[0] / span_tmp[0] -
        matrix[1] * orig_tmp[1] / span_tmp[1] -
        matrix[2] * orig_tmp[2] / span_tmp[2] + matrix[3];
  x21 = matrix[4] / span_tmp[0];
  x22 = matrix[5] / span_tmp[1];
  x23 = matrix[6] / span_tmp[2];
  x24 = -matrix[4] * orig_tmp[0] / span_tmp[0] -
        matrix[5] * orig_tmp[1] / span_tmp[1] -
        matrix[6] * orig_tmp[2] / span_tmp[2] + matrix[7];
  x31 = matrix[8] / span_tmp[0];
  x32 = matrix[9] / span_tmp[1];
  x33 = matrix[10] / span_tmp[2];
  x34 = -matrix[8] * orig_tmp[0] / span_tmp[0] -
        matrix[9] * orig_tmp[1] / span_tmp[1] -
        matrix[10] * orig_tmp[2] / span_tmp[2] + matrix[11];
  x41 = matrix[12] / span_tmp[0];
  x42 = matrix[13] / span_tmp[1];
  x43 = matrix[14] / span_tmp[2];
  x44 = -matrix[12] * orig_tmp[0] / span_tmp[0] -
        matrix[13] * orig_tmp[1] / span_tmp[1] -
        matrix[14] * orig_tmp[2] / span_tmp[2] + matrix[15];

  x11 = x11 * span_tmp[0] + x41 * orig_tmp[0];
  x12 = x12 * span_tmp[0] + x42 * orig_tmp[0];
  x13 = x13 * span_tmp[0] + x43 * orig_tmp[0];
  x14 = x14 * span_tmp[0] + x44 * orig_tmp[0];
  x21 = x21 * span_tmp[1] + x41 * orig_tmp[1];
  x22 = x22 * span_tmp[1] + x42 * orig_tmp[1];
  x23 = x23 * span_tmp[1] + x43 * orig_tmp[1];
  x24 = x24 * span_tmp[1] + x44 * orig_tmp[1];
  x31 = x31 * span_tmp[2] + x41 * orig_tmp[2];
  x32 = x32 * span_tmp[2] + x42 * orig_tmp[2];
  x33 = x33 * span_tmp[2] + x43 * orig_tmp[2];
  x34 = x34 * span_tmp[2] + x44 * orig_tmp[2];

  fprintf(fp, "%f %f %f \n", x11, x12, x13);
  fprintf(fp, "%f %f %f \n", x21, x22, x23);
  fprintf(fp, "%f %f %f \n", x31, x32, x33);
  fprintf(fp, "%f %f %f \n\n", x14, x24, x34);
}

}; // namespace SegSubunit
