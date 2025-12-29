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
#include <sys/stat.h>
#include <sys/types.h>

#define _LITTLE_ENDIAN 1

#define IndexVect(i, j, k) ((k) * XDIM * YDIM + (j) * XDIM + (i))
#define max2(x, y) ((x > y) ? (x) : (y))
#define min2(x, y) ((x < y) ? (x) : (y))
#define PIE 3.1415926f
#define ANGL1 1.107149f
#define ANGL2 2.034444f

namespace SegCapsid {

typedef struct {
  float x;
  float y;
  float z;
} VECTOR;

static int XDIM, YDIM, ZDIM;
static float maxraw;
static float minraw;

static float minext[3], maxext[3];
static int nverts, ncells;
static unsigned int dim[3];
static float orig[3], span[3];

void swap_buffer(char *buffer, int count, int typesize);

void get_header(int *xd, int *yd, int *zd, float *orig_t, FILE *fp) {
  size_t fread_return = 0;

  /* reading RAWIV header */
  fread_return = fread(minext, sizeof(float), 3, fp);
  fread_return = fread(maxext, sizeof(float), 3, fp);
  fread_return = fread(&nverts, sizeof(int), 1, fp);
  fread_return = fread(&ncells, sizeof(int), 1, fp);
  fread_return = fread(dim, sizeof(unsigned int), 3, fp);
  fread_return = fread(orig, sizeof(float), 3, fp);
  fread_return = fread(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  *xd = dim[0];
  *yd = dim[1];
  *zd = dim[2];
  orig_t[0] = minext[0]; // orig[0];
  orig_t[1] = minext[1]; // orig[1];
  orig_t[2] = minext[2]; // orig[2];
}

void read_data(int *xd, int *yd, int *zd, float **data, float *span_t,
               float *orig_t, char *input_name, int halfsphere,
               int background) {
  float c_float;
  unsigned char c_unchar;
  unsigned short c_unshort;
  int i, j, k;
  float *dataset;

  struct stat filestat;
  size_t size[3];
  int datatype = 0;
  int found;
  FILE *fp;

  size_t fread_return = 0;

  if ((fp = fopen(input_name, "rb")) == NULL) {
    printf("read error...\n");
    exit(0);
  }
  stat(input_name, &filestat);

  /* reading RAWIV header */
  fread_return = fread(minext, sizeof(float), 3, fp);
  fread_return = fread(maxext, sizeof(float), 3, fp);
  fread_return = fread(&nverts, sizeof(int), 1, fp);
  fread_return = fread(&ncells, sizeof(int), 1, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
#endif

  size[0] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
            nverts * sizeof(unsigned char);
  size[1] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
            nverts * sizeof(unsigned short);
  size[2] = 12 * sizeof(float) + 2 * sizeof(int) + 3 * sizeof(unsigned int) +
            nverts * sizeof(float);

  found = 0;
  for (i = 0; i < 3; i++)
    if (size[i] == (unsigned int)filestat.st_size) {
      if (found == 0) {
        datatype = i;
        found = 1;
      }
    }
  if (found == 0) {
    printf("Corrupted file or unsupported dataset type\n");
    exit(5);
  }

  fread_return = fread(dim, sizeof(unsigned int), 3, fp);
  fread_return = fread(orig, sizeof(float), 3, fp);
  fread_return = fread(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  span_t[0] = span[0];
  span_t[1] = span[1];
  span_t[2] = span[2];
  orig_t[0] = minext[0]; // orig[0];
  orig_t[1] = minext[1]; // orig[1];
  orig_t[2] = minext[2]; // orig[2];

  XDIM = dim[0];
  YDIM = dim[1];

  if (halfsphere == 0) {
    ZDIM = dim[2];
    dataset = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

    maxraw = -99999999.f;
    minraw = 99999999.f;

    if (datatype == 0) {
      printf("data type: unsigned char \n");
      for (i = 0; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++) {
            fread_return = fread(&c_unchar, sizeof(unsigned char), 1, fp);
            dataset[IndexVect(k, j, i)] = (float)c_unchar;

            if (c_unchar > maxraw)
              maxraw = c_unchar;
            if (c_unchar < minraw)
              minraw = c_unchar;
          }
    } else if (datatype == 1) {
      printf("data type: unsigned short \n");
      for (i = 0; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++) {
            fread_return = fread(&c_unshort, sizeof(unsigned short), 1, fp);
#ifdef _LITTLE_ENDIAN
            swap_buffer((char *)&c_unshort, 1, sizeof(unsigned short));
#endif
            dataset[IndexVect(k, j, i)] = (float)c_unshort;

            if (c_unshort > maxraw)
              maxraw = c_unshort;
            if (c_unshort < minraw)
              minraw = c_unshort;
          }
    } else if (datatype == 2) {
      printf("data type: float \n");
      for (i = 0; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++) {
            fread_return = fread(&c_float, sizeof(float), 1, fp);
#ifdef _LITTLE_ENDIAN
            swap_buffer((char *)&c_float, 1, sizeof(float));
#endif
            dataset[IndexVect(k, j, i)] = c_float;

            if (c_float > maxraw)
              maxraw = c_float;
            if (c_float < minraw)
              minraw = c_float;
          }
    }

    else {
      printf("error\n");
      fclose(fp);
      exit(1);
    }

    fclose(fp);

    if (background) {
      for (i = 0; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++)
            dataset[IndexVect(k, j, i)] =
                255 * (dataset[IndexVect(k, j, i)] - minraw) /
                (maxraw - minraw);
    } else {
      for (i = 0; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++)
            dataset[IndexVect(k, j, i)] =
                255 - 255 * (dataset[IndexVect(k, j, i)] - minraw) /
                          (maxraw - minraw);
    }

    printf("minimum = %f,   maximum = %f \n", minraw, maxraw);
  } else {
    ZDIM = 2 * dim[2] - 1;
    dim[2] = ZDIM;
    maxext[2] = minext[2] + 2 * (maxext[2] - minext[2]);
    nverts = 2 * nverts - XDIM * YDIM;
    ncells = 2 * ncells;

    dataset = (float *)malloc(sizeof(float) * XDIM * YDIM * ZDIM);

    maxraw = -99999999.f;
    minraw = 99999999.f;

    if (datatype == 0) {
      printf("data type: unsigned char \n");
      for (i = (ZDIM - 1) / 2; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++) {
            fread_return = fread(&c_unchar, sizeof(unsigned char), 1, fp);
            dataset[IndexVect(k, j, i)] = (float)c_unchar;

            if (c_unchar > maxraw)
              maxraw = c_unchar;
            if (c_unchar < minraw)
              minraw = c_unchar;
          }
    } else if (datatype == 1) {
      printf("data type: unsigned short \n");
      for (i = (ZDIM - 1) / 2; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++) {
            fread_return = fread(&c_unshort, sizeof(unsigned short), 1, fp);
#ifdef _LITTLE_ENDIAN
            swap_buffer((char *)&c_unshort, 1, sizeof(unsigned short));
#endif
            dataset[IndexVect(k, j, i)] = (float)c_unshort;

            if (c_unshort > maxraw)
              maxraw = c_unshort;
            if (c_unshort < minraw)
              minraw = c_unshort;
          }
    } else if (datatype == 2) {
      printf("data type: float \n");
      for (i = (ZDIM - 1) / 2; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++) {
            fread_return = fread(&c_float, sizeof(float), 1, fp);
#ifdef _LITTLE_ENDIAN
            swap_buffer((char *)&c_float, 1, sizeof(float));
#endif
            dataset[IndexVect(k, j, i)] = c_float;

            if (c_float > maxraw)
              maxraw = c_float;
            if (c_float < minraw)
              minraw = c_float;
          }
    }

    else {
      printf("error\n");
      fclose(fp);
      exit(1);
    }

    fclose(fp);

    if (background) {
      for (i = (ZDIM - 1) / 2; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++)
            dataset[IndexVect(k, j, i)] =
                255 * (dataset[IndexVect(k, j, i)] - minraw) /
                (maxraw - minraw);
    } else {
      for (i = (ZDIM - 1) / 2; i < ZDIM; i++)
        for (j = 0; j < YDIM; j++)
          for (k = 0; k < XDIM; k++)
            dataset[IndexVect(k, j, i)] =
                255 - 255 * (dataset[IndexVect(k, j, i)] - minraw) /
                          (maxraw - minraw);
    }

    printf("minimum = %f,   maximum = %f \n", minraw, maxraw);

    for (i = 0; i < (ZDIM - 1) / 2; i++)
      for (j = 0; j < YDIM; j++)
        for (k = 0; k < XDIM; k++)
          dataset[IndexVect(k, j, i)] =
              dataset[IndexVect(k, YDIM - 1 - j, ZDIM - 1 - i)];
  }

  *xd = XDIM;
  *yd = YDIM;
  *zd = ZDIM;
  *data = dataset;
}

void write_rawiv_char(unsigned char *result, FILE *fp) {
  int i, j, k;
  unsigned char c_char;

  size_t fwrite_return = 0;

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  fwrite_return = fwrite(minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {

        c_char = result[IndexVect(i, j, k)];

#ifdef _LITTLE_ENDIAN
        swap_buffer((char *)&c_char, 1, sizeof(unsigned char));
#endif
        fwrite_return = fwrite(&c_char, sizeof(unsigned char), 1, fp);
#ifdef _LITTLE_ENDIAN
        swap_buffer((char *)&c_char, 1, sizeof(unsigned char));
#endif
      }

  fclose(fp);
}

void write_rawiv_short(unsigned short *result, FILE *fp) {
  int i, j, k;
  unsigned short c_short;

  size_t fwrite_return = 0;

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  fwrite_return = fwrite(minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {

        c_short = result[IndexVect(i, j, k)];

#ifdef _LITTLE_ENDIAN
        swap_buffer((char *)&c_short, 1, sizeof(unsigned short));
#endif
        fwrite_return = fwrite(&c_short, sizeof(unsigned short), 1, fp);
#ifdef _LITTLE_ENDIAN
        swap_buffer((char *)&c_short, 1, sizeof(unsigned short));
#endif
      }

  fclose(fp);
}

void write_rawiv_float(float *result, FILE *fp) {
  int i, j, k;
  float c_float;

  size_t fwrite_return = 0;

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  fwrite_return = fwrite(minext, sizeof(float), 3, fp);
  fwrite_return = fwrite(maxext, sizeof(float), 3, fp);
  fwrite_return = fwrite(&nverts, sizeof(int), 1, fp);
  fwrite_return = fwrite(&ncells, sizeof(int), 1, fp);
  fwrite_return = fwrite(dim, sizeof(unsigned int), 3, fp);
  fwrite_return = fwrite(orig, sizeof(float), 3, fp);
  fwrite_return = fwrite(span, sizeof(float), 3, fp);
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)minext, 3, sizeof(float));
  swap_buffer((char *)maxext, 3, sizeof(float));
  swap_buffer((char *)&nverts, 1, sizeof(int));
  swap_buffer((char *)&ncells, 1, sizeof(int));
  swap_buffer((char *)dim, 3, sizeof(unsigned int));
  swap_buffer((char *)orig, 3, sizeof(float));
  swap_buffer((char *)span, 3, sizeof(float));
#endif

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {

        c_float = result[IndexVect(i, j, k)];

#ifdef _LITTLE_ENDIAN
        swap_buffer((char *)&c_float, 1, sizeof(float));
#endif
        fwrite_return = fwrite(&c_float, sizeof(float), 1, fp);
#ifdef _LITTLE_ENDIAN
        swap_buffer((char *)&c_float, 1, sizeof(float));
#endif
      }

  fclose(fp);
}

void write_rawv(float *data, unsigned short *classify, int numfd, int numfd5,
                int random_color, FILE *fp) {
  int i, j, k;
  int m;
  unsigned short c_unshort;
  unsigned char c_unchar;
  VECTOR color[5000];
  float r[10], g[10], b[10];
  float red, green, blue;
  int color_index;
  int numfold;
  unsigned int xdim, ydim, zdim;

  unsigned int MagicNumW = 0xBAADBEEF;
  unsigned int NumTimeStep, NumVariable;
  float MinXYZT[4], MaxXYZT[4];
  unsigned char VariableType[100];
  char *VariableName[100];

  size_t fwrite_return = 0;

  NumTimeStep = 1;
  NumVariable = 4;
  MinXYZT[0] = 0;
  MinXYZT[1] = 0;
  MinXYZT[2] = 0;
  MinXYZT[3] = 0;
  MaxXYZT[0] = XDIM - 1.f;
  MaxXYZT[1] = YDIM - 1.f;
  MaxXYZT[2] = ZDIM - 1.f;
  MaxXYZT[3] = 1;
  VariableType[0] = 1;
  VariableName[0] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[0], "red");
  VariableType[1] = 1;
  VariableName[1] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[1], "green");
  VariableType[2] = 1;
  VariableName[2] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[2], "blue");
  VariableType[3] = 1;
  VariableName[3] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[3], "alpha");

  xdim = XDIM;
  ydim = YDIM;
  zdim = ZDIM;
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)&MagicNumW, 1, sizeof(unsigned int));
  swap_buffer((char *)&xdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&ydim, 1, sizeof(unsigned int));
  swap_buffer((char *)&zdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumTimeStep, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumVariable, 1, sizeof(unsigned int));
  swap_buffer((char *)MinXYZT, 4, sizeof(float));
  swap_buffer((char *)MaxXYZT, 4, sizeof(float));
#endif

  fwrite_return = fwrite(&MagicNumW, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&xdim, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&ydim, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&zdim, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&NumTimeStep, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&NumVariable, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(MinXYZT, sizeof(float), 4, fp);
  fwrite_return = fwrite(MaxXYZT, sizeof(float), 4, fp);

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)&MagicNumW, 1, sizeof(unsigned int));
  swap_buffer((char *)&xdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&ydim, 1, sizeof(unsigned int));
  swap_buffer((char *)&zdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumTimeStep, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumVariable, 1, sizeof(unsigned int));
  swap_buffer((char *)MinXYZT, 4, sizeof(float));
  swap_buffer((char *)MaxXYZT, 4, sizeof(float));
#endif

  for (m = 0; m < 4; m++) {
    fwrite_return = fwrite(&VariableType[m], sizeof(unsigned char), 1, fp);
    fwrite_return = fwrite(VariableName[m], sizeof(unsigned char), 64, fp);
  }

  r[0] = 255;
  g[0] = 0;
  b[0] = 0;
  r[1] = 0;
  g[1] = 255;
  b[1] = 0;
  r[2] = 0;
  g[2] = 0;
  b[2] = 255;
  r[3] = 255;
  g[3] = 255;
  b[3] = 0;
  r[4] = 0;
  g[4] = 255;
  b[4] = 255;
  r[5] = 255;
  g[5] = 0;
  b[5] = 255;
  r[6] = 255;
  g[6] = 200;
  b[6] = 100;
  r[7] = 100;
  g[7] = 255;
  b[7] = 200;
  r[8] = 200;
  g[8] = 100;
  b[8] = 255;
  r[9] = 200;
  g[9] = 200;
  b[9] = 100;

  if (numfd5 > 0) {
    numfold = numfd * 60 + 12;
    color_index = 1;
    for (k = 0; k < 12; k++) {
      color[k].x = r[0];
      color[k].y = g[0];
      color[k].z = b[0];
    }
  } else {
    numfold = numfd * 60;
    color_index = 0;
  }
  i = numfd;
  j = 0;
  if (random_color == 0) {
    while (j < i) {

      if (color_index < 10) {
        red = r[color_index];
        green = g[color_index];
        blue = b[color_index];
        color_index++;
      } else {
        red = (float)(255.0 * rand() / (float)RAND_MAX);
        green = (float)(255.0 * rand() / (float)RAND_MAX);
        blue = (float)(255.0 * rand() / (float)RAND_MAX);
      }

      for (k = j * 60 + 12 * numfd5; k < j * 60 + 60 + 12 * numfd5; k++) {
        color[k].x = red;
        color[k].y = green;
        color[k].z = blue;
      }
      j++;
    }
  } else {
    while (j < i) {
      for (k = 12 * numfd5 + j * 60; k < j * 60 + 60 + 12 * numfd5; k++) {
        color[k].x = (float)(255.0 * rand() / (float)RAND_MAX);
        color[k].y = (float)(255.0 * rand() / (float)RAND_MAX);
        color[k].z = (float)(255.0 * rand() / (float)RAND_MAX);
      }
      j++;
    }
  }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < numfold && data[IndexVect(i, j, k)] > 0) {
          c_unchar = (unsigned char)(color[c_unshort].x);
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < numfold && data[IndexVect(i, j, k)] > 0) {
          c_unchar = (unsigned char)(color[c_unshort].y);
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < numfold && data[IndexVect(i, j, k)] > 0) {
          c_unchar = (unsigned char)(color[c_unshort].z);
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < numfold && data[IndexVect(i, j, k)] > 0) {
          c_unchar = 255;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  fclose(fp);
}

void write_asym_rawv(float *data, unsigned short *classify, FILE *fp) {
  int i, j, k;
  int m;
  unsigned short c_unshort;
  unsigned char c_unchar;
  VECTOR color[60];
  unsigned int xdim, ydim, zdim;

  unsigned int MagicNumW = 0xBAADBEEF;
  unsigned int NumTimeStep, NumVariable;
  float MinXYZT[4], MaxXYZT[4];
  unsigned char VariableType[100];
  char *VariableName[100];

  size_t fwrite_return = 0;

  NumTimeStep = 1;
  NumVariable = 4;
  MinXYZT[0] = 0;
  MinXYZT[1] = 0;
  MinXYZT[2] = 0;
  MinXYZT[3] = 0;
  MaxXYZT[0] = XDIM - 1.f;
  MaxXYZT[1] = YDIM - 1.f;
  MaxXYZT[2] = ZDIM - 1.f;
  MaxXYZT[3] = 1;
  VariableType[0] = 1;
  VariableName[0] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[0], "red");
  VariableType[1] = 1;
  VariableName[1] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[1], "green");
  VariableType[2] = 1;
  VariableName[2] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[2], "blue");
  VariableType[3] = 1;
  VariableName[3] = (char *)malloc(sizeof(char) * 64);
  strcpy(VariableName[3], "alpha");

  xdim = XDIM;
  ydim = YDIM;
  zdim = ZDIM;
#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)&MagicNumW, 1, sizeof(unsigned int));
  swap_buffer((char *)&xdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&ydim, 1, sizeof(unsigned int));
  swap_buffer((char *)&zdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumTimeStep, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumVariable, 1, sizeof(unsigned int));
  swap_buffer((char *)MinXYZT, 4, sizeof(float));
  swap_buffer((char *)MaxXYZT, 4, sizeof(float));
#endif

  fwrite_return = fwrite(&MagicNumW, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&xdim, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&ydim, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&zdim, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&NumTimeStep, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(&NumVariable, sizeof(unsigned int), 1, fp);
  fwrite_return = fwrite(MinXYZT, sizeof(float), 4, fp);
  fwrite_return = fwrite(MaxXYZT, sizeof(float), 4, fp);

#ifdef _LITTLE_ENDIAN
  swap_buffer((char *)&MagicNumW, 1, sizeof(unsigned int));
  swap_buffer((char *)&xdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&ydim, 1, sizeof(unsigned int));
  swap_buffer((char *)&zdim, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumTimeStep, 1, sizeof(unsigned int));
  swap_buffer((char *)&NumVariable, 1, sizeof(unsigned int));
  swap_buffer((char *)MinXYZT, 4, sizeof(float));
  swap_buffer((char *)MaxXYZT, 4, sizeof(float));
#endif

  for (m = 0; m < 4; m++) {
    fwrite_return = fwrite(&VariableType[m], sizeof(unsigned char), 1, fp);
    fwrite_return = fwrite(VariableName[m], sizeof(unsigned char), 64, fp);
  }

  for (k = 0; k < 60; k++) {
    color[k].x = (float)(255.0 * rand() / (float)RAND_MAX);
    color[k].y = (float)(255.0 * rand() / (float)RAND_MAX);
    color[k].z = (float)(255.0 * rand() / (float)RAND_MAX);
  }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < 255 && data[IndexVect(i, j, k)] > 0) {
          c_unchar = (unsigned char)(color[c_unshort].x);
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < 255 && data[IndexVect(i, j, k)] > 0) {
          c_unchar = (unsigned char)(color[c_unshort].y);
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < 255 && data[IndexVect(i, j, k)] > 0) {
          c_unchar = (unsigned char)(color[c_unshort].z);
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  for (k = 0; k < ZDIM; k++)
    for (j = 0; j < YDIM; j++)
      for (i = 0; i < XDIM; i++) {
        c_unshort = classify[IndexVect(i, j, k)];
        if (c_unshort < 255 && data[IndexVect(i, j, k)] > 0) {
          c_unchar = 255;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        } else {
          c_unchar = 0;
          fwrite_return = fwrite(&c_unchar, sizeof(unsigned char), 1, fp);
        }
      }

  fclose(fp);
}

void swap_buffer(char *buffer, int count, int typesize) {
  char sbuf[4];
  int i;
  int temp = 1;
  unsigned char *chartempf = (unsigned char *)&temp;
  if (chartempf[0] > '\0') {

    // swapping isn't necessary on single byte data
    if (typesize == 1)
      return;

    for (i = 0; i < count; i++) {
      memcpy(sbuf, buffer + (i * typesize), typesize);

      switch (typesize) {
      case 2: {
        buffer[i * typesize] = sbuf[1];
        buffer[i * typesize + 1] = sbuf[0];
        break;
      }
      case 4: {
        buffer[i * typesize] = sbuf[3];
        buffer[i * typesize + 1] = sbuf[2];
        buffer[i * typesize + 2] = sbuf[1];
        buffer[i * typesize + 3] = sbuf[0];
        break;
      }
      default:
        break;
      }
    }
  }
}

}; // namespace SegCapsid
