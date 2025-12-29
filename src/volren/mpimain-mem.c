/*
  Copyright 2000-2002 The University of Texas at Austin

        Authors: Sanghun Park <hun@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of volren.

  volren is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  volren is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <volren/vr.h>

void usage(char *app, int rank) {
  if (rank == 0) {
    fprintf(stderr, "Usage: %s <config file> <output filename>\n", app);
    fprintf(stderr,
            "<config file> - cnf file describing rendering parameters\n");
    fprintf(stderr, "<output filename> - PPM file of the image rendered\n");
  }
  exit(-1);
}

int img_cmp(const void *a, const void *b) {
  ImageBuffer *imga = *((ImageBuffer **)a);
  ImageBuffer *imgb = *((ImageBuffer **)b);
  if (imga->dist == imgb->dist)
    return 0;
  else if (imga->dist < imgb->dist)
    return -1;
  else
    return 1;
}

int main(int argc, char *argv[]) {
  int rank, size, pid, i, j;
  double start, end;
  VolRenEnv *env;
  MultiVolRenEnv *menv;
  ImageBuffer **img;
  ImageBuffer *subimg;
  unsigned char *image;
  MPI_Status status;
  FILE *fp;
  Volume *vol;
  int sub_orig[3], sub_ext[3];
  Point3d minb, maxb, center;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc < 3)
    usage(argv[0], rank);

  if (rank == 0) {
    printf("start volren...\n");
  }

  menv = vrCreateMultiEnv();
  if (rank != 0) {
    printf("Rank: %d\n", rank);
    for (i = 1; i < argc - 1; i++) {
      env = vrCreateEnvFromFile(argv[i], rank - 1, size - 1);
      if (env == NULL || !env->valid) {
        printf("Could not create env from file %s\n", argv[i]);
        vrCleanMultiEnv(menv);
        vrCleanEnv(env);
        MPI_Finalize();
        return -1;
      }
      vrAddEnv(menv, env);
    }
  } else {
    for (i = 1; i < argc - 1; i++) {
      env = vrCreateEnv(); /* the root process does not need the volume in
                              memory */
      if (env == NULL) {
        vrCleanMultiEnv(menv);
        fprintf(stderr, "Could not allocate memory for environment\n");
        MPI_Finalize();
        return -1;
      }
      if ((fp = fopen(argv[i], "r")) == NULL) {
        fprintf(stderr, "Cannot open file %s\n", argv[i]);
        vrCleanMultiEnv(menv);
        vrCleanEnv(env);
        MPI_Finalize();
        return -1;
      }
      vrReadConfig(env, fp);
      fclose(fp);
      if (!vrVolumeExists(env))
        env->valid = 0;
      if (env == NULL || !env->valid) {
        fprintf(stderr, "Could not create env from file %s\n", argv[i]);
        vrCleanMultiEnv(menv);
        vrCleanEnv(env);
        MPI_Finalize();
        return -1;
      }
      vrAddEnv(menv, env);
    }

    printf("render mode: ");
    for (i = 0; i < menv->n_env; i++) {
      switch (menv->env[i]->rend->render_mode) {
      case RAY_CASTING:
        printf("ray casting; ");
        break;
      case ISO_SURFACE:
        printf("iso-surfacing; ");
        break;
      case ISO_AND_RAY:
        printf("iso-surfacing and ray casting; ");
        break;
      case COL_DENSITY:
        printf("color-density map; ");
        break;
      case COLDEN_AND_RAY:
        printf("color-density map and ray casting; ");
        break;
      case COLDEN_AND_ISO:
        printf("color-density map and iso-surfacing; ");
        break;
      case COLDEN_AND_RAY_AND_ISO:
        printf("color-density map, ray casting, and iso-surfacing; ");
        break;
      }
    }
    printf("\n");
  }
  fflush(stdout);

  env = menv->env[0];
  start = MPI_Wtime();
  if (rank == 0) /* allocate some memory */
  {
    img = (ImageBuffer **)calloc(size - 1, sizeof(ImageBuffer *));
    for (i = 0; i < size - 1; i++) {
      img[i] = (ImageBuffer *)calloc(1, sizeof(ImageBuffer));
      img[i]->width = env->view->pix_width;
      img[i]->height = env->view->pix_height;
    }
    image = (unsigned char *)calloc(1, sizeof(char) * env->view->pix_width *
                                           env->view->pix_height * RGBA_SIZE);
  }

  if (rank == 0) {
    printf("number of rendering processors = %d\n", size - 1);
    fflush(stdout);

    for (pid = 1; pid < size; pid++) {
      MPI_Recv(image,
               sizeof(char) * env->view->pix_width * env->view->pix_height *
                   RGBA_SIZE,
               MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
               &status);
      printf("Received image from processor %d\n", status.MPI_SOURCE);
      img[pid - 1]->buffer =
          (unsigned char *)calloc(1, sizeof(char) * env->view->pix_width *
                                         env->view->pix_height * RGBA_SIZE);
      memcpy(img[pid - 1]->buffer, image,
             sizeof(char) * env->view->pix_width * env->view->pix_height *
                 RGBA_SIZE);
      img[pid - 1]->id = status.MPI_SOURCE - 1;

      /* calculate the distance between the centerpoint of this slab and the
       * eye point for image composition ordering */
      vol = env->vol;
      memcpy(sub_orig, vol->sub_orig, 3 * sizeof(int));
      memcpy(sub_ext, vol->sub_ext, 3 * sizeof(int));
      sub_orig[2] = (sub_ext[2] = (sub_ext[2] - sub_orig[2]) / (size - 1)) *
                        (status.MPI_SOURCE - 1) +
                    sub_orig[2];
      sub_ext[2] += SUB_EXT_Z_GAP;
      for (i = 0; i < 3; i++) /* make sure we dont over extend ourselves */
      {
        if (sub_orig[i] < 0)
          sub_orig[i] = 0;
        if (sub_orig[i] + sub_ext[i] >= vol->dim[i])
          sub_ext[i] = vol->dim[i] - sub_orig[i];
      }
      minb.x = vol->orig.x + sub_orig[0] * vol->span[0];
      minb.y = vol->orig.y + sub_orig[1] * vol->span[1];
      minb.z = vol->orig.z + sub_orig[2] * vol->span[2];
      maxb.x = minb.x + sub_ext[0] * vol->span[0];
      maxb.y = minb.y + sub_ext[1] * vol->span[1];
      maxb.z = minb.z + sub_ext[2] * vol->span[2];
      center.x = (minb.x + maxb.x) / 2;
      center.y = (minb.y + maxb.y) / 2;
      center.z = (minb.z + maxb.z) / 2;
      img[pid - 1]->dist = (float)sqrt(
          (env->view->eye.x - center.x) * (env->view->eye.x - center.x) +
          (env->view->eye.y - center.y) * (env->view->eye.y - center.y) +
          (env->view->eye.z - center.z) * (env->view->eye.z - center.z));
    }
    fflush(stdout);
  } else {
    printf("Rendering image %d\n", rank - 1);
    fflush(stdout);
    subimg = vrRayTracing(menv, 1);
    MPI_Send(subimg->buffer,
             sizeof(char) * env->view->pix_width * env->view->pix_height *
                 RGBA_SIZE,
             MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    ImageBuffer *test = (ImageBuffer *)calloc(1, sizeof(ImageBuffer));
    float opac, r, g, b, a;
    // char filename[256];

    end = MPI_Wtime();
    printf("Time : %lf\n", end - start);

    /* do the composite */
    printf("Compositing...\n");
    memset(image, 0,
           env->view->pix_width * env->view->pix_height * RGBA_SIZE *
               sizeof(char));

    qsort(img, size - 1, sizeof(ImageBuffer *), img_cmp);

    /* debug the composite */
    /*for(i=0; i<size-1; i++)
      {
        sprintf(filename,"%s.%d.ppm",argv[argc-1],i);
        vrSaveImg2PPM(img[i], filename);
      }
    */

    for (i = 0; i < env->view->pix_width * env->view->pix_height *
                        sizeof(char) * RGBA_SIZE;
         i += RGBA_SIZE) {
      opac = r = g = b = a = 0.0f;
      for (j = 0; j < size - 1; j++) {

        /* front to back image composition */
        opac = img[j]->buffer[i + 3] / 255.0f;
        r = r + (1 - a) * opac * img[j]->buffer[i];
        g = g + (1 - a) * opac * img[j]->buffer[i + 1];
        b = b + (1 - a) * opac * img[j]->buffer[i + 2];
        a = a + (1 - a) * opac;

        if (a > THRESHOLD_OPC)
          j = size - 1;
      }
      /* the background color */
      r = r + (1 - a) * env->misc->back_color[0];
      g = g + (1 - a) * env->misc->back_color[1];
      b = b + (1 - a) * env->misc->back_color[2];

      image[i] = (unsigned char)MIN2(MAX_COLOR, r);
      image[i + 1] = (unsigned char)MIN2(MAX_COLOR, g);
      image[i + 2] = (unsigned char)MIN2(MAX_COLOR, b);
      image[i + 3] = (unsigned char)(a * 255);
      // printf("a == %f\n",a);
    }
    free(test);

    memcpy(img[0]->buffer, image,
           sizeof(char) * env->view->pix_width * env->view->pix_height *
               RGBA_SIZE);
    printf("Writing Image...\n");
    vrSaveImg2PPM(img[0], argv[argc - 1]);

    /* clean up */
    for (i = 0; i < size - 1; i++)
      vrDestroyImage(img[i]);
    free(img);
    free(image);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  vrCleanMultiEnv(menv);

  MPI_Finalize();

  return 0;
}
