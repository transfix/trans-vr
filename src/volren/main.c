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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#include <volren/vr.h>

void usage(char *app) {
#ifdef USE_OMP
  fprintf(stderr, "Using OpenMP\n");
#endif

  fprintf(stderr,
          "Usage: %s <config file> [<config file> <config file> .. ] <output "
          "filename>\n",
          app);
  fprintf(stderr,
          "<config file> - cnf file describing rendering parameters\n");
  fprintf(stderr, "<output filename> - PPM file of the image rendered\n");
  exit(-1);
}

int main(int argc, char *argv[]) {
  int i;
  MultiVolRenEnv *menv;
  VolRenEnv *env;
  ImageBuffer *img;
  double start, end;

  if (argc < 3)
    usage(argv[0]);

  printf("start volren...\n");

  start = vrGetTime();

  menv = vrCreateMultiEnv();
  for (i = 1; i < argc - 1; i++) {
    env = vrCreateEnvFromFile(argv[i], 0, 1);
    if (env == NULL || !env->valid) {
      printf("Could not create env from file %s\n", argv[i]);
      vrCleanMultiEnv(menv);
      vrCleanEnv(env);
      return -1;
    }
    vrAddEnv(menv, env);
  }
  printf("orig: %f %f %f\n", menv->metavol->orig.x, menv->metavol->orig.y,
         menv->metavol->orig.z);
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

#ifdef USE_OMP
  {
    Viewing *view;
    int ntx, nty, tid, tid_, i, *tids, tmp, num_threads, thread_num,
        chunk_size;
    MultiVolRenEnv *local_menv;
    unsigned char tile_image[TILE_RES * RGBA_SIZE];

    view = menv->env[0]->view;

    img = (ImageBuffer *)calloc(1, sizeof(ImageBuffer));
    vrCreateImage(img, view->pix_width, view->pix_height);
    ntx = img->width / TILE_SIZE;
    nty = img->height / TILE_SIZE;

    /* randomized load balancing... */
    tids = (int *)malloc(sizeof(int) * ntx * nty);
    for (i = 0; i < ntx * nty; i++)
      tids[i] = i;
    for (i = 0; i < ntx * nty; i++) {
      tid = rand() % (ntx * nty);
      tid_ = ntx * nty - tid;
      tmp = tids[tid];
      tids[tid] = tids[tid_];
      tids[tid_] = tmp;
    }

#pragma omp parallel default(shared) private(                                \
        tid, i, tile_image, thread_num, num_threads, chunk_size, local_menv)
    {
      thread_num = omp_get_thread_num();
      num_threads = omp_get_num_threads();
      chunk_size = (ntx * nty) / num_threads;
      local_menv =
          vrCopyMenv(menv); /* maybe fix this later, might cause memory leaks,
                               though right now it wont matter */
      for (i = chunk_size * thread_num; i < chunk_size * (thread_num + 1);
           i++) {
        tid = tids[i];
        vrTracingTile(local_menv, tid, tile_image);
        vrCopyTile(img, tid, tile_image);

        printf("tile %d \n", tid);
        fflush(stdout);
      }
      free(local_menv);
    }

    free(tids);
  }
#else
  img = vrRayTracing(menv, 0); /* single threaded */
#endif

  vrSaveImg2PPM(img, argv[argc - 1]);
  vrDestroyImage(img);

  end = vrGetTime();
  printf("running time = %f\n", end - start);

  vrCleanMultiEnv(menv);
  return 0;
}
