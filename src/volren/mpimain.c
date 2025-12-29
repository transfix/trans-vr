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
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char *argv[]) {
  int rank = -1, size = -1, pid, tid, i;
  double start, end;
  VolRenEnv *env;
  MultiVolRenEnv *menv;
  ImageBuffer *img;
  unsigned char tile_image[TILE_RES * RGBA_SIZE];
  MPI_Status status;
  int endmsg = -1;
  FILE *fp;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size == 0) {
    fprintf(stderr, "Error: number of processors is zero\n");
    MPI_Finalize();
    return -1;
  }

  if (argc < 3)
    usage(argv[0], rank);

  if (rank == 0) {
    printf("start volren...\n");
  }

  menv = vrCreateMultiEnv();
  if (rank != 0) {
    for (i = 1; i < argc - 1; i++) {
      env = vrCreateEnvFromFile(argv[i], 0, 1);
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
  // MPI_Barrier(MPI_COMM_WORLD);

  env = menv->env[0];
  start = MPI_Wtime();
  if (rank == 0) {
    img = (ImageBuffer *)calloc(1, sizeof(ImageBuffer));
    vrCreateImage(img, env->view->pix_width, env->view->pix_height);
  }
  //  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    int ntile = env->view->pix_width * env->view->pix_height / TILE_RES;
    printf("number of tiles = %d, number of processors = %d\n", ntile, size);
    /* send out the first tile */
    for (pid = 1, tid = 0; pid < size; pid++) {
      if (tid < ntile) {
        MPI_Send(&tid, 1, MPI_INT, pid, 10, MPI_COMM_WORLD);
        tid++;
      } else {
        MPI_Send(&endmsg, 1, MPI_INT, pid, 10, MPI_COMM_WORLD);
      }
    }

    for (i = 0; i < ntile; i++) {
      MPI_Recv(tile_image, TILE_RES * RGBA_SIZE, MPI_CHAR, MPI_ANY_SOURCE,
               MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (tid < ntile) {
        MPI_Send(&tid, 1, MPI_INT, status.MPI_SOURCE, 10, MPI_COMM_WORLD);
        tid++;
      } else {
        MPI_Send(&endmsg, 1, MPI_INT, status.MPI_SOURCE, 10, MPI_COMM_WORLD);
      }
      vrCopyTile(img, status.MPI_TAG, tile_image);
    }
  } else {
    while (1) {
      MPI_Recv(&tid, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
      if (tid == -1)
        break;
      else {
        printf("processor %d rendering tile %d\n", rank, tid);
        fflush(stdout);
        vrTracingTile(menv, tid, tile_image);
        MPI_Send(tile_image, TILE_RES * RGBA_SIZE, MPI_CHAR, 0, tid,
                 MPI_COMM_WORLD);
      }
    }
  }
  end = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    vrSaveImg2PPM(img, argv[argc - 1]);
    vrDestroyImage(img);
    printf("Time : %lf\n", end - start);
  }

  vrCleanMultiEnv(menv);

  MPI_Finalize();

  return 0;
}
