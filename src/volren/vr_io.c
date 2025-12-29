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
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
/* #include <unistd.h> */

#include <volren/vr.h>

void vrReadConfig(VolRenEnv *env, FILE *fp) {
  char name[MAX_STRING];
  char type[MAX_STRING];
  float span[3];
  float x, y, z;
  int dim[3];
  int org[3], ext[3];
  int win_width, win_height, pix_width, pix_height;
  int persp;
  float fov;
  int num, idx;
  float opacity, isoval;
  int start, up, down, end;
  int kar, kag, kab, kdr, kdg, kdb, ksr, ksg, ksb, kn;
  int r, g, b, a, den;
  int render_mode, cutplane, light_both, color_channel;
  int step_size;
  int grad_ramp0, grad_ramp1, grad_ramp2;

  if (env == NULL)
    return;

  /* Read Volume */
  fgets(name, MAX_STRING, fp);
  fscanf(fp, "%s", type);
  fscanf(fp, "%f %f %f", span, span + 1, span + 2);
  fscanf(fp, "%d %d %d", dim, dim + 1, dim + 2);
  fscanf(fp, "%d %d %d %d %d %d", &org[0], &org[1], &org[2], &ext[0], &ext[1],
         &ext[2]);
  vrSetVolumeInfo(env->vol, name, type, span, dim);
  vrSetSubVolume(env->vol, org, ext);

  /* Read Viewing Parameters*/
  fscanf(fp, "%d %f", &persp, &fov);
  vrSetPerspective(env->view, persp, fov);
  fscanf(fp, "%f %f %f", &x, &y, &z);
  vrSetEyePoint(env->view, x, y, z);
  fscanf(fp, "%f %f %f", &x, &y, &z);
  vrSetUpVector(env->view, x, y, z);
  fscanf(fp, "%f %f %f", &x, &y, &z);
  vrSetPlaneNormVector(env->view, x, y, z);
  fscanf(fp, "%d %d", &win_width, &win_height);
  vrSetViewWindowSize(env->view, win_width, win_height);
  fscanf(fp, "%d %d", &pix_width, &pix_height);
  vrSetPixelWindowSize(env->view, pix_width, pix_height);
  vrComputeView(env->view);

  /* Materials */
  fscanf(fp, "%d", &num);
  assert(num <= MAX_MATERIAL);
  vrInitMaterial(env->rend, env->vol);
  for (idx = 0; idx < num; idx++) {
    Shading shade;
    fscanf(fp, "%f %d %d %d %d", &opacity, &start, &up, &down, &end);
    fscanf(fp, "%d %d %d %d %d %d %d %d %d %d", &kar, &kag, &kab, &kdr, &kdg,
           &kdb, &ksr, &ksg, &ksb, &kn);
    shade.ambient.r = kar;
    shade.ambient.g = kag;
    shade.ambient.b = kab;
    shade.diffuse.r = kdr;
    shade.diffuse.g = kdg;
    shade.diffuse.b = kdb;
    shade.specular.r = ksr;
    shade.specular.g = ksg;
    shade.specular.b = ksb;
    shade.shining = kn;
    vrAddMaterial(env->rend, env->vol, &shade, opacity, start, up, down, end);
  }

  /* Isosurfaces */
  fscanf(fp, "%d", &num);
  assert(num <= MAX_SURFACE);
  for (idx = 0; idx < num; idx++) {
    Shading shade;
    fscanf(fp, "%f %f", &opacity, &isoval);
    fscanf(fp, "%d %d %d %d %d %d %d %d %d %d", &kar, &kag, &kab, &kdr, &kdg,
           &kdb, &ksr, &ksg, &ksb, &kn);
    shade.ambient.r = kar;
    shade.ambient.g = kag;
    shade.ambient.b = kab;
    shade.diffuse.r = kdr;
    shade.diffuse.g = kdg;
    shade.diffuse.b = kdb;
    shade.specular.r = ksr;
    shade.specular.g = ksg;
    shade.specular.b = ksb;
    shade.shining = kn;
    vrAddIsoSurface(env->rend, &shade, opacity, isoval);
  }

  /* Color-Density Table */
  fscanf(fp, "%d", &num);
  assert(num <= MAX_COLDEN_RAMP);
  for (idx = 0; idx < num; idx++) {
    fscanf(fp, "%d %d %d %d %d", &den, &r, &g, &b, &a);
    vrAddColDenRamp(env->rend, den, r, g, b, a);
  }
  vrComputeColDen(env->rend, env->vol);

  /* Lights */
  fscanf(fp, "%d", &num);
  assert(num <= MAX_LIGHT);
  for (idx = 0; idx < num; idx++) {
    fscanf(fp, "%d %d %d", &r, &g, &b);
    fscanf(fp, "%f %f %f", &x, &y, &z);
    vrAddLight(env->rend, r, g, b, x, y, z);
  }

  /* Cutting Planes */
  fscanf(fp, "%d", &num);
  assert(num < MAX_PLANE);
  for (idx = 0; idx < num; idx++) {
    Point3d orig;
    Vector3d norm;
    fscanf(fp, "%f %f %f", &x, &y, &z);
    orig.x = x;
    orig.y = y;
    orig.z = z;
    fscanf(fp, "%f %f %f", &x, &y, &z);
    norm.x = x;
    norm.y = y;
    norm.z = z;
    vrAddCuttingPlane(env->rend, &orig, &norm);
  }

  /* Rendering modes */
  fscanf(fp, "%d %d %d %d", &render_mode, &cutplane, &light_both,
         &color_channel);
  vrSetRenderingMode(env->rend, render_mode);
  vrToggleCuttingPlane(env->rend, cutplane);
  vrToggleLightBoth(env->rend, light_both);
  vrSetColorMode(env->rend, color_channel);

  /* RayMisc */
  fscanf(fp, "%d", &step_size);
  vrSetStepSize(env, step_size);
  fscanf(fp, "%d %d %d", &grad_ramp0, &grad_ramp1, &grad_ramp2);
  vrSetGradientTable(env->misc, grad_ramp0, grad_ramp1, grad_ramp2);
  fscanf(fp, "%d %d %d", &r, &g, &b);
  vrSetBackColor(env->misc, r, g, b);
}

void vrWriteConfig(VolRenEnv *env, FILE *fp) {
  Volume *vol;
  Viewing *view;
  Rendering *rend;
  Shading shade;
  RayMisc *misc;
  int i, orig[3], ext[3];

  vol = env->vol;
  view = env->view;
  rend = env->rend;
  misc = env->misc;

  /* Write volume info */
  fprintf(fp, "%s\n", env->vol->fname);
  switch (env->vol->orig_type) {
  case TYPE_8:
    fprintf(fp, "8\n");
    break;
  case TYPE_12:
    fprintf(fp, "12\n");
    break;
  case TYPE_16:
    fprintf(fp, "16\n");
    break;
  case TYPE_SLC8:
    fprintf(fp, "SLC8\n");
    break;
  case TYPE_SLC12:
    fprintf(fp, "SLC12\n");
    break;
  case TYPE_VTM12:
    fprintf(fp, "VTM12\n");
    break;
  case TYPE_VTM15:
    fprintf(fp, "VTM15\n");
    break;
  case TYPE_VSTL:
    fprintf(fp, "VSTL\n");
    break;
  case TYPE_FLOAT:
    fprintf(fp, "FLOAT\n");
    break;
  case TYPE_RAWV:
    fprintf(fp, "RAWV\n");
    break;
  default:
    printf("env->vol->orig_type == %d\n", env->vol->orig_type);
    assert(0);
  }
  fprintf(fp, "%f %f %f\n", vol->span[0], vol->span[1], vol->span[2]);
  fprintf(fp, "%d %d %d\n", vol->dim[0], vol->dim[1], vol->dim[2]);
  orig[0] = (int)((vol->minb.x - vol->orig.x) / vol->span[0]);
  orig[1] = (int)((vol->minb.y - vol->orig.y) / vol->span[1]);
  orig[2] = (int)((vol->minb.z - vol->orig.z) / vol->span[2]);
  ext[0] = (int)((vol->maxb.x - vol->minb.x) / vol->span[0]);
  ext[1] = (int)((vol->maxb.y - vol->minb.y) / vol->span[1]);
  ext[2] = (int)((vol->maxb.z - vol->minb.z) / vol->span[2]);
  fprintf(fp, "%d %d %d %d %d %d\n", orig[0], orig[1], orig[2], ext[0],
          ext[1], ext[2]);

  fprintf(fp, "\n");

  /* Write viewing parameters */
  fprintf(fp, "%d %f\n", view->persp, view->fov * (180 / PI));
  fprintf(fp, "%f %f %f\n", view->eye.x, view->eye.y, view->eye.z);
  fprintf(fp, "%f %f %f\n", view->vup.x, view->vup.y, view->vup.z);
  fprintf(fp, "%f %f %f\n", view->vpn.x, view->vpn.y, view->vpn.z);
  fprintf(fp, "%d %d\n", view->win_width, view->win_height);
  fprintf(fp, "%d %d\n", view->pix_width, view->pix_height);

  fprintf(fp, "\n");

  /* Materials */
  fprintf(fp, "%d\n", rend->n_material);
  for (i = 0; i < rend->n_material; i++) {
    vrCopyShading(&shade, &(rend->mat[i].shading));
    fprintf(fp, "%f %d %d %d %d\n", rend->mat[i].opacity, rend->mat[i].start,
            rend->mat[i].up, rend->mat[i].down, rend->mat[i].end);
    fprintf(fp, "%d %d %d %d %d %d %d %d %d %d\n", shade.ambient.r,
            shade.ambient.g, shade.ambient.b, shade.diffuse.r,
            shade.diffuse.g, shade.diffuse.b, shade.specular.r,
            shade.specular.g, shade.specular.b, shade.shining);
  }

  fprintf(fp, "\n");

  /* Isosurfaces */
  fprintf(fp, "%d\n", rend->n_surface);
  for (i = 0; i < rend->n_surface; i++) {
    vrCopyShading(&shade, &(rend->surf[i].shading));
    fprintf(fp, "%f %f\n", rend->surf[i].opacity, rend->surf[i].value);
    fprintf(fp, "%d %d %d %d %d %d %d %d %d %d\n", shade.ambient.r,
            shade.ambient.g, shade.ambient.b, shade.diffuse.r,
            shade.diffuse.g, shade.diffuse.b, shade.specular.r,
            shade.specular.g, shade.specular.b, shade.shining);
  }

  fprintf(fp, "\n");

  /* Color-Density Table */
  fprintf(fp, "%d\n", rend->n_colden);
  for (i = 0; i < rend->n_colden; i++)
    fprintf(fp, "%d %d %d %d %d\n", rend->coldenramp[i].d,
            rend->coldenramp[i].r, rend->coldenramp[i].g,
            rend->coldenramp[i].b, rend->coldenramp[i].a);

  fprintf(fp, "\n");

  /* Lights */
  fprintf(fp, "%d\n", rend->n_light);
  for (i = 0; i < rend->n_light; i++) {
    fprintf(fp, "%d %d %d\n", rend->light[i].color.r, rend->light[i].color.g,
            rend->light[i].color.b);
    fprintf(fp, "%f %f %f\n", rend->light[i].dir.x, rend->light[i].dir.y,
            rend->light[i].dir.z);
  }

  fprintf(fp, "\n");

  /* Cutting Planes */
  fprintf(fp, "%d\n", rend->n_plane);
  for (i = 0; i < rend->n_plane; i++) {
    fprintf(fp, "%f %f %f\n", rend->plane[i].point.x, rend->plane[i].point.y,
            rend->plane[i].point.z);
    fprintf(fp, "%f %f %f\n", rend->plane[i].normal.x,
            rend->plane[i].normal.y, rend->plane[i].normal.z);
  }

  fprintf(fp, "\n");

  /* Rendering modes */
  fprintf(fp, "%d %d %d %d\n", rend->render_mode, rend->cut_plane,
          rend->light_both, rend->color_mode);

  fprintf(fp, "\n");

  /* RayMisc */
  fprintf(fp, "%d\n", misc->step_size);
  fprintf(fp, "%d %d %d\n", misc->grad_ramp[0], misc->grad_ramp[1],
          misc->grad_ramp[2]);
  fprintf(fp, "%d %d %d\n", misc->back_color[0], misc->back_color[1],
          misc->back_color[2]);
}

void vrLoadVolume(VolRenEnv *env) {
  FILE *fp;
  char fname[MAX_STRING];
  Volume *vol;
  int i, slc, slc_size, slength, length;
  unsigned short max = 0, min = USHRT_MAX;
  unsigned int test = 0xABCDEF01; /* for endedness checking */
  Rendering *rend;
  size_t read_bytes;

  vol = env->vol;
  rend = env->rend;

  if (vol->den.us_ptr != NULL)
    free(vol->den.us_ptr);
  if (vol->den.uc_ptr != NULL)
    free(vol->den.uc_ptr);

  if ((fp = fopen(vol->fname, "rb")) == NULL) {
    printf("ERROR : Fopen(VolumeData)\n");
    env->valid = 0;
    return;
  }

  if (strlen(vol->fname) > 6 &&
      strcmp(vol->fname + strlen(vol->fname) - 6, ".rawiv") == 0)
    fseek(fp, 68, SEEK_SET);
  else if (strlen(vol->fname) > 4 &&
           strcmp(vol->fname + strlen(vol->fname) - 4, ".sds") == 0)
    fseek(fp, 224, SEEK_SET);

  /*
     TODO: fix subvolume loading for RAWV
     Update: should work now, however testing needed...
  */

  /* only load the subvolume that which this process renders */
  vol->sub_ext[2] =
      vol->sub_ext[2] /
      vol->num_slc; /* divide the whole subvolume into sub-subvolumes... */
  vol->sub_orig[2] = vol->sub_ext[2] * vol->slc_id +
                     vol->sub_orig[2]; /* calculate the new sub origin
                                          according to slab id */
  slength = vol->sub_ext[0] * vol->sub_ext[1] *
            vol->sub_ext[2];        /* seek this far for each slab */
  vol->sub_ext[2] += SUB_EXT_Z_GAP; /* get rid of the gap */
  vrSetSubVolume(vol, vol->sub_orig,
                 vol->sub_ext); /* apply changes to subvolume to reflect what
                                   this particular process will render */
  // vrSetStepSize(env,env->misc->step_size);
  length = vol->sub_ext[0] * vol->sub_ext[1] *
           vol->sub_ext[2]; /* load this much of the volume */

  printf("length = %d,sub_ext[2] = %d, sub_orig[2] = %d\n", length,
         vol->sub_ext[2], vol->sub_orig[2]);

  switch (vol->type) {
  case TYPE_8:
    printf("Load Volume Data (TYPE_8): %s\n", vol->fname);
    vol->type = UCHAR;
    vol->den.uc_ptr = (unsigned char *)calloc(1, sizeof(char) * length);
    fseek(fp, sizeof(char) * slength * vol->slc_id,
          SEEK_CUR); /* seek to the start of the slice we want to load */
    read_bytes = fread(vol->den.uc_ptr, 1, sizeof(char) * length, fp);
    printf("Seeking: %d, Slab size: %d bytes\n",
           (int)(sizeof(char) * length * vol->slc_id), (int)read_bytes);
    if (ferror(fp))
      printf("I/O error\n");
    for (i = 0; i < length; i++) {
      if (vol->den.uc_ptr[i] < min)
        min = vol->den.uc_ptr[i];
      if (vol->den.uc_ptr[i] > max)
        max = vol->den.uc_ptr[i];
    }

    break;

  case TYPE_12:
    printf("Load Volume Data (TYPE_12): %s\n", vol->fname);
    goto skip;
  case TYPE_16:
    printf("Load Volume Data (TYPE_16): %s\n", vol->fname);
  skip: {
    unsigned char *c, d;

    vol->type = USHORT;
    vol->den.us_ptr = (unsigned short *)malloc(sizeof(short) * length);
    fseek(fp, sizeof(short) * slength * vol->slc_id, SEEK_CUR);
    read_bytes = fread(vol->den.us_ptr, sizeof(short) * length, 1, fp);
    printf("Seeking: %d, Slab size: %d bytes\n",
           (int)(sizeof(char) * length * vol->slc_id), (int)read_bytes);

    /* check endedness (data is stored on disk as big endian) */
    c = (unsigned char *)&test;
    if (c[0] == 0x01 && c[1] == 0xEF && c[2] == 0xCD &&
        c[3] == 0xAB) // little endian
    {
      printf("Converting to little endian\n");
      for (i = 0; i < length; i++) {
        c = (unsigned char *)&(vol->den.us_ptr[i]);

        d = c[0];
        c[0] = c[1];
        c[1] = d;

        if (vol->den.us_ptr[i] < min)
          min = vol->den.us_ptr[i];
        if (vol->den.us_ptr[i] > max)
          max = vol->den.us_ptr[i];
      }
    } else {
      for (i = 0; i < length; i++) {
        if (vol->den.us_ptr[i] < min)
          min = vol->den.us_ptr[i];
        if (vol->den.us_ptr[i] > max)
          max = vol->den.us_ptr[i];
      }
    }
  }

  break;

  case TYPE_SLC8:
    printf("Load Volume Data (TYPE_SLC8): %s\n", vol->fname);
    /* to be added */
    break;

  case TYPE_SLC12:
    /***** For Loading Original Visible Human Data Slices Files *****/
    {
      int last_slc, VH_FIRST_FILE = 1012;
      unsigned short *ptr;
      printf("Loading Volume Data (TYPE_SLC12): %s\n", vol->fname);
      slc_size = vol->slc_size;
      last_slc = VH_FIRST_FILE + vol->dim[2];
      for (i = VH_FIRST_FILE, slc = 0; i < last_slc; i++, slc++) {
#ifdef WIN32
        sprintf(fname, "%s\\c_vm%d.fre\0", vol->fname, i);
#else
        sprintf(fname, "%s/c_vm%d.fre", vol->fname, i);
#endif
        vol->type = USHORT;
        vol->den.us_ptr = (unsigned short *)malloc(
            sizeof(short) * vol->dim[0] * vol->dim[1] * vol->dim[2]);
        ptr = vol->den.us_ptr;
        assert(ptr);
        // printf(" *** Reading %s *** \n", fname);
        fread(ptr + ((vol->dim[2] - 1 - slc) * slc_size), sizeof(short),
              slc_size, fp);
      }
    }
    break;

  case TYPE_VTM12:
    printf("Load Volume Data (TYPE_VTM12): %s\n", vol->fname);
    /* to be added */
    break;

  case TYPE_VTM15:
    printf("Load Volume Data (TYPE_VTM15): %s\n", vol->fname);
    /* to be added */
    break;

  case TYPE_VSTL:
    printf("Load Volume Data (TYPE_VSTL): %s\n", vol->fname);
    /* to be added */
    break;

  case TYPE_FLOAT:
    printf("Load Volume Data (TYPE_FLOAT): %s\n", vol->fname);
    {
      float fmax, fmin;
      float *fptr;
      unsigned char *c, buf[4];

      vol->type = USHORT;
      fptr = (float *)malloc(sizeof(float) * length);
      fseek(fp, sizeof(float) * slength * vol->slc_id, SEEK_CUR);
      read_bytes = fread(fptr, length * sizeof(float), 1, fp);
      printf("Seeking: %d, Slab size: %d bytes\n",
             (int)(sizeof(char) * length * vol->slc_id), (int)read_bytes);

      /* check endedness (data is stored on disk as big endian) */
      c = (unsigned char *)&test;
      if (c[0] == 0x01 && c[1] == 0xEF && c[2] == 0xCD &&
          c[3] == 0xAB) /* little endian */
      {
        printf("Converting to little endian\n");

        /* swap the first float so we can get the correct fmax and fmin */
        c = (unsigned char *)fptr;
        buf[0] = *c;
        buf[1] = *(c + 1);
        buf[2] = *(c + 2);
        buf[3] = *(c + 3);
        *c = buf[3];
        *(c + 1) = buf[2];
        *(c + 2) = buf[1];
        *(c + 3) = buf[0];
        fmax = fmin = *fptr;
        /* swap the rest */
        for (i = 1; i < length; i++) {
          c = (unsigned char *)(&fptr[i]);
          buf[0] = *c;
          buf[1] = *(c + 1);
          buf[2] = *(c + 2);
          buf[3] = *(c + 3);
          *c = buf[3];
          *(c + 1) = buf[2];
          *(c + 2) = buf[1];
          *(c + 3) = buf[0];

          if (fmax < fptr[i])
            fmax = fptr[i];
          if (fmin > fptr[i])
            fmin = fptr[i];
        }
      } else {
        for (i = 0; i < length; i++) {
          if (fmax < fptr[i])
            fmax = fptr[i];
          if (fmin > fptr[i])
            fmin = fptr[i];
        }
      }

      vol->den.us_ptr = (unsigned short *)
          fptr;               /* we can use this already allocated memory */
      if (fmax - fmin == 0.0) /* avoid divide by zero */
        memset(vol->den.us_ptr, 0, length * sizeof(short));
      else
        for (i = 0; i < length; i++) /* normalize */
        {
          vol->den.us_ptr[i] =
              (unsigned short)(((fptr[i] - fmin) / (fmax - fmin)) * 4096);

          if (vol->den.us_ptr[i] < min)
            min = vol->den.us_ptr[i];
          if (vol->den.us_ptr[i] > max)
            max = vol->den.us_ptr[i];
        }

      /* reclaim unused memory */
      realloc(vol->den.us_ptr, sizeof(short) * length);
    }

    break;
  case TYPE_RAWV: {
    unsigned int magic;
    float min_var[4], max_var[4];
    unsigned char *c, type, *rawv_ptr_cur;
    unsigned int cellsize, v, t;
    int little_endian;
    printf("Load Volume Data (TYPE_RAWV): %s\n", vol->fname);
    vol->type = RAWV;

    vol->den.r_var = 0;
    vol->den.g_var = 1;
    vol->den.b_var = 2;
    vol->den.den_var = 3;

    /* endian test */
    c = (unsigned char *)&test;
    little_endian =
        c[0] == 0x01 && c[1] == 0xEF && c[2] == 0xCD && c[3] == 0xAB;

    fread(&magic, 4, 1, fp);
    if (little_endian)
      SWAP_32(&magic);
    if (magic != 0xBAADBEEF) {
      fprintf(stderr, "Not a rawv file!\n");
      env->valid = 0;
      break;
    }
    fread(vol->dim, 4, 3, fp);
    if (little_endian) {
      SWAP_32(&vol->dim[0]);
      SWAP_32(&vol->dim[1]);
      SWAP_32(&vol->dim[2]);
    }
    vol->slc_size = vol->dim[0] * vol->dim[1];
    vol->vol_size = vol->slc_size * vol->dim[2];
    fread(&vol->den.rawv_num_timesteps, 4, 1, fp);
    fread(&vol->den.rawv_num_vars, 4, 1, fp);
    fread(min_var, 4, 4, fp);
    fread(max_var, 4, 4, fp);
    if (little_endian) {
      SWAP_32(&vol->den.rawv_num_timesteps);
      SWAP_32(&vol->den.rawv_num_vars);
      for (i = 0; i < 4; i++) {
        SWAP_32(&min_var[i]);
        SWAP_32(&max_var[i]);
      }
    }
    /* calculate span */
    for (i = 0; i < 3; i++)
      vol->span[i] = (max_var[i] - min_var[i]) / vol->dim[i];

    /* recalculate volume alignment */
    vol->orig.x = -(vol->dim[0] / 2.0f) * vol->span[0];
    vol->orig.y = -(vol->dim[1] / 2.0f) * vol->span[1];
    vol->orig.z = -(vol->dim[2] / 2.0f) * vol->span[2];

    /* reset the subvolume */
    vrSetSubVolume(vol, vol->sub_orig, vol->sub_ext);

    // vol->xinc = (int *)realloc(vol->xinc,sizeof(int) * vol->dim[1]);
    // vol->yinc = (int *)realloc(vol->yinc,sizeof(int) * vol->dim[2]);
    /* recalculate xinc and yinc */
    // for(i = 0; i < vol->dim[1]; i++)
    //   vol->xinc[i] = i * vol->dim[0];
    // for(i = 0; i < vol->dim[2]; i++)
    //   vol->yinc[i] = i * vol->slc_size;

    /* recalculate subvolume bounding box */
    /*vol->minb.x = vol->orig.x;
      vol->minb.y = vol->orig.y;
      vol->minb.z = vol->orig.z;
      vol->maxb.x = vol->minb.x + vol->dim[0]*vol->span[0];
      vol->maxb.y = vol->minb.y + vol->dim[1]*vol->span[1];
      vol->maxb.z = vol->minb.z + vol->dim[2]*vol->span[2];*/

    /* get variables types and names */
    vol->den.rawv_var_types =
        (RawVType *)malloc(sizeof(RawVType) * vol->den.rawv_num_vars);
    vol->den.rawv_var_names = (unsigned char **)malloc(
        sizeof(unsigned char *) * vol->den.rawv_num_vars);
    if (vol->den.rawv_var_types == NULL || vol->den.rawv_var_names == NULL) {
      env->valid = 0;
      fprintf(stderr, "Cannot allocate memory for RawV header!\n");
      break;
    }
    vol->den.rawv_sizeof_cell = 0;
    for (i = 0; i < vol->den.rawv_num_vars; i++) {
      fread(&type, 1, 1, fp);
      vol->den.rawv_var_types[i] = (RawVType)type;
      // fread(&vol->den.rawv_var_types[i],1,1,fp);
      vol->den.rawv_var_names[i] =
          (unsigned char *)malloc(sizeof(unsigned char) * RAWV_VAR_NAME_LEN);
      if (vol->den.rawv_var_names[i] == NULL) {
        env->valid = 0;
        fprintf(stderr, "Cannot allocate memory for RawV header!\n");
        break;
      }
      fread(vol->den.rawv_var_names[i], 1, RAWV_VAR_NAME_LEN, fp);
      vol->den.rawv_sizeof_cell += vrRawVSizeOf(vol->den.rawv_var_types[i]);
    }
    /* now allocate memory and read the data */
    cellsize = vol->den.rawv_sizeof_cell;
    vol->den.rawv_ptr = (unsigned char *)malloc(vol->den.rawv_num_timesteps *
                                                cellsize * length);
    rawv_ptr_cur = vol->den.rawv_ptr;
    if (rawv_ptr_cur == NULL) {
      env->valid = 0;
      fprintf(stderr, "Cannot allocate memory for RawV data!\n");
      break;
    }
    for (v = 0; v < vol->den.rawv_num_vars; v++)
      for (t = 0; t < vol->den.rawv_num_timesteps; t++) {
        fseek(fp,
              vrRawVSizeOf(vol->den.rawv_var_types[v]) * slength *
                  vol->slc_id,
              SEEK_CUR); /* seek to the start of the slab we want to load */
        fread(rawv_ptr_cur, vrRawVSizeOf(vol->den.rawv_var_types[v]), length,
              fp);
        if (little_endian && vrRawVSizeOf(vol->den.rawv_var_types[v]) > 1) {
          for (i = 0;
               i < vol->vol_size * vrRawVSizeOf(vol->den.rawv_var_types[v]);
               i += vrRawVSizeOf(vol->den.rawv_var_types[v])) {
            switch (vol->den.rawv_var_types[v]) {
            case RAWV_USHORT:
              SWAP_16(&rawv_ptr_cur[i]);
              break;
            case RAWV_UINT:
            case RAWV_FLOAT:
              SWAP_32(&rawv_ptr_cur[i]);
              break;
            case RAWV_DOUBLE:
              SWAP_64(&rawv_ptr_cur[i]);
            default:
              break;
            }
          }
        }
        /* find max and min if we're at the density variable */
        if (v == vol->den.den_var) {
          /* set max_dens */
          switch (vol->den.rawv_var_types[v]) {
          case RAWV_UCHAR:
            vol->max_dens = UCHAR_MAX + 1;
            break;
          case RAWV_UINT:
          case RAWV_USHORT:
            vol->max_dens = USHRT_MAX + 1;
            break;
          case RAWV_DOUBLE:
          case RAWV_FLOAT: {
            double dmax, dmin;

            /* normalize */
            for (i = 0;
                 i < vol->vol_size * vrRawVSizeOf(vol->den.rawv_var_types[v]);
                 i += vrRawVSizeOf(vol->den.rawv_var_types[v])) {
#define RAWV_LOAD_NORM_MAX_MIN(t)                                            \
  if ((double)*((t *)&rawv_ptr_cur[i]) > dmax)                               \
    dmax = (double)*((t *)&rawv_ptr_cur[i]);                                 \
  if ((double)*((t *)&rawv_ptr_cur[i]) < dmin)                               \
    dmin = (double)*((t *)&rawv_ptr_cur[i]);

              RAWV_CAST(vol->den.rawv_var_types[v], RAWV_LOAD_NORM_MAX_MIN);
            }
            for (i = 0;
                 i < vol->vol_size * vrRawVSizeOf(vol->den.rawv_var_types[v]);
                 i += vrRawVSizeOf(vol->den.rawv_var_types[v])) {
#define RAWV_LOAD_NORM(t)                                                    \
  *((t *)&rawv_ptr_cur[i]) =                                                 \
      (t)((((double)*((t *)&rawv_ptr_cur[i]) - dmin) / (dmax - dmin)) *      \
          4096);

              RAWV_CAST(vol->den.rawv_var_types[v], RAWV_LOAD_NORM);
            }

            vol->max_dens = 4096;
            break;
          }
          default:
            break;
          }

          for (i = 0;
               i < vol->vol_size * vrRawVSizeOf(vol->den.rawv_var_types[v]);
               i += vrRawVSizeOf(vol->den.rawv_var_types[v])) {
#define RAWV_LOAD_MAX_MIN(t)                                                 \
  if ((unsigned short)*((t *)&rawv_ptr_cur[i]) > max)                        \
    max = (unsigned short)*((t *)&rawv_ptr_cur[i]);                          \
  if ((unsigned short)*((t *)&rawv_ptr_cur[i]) < min)                        \
    min = (unsigned short)*((t *)&rawv_ptr_cur[i]);

            RAWV_CAST(vol->den.rawv_var_types[v], RAWV_LOAD_MAX_MIN);
          }
        }
        rawv_ptr_cur +=
            vol->vol_size * vrRawVSizeOf(vol->den.rawv_var_types[v]);
      }

    /* recalculate the color density table since max_dens might have changed
     */
    vrComputeColDen(env->rend, vol);

    break;
  }
  default:
    fflush(stdout);
    fprintf(stderr, "ERROR : unknown type %d!\n", vol->type);
    break;
    ;
  }

  fclose(fp);

  rend->min_den = min;
  rend->max_den = max;

  printf("min_den: %d max_den: %d\n", rend->min_den, rend->max_den);
}

int vrVolumeExists(VolRenEnv *env) {
  struct stat s;
  if (stat(env->vol->fname, &s) == -1)
    return 0;
  return 1;
}

/*
SubVol* vrGetSubVolData(VolRenEnv* env, float min[3], float max[3])
{
  Volume* vol;
  SubVol* subvol;
  int i, j, k, m, n, size, shift;
  int lower[3], upper[3];

  vol = env->vol;

  lower[0] = (int)((min[0] - vol->orig.x) / vol->span[0] - 0.5);
  lower[1] = (int)((min[1] - vol->orig.y) / vol->span[1] - 0.5);
  lower[2] = (int)((min[2] - vol->orig.z) / vol->span[2] - 0.5);

  upper[0] = (int)((max[0] - vol->orig.x) / vol->span[0] + 0.5);
  upper[1] = (int)((max[1] - vol->orig.y) / vol->span[1] + 0.5);
  upper[2] = (int)((max[2] - vol->orig.z) / vol->span[2] + 0.5);

  subvol = (SubVol *)malloc(sizeof(SubVol));
  for(i = 0; i < 3; i++) {
    lower[i] = (lower[i] >= 0)? lower[i]:0;
    upper[i] = (upper[i] < vol->dim[i])? upper[i]:vol->dim[i]-1;
    subvol->dim[i] = upper[i] - lower[i] + 1;
  }
  size = subvol->dim[0] * subvol->dim[1] * subvol->dim[2];
  subvol->data = (unsigned char*) malloc(sizeof(char)*size);

  n = 0;
  shift = vol->max_dens >> 8;
  for(k = lower[2]; k <= upper[2] ; k++) {
    for(j = lower[1]; j <= upper[1]; j++) {
      for(i = lower[0], m = lower[0]+j*vol->dim[0]+k*vol->slc_size; i <=
upper[0]; i++, m++) { subvol->data[n] = (vol->den.us_ptr[m]) >> shift; n++;
      }
    }
  }
  return subvol;
}
*/

void vrCreateImage(ImageBuffer *img, int w, int h) {
  img->width = w;
  img->height = h;
  img->buffer = (unsigned char *)calloc(w * h, sizeof(char) * RGBA_SIZE);
}

void vrDestroyImage(ImageBuffer *img) {
  free(img->buffer);
  free(img);
}

void vrSaveImg2PPM(ImageBuffer *img, char *fname) {
  FILE *fp;
  int i;

  if ((fp = fopen(fname, "wb")) == NULL) {
    fprintf(stderr, "ERROR : Fopen(%s)\n", fname);
    return;
  }

  fprintf(fp, "P6\n");
  fprintf(fp, "%d %d\n", img->width, img->height);
  fprintf(fp, "%d\n", MAX_COLOR);

  // fwrite(img->buffer, sizeof(char) * 3 * img->width * img->height, 1, fp);
  /* only write rgb data */
  for (i = 0; i < img->width * img->height * sizeof(char) * RGBA_SIZE;
       i += RGBA_SIZE)
    fwrite(img->buffer + i, sizeof(char) * RGB_SIZE, 1, fp);

  fclose(fp);
}
