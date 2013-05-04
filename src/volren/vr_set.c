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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include <volren/vr.h>

void vrSetVolumeInfo(Volume* vol, char* fname, char* type, float span[3], int dim[3])
{
  int i;

  fname[strlen(fname)-1] = '\0';
  strcpy(vol->fname, fname);
  assert(strlen(vol->fname) < MAX_STRING);
  assert(vol->xinc == NULL && vol->yinc == NULL);

  if (!strcmp(type, "8")) {
    vol->type = TYPE_8;
    vol->max_dens = 256;
  } else if (!strcmp(type, "12")) {
    vol->type = TYPE_12;
    vol->max_dens = 4096;
  } else if (!strcmp(type, "16")) {
    vol->type = TYPE_16;
    vol->max_dens = 65536;
  } else if (!strcmp(type, "SLC8")) {
    vol->type = TYPE_SLC8;
    vol->max_dens = 256;
  } else if (!strcmp(type, "SLC12")) {
    vol->type = TYPE_SLC12;
    vol->max_dens = 4096;
  } else if (!strcmp(type, "VTM12")) {
    vol->type = TYPE_VTM12;
    vol->max_dens = 4096;
  } else if (!strcmp(type, "VTM15")) {
    vol->type = TYPE_VTM15;
    vol->max_dens = 32768;
  } else if (!strcmp(type, "VSTL")) {
    vol->type = TYPE_VSTL;
    vol->max_dens = 4096;
  } else if(!strcmp(type, "FLOAT") || !strcmp(type, "FLT_12")) {
    vol->type = TYPE_FLOAT;
    vol->max_dens = 4096;		/* normalize float to (0, 4096) */
  } else if(!strcmp(type, "RAWV")) {
    vol->type = TYPE_RAWV;
    vol->max_dens = 65536; /* this may change later */
  } else {
    fprintf(stderr, "unknow type: %s\n", type);
    assert(0);
  }
  vol->orig_type = vol->type;
  for(i = 0; i < 3; i++) {
    vol->span[i] = span[i];
    vol->dim[i] = dim[i];
  }

  /* aline the center of volume to (0, 0, 0) */
  vol->orig.x = -(dim[0]  / 2.0f) * span[0];
  vol->orig.y = -(dim[1]  / 2.0f) * span[1];
  vol->orig.z = -(dim[2]  / 2.0f) * span[2];
}

void vrSetSubVolume(Volume* vol, int orig[3], int ext[3])
{
  int i;

  for(i = 0; i < 3; i++) {
    if(orig[i] < 0) orig[i] = 0;
    if(orig[i]+ext[i] >= vol->dim[i]) ext[i] = vol->dim[i]-orig[i];
    vol->sub_orig[i] = orig[i];
    vol->sub_ext[i] = ext[i];
  }

  vol->slc_size = ext[0] * ext[1];
  vol->vol_size = vol->slc_size * ext[2];
  vol->xinc = (int *)realloc(vol->xinc,sizeof(int) * ext[1]);
  vol->yinc = (int *)realloc(vol->yinc,sizeof(int) * ext[2]);
  for(i = 0; i < ext[1]; i++) {
    vol->xinc[i] = i * ext[0];
  }
  for(i = 0; i < ext[2]; i++) {
    vol->yinc[i] = i * vol->slc_size;
  }

  vol->minb.x = vol->orig.x + orig[0]*vol->span[0];
  vol->minb.y = vol->orig.y + orig[1]*vol->span[1];
  vol->minb.z = vol->orig.z + orig[2]*vol->span[2];
  vol->maxb.x = vol->minb.x + ext[0]*vol->span[0];
  vol->maxb.y = vol->minb.y + ext[1]*vol->span[1];
  vol->maxb.z = vol->minb.z + ext[2]*vol->span[2];
}

void vrSetPerspective(Viewing* view, int persp, float fov)
{
  view->persp = persp;

  if(fov <= 0.0f || fov >= 180.0f) {
    view->fov = (float)(PI / 2.0);		/* default fov */
  } else {
    view->fov = (float)((PI * fov) / 180.0);
  }
}

void vrSetEyePoint(Viewing* view, float x, float y, float z)
{
  view->eye.x = x;
  view->eye.y = y;
  view->eye.z = z;
}

void vrSetUpVector(Viewing* view, float x, float y, float z)
{
  view->vup.x = x;
  view->vup.y = y;
  view->vup.z = z;
  vrNormalize(&(view->vup));
}

void vrSetPlaneNormVector(Viewing* view, float x, float y, float z)
{
  view->vpn.x = x;
  view->vpn.y = y;
  view->vpn.z = z;
  vrNormalize(&(view->vpn));
}

void vrSetViewWindowSize(Viewing* view, int width, int height)
{
  view->win_width = width;
  view->win_height = height;
}

void vrSetPixelWindowSize(Viewing* view, int width, int height)
{
  /* make sure the image is square and nonzero in size */
  if(width>=0)
    view->pix_width = (width % TILE_SIZE)!=0 ? width - (width%TILE_SIZE) + TILE_SIZE : width;
  else
    view->pix_width = TILE_SIZE;
  if(height>=0)
    view->pix_height = (height % TILE_SIZE)!=0 ? height - (height%TILE_SIZE) + TILE_SIZE : height;
  else
    view->pix_height = TILE_SIZE;
}

void vrComputeView(Viewing* view)
{
  float w, h;

  view->vpu.x = view->vup.y * view->vpn.z - view->vup.z * view->vpn.y;   
  view->vpu.y = view->vup.z * view->vpn.x - view->vup.x * view->vpn.z;   
  view->vpu.z = view->vup.x * view->vpn.y - view->vup.y * view->vpn.x;
  vrNormalize(&(view->vpu));

  view->pix_sz_x = (float)view->win_width / view->pix_width;
  view->pix_sz_y = (float)view->win_height / view->pix_height;

  /* set view window origin */
  w = (float)view->win_width;	h = (float)view->win_height;
  if(view->persp) {
    Point3d	center;
    float	half_win_h, half_win_w, d;
	
    half_win_h = view->win_height / 2.0f;
    half_win_w = view->win_width  / 2.0f;

    /* distance from the eye to the center of view plane */
    d = (float)(half_win_h / tan(view->fov / 2.0f));

    center.x = view->eye.x - d * view->vpn.x;
    center.y = view->eye.y - d * view->vpn.y;
    center.z = view->eye.z - d * view->vpn.z;

    view->win_sp.x = center.x - half_win_w * view->vpu.x - half_win_h * view->vup.x;
    view->win_sp.y = center.y - half_win_w * view->vpu.y - half_win_h * view->vup.y;
    view->win_sp.z = center.z - half_win_w * view->vpu.z - half_win_h * view->vup.z;
  } else {
    view->win_sp.x = view->eye.x - 0.5f * (w*view->vpu.x + h*view->vup.x);
    view->win_sp.y = view->eye.y - 0.5f * (w*view->vpu.y + h*view->vup.y);
    view->win_sp.z = view->eye.z - 0.5f * (w*view->vpu.z + h*view->vup.z);
  }
}

void vrInitMaterial(Rendering* rend, Volume* vol)
{
  if(rend->opctbl != NULL) free(rend->opctbl);
  rend->opctbl = (float *)calloc(vol->max_dens, sizeof(float));
}

int vrAddMaterial(Rendering* rend, Volume* vol, Shading* shade, 
		  float opac, int start, int up, int down, int end)
{
  int n;
  n = rend->n_material;
  if(n >= MAX_MATERIAL) return 0;
  rend->n_material++;

  return vrSetMaterial(rend, vol, n, shade, opac, start, up, down, end);
}

int vrSetMaterial(Rendering* rend, Volume* vol, int n, Shading* shade, 
		  float opac, int start, int up, int down, int end)
{
  int i;
  if(n >= rend->n_material) return 0;
  rend->mat[n].opacity = opac;
  rend->mat[n].start = start;
  rend->mat[n].up = up;
  rend->mat[n].end = end;
  rend->mat[n].down = down;
  vrCopyShading(&(rend->mat[n].shading), shade);

  /* setup the opacity table */
  n = vol->max_dens;
  for(i = start; i < up && i < n; i++) {
    rend->opctbl[i] = opac * (i - start) / (up - start);
  }
  for(i = up; i < down && i < n; i++) {
    rend->opctbl[i] = opac;
  }	
  for (i = down; i < end && i < n; i++) {
    rend->opctbl[i] = opac * (end - i) / (end - down);
  }
  return 1;
}

int vrAddIsoSurface(Rendering* rend, Shading* shade, float opac, float isoval)
{
  int n;

  n = rend->n_surface;
  if(n >= MAX_SURFACE) return 0;
  rend->n_surface++;

  return vrSetIsoSurface(rend, n, shade, opac, isoval);
}

int vrSetIsoSurface(Rendering* rend, int id, Shading* shade, float opac, float isoval)
{
  if(id >= rend->n_surface) return 0;

  rend->surf[id].opacity = opac;
  rend->surf[id].value = isoval;
  vrCopyShading(&(rend->surf[id].shading), shade);

  return 1;
}

int vrAddColDenRamp(Rendering* rend, int den, int r, int g, int b, int a)
{
  int n;

  n = rend->n_colden;
  if(n >= MAX_COLDEN_RAMP) return 0;
  rend->n_colden++;

  return vrSetColDenRamp(rend, n, den, r, g, b, a);
}

void vrSetColDenNum(Rendering *rend, int nd) {
  if(nd >= MAX_COLDEN_RAMP) rend->n_colden = MAX_COLDEN_RAMP;
  else rend->n_colden = nd;
}

int vrSetColDenRamp(Rendering* rend, int id, int den, int r, int g, int b, int a)
{
  if(id >= rend->n_colden) return 0;

  rend->coldenramp[id].d = den;
  rend->coldenramp[id].r = r;
  rend->coldenramp[id].g = g;
  rend->coldenramp[id].b = b;
  rend->coldenramp[id].a = a;

  return 1;	
}

void vrComputeColDen(Rendering* rend, Volume* vol)
{
  int i, j;
  unsigned char *ptr;
  float t;
	
  if(rend->coldentbl != NULL) free(rend->coldentbl);

  rend->coldentbl = (unsigned char*)calloc(vol->max_dens, sizeof(char)*RGBA_SIZE);
  fflush(stdout);
  assert(rend->coldentbl);
  if(rend->n_colden > 0) rend->min_den = rend->max_den = rend->coldenramp[0].d;
  else return;
  
  /* the density values in the coldenramp array should be sorted */
  if(rend->min_den > 0) {
    for(j = 0; j < rend->min_den; j++) {
      if(j==vol->max_dens)
	{
	  fprintf(stderr,"Transfer function error: transfer function densities exceed the maximum density for this datatype (%d).\n",j);
	  return;
	}
      t = (float)(j) / rend->min_den;
      ptr = rend->coldentbl + RGBA_SIZE*j;
      *ptr = (unsigned char)(t*rend->coldenramp[0].r);
      *(ptr+1) = (unsigned char)(t*rend->coldenramp[0].g);
      *(ptr+2) = (unsigned char)(t*rend->coldenramp[0].b);
      *(ptr+3) = (unsigned char)(t*rend->coldenramp[0].a);
    }
    rend->min_den = 0;
  }
  for(i = 0; i < rend->n_colden - 1; i++) {
    int d1, d2;
    d1 = rend->coldenramp[i].d; d2 = rend->coldenramp[i+1].d;
    if(d2 < rend->min_den) rend->min_den = d2;
    if(d2 > rend->max_den) rend->max_den = d2;
    for(j = d1; j < d2; j++) {
      if(j==vol->max_dens)
	{
	  fprintf(stderr,"Transfer function error: transfer function densities exceed the maximum density for this datatype (%d).\n",j);
	  return;
	}
      ptr = rend->coldentbl + RGBA_SIZE*j;
      t = (float)(j-d1) / (d2-d1);
      *ptr = (unsigned char)((1-t)*rend->coldenramp[i].r + t*rend->coldenramp[i+1].r);
      *(ptr+1) = (unsigned char)((1-t)*rend->coldenramp[i].g + t*rend->coldenramp[i+1].g);
      *(ptr+2) = (unsigned char)((1-t)*rend->coldenramp[i].b + t*rend->coldenramp[i+1].b);
      *(ptr+3) = (unsigned char)((1-t)*rend->coldenramp[i].a + t*rend->coldenramp[i+1].a);
    }
  }
  if(rend->max_den < vol->max_dens-1) {
    for(j = rend->max_den; j < vol->max_dens-1; j++) {
      ptr = rend->coldentbl + RGBA_SIZE*j;
      t = (float)(j-rend->max_den) / (vol->max_dens-1 - rend->max_den);
      *ptr = (unsigned char)((1-t)*rend->coldenramp[rend->n_colden].r);
      *(ptr+1) = (unsigned char)((1-t)*rend->coldenramp[rend->n_colden].g);
      *(ptr+2) = (unsigned char)((1-t)*rend->coldenramp[rend->n_colden].b);
      *(ptr+3) = (unsigned char)((1-t)*rend->coldenramp[rend->n_colden].a);
    }
    rend->max_den = vol->max_dens-1;
  }

  fflush(stdout);
}

int vrAddLight(Rendering* rend, int r, int g, int b, float x, float y, float z)
{
  int n;
  n = rend->n_light;
  if(n >= MAX_LIGHT) return 0;
  rend->n_light ++;

  return vrSetLight(rend, n, r, g, b, x, y, z);
}

int vrSetLight(Rendering* rend, int id, int r, int g, int b, float x, float y, float z)
{
  if(id >= rend->n_light) return 0;

  rend->light[id].color.r = r;
  rend->light[id].color.g = g;
  rend->light[id].color.b = b;
  rend->light[id].dir.x = x;
  rend->light[id].dir.y = y;
  rend->light[id].dir.z = z;
  vrNormalize(&(rend->light[id].dir));

  return 1;
}

int vrAddCuttingPlane(Rendering* rend, Point3d* orig, Vector3d* norm)
{
  int n;
  n = rend->n_plane;
  if(n >= MAX_PLANE) return 0;
  rend->n_plane ++;

  return vrSetCuttingPlane(rend, n, orig, norm);
}

int vrSetCuttingPlane(Rendering* rend, int id, Point3d* orig, Vector3d* norm)
{
  if(id >= rend->n_plane) return 0;

  rend->plane[id].point = *orig;
  rend->plane[id].normal = *norm;

  return 1;
}

void vrSetRenderingMode(Rendering* rend, int mode)
{
  rend->render_mode = (RenderMode) mode;
}

void vrToggleCuttingPlane(Rendering* rend, int cut)
{
  rend->cut_plane = cut;
}

void vrToggleLightBoth(Rendering* rend, int both)
{
  rend->light_both = both;
}

void vrSetColorMode(Rendering* rend, int colmode)
{
  rend->color_mode = (ColorMode) colmode;
}

void vrSetBackColor(RayMisc* misc, int r, int g, int b)
{
  misc->back_color[0] = r;
  misc->back_color[1] = g;
  misc->back_color[2] = b;
}

void vrSetStepSize(VolRenEnv* env, int nstep)
{
  RayMisc* misc;
  Volume* vol;
  float dx, dy, dz;

  misc = env->misc;
  vol = env->vol;
  misc->step_size = nstep;
  //misc->step_size = 2 * MAX2(vol->dim[0], MAX2(vol->dim[1], vol->dim[2]));
  dx = vol->maxb.x - vol->minb.x;
  dy = vol->maxb.y - vol->minb.y;
  dz = vol->maxb.z - vol->minb.z;
  misc->unit_step = (float)(sqrt(dx*dx+dy*dy+dz*dz)) / misc->step_size;
}

void vrSetRayMisc(VolRenEnv* env)
{
  RayMisc* misc;
  Viewing* view;
  float unit_step;

  misc = env->misc;
  view = env->view;
  unit_step = misc->unit_step;

  if (view->persp) {
    misc->step_x = -view->raydir.x * unit_step;
    misc->step_y = -view->raydir.y * unit_step;
    misc->step_z = -view->raydir.z * unit_step;
  } else {
    misc->step_x = -view->vpn.x * unit_step;
    misc->step_y = -view->vpn.y * unit_step;
    misc->step_z = -view->vpn.z * unit_step;
  }

  if (misc->step_x > misc->step_y) {
    if (misc->step_x > misc->step_z)
      misc->axis = 'X';
    else
      misc->axis = 'Z';
  } else {
    if (misc->step_y > misc->step_z)
      misc->axis = 'Y';
    else
      misc->axis = 'Z';
  }
}

void vrSetGradientTable(RayMisc* misc, int ramp0, int ramp1, int ramp2)
{
  int		grad;
  float   factor = 0.9f;

  misc->grad_ramp[0] = ramp0;
  misc->grad_ramp[1] = ramp1;
  misc->grad_ramp[2] = ramp2;

  for (grad = 0; grad < MAX_GRADIENT; grad++) {
    if ((ramp0 <= grad) && (grad <= ramp1))
      misc->gradtbl[grad] = factor * (float)(grad - ramp0) / (float)(ramp1 - ramp0);
    else if ((ramp1 < grad && grad <= ramp2))
      misc->gradtbl[grad] = factor * 1.0f;
    else
      misc->gradtbl[grad] = 0.0f;
  }
}

void vrSetContourRay(iRay* ray, Viewing* view, float pnt[3])
{
 if(view->persp) {
   ray->dir[0] = -view->raydir.x;
   ray->dir[1] = -view->raydir.y;
   ray->dir[2] = -view->raydir.z;
 } else {
   ray->dir[0] = -view->vpn.x;
   ray->dir[1] = -view->vpn.y;
   ray->dir[2] = -view->vpn.z;
 }
 ray->orig[0] = pnt[0];
 ray->orig[1] = pnt[1];
 ray->orig[2] = pnt[2];
}


