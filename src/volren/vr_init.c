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

#include <volren/vr.h>

VolRenEnv* vrCreateEnv(void)
{
  VolRenEnv* env = (VolRenEnv *)calloc(1,sizeof(VolRenEnv));
  env->vol = (Volume *)calloc(1,sizeof(Volume));
  env->view = (Viewing *)calloc(1,sizeof(Viewing));
  env->rend = (Rendering *)calloc(1,sizeof(Rendering));
  env->misc = (RayMisc *)calloc(1,sizeof(RayMisc));
  assert(env && env->vol && env->view && env->rend && env->misc);

  env->valid = 1;
	
  return env;
}

VolRenEnv* vrCreateEnvFromFile(char *fname, unsigned int slc_id, unsigned int num_slc)
{
  FILE* fp;
  VolRenEnv* env;

  if((fp = fopen(fname, "r")) == NULL) return NULL;
  env = vrCreateEnv();
  vrReadConfig(env, fp);
  env->vol->slc_id = slc_id;
  env->vol->num_slc = num_slc;
  printf("Slab id: %d ",slc_id);
  printf("Number of slabs: %d\n",num_slc);
  fflush(stdout);
  vrLoadVolume(env);
  fclose(fp);

  if(env->vol->den.us_ptr == NULL && env->vol->den.uc_ptr == NULL && env->vol->den.rawv_ptr == NULL)
    {
      fprintf(stderr,"vrCreateEnvFromFile(): Error: unable to allocate memory for volume data\n");
      env->valid = 0;
    }

  return env;
}

MultiVolRenEnv *vrCreateMultiEnv()
{
  MultiVolRenEnv *menv;
  menv = (MultiVolRenEnv *)calloc(1,sizeof(MultiVolRenEnv));
  menv->metavol = (Volume *)calloc(1,sizeof(Volume));
  return menv;
}

void vrAddEnv(MultiVolRenEnv *menv, VolRenEnv *env)
{
  if(menv == NULL || env == NULL) return;
  if(menv->n_env < MAX_ENV) menv->env[menv->n_env++] = env;
  if(menv->metavol->minb.x > env->vol->minb.x) menv->metavol->minb.x = env->vol->minb.x;
  if(menv->metavol->minb.y > env->vol->minb.y) menv->metavol->minb.y = env->vol->minb.y;
  if(menv->metavol->minb.z > env->vol->minb.z) menv->metavol->minb.z = env->vol->minb.z;
  if(menv->metavol->maxb.x < env->vol->maxb.x) menv->metavol->maxb.x = env->vol->maxb.x;
  if(menv->metavol->maxb.y < env->vol->maxb.y) menv->metavol->maxb.y = env->vol->maxb.y;
  if(menv->metavol->maxb.z < env->vol->maxb.z) menv->metavol->maxb.z = env->vol->maxb.z;
  if(menv->metavol->orig.x > env->vol->orig.x) menv->metavol->orig.x = env->vol->orig.x;
  if(menv->metavol->orig.y > env->vol->orig.y) menv->metavol->orig.y = env->vol->orig.y;
  if(menv->metavol->orig.z > env->vol->orig.z) menv->metavol->orig.z = env->vol->orig.z;
}

void vrCleanEnv(VolRenEnv* env)
{
  int i;
  if(env == NULL) return;
  switch(env->vol->type) {
  case UCHAR:
    if(env->vol->den.uc_ptr != NULL) free(env->vol->den.uc_ptr);
    break;
  case USHORT:
    if(env->vol->den.us_ptr != NULL) free(env->vol->den.us_ptr);
    break;
  case FLOAT:
    if(env->vol->den.f_ptr != NULL) free(env->vol->den.f_ptr);
    break;
  case RAWV:
    if(env->vol->den.rawv_ptr != NULL) free(env->vol->den.rawv_ptr);
    if(env->vol->den.rawv_var_types != NULL) free(env->vol->den.rawv_var_types);
    if(env->vol->den.rawv_var_names != NULL)
      for(i=0;i<env->vol->den.rawv_num_vars;i++)
	if(env->vol->den.rawv_var_names[i] != NULL) free(env->vol->den.rawv_var_names[i]);
    if(env->vol->den.rawv_var_names != NULL) free(env->vol->den.rawv_var_names);
  default:
    break;
  }
  if(env->vol->xinc) free(env->vol->xinc);
  if(env->vol->yinc) free(env->vol->yinc);
  free(env->vol);

  if(env->rend->opctbl != NULL) free(env->rend->opctbl);
  if(env->rend->coldentbl != NULL) free(env->rend->coldentbl);
  free(env->rend);

  free(env->misc);
  free(env->view);

  free(env);
}

void vrCleanMultiEnv(MultiVolRenEnv *menv)
{
  int i;
  for(i=0;i<menv->n_env; i++)
    {
      vrCleanEnv(menv->env[i]);
      //      free(menv->env[i]);
    }
  free(menv->metavol);
  free(menv);
}
