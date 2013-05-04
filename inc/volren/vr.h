/*
  Copyright 2000-2002,2004-2005 The University of Texas at Austin

	Authors: Sanghun Park <hun@ices.utexas.edu>
                 Joe Rivera <transfix@ices.utexas.edu>
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

/**
 * This file defines basic constants and data structures for volume rendering
 */

#ifndef VOLREN_H
#define VOLREN_H

#include <stdio.h>
#include <libiso/iso.h>
                            
#ifdef __cplusplus
extern "C" {
#endif

#define PI 3.1415926

#define TILE_SIZE 32							/* default image tile size */
#define TILE_RES (TILE_SIZE*TILE_SIZE)

#define MAX_GRADIENT	256
#define MAX_GRADIENT_M1 (MAX_GRADIENT-1)

#define	THRESHOLD_OPC	0.95                    /* opacity threshold for early termination */
#define	MAX_COLOR	255
#define	MAX_COLOR_P1	256
#define MAX_MATERIAL	10           
#define MAX_SURFACE	5	
#define MAX_LIGHT	5                           /* maximum number of lights */                          
#define MAX_PLANE	5                           /* maximum number of cutting planes */
#define MAX_ENV         5                           /* maximum number of volumes to render */
#define	MAX_COLDEN_RAMP	1000
#define MAX_STRING	100

#define RGB_SIZE  3
#define RGBA_SIZE 4

#define MICSEC_PER_SEC	1000000.0
#define LARGEST		1000000.0
#define SMALLEST	0.00001f

#define RAWV_VAR_NAME_LEN 64

#define SUB_EXT_Z_GAP   2

  /* some useful macros */                                              
#define VERYSMALL(a)	((-SMALLEST <= (a)) && ((a) <= SMALLEST))
#define ABS(a)		((a) > 0.0 ? (a) : (-(a)))
#define MAX2(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN2(a, b)	(((a) < (b)) ? (a) : (b))
#define INNERPROD(u, v)	((u).x*(v).x + (u).y*(v).y + (u).z*(v).z)
#define SWAP_64(a) \
  { \
    unsigned char tmp[8]; \
    unsigned char *ch; \
    ch = (unsigned char *)a; \
    tmp[0] = ch[0]; tmp[1] = ch[1]; tmp[2] = ch[2]; tmp[3] = ch[3]; \
    tmp[4] = ch[4]; tmp[5] = ch[5]; tmp[6] = ch[6]; tmp[7] = ch[7]; \
    ch[0] = tmp[7]; ch[1] = tmp[6]; ch[2] = tmp[5]; ch[3] = tmp[4]; \
    ch[4] = tmp[3]; ch[5] = tmp[2]; ch[6] = tmp[1]; ch[7] = tmp[0]; \
  }
#define SWAP_32(a) \
  { \
    unsigned char tmp[4]; \
    unsigned char *ch; \
    ch = (unsigned char *)a; \
    tmp[0] = ch[0]; tmp[1] = ch[1]; tmp[2] = ch[2]; tmp[3] = ch[3]; \
    ch[0] = tmp[3]; ch[1] = tmp[2]; ch[2] = tmp[1]; ch[3] = tmp[0]; \
  }
#define SWAP_16(a) \
  { \
    unsigned char d; \
    unsigned char *ch; \
    ch = (unsigned char *)a; \
    d = ch[0]; \
    ch[0] = ch[1]; \
    ch[1] = d; \
  }
#define RAWV_CAST(a,c) \
  switch(a) \
    { \
    case RAWV_UCHAR: c(unsigned char); break; \
    case RAWV_USHORT: c(unsigned short); break; \
    case RAWV_UINT: c(unsigned int); break; \
    case RAWV_FLOAT: c(float); break; \
    case RAWV_DOUBLE: c(double); break; \
    default: break; \
    }
#define RAWV_CHECK(v) \
  if(v->type != RAWV || v->den.rawv_ptr == NULL) \
    { \
      fprintf(stderr,"Error: RawV function call on non RawV dataset\n"); \
      return; \
    }

  typedef struct               /* Vector defined in 3-dimensional space     */
  {
    float x;
    float y;
    float z;
  } Vector3d, Point3d;

  typedef struct               /* Color                                     */
  {
    int r;
    int g;
    int b;
  } Color;

  typedef enum {
    UCHAR = 0,
    USHORT,
    FLOAT,
    RAWV,
    TYPE_8,
    TYPE_12,
    TYPE_16,
    TYPE_SLC8,
    TYPE_SLC12,
    TYPE_VTM12,
    TYPE_VTM15,
    TYPE_VSTL,
    TYPE_FLOAT,
    TYPE_RAWV,
  } DataType;

  typedef enum {
    RAWV_UCHAR = 1,
    RAWV_USHORT,
    RAWV_UINT,
    RAWV_FLOAT,
    RAWV_DOUBLE,
  } RawVType;

  typedef struct {
    unsigned char *uc_ptr;
    unsigned short *us_ptr;
    float   *f_ptr;

    /* ****** rawv specific ******* */
    unsigned char *rawv_ptr; /* rawv data */
    /* rawv types:
     * 1 for unsigned char (1 byte)
     * 2 for unsigned short (2 bytes)
     * 3 for unsigned int/long (4 bytes)
     * 4 for float (4 bytes)
     * 5 for double (8 bytes)
     */
    RawVType *rawv_var_types;
    unsigned char **rawv_var_names;
    unsigned int rawv_num_vars;
    unsigned int rawv_num_timesteps; /* just render the first one for now */
    unsigned int rawv_sizeof_cell; /* variable byte lengths all summed up */
    unsigned int r_var; /* variable number of the variable interpreted as red */
    unsigned int g_var; /* variable number of the variable interpreted as green */
    unsigned int b_var; /* variable number of the variable interpreted as blue */
    unsigned int den_var; /* variable number of the variable interpreted as density */

    /* chunks extend in the z direction of the volume */
    unsigned int chunk_id; /* the chunk that this process has in memory */
    unsigned int max_chunk;
  } DataPtr;

  typedef enum {
    RAY_CASTING = 1,
    ISO_SURFACE = 2,
    ISO_AND_RAY = 3,
    COL_DENSITY = 4,
    COLDEN_AND_RAY = 5,
    COLDEN_AND_ISO = 6,
    COLDEN_AND_RAY_AND_ISO = 7
  } RenderMode;

#define ISOSURFACE(x) (x & ISO_SURFACE)
#define RAYSHADING(x) (x & RAY_CASTING)
#define COLDEN(x) (x & COL_DENSITY)

  typedef enum {
    GRAY_SCALE = 0,
    RGB = 1
  } ColorMode;

  /**
   * Image buffer
   */
  typedef struct {
    int id;
    float dist;
    int width, height;
    unsigned char* buffer;
  } ImageBuffer;

  /**
   * One point of the Color Density Ramp of a transfer function.
   * @note maybe changed with a data structure for transfer function
   */
  typedef struct {
    unsigned short d;
    unsigned char  r, g, b, a;
  } ColDenRamp;

  typedef struct {
    Color ambient, diffuse, specular;
    int shining;
  } Shading;

  typedef struct
  {
    float opacity;            /* Opacity value of the material graph       */

    int start;                /* Each denotes a point where the direction  */
    int up;                   /* of the material graph changes             */
    int down;
    int end;

    float *opctbl; 			/* Opacity Table */

#ifdef COLOR_TABLE
    unsigned char *clrtbl;	/* Color Table */
#endif
    Shading shading;
  } Material;

  typedef struct {
    float opacity;			/* Opacity value of the material graph       */
    float value;             /* the Isovalue */

#ifdef COLOR_TABLE
    unsigned char *clrtbl;   /* Color Table */
#endif

    Shading shading;
  } Surface;

  /**
   * Point Light
   */
  typedef struct {
    Color color;  
    Vector3d dir;			/* Light source position */ 
  } Light;

  /**
   * Cutting Plane
   */
  typedef struct {                              
    Point3d point;           /* A point passed by the plane */
    Vector3d normal;			/* Plane normal */
  } Plane;                    

  typedef struct {
    int   step_size;			/* max number of steps on one ray */
    float unit_step;         /* length of one step */
    float step_x;
    float step_y;
    float step_z;
    char	 axis;				/* Skip sampling by the unit: FireOneRay() */
   
    int   grad_ramp[3];
    float gradtbl[MAX_GRADIENT];
  
    unsigned char back_color[3];
  } RayMisc;

  /**
   * Volume Info.
   */
  typedef struct {
    char fname[MAX_STRING];
    DataType type;
    DataType orig_type;         /* the type of the volume data file */
    DataPtr den;		/* Density volume data */
    Point3d	orig;		/* Origin of volume data                */

    float	     span[3];	/* Interval of slices along x, y and z axis  */
    unsigned int     dim[3];	/* Dimension of the volume */
    unsigned int     slc_size;	/* Size of a single slice = dim[0]*dim[1] */
    unsigned int     vol_size;  /* Size of the entire volume = dim[0]*dim[1]*dim[2] */
    unsigned int     slc_id;    /* which slice this is of the whole volume */
    unsigned int     num_slc;   /* number of slices */

    int     sub_orig[3];	        /* Origin of subvolume */
    int     sub_ext[3];	        /* Extension of subvolume */
    Point3d minb, maxb;         /* left and right corners of the subvolume bounding box */

    int   max_dens;		/* maximum density value */
    int   *xinc, *yinc; /* x and y incremental help table */
  } Volume;

  /**
   * Viewing Parameters
   */
  typedef struct {
    int persp;			/* Perspective or Orthographics Projection */

    float fov;			/* Field of View */
    Vector3d raydir;	/* Ray direction vector	*/
    Point3d eye;        /* Viewing eye point */

    Vector3d vup;		/* View up vector */
    Vector3d vpn;       /* View plane normal */
    Vector3d vpu;       /* vpu = vup x vpn */

    Point3d win_sp;				/* Start point of a window in the world coordinates */                        
    int win_width, win_height;  /* Height & Width of a view window */
    int pix_width, pix_height;	/* Height & Width of pixels in the rendered image */
    float pix_sz_x, pix_sz_y;   /* pixel size: pix_sz_x = win_width / pix_width ... */
  } Viewing;

  /**
   * Rendering Options
   */
  typedef struct {
    int n_material;			/* number of material types in the ray_casting rendering mode */
    int n_light;			/* number of lights */
    int n_plane;			/* number of cutting planes */
    int n_surface;			/* number of iso-surfaces */
    int n_colden;			/* number of density points in the transfer function */

    Material	mat[MAX_MATERIAL];
    Surface	surf[MAX_SURFACE];
    Light	light[MAX_LIGHT];
    Plane	plane[MAX_PLANE];

    RenderMode render_mode;
    int cut_plane;			/* Cutting Plane Option */
    int light_both;			/* Allow Light Both Direction */
    ColorMode color_mode;	/* Gray Scale or RGB Color */
 
    ColDenRamp coldenramp[MAX_COLDEN_RAMP];

    int min_den, max_den;
    float *opctbl;			/* opacity table */
    unsigned char *coldentbl;	/* Color map table for transfer function */

  } Rendering;

  /**
   * Volume Rendering Environment
   */
  typedef struct {
    Volume* vol;
    Viewing* view;
    Rendering* rend;
    RayMisc* misc;
    int	valid;		// error flag

    /* vrSplineNorm() vars (used to provide reentrancy...) */
    float val[4][4][4];
    int curidx[3];
    float Derivative[4][4][3][3];
    int notfirst;

  } VolRenEnv;

  /**
   * Multi-Volume Rendering
   */
  typedef struct {
    VolRenEnv *env[MAX_ENV];
    Volume* metavol; /* A volume that contains all the volumes.  This volume does
			not have it's own density values */
    int n_env;
  } MultiVolRenEnv;

  /**
   * Create a default volume rendering environment.
   * @note this function is not very useful; normally you should call vrCreateEnvFromFile()
   */
  VolRenEnv *vrCreateEnv(void);

  /**
   * Create a volume rendering environment from a config file.
   * slc_id is the slice id that this env will load.
   * num_slc is the number of slices that will be loaded over all processes.
   * @return Null if the creation failed
   */
  VolRenEnv *vrCreateEnvFromFile(char* fname, unsigned int slc_id, unsigned int num_slc);

  /**
   * Create a multiple volume rendering environment.
   */
  MultiVolRenEnv *vrCreateMultiEnv();

  /**
   * Add env to MultiVolRenEnv
   */
  void vrAddEnv(MultiVolRenEnv *menv, VolRenEnv *env);

  /**
   * Free memory associated with VolRenEnv
   */
  void vrCleanEnv(VolRenEnv* env);

  /**
   * Free memory associated with each env
   */
  void vrCleanMultiEnv(MultiVolRenEnv *menv);

  typedef struct _SubVolData{
    unsigned char  *data;
    int	 dim[3];
  } SubVol;

  /**
   * Read Config file.
   */
  void vrReadConfig(VolRenEnv* env, FILE* fp);

  /**
   * Write config file.
   */
  void vrWriteConfig(VolRenEnv *env, FILE *fp);

  /**
   * Load actual volume data.
   */
  void vrLoadVolume(VolRenEnv* env);

  /**
   * Test for existance of the volume file
   *
   */
  int vrVolumeExists(VolRenEnv *env);

  /**
   * Get the data of a subvolume.
   */
  SubVol* vrGetSubVolData(VolRenEnv* env, float min[3], float max[3]);

  /**
   * Create a new Image.
   */
  void vrCreateImage(ImageBuffer* img, int w, int h);

  void vrDestroyImage(ImageBuffer* img);

  /**
   * Save an image to a ppm file.
   */
  void vrSaveImg2PPM(ImageBuffer* img, char* fname);


#define QUANTIZE_SIZE 256
#define SHINING 10

  static float spec_val[QUANTIZE_SIZE+1] = {
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000000f, 
    0.0000000001f, 
    0.0000000001f, 
    0.0000000001f, 
    0.0000000002f, 
    0.0000000002f, 
    0.0000000003f, 
    0.0000000005f, 
    0.0000000007f, 
    0.0000000009f, 
    0.0000000013f, 
    0.0000000017f, 
    0.0000000023f, 
    0.0000000030f, 
    0.0000000040f, 
    0.0000000052f, 
    0.0000000067f, 
    0.0000000087f, 
    0.0000000111f, 
    0.0000000141f, 
    0.0000000179f, 
    0.0000000225f, 
    0.0000000282f, 
    0.0000000351f, 
    0.0000000435f, 
    0.0000000537f, 
    0.0000000660f, 
    0.0000000808f, 
    0.0000000985f, 
    0.0000001196f, 
    0.0000001447f, 
    0.0000001744f, 
    0.0000002095f, 
    0.0000002509f, 
    0.0000002995f, 
    0.0000003564f, 
    0.0000004228f, 
    0.0000005002f, 
    0.0000005901f, 
    0.0000006943f, 
    0.0000008147f, 
    0.0000009537f, 
    0.0000011136f, 
    0.0000012973f, 
    0.0000015078f, 
    0.0000017486f, 
    0.0000020234f, 
    0.0000023366f, 
    0.0000026927f, 
    0.0000030969f, 
    0.0000035549f, 
    0.0000040730f, 
    0.0000046581f, 
    0.0000053179f, 
    0.0000060605f, 
    0.0000068952f, 
    0.0000078320f, 
    0.0000088818f, 
    0.0000100566f, 
    0.0000113694f, 
    0.0000128346f, 
    0.0000144675f, 
    0.0000162851f, 
    0.0000183056f, 
    0.0000205491f, 
    0.0000230371f, 
    0.0000257929f, 
    0.0000288420f, 
    0.0000322117f, 
    0.0000359318f, 
    0.0000400341f, 
    0.0000445532f, 
    0.0000495264f, 
    0.0000549937f, 
    0.0000609983f, 
    0.0000675867f, 
    0.0000748087f, 
    0.0000827181f, 
    0.0000913722f, 
    0.0001008329f, 
    0.0001111662f, 
    0.0001224429f, 
    0.0001347390f, 
    0.0001481355f, 
    0.0001627189f, 
    0.0001785821f, 
    0.0001958237f, 
    0.0002145493f, 
    0.0002348714f, 
    0.0002569098f, 
    0.0002807920f, 
    0.0003066542f, 
    0.0003346407f, 
    0.0003649054f, 
    0.0003976115f, 
    0.0004329327f, 
    0.0004710532f, 
    0.0005121685f, 
    0.0005564857f, 
    0.0006042250f, 
    0.0006556189f, 
    0.0007109142f, 
    0.0007703720f, 
    0.0008342684f, 
    0.0009028956f, 
    0.0009765625f, 
    0.0010555953f, 
    0.0011403387f, 
    0.0012311566f, 
    0.0013284330f, 
    0.0014325730f, 
    0.0015440037f, 
    0.0016631753f, 
    0.0017905623f, 
    0.0019266641f, 
    0.0020720069f, 
    0.0022271443f, 
    0.0023926585f, 
    0.0025691618f, 
    0.0027572985f, 
    0.0029577450f, 
    0.0031712120f, 
    0.0033984459f, 
    0.0036402307f, 
    0.0038973885f, 
    0.0041707819f, 
    0.0044613164f, 
    0.0047699404f, 
    0.0050976477f, 
    0.0054454808f, 
    0.0058145304f, 
    0.0062059397f, 
    0.0066209044f, 
    0.0070606768f, 
    0.0075265658f, 
    0.0080199419f, 
    0.0085422369f, 
    0.0090949470f, 
    0.0096796378f, 
    0.0102979429f, 
    0.0109515702f, 
    0.0116423015f, 
    0.0123719964f, 
    0.0131425979f, 
    0.0139561314f, 
    0.0148147102f, 
    0.0157205369f, 
    0.0166759100f, 
    0.0176832248f, 
    0.0187449735f, 
    0.0198637564f, 
    0.0210422818f, 
    0.0222833678f, 
    0.0235899501f, 
    0.0249650832f, 
    0.0264119450f, 
    0.0279338416f, 
    0.0295342132f, 
    0.0312166363f, 
    0.0329848267f, 
    0.0348426551f, 
    0.0367941335f, 
    0.0388434343f, 
    0.0409948975f, 
    0.0432530195f, 
    0.0456224754f, 
    0.0481081195f, 
    0.0507149920f, 
    0.0534483157f, 
    0.0563135147f, 
    0.0593162142f, 
    0.0624622516f, 
    0.0657576770f, 
    0.0692087561f, 
    0.0728220046f, 
    0.0766041428f, 
    0.0805621594f, 
    0.0847032964f, 
    0.0890350342f, 
    0.0935651362f, 
    0.0983016342f, 
    0.1032528430f, 
    0.1084273756f, 
    0.1138341427f, 
    0.1194823608f, 
    0.1253815740f, 
    0.1315416247f, 
    0.1379727423f, 
    0.1446854621f, 
    0.1516907066f, 
    0.1589997262f, 
    0.1666242033f, 
    0.1745761633f, 
    0.1828680634f, 
    0.1915127486f, 
    0.2005234957f, 
    0.2099140435f, 
    0.2196985334f, 
    0.2298915833f, 
    0.2405083179f, 
    0.2515642941f, 
    0.2630755901f, 
    0.2750587761f, 
    0.2875310481f, 
    0.3005099893f, 
    0.3140138686f, 
    0.3280614316f, 
    0.3426720798f, 
    0.3578657508f, 
    0.3736630976f, 
    0.3900852799f, 
    0.4071542025f, 
    0.4248923957f, 
    0.4433231056f, 
    0.4624702632f, 
    0.4823584855f, 
    0.5030131936f, 
    0.5244604945f, 
    0.5467272997f, 
    0.5698413849f, 
    0.5938313007f, 
    0.6187263727f, 
    0.6445568204f, 
    0.6713537574f, 
    0.6991492510f, 
    0.7279761434f, 
    0.7578684092f, 
    0.7888609171f, 
    0.8209894300f, 
    0.8542908430f, 
    0.8888031244f, 
    0.9245651364f, 
    0.9616170526f, 
    1.0000000000f
  };


  /**
   * Do ray tracing for a tile.
   * @return the rendered image is stored in tile_image
   */
  void vrTracingTile(MultiVolRenEnv *menv, int tid, unsigned char* tile_image);

  /**
   * Do ray tracing for a single ray.
   * @param offset - offset in the tile_image
   */
  void vrFireOneRay(MultiVolRenEnv* menv, int nx, int ny, unsigned char* tile_image, int offset);

  /**
   * Compute an image with ray tracing.
   * @note Caller should release the returned image buffer 
   */
  ImageBuffer* vrRayTracing(MultiVolRenEnv* menv,int quiet);

  /**
   * Set Volume layout
   */
  void vrSetVolumeInfo(Volume* vol, char* fname, char* type, float span[3], int dim[3]);

  /**
   * Set SubVolume
   */
  void vrSetSubVolume(Volume* vol, int orig[3], int ext[3]);

  /**
   * Set Perspective projection properties.
   */
  void vrSetPerspective(Viewing* view, int persp, float fov);

  void vrSetEyePoint(Viewing* view, float x, float y, float z);

  void vrSetUpVector(Viewing* view, float x, float y, float z);

  void vrSetPlaneNormVector(Viewing* view, float x, float y, float z);

  void vrSetViewWindowSize(Viewing* view, int width, int height);

  void vrSetPixelWindowSize(Viewing* view, int width, int height);

  /**
   * Compute Other viewing parameters.
   * @note it should be called after call one up set method.
   */
  void vrComputeView(Viewing* view);

  /**
   * Add a new material type
   * @return 1 if successful, 0 otherwise
   */
  int vrAddMaterial(Rendering* rend, Volume* vol, Shading* shade, 
		    float opac, int start, int up, int down, int end);

  int vrSetMaterial(Rendering* rend, Volume* vol, int id, Shading* shade, 
		    float opac, int start, int up, int down, int end);

  void vrInitMaterial(Rendering* rend, Volume* vol);

  /**
   * Add a point in the density transfer curve
   * @return 1 if successful, 0 otherwise
   */
  int vrAddColDenRamp(Rendering* rend, int den, int r, int g, int b, int a);

  int vrSetColDenRamp(Rendering* rend, int id, int den, int r, int g, int b, int a);

  void vrSetColDenNum(Rendering *rend, int nd);

  void vrComputeColDen(Rendering* rend, Volume* vol);

  /**
   * Add a new isosurface
   * @return 1 if successful, 0 otherwise
   */
  int vrAddIsoSurface(Rendering* rend, Shading* shade, float opac, float isoval);

  int vrSetIsoSurface(Rendering* rend, int id, Shading* shade, float opac, float isoval);

  /**
   * Add a new Light
   * @return 1 if successful, 0 otherwise
   */
  int vrAddLight(Rendering* rend, int r, int g, int b, float x, float y, float z);

  int vrSetLight(Rendering* rend, int id, int r, int g, int b, float x, float y, float z);

  /**
   * Add a cutting plane
   * @return 1 if successful, 0 otherwise
   */
  int vrAddCuttingPlane(Rendering* rend, Point3d* orig, Vector3d* norm);

  int vrSetCuttingPlane(Rendering* rend, int id, Point3d* orig, Vector3d* norm);

  /**
   * Set rendering mode
   */
  void vrSetRenderingMode(Rendering* rend, int mode);

  void vrToggleCuttingPlane(Rendering* rend, int cut);

  void vrToggleLightBoth(Rendering* rend, int both);

  void vrSetColorMode(Rendering* rend, int colmode);

  /**
   * Set Misc options about the ray
   */
  void vrSetBackColor(RayMisc* misc, int r, int g, int b);

  void vrSetStepSize(VolRenEnv* env, int nstep);

  void vrSetRayMisc(VolRenEnv* env);

  /**
   * Set Gradient Table.
   * @note not sure what it does (xiaoyu)
   */
  void vrSetGradientTable(RayMisc* misc, int ramp0, int ramp1, int ramp2);

  /**
   * Set up ray parameters for isocontour computation.
   */
  void vrSetContourRay(iRay* ray, Viewing* view, float pnt[3]);

  /**
   * Normalize a vector.
   * @return the length of the vector
   */
  float vrNormalize(Vector3d *vector);

  /**
   * Copy a shading variable
   */
  void vrCopyShading(Shading* dest, Shading* src);

  /**
   * Compute the world coordinates of a pixel on the viewing window.
   */
  void vrGetPixCoord(Viewing* view, int nx, int ny, Point3d* pnt);

  /**
   * Compute the intersection points of a ray with the volume.
   * @return 1 if the ray intersects, 0 otherwise
   */
  int vrComputeIntersection(Volume* vol, Point3d* eye_p, Vector3d* dir_p, 
			    Point3d* start_pnt_p, Point3d* end_pnt_p);

  /** 
   * These functions are here to avoid calculating an x, y, or z value twice.
   * Used only in vrComputeIntersection()
   */
  void vrCalcIntersectTx(Point3d *ray_org, Vector3d *dir, float px, Point3d *pnt);

  void vrCalcIntersectTy(Point3d *ray_org, Vector3d *dir, float py, Point3d *pnt);

  void vrCalcIntersectTz(Point3d *ray_org, Vector3d *dir, float pz, Point3d *pnt);

  /**
   * Convert a point into 3D index of the volume.
   */
  //void vrPoint2Index(Volume* vol, Point3d* pnt, int idx[3]);

  /**
   * Get Density of normals of a point within a cell.
   */
  void vrGetDenNorm(VolRenEnv* env, float pnt[3], int idx[3], float* den, Vector3d *norm,
		    unsigned char *flag, float vals[8]);

  /**
   * Get densities on the eight vertices of a cell
   */
  void vrGetVertDensities(Volume* vol, int idx[3], float vals[8]);

  /**
   * Compute gradient of a given point using spline approximation
   */
  void vrSplineNorm(VolRenEnv* env, float w[3], int idx[3], Vector3d* norm);

  /**
   * Compute color and opacity at an intersection point with phong shading
   */
  void vrComputeColorOpacity(VolRenEnv* env, float pnt[3], int idx[3], float den, Vector3d* norm, float vals[8], float color[3], float* opac);

  /**
   * Compute color and opacity at an intersection point with interpolation.
   */
  void vrComputeColorOpacityInterp(VolRenEnv* env, float pnt[3], int idx[3], 
				   float vals[8], float color[3], float *opac);

  /**
   * Compute color and opacity for an isosurface.
   * @param n the nth isosurface
   */
  void vrComputeColorOpacitySurf(VolRenEnv* env, float pnt[3], int idx[3], int n, iRay *ray, Cell* cell, 
				 float vals[8], float color[3], float* opac);

  /**
   * Compute color with Phong shading.
   */
  void vrPhongShading(VolRenEnv *env, float color[3], Shading* shade, Vector3d* norm);

  /**
   * Copy the tile into img at the location specified by tid
   */
  void vrCopyTile(ImageBuffer* img, int tid, unsigned char* tile);

  /**
   * Trilinear interpolaton.
   */
  float vrTriInterp(float w[3], float vals[8]);

  /**
   * timing
   */
  double vrGetTime();

  /**
   * Returns the size of a RawV datatype
   */
  static int rawv_sizes[] = { 1, 2, 4, 4, 8 };
#define vrRawVSizeOf(rvt) rawv_sizes[(int)rvt-1]

  /**
   * Returns a pointer to the value for a variable at a specific index
   */
  void *vrRawVGetValue(Volume *vol, int timestep, unsigned int var, int idx[3]);

  /**
   * Gets the color of an index using the first 3 variables as color values
   */
  void vrRawVGetColor(Volume *vol, int timestep, int idx[3], float color[3]);

  /**
   * Gets the color of all surrounding verticies.
   */
  void vrRawVGetVertColor(Volume *vol, int timestep, int idx[3], float color[3][8]);

  /**
   * Gets the density value at an index
   */
  void vrRawVGetDensity(Volume *vol, int timestep, int idx[3], float *dens);

  /**
   * Returns a pointer to a copy of menv
   */
  MultiVolRenEnv *vrCopyMenv(MultiVolRenEnv *menv);

  /**
   * Returns a pointer to a copy of env
   */
  VolRenEnv *vrCopyEnv(VolRenEnv *env);
#ifdef __cplusplus
}
#endif

#endif
