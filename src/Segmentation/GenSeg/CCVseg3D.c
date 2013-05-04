/**************************************
 * Zeyun Yu (zeyun@cs.utexas.edu)    *
 * Department of Computer Science    *
 * University of Texas at Austin     *
 **************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/times.h>

#define IndexVect(i,j,k) ((k)*xdim*ydim + (j)*xdim + (i))
#define max(x, y) ((x>y) ? (x):(y))
#define min(x, y) ((x<y) ? (x):(y))

typedef struct SeedPoint SDPNT;
struct SeedPoint{
  int x;
  int y;
  int z;
  SDPNT *next;
};


void Segment(int, int,int, float *, float, float, SDPNT **, int);
void read_data(int* , int*, int*, float**, const char *);



int main(int argc, char *argv[])
{
  int xdim,ydim,zdim;
  float *image;
  SDPNT **seed_list;
  float tlow, thigh;
  int num;


  if (argc != 4){
    printf("Usage: CCVseg <input_filename> <tlow> <thigh>\n");
    printf("       <input_filename>:   RAWIV file \n");
    printf("       <tlow>: thresholds for segemntation (0-255) \n");
    printf("       <thigh>: thresholds (0-255) for valid seed points \n");
    exit(0);              
  }

  
  printf("begin reading rawiv.... \n");
  read_data(&xdim,&ydim,&zdim,&image,argv[1]);

    
  tlow = atof(argv[2]);  
  thigh = atof(argv[3]);
  

  /* seed_list is the list of seed points. num is the number of classes */
  printf("begin segmentation ....\n");
  num = 4;
  Segment(xdim,ydim,zdim, image, tlow, thigh, seed_list,num);
  
  
  
  return(0);
}

