#ifndef _UTILITIES_H
#define _UTILITIES_H
//#include <stdio.h>
//#include <math.h>
//#include <string.h>
#include <sys/stat.h>


#include <fftw3.h>
#include <Reconstruction/Reconstruction.h>


//#define PI  3.14159265358979323846264338327950288
#define PI2 6.28318530717958647692528676655900576
#define SQRTPI2 2.5066282746310005024157

//#define CVC_DBL_EPSILON 1.0e-9
#define CVC_DBL_EPSILON 1.0e-16
#define SMALLFLOAT              1e-5
#define TRIGPRECISION   1e-5
#define MAXFLOAT  1e30



#ifndef M_PI 
#define M_PI    3.14159265358979323846264338327950288
#endif
#ifndef M_PI_2 
#define M_PI_2  1.57079632679489661923132169163975144
#endif
#ifndef TWOPI 
#define TWOPI   6.28318530717958647692528676655900576
#endif

#define DEG2RAD(d) ((d) * M_PI / 180.0)



/*

Kot[0] = (-0.8611363115940526L + 1.0L)/2.0L;
Kot[1] = (-0.3399810435848563L + 1.0L)/2.0L;
Kot[2] =  (0.3399810435848563L + 1.0L)/2.0L;
Kot[3] =  (0.8611363115940526L + 1.0L)/2.0L;

Weit[0] = 0.3478548451374539L/2.0L;
Weit[1] = 0.6521451548625461L/2.0L;
Weit[2] = 0.6521451548625461L/2.0L;
Weit[3] = 0.3478548451374539L/2.0L;
*/

//float*   VertArray;
//int*     FaceArray,numbpts,numbtris;
struct Oimage 
{
   int     nx, ny, nz, c; // Dimensions ,x y z and channels.
    float  avg, std;   // Average and standard deviation. 
   float   ux, uy, uz; // Voxel units (angstrom/pixel edge)
   float*  data;                // pixel values at each grid.
};


static struct RawIVHeader
{
  float min[3];
  float max[3];
  unsigned int numVerts;
  unsigned int numCells;
  unsigned int dim[3];
  float origin[3];
  float span[3];
} rawivHeader;


  typedef struct
    {
        /* 1 */
        float fNslice; // NUMBER OF SLICES (PLANES) IN VOLUME
        // (=1 FOR AN IMAGE)  FOR NEW LONG LABEL
        // FORMAT THE VALUE OF NSLICE STORED IN
        // THE FILE IS NEGATIVE.

        /* 2 */
        float fNrow; // NUMBER OF ROWS PER SLICE (Y)

        /* 3 */
        float fNrec; // TOTAL NUMBER OF RECORDS (SEE NOTE #3).

        /* 4 */
        float fNlabel; // AUXILIARY NUMBER TO COMPUTE TOTAL NUMBER OF RECS

        /* 5 */
        float fIform; // FILE TYPE SPECIFIER.
        // +3 FOR A 3-D FILE  (FLOAT)
        // +1 FOR A 2-D IMAGE (FLOAT)
        // -1 FOR A 2-D FOURIER TRANSFORM
        // -3 FOR A 3-D FOURIER TRANSFORM
        // -5 FOR A NEW 2-D FOURIER TRANSFORM
        // -7 FOR A NEW 3-D FOURIER TRANSFORM
        // +8 FOR A 2-D EIGHT BIT IMAGE FILE
        // +9 FOR A 2-D INT IMAGE FILE
        // 10 FOR A 3-D INT IMAGE FILE
        // 11 FOR A 2-D EIGHT BIT COLOR IMAGE FILE

        /* 6 */
        float fImami; // MAXIMUM/MINIMUM FLAG. IS SET AT 0 WHEN THE
        // FILE IS CREATED, AND AT 1 WHEN THE MAXIMUM AND
        // MINIMUM HAVE BEEN COMPUTED, AND HAVE BEEN STORED
        // INTO THIS LABEL RECORD (SEE FOLLOWING WORDS)

        /* 7 */
        float fFmax; // MAXIMUM VALUE

        /* 8 */
        float fFmin; // MINIMUM VALUE

        /* 9 */
        float fAv; // AVERAGE VALUE

        /* 10*/
        float fSig; // STANDARD DEVIATION. A VALUE OF -1. INDICATES
        // THAT SIG HAS NOT BEEN COMPUTED PREVIOUSLY.

        /* 11*/
        float fIhist; // FLAG INDICATING IF THE HISTOGRAM HAS BE
        // COMPUTED. NOT USED IN 3D FILES!

        /* 12*/
        float fNcol; // NUMBER OF PIXELS PER LINE (Columns X)

        /* 13*/
        float fLabrec; // NUMBER OF LABEL RECORDS IN FILE HEADER

        /* 14*/
        float fIangle; // FLAG THAT TILT ANGLES HAVE BEEN FILLED

        /* 15*/
        float fPhi; // EULER: ROTATIONAL ANGLE

        /* 16*/
        float fTheta; // EULER: TILT ANGLE

        /* 17*/
        float fPsi; // EULER: PSI  = TILT ANGLE

        /* 18*/
        float fXoff; // X TRANSLATION

        /* 19*/
        float fYoff; // Y TRANSLATION

        /* 20*/
        float fZoff; // Z TRANSLATION

        /* 21*/
        float fScale; // SCALE

        /* 22*/
        float fLabbyt; // TOTAL NUMBER OF BYTES IN LABEL

        /* 23*/
        float fLenbyt; // RECORD LENGTH IN BYTES
        char  fNada[24]; // this is a spider incongruence

        /* 30*/
        float fFlag; // THAT ANGLES ARE SET. 1 = ONE ADDITIONAL
        // ROTATION IS PRESENT, 2 = ADDITIONAL ROTATION
        // THAT PRECEEDS THE ROTATION THAT WAS STORED IN
        // 15 FOR DETAILS SEE MANUAL CHAPTER VOCEUL.MAN

        /* 31*/
        float fPhi1;

        /* 32*/
        float fTheta1;

        /* 33*/
        float fPsi1;

        /* 34*/
        float fPhi2;

        /* 35*/
        float fTheta2;

        /* 36*/
        float fPsi2;

        double fGeo_matrix[3][3]; // x9 = 72 bytes: Geometric info
        float fAngle1; // angle info

        float fr1;
        float fr2; // lift up cosine mask parameters

        /** Fraga 23/05/97  For Radon transforms **/
        float RTflag; // 1=RT, 2=FFT(RT)
        float Astart;
        float Aend;
        float Ainc;
        float Rsigma; // 4*7 = 28 bytes
        float Tstart;
        float Tend;
        float Tinc; // 4*3 = 12, 12+28 = 40B

        /** Sjors Scheres 17/12/04 **/
        float Weight; // For Maximum-Likelihood refinement
        float Flip; // 0=no flipping operation (false), 1=flipping (true)

        char fNada2[576]; // empty 700-76-40=624-40-8= 576 bytes

        /*212-214*/
        char szIDat[12]; // LOGICAL * 1 ARRAY DIMENSIONED 10, CONTAINING
        // THE DATE OF CREATION (10 CHARS)

        /*215-216*/
        char szITim[8]; // LOGICAL * 1 ARRAY DIMENSIONED 8, CONTAINING
        // THE TIME OF CREATION (8 CHARS)

        /*217-256*/
        char szITit[160]; // LOGICAL * 1 ARRAY DIMENSIONED 160
    }
    SpiderHeader;






void ObtainRotationMatFromViews(float* Rmat, struct Views* view);
void  InverseRotationMatrix(float *Rmat,int nv);
float angle_set_negPI_to_PI(float angle);
float bcos(float x);
float bsin(float x);





void ObtainRotationMatrixAroundOrigin(float* rotmat,struct Views* v);
void Matrix3Transpose(float* matrix, float* trmatrix);
void MatrixTranspose(float *A,int m,int n);

void MatrixMultiply(float* A1,int m1,int n1,float* A2,int m2,int n2,float* Affi);

Views*  ObtainViewVectorFromTriangularSphere(char *filename);
void    ReadRawFile(char *filename);
float   InnerProduct(float u[],float v[]);
long double InnerProductl(long double *u,long double *v, int n);
void    gaussinverse (long double* a, int n, long double eps,int* message);
void    Exchangerowcolumn(long double* a, int n,int k, int ik, int jk);
Views*  quaternion_from_view_vector(Views* view);
float  Angle_Of_Two_Vectors(float* p12, float* p32);
void RotateMatrix_z(float nx, float ny, float nz, float *matrix);

float TrilinearInterpolation8(float xd, float yd, float zd, float x1, float x2, float x3, float x4, float x5, float x6, float x7, float x8);


bool fft1D_shift(fftw_complex *Vec,int n);

bool fft2D_shift(fftw_complex *Vec,int m, int n);


void EulerMatrice(float *Rmat,struct EulerAngles *Eulers);

void euler2matrix(float alpha, float beta, float gamma, float A[9]);

inline void euler2view(float rot, float tilt, float psi, float view[3]);

EulerAngles* phantomEulerAngles(float p1, float p2, float p3);

EulerAngles* phantomEulerLimitedAngles(float p1, float p2, float p3);


void ProjectPoint2Plane(float point[3], float e1[3], float e2[3]);


float MaxError(fftw_complex * ar1, fftw_complex * ar2, int size);

float MaxError_L2(fftw_complex * ar1, fftw_complex * ar2, int size);

void  Volume_Projection(Oimage *Volume, float rotmat[9], int sample_num, float *prjimg);

void  Volume_GridProjection(Oimage* Volume, float rotmat[9], int sample_num, float *prjimg);


void Imageinterpo_BiLinear(float *data, int nx, int ny);
void Imageinterpo_BiLinear(float *data, int nx, int ny, float *result);


void WriteSpiFile(char* filename, float* data, int nx, int ny, int nz, float rot, float tilt, float psi);
//void  readSpiFile(float* data, long memsize, char* spifile, EulerAngles* eulers);
size_t FREAD(void *dest, size_t size, size_t nitems, FILE *&fp, int reverse);
int readSpiderFile(const char *filename, char filetype, int dim[], float **data, EulerAngles* eulers);



void ReadRawIVFile(const char* filename, const char* path, float *data); 
#endif
