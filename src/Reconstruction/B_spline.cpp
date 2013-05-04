/*
All right are reserved by The National Key Lab of Scientific and Engineering Computing,
Chinese Academy of Sciences.

        Author: Ming Li <ming@lsec.cc.ac.cn>
        Advisor: Guoliang Xu <xuguo@lsec.cc.ac.cn>

This file is part of CIMOR. CIMOR stands for:
        Computational Inverse Methods of Reconstruction

*/



#include <stdio.h>
#include <math.h>
//#include <malloc.h>
#include <stdlib.h>
#include <Reconstruction/B_spline.h>
#include <Reconstruction/utilities.h>

//int N = 2; 
//int M = 7;
//int N3 = 3*(N+1);
//int M1 = 3*(M+1);
//int N1 = (N + 1);

extern int PRO_LENGTH, PRO_LENGTH_SUB, SUB;

Bspline::Bspline()
{

Kot[0] = (-0.8611363115940526L + 1.0L)/2.0L;
Kot[1] = (-0.3399810435848563L + 1.0L)/2.0L;
Kot[2] =  (0.3399810435848563L + 1.0L)/2.0L;
Kot[3] =  (0.8611363115940526L + 1.0L)/2.0L;

Weit[0] = 0.3478548451374539L/2.0L;
Weit[1] = 0.6521451548625461L/2.0L;
Weit[2] = 0.6521451548625461L/2.0L;
Weit[3] = 0.3478548451374539L/2.0L;


 BernBase       = NULL;
 BBaseGN        = NULL;
 OrthoBBaseGrid = NULL;
 OrthoBBaseGN   = NULL; 
 SchmidtMat     = NULL;
 InvSchmidtMat  = NULL;
}

Bspline::~Bspline()
{
//delete [] BernBase;
//delete [] OrthoBBaseGN ;
//delete [] OrthoBBaseGrid;
//delete [] SchmidtMat;
}

/************************************************************************/
long double* Bspline::GramSchmidtBsplineBaseAtGrids(int bN, int nx, int gM)
{
   long double *u, **v, **u1, *v1;
   long double x, xx, inn, norm, values[6], sum;
   long double eps = 1.0L, invdelx;
  int         i, j, k;
  int         message;

  invdelx -= 1.0L/delx;

 v  = (long double**)malloc((bN+1) *sizeof(long double));
 u1 = (long double**)malloc((bN+1) *sizeof(long double));

 u  = (long double*)malloc( (bN+1) *nx*sizeof(long double));
 v1 = (long double*)malloc( (bN+1) *nx*sizeof(long double));


 SchmidtMatG = new long double [(bN+1)*(bN+1)];    
 for( i = 0; i < (bN+1)*(bN+1); i++ ) SchmidtMatG[i] = 0.0L;


 for ( i = 0; i <= bN; i++ )
    v[i] = (long double *)malloc(nx*sizeof(long double));

 for ( i = 0; i <=bN; i++ )
    u1[i] = (long double *)malloc(nx*sizeof(long double));


 for ( i = 0; i <=bN; i++ )
    for ( j = 0; j < nx; j++ )
       v[i][j] = 0.0L;

 // delx = 1.0L/(nx-1.0L);  

for (i = 0; i < nx; i++) 
    {

     xx = delx*i;
	 // printf("\nxx=%Lf ", xx);getchar();
     for (j = 0; j <= bN; j++) 
        {
		  x = invdelx*xx - j + 1.0L;  //j-1  is the shift for prof.xu's \beta^3(x,shift) bspline.

         Spline_N_Base(x, nx-1, values);
		 values[1] = values[1]*invdelx;
		 values[2] = values[2]*invdelx*invdelx;

  
         u[i*(bN+1)+j] = values[0]; 
         v[j][i]       = values[0];
         u1[j][i]      = values[0];
         //if( j == 8 ) printf("\nj = %d v[j][i] = %Lf ", j,v[j][i]);
        }
    }


for ( j = 0; j <= bN; j++ )
    {
     if( j == 0 ) 
        {
         for( k = 0; k < nx; k++ )
           {
            v[j][k] = u[k*(bN+1)+j];
           } 
        }

     else 
        {
 
         for ( i = 0; i < j; i++ )
            {
               for ( k = 0; k < nx; k++ )
			   printf("\nv[i] = %Lf, v[j] = %Lf ", v[i][k], v[j][k]);

			  inn  = InnerProductl(v[i], v[j], nx);
			  printf("\ninn=%Lf  i=%d  j=%d", inn, i,j);//getchar();

          
             for ( k = 0; k < nx; k++ )
                 v[j][k] = v[j][k] -  inn * v[i][k];
            }
        }

   
     norm = InnerProductl(v[j], v[j], nx);
	 printf("\nj = %d norm=%Lf ", j,norm);
     norm = sqrtl(norm); 

     SchmidtMatG[j*(bN+1)+j] = norm;

     for ( i = 0; i <j; i++ )
        {
		  inn = InnerProductl(u1[j], v[i], nx);
         SchmidtMatG[j*(bN+1)+i] = inn;

        }     



     for ( k = 0; k < nx; k++ )
        {
		  v[j][k] = v[j][k]/norm;
        }
     }



 for ( j = 0; j <= bN; j++ )
    for ( k = 0; k < nx; k++ )
        {
         v1[j*nx+k] = v[j][k];
         
        }

printf("\nTest for orthogonal.\n");

for( i = 0; i <bN; i++ )
   for( j = i+1; j <=bN; j++)
   {
	 norm = InnerProductl(v[i], v[j], nx);
    if(fabsl(norm) > SMALLFLOAT)  
    printf("\n v[%d] and  v[%d]  are not orthogonal" , i, j);
   }


printf("\n original functions u.");
for ( i = 0; i <=bN; i++)
    {
     printf("\n");
     for ( j = 0; j < nx; j++ )
       printf("%Lf ",u[j*(bN+1)+i]);
    }


/*
printf("\n \n  Test inverse Schmidt Matrix  u = SchmidtMat * v. the output is u ");
for( i = 0; i <=bN; i++ )
   {
    printf("\n");
   for( k = 0; k < nx; k++)
      {
       sum = 0.0L;
       for( j = 0; j <= bN; j++ )
          sum = sum + SchmidtMat[i*(bN+1)+j]*v1[j*nx+k];
 
	    if(fabs(u[k*(bN+1)+i]-sum)>SMALLFLOAT)  
	   printf("%Lf " , sum);//, u[k*(bN+1)+i]);
      }
   }

 getchar();
*/

gaussinverse(SchmidtMatG, bN+1, eps,&message); 
//printf("\nOK.");getchar();

if ( message == 0) 
   {
    printf("message = 0 ");return NULL;
   }

return SchmidtMatG;
  
}

/******************************************************************************
Descriptions:

Arguments:
   gM         :  The total Gauss nodes 0,1,...,gM.
   bN         :  The total Bspline Base Functions number, bN + 1 on [0,1].
   u          :  The total input Bspline Base Functions values at all Gauss nodes.
   v          :  The results of Gram Schmidt Orthogonalization.
   SchmidtMat :  The output result is the Schmidt process matrix. v = SchmidtMat * u.
   

*******************************************************************************/
long double* Bspline::GramSchmidtofBsplineBaseFunctions(int bN,int gM)
{
 unsigned long  i,j,k,ii, numb_in, mod;
 long double    *v1, *u, x, xx,inn,norm, values[6],sum;
 long double    **v,**u1;
 long double    eps = 0.0L, invdelx;
 int            message;

 invdelx = 1.0L/delx;

 mod = (gM + 1)%4;
 if ( mod != 0) {printf("\n Error. (%d+1) must is not the multiple of 4.", gM);return NULL;}

 if ( (nx-1)*4 != (gM+1)) {printf("\n Error. Not a complete Bspline Base Function set."); return NULL;}

 v  = (long double**)malloc((bN+1) *sizeof(long double));
 u  = (long double*)malloc( (bN+1) *(gM + 1)*sizeof(long double));
 v1 = (long double*)malloc( (bN+1) *(gM + 1)*sizeof(long double));
 u1 = (long double**)malloc((bN+1) *sizeof(long double));

 SchmidtMat = new long double [(bN+1)*(bN+1)];            //(long double *)malloc((bN+1)*(bN+1)*sizeof(long double));
 InvSchmidtMat = new long double [(bN + 1) * (bN + 1)];

 for( i = 0; i < (bN+1)*(bN+1); i++ ) SchmidtMat[i] = 0.0L;

 for ( i = 0; i <=bN; i++ )
    v[i] = (long double *)malloc((gM+1)*sizeof(long double));

 for ( i = 0; i <=bN; i++ )
    u1[i] = (long double *)malloc((gM+1)*sizeof(long double));


 for ( i = 0; i <=bN; i++ )
    for ( j = 0; j <= gM; j++ )
       v[i][j] = 0.0L;


 //delx = 4.0L/(gM+1.0L);  
 for (i = 0; i <= gM; i++) 
    {

     numb_in = i/4;
     ii = i - 4*numb_in;
     xx = delx*numb_in + delx*Kot[ii];

     for (j = 0; j <= bN; j++) 
        {

		  // x = (bN-2) * xx - j + 1.0L; //j-1 is the shift for Prof. Xu 's Bspline(u, shift).

		  x = invdelx * xx -j;  //j is the bspline \Beta(x,shift)'s shift when nx = N+1;

           Spline_N_Base(x, nx-1, values);
		   values[1] = values[1]*invdelx;
		   values[2] = values[2]*invdelx*invdelx;

		  //		  Spline_N_i(xx, j, bN-2, values); // bspline for boundary conditions.
  
         u[i*(bN+1)+j] = values[0]; 
         v[j][i]       = values[0];
         u1[j][i]      = values[0];
        }
    }



 for ( j = 0; j <= bN; j++ )
    {
     if( j == 0 ) 
        {
         for( k = 0; k <= gM; k++ )
           {
            v[j][k] = u[k*(bN+1)+j];
            //printf("\n k*(bN+1)+j=%d  u =%Lf  v = %Lf ",k*(bN+1)+j,u[k*(bN+1)+j], v[j][k]);
           } 
        }

     else 
        {
 
         for ( i = 0; i < j; i++ )
            {
			  // inn  = InnerProductl(v[i], v[j],gM);
			  inn = EvaluateIntegralInnerProducts(v[i], v[j], gM);

             for ( k = 0; k <= gM; k++ )
                 v[j][k] = v[j][k] -  inn * v[i][k];
            }
        }


       
     //norm = InnerProductl(v[j], v[j],gM);
	 norm = EvaluateIntegralInnerProducts(v[j], v[j], gM);

     norm = sqrtl(norm); 

     SchmidtMat[j*(bN+1)+j] = norm;
     for ( i = 0; i <j; i++ )
        {
		  //inn =InnerProductl(u1[j], v[i],gM); 
		  inn = EvaluateIntegralInnerProducts(u1[j], v[i], gM);
         SchmidtMat[j*(bN+1)+i] = inn;

        }     


   //printf("\n j = %d norm = %Lf ",j, norm);


     for ( k = 0; k <= gM; k++ )
        {
         v[j][k] = v[j][k]/norm;
         //printf("\n v/norm = %Lf ", v[j][k]);
        }
    }



 for ( j = 0; j <= bN; j++ )
    for ( k = 0; k <= gM; k++ )
        {
         v1[j*(gM+1)+k] = v[j][k];
         //printf("\n v1 = %Lf ", v1[j*(gM+1)+k]);
        }


printf("\nTest for orthogonal.\n");

for( i = 0; i <bN; i++ )
   for( j = i+1; j <=bN; j++)
   {
	 //norm = InnerProductl(v[i], v[j],gM);
	 norm = EvaluateIntegralInnerProducts(v[i], v[j], gM);
    if(fabsl(norm) > 0.000001L)  
    printf("\nv[%d] and  v[%d]  are not orthogonal" , i, j);
   }




printf("\n original functions u.");
for ( i = 0; i <=bN; i++)
    {
     printf("\n");
     for ( j = 0; j <=gM; j++ )
       printf("%Lf ",u[j*(bN+1)+i]);
    }

/*

printf("\n \n  Test inverse Schmidt Matrix  u = SchmidtMat * v. the output is u ");
for( i = 0; i <=bN; i++ )
   {
    printf("\n");
   for( k = 0; k <=gM; k++)
      {
       sum = 0.0L;
       for( j = 0; j <= bN; j++ )
          sum = sum + SchmidtMat[i*(bN+1)+j]*v1[j*(gM+1)+k];
 
	   // if(fabs(u[k*(bN+1)+i]-sum)>SMALLFLOAT)  
	   printf("%Lf " , sum);//, u[k*(bN+1)+i]);
      }
   }

*/


 for ( i = 0; i < (bN + 1)*(bN + 1); i++ )
   InvSchmidtMat[i] = SchmidtMat[i];
 
gaussinverse(SchmidtMat, bN+1, eps,&message); 

/*


 printf("\nv1");
   for( j = 0; j <=bN; j++)
      {
		printf("\n");
       for( k = 0; k <= gM; k++ )
		 printf("%Lf " ,v1[j*(gM+1)+k]);
      }
   

printf("\n \n  Test Schmidt Matrix  v = SchmidtMat * u. the output is v ");
for( i = 0; i <=bN; i++ )
   {
    printf("\n");
   for( k = 0; k <=gM; k++)
      {
       sum = 0.0L;
       for( j = 0; j <= bN; j++ )
          sum = sum + SchmidtMat[i*(bN+1)+j]*u[k*(bN+1)+j];
 
	    if(fabs(v1[i*(gM+1)+k]-sum)>SMALLFLOAT)  
	   printf("%Lf " , sum);//, u[k*(bN+1)+i]);
      }
   }


 printf("\nInvSchmidtMat");
 for ( i = 0; i <= bN; i++ )
   {
    printf("\n");
   for ( j = 0; j <= bN; j++ )
	 printf(" %Lf ", InvSchmidtMat[i*(bN+1)+j]);
   }

*/
 if ( message == 0) 
   {
    printf("message = 0 ");return NULL;
   }

 /*
printf("\n\n  Schmidt Matrix  A :");

for( i = 0; i <=bN; i++ )
   {
    printf("\n");

    for( j = 0; j <= bN; j++ )
       {
        printf(" %Lf  ", SchmidtMat[i*(bN+1)+j]);
       }
   }

 printf("\nOK!");
 getchar();
 */
/*
printf("\n\n Test Schmidt Matrix A. v = A u. the output is v ");
for( i = 0; i <=bN; i++ )
   {
    printf("\n");
   for( k = 0; k <=gM; k++)
      {
       sum = 0.0L;
       for( j = 0; j <= bN; j++ )
          sum = sum + SchmidtMat[i*(bN+1)+j]*u[k*(bN+1)+j];
       printf("%Lf  " , sum);
      }
   }

*/
/*
 for ( i = 0; i <=bN; i++ )
    free(v[i]);

 for ( i = 0; i <=bN; i++ )
    free(u1[i]);


free(u); free(v); free(u1); free(v1);
*/
return SchmidtMat;


}

//Type 2. orthogonorize at interval -2,-1, 0 , 1, 2, 3, ..., nx, nx+1, nx+2.
long double* Bspline::GramSchmidtofBsplineBaseFunctions2()
{

 int            i,j,k,ii, numb_in, mod;
 long double    *v1, *u,  x, xx,inn,norm, values[6],sum;
 long double    **v,**u1, invdelx;
 long double    eps = 0.0L;
 int            message;
 int  gM2, usedN;

 //gM2 = 4*(nx+3)-1; 
 
 usedN = N - 4;
 invdelx = 1.0L/delx;


 mod = (M + 1)%4;
 if ( mod != 0) {printf("\n Error. (%d+1) must is not the multiple of 4.", M);return NULL;}

 if ( (nx-1)*4 != (M+1)) {printf("\n Error. Not a complete Bspline Base Function set."); return NULL;}

 v  = (long double**)malloc((usedN+1) *sizeof(long double));
 u1 = (long double**)malloc((usedN+1) *sizeof(long double));

 u  = (long double*)malloc( (usedN+1) *(M + 1)*sizeof(long double));
 v1 = (long double*)malloc( (usedN+1) *(M + 1)*sizeof(long double));

 SchmidtMat = new long double [(usedN+1)*(usedN+1)];            //(long double *)malloc((bN+1)*(bN+1)*sizeof(long double));
 InvSchmidtMat = new long double [(usedN + 1) * (usedN + 1)];

 for( i = 0; i < (usedN+1)*(usedN+1); i++ ) 
   {SchmidtMat[i] = 0.0L;InvSchmidtMat[i] = 0.0L;}

 for ( i = 0; i <=usedN; i++ )
    v[i] = (long double *)malloc((M+1)*sizeof(long double));

 for ( i = 0; i <=usedN; i++ )
    u1[i] = (long double *)malloc((M+1)*sizeof(long double));


 for ( i = 0; i <=usedN; i++ )
    for ( j = 0; j <= M; j++ )
       v[i][j] = 0.0L;




 //delx = 4.0L/(gM+1.0L);  
 // for (i = 0; i <= gM2; i++) 
 for (i = 0; i <= M; i++)  
   {

	 numb_in = i/4;   
	 ii = i - 4*numb_in;
	 
     xx = delx*(numb_in) + delx*Kot[ii];
	 // printf("\ndelx = %Lf, xx=%Lf  ", delx, xx);
	 
	 for (j = 2; j <= N - 2; j++) 
	   //for ( j = StartX; j <= FinishX; j++ )
	   {
		 x = invdelx * xx -j; 
		 // printf("x = %Lf", x);		  
		 Spline_N_Base(x, nx-1, values);
		 values[1] = values[1]*invdelx;
		 values[2] = values[2]*invdelx*invdelx;
		 
		 
		 u[i*(usedN+1)+j-2] = values[0]; 
		 v[j-2][i]       = values[0];
		 u1[j-2][i]      = values[0];
	   }
   }
 



 for ( j = 0; j <= usedN; j++ )
   {
     if( j == 0 ) 
	   {
         for( k = 0; k <= M; k++ )
           {
			 v[j][k] = u[k*(usedN+1)+j];
		   } 
	   }
     else 
        {
 
         for ( i = 0; i < j; i++ )
            {

			  inn = EvaluateIntegralInnerProducts2(v[i], v[j], M, M);
			  for ( k = 0; k <= M; k++ )
				v[j][k] = v[j][k] -  inn * v[i][k];
            }
        }

	 norm = EvaluateIntegralInnerProducts2(v[j], v[j], M, M);

     norm = sqrtl(norm); 

     SchmidtMat[j*(usedN+1)+j] = norm;
     for ( i = 0; i <j; i++ )
        {
		  //inn =InnerProductl(u1[j], v[i],gM); 
		  inn = EvaluateIntegralInnerProducts2(u1[j], v[i], M, M);
         SchmidtMat[j*(usedN+1)+i] = inn;

        }
     
    for ( k = 0; k <= M; k++ )
        {
         v[j][k] = v[j][k]/norm;
         //printf("\n v/norm = %Lf ", v[j][k]);
        }
    }


	
 for ( j = 0; j <= usedN; j++ )
    for ( k = 0; k <= M; k++ )
        {
         v1[j*(M+1)+k] = v[j][k];
         //printf("\n v1 = %Lf ", v1[j*(gM+1)+k]);
        }



 
 /*
printf("\nTest for orthogonal.\n");

for( i = 0; i <usedN; i++ )
   for( j = i+1; j <=usedN; j++)
   {
	 norm = EvaluateIntegralInnerProducts2(v[i], v[i], M, M);
	 //norm = EvaluateIntegralInnerProducts2(v[i], v[j], M, M);
	 //if(fabsl(norm) > 0.000001)  
	 //printf("\nv[%d] and  v[%d]  are not orthogonal" , i, j);
	 printf("\nnorm = %Lf", norm);
   }

 getchar();

printf("\n original functions u.");
for ( i = 0; i <=usedN; i++)
    {
     printf("\n");
     for ( j = 0; j <=M; j++ )
       printf("%Lf ",u[j*(usedN+1)+i]);
    }
 
 getchar();
 */
 for ( i = 0; i < (usedN + 1)*(usedN + 1); i++ )
   InvSchmidtMat[i] = SchmidtMat[i];
 
gaussinverse(SchmidtMat, usedN+1, eps,&message); 

 if ( message == 0) 
   {
    printf("message = 0 ");return NULL;
   }

 /*
for( i = 0; i <=usedN; i++ )
   {
    printf("\n");

    for( j = 0; j <= usedN; j++ )
       {
        printf(" %Lf  ", SchmidtMat[i*(usedN+1)+j]);
       }
   }

 printf("\nOK!");
 getchar();
 
 
printf("\n \n  Test Schmidt Matrix  v = SchmidtMat * u. the output is v ");
for( i = 0; i <=usedN; i++ )
   {
    printf("\n");
   for( k = 0; k <=M; k++)
      {
       sum = 0.0L;
       for( j = 0; j <= usedN; j++ )
          sum = sum + SchmidtMat[i*(usedN+1)+j]*u[k*(usedN+1)+j];
 
	    if(fabs(v1[i*(M+1)+k]-sum)>SMALLFLOAT)  
	   printf("%Lf " , sum);//, u[k*(N+1)+i]);
      }
   }

 printf("\nOK.");
 getchar();
 
 */
}



/******************************************************************************
Descriptions:
    Evalute inner product in function space.
    <f1, f2> = \integral f1 * f2 dx;

Algorithms: Use Gauss quadrature to compute the integration.
            discretize each function into vector, and then use gauss quadrature
            formula.
*******************************************************************************/
long double Bspline::EvaluateIntegralInnerProducts(long double *u1, long double *u2, int gM)
{

 unsigned long  i,j,k, ii, numb_in;
 long double sum = 0.0L;
 
//for( i = 0; i <= gM; i++ )
//   printf( "\n u1 = %Lf  u2 = %Lf ", u1[i], u2[i]);
  
//delx = 4.0L/(gM+1.0L);    //The length of the interval


for (i = 0; i <= gM; i++) {
   numb_in = i/4;
   ii = i - 4*numb_in;
//   xx = delx*numb_in + delx*Kot[ii];

   sum = sum + Weit[ii]*u1[i]*u2[i];

}
//printf("\n sum = %Lf ", sum);
return sum;

}

//Type 2.
long double Bspline::EvaluateIntegralInnerProducts2(long double *u1, long double *u2, int gM, int gM2)
{
  int  i,j,k, ii, numb_in;
  long double sum = 0.0L;


  //delx = 4.0L/(gM+1.0L); 

  for (i = 0; i <= gM2; i++) 
	{
	  numb_in = i/4;
	  ii      = i - 4*numb_in;
	  sum     = sum + delx*Weit[ii]*u1[i]*u2[i];
	}
  //printf("\n sum = %Lf ", sum);
  return sum;
  
}


void Bspline::Evaluate_BSpline_Basis_AtImgGrid()
{
  long double  xx, x, values[6];
  float       *vec, *vec1, *vec2, *vec3, norm;
  int   i, j, k, iN3, iN1, jN1, j3, usedN, usedN3, usedN1;
  long double  invdelx;

  invdelx = 1.0L/delx;

  usedN  = N - 4;
  usedN1 = usedN + 1;
  usedN3 = 3 * (usedN + 1); 
 
//  for ( i = 0; i < ImgNx; i++ )
  for ( i = 0; i <= ImgNx; i++ )
	for ( j = 0; j <= usedN; j++ )
	  {
		BBaseImgGrid[i*usedN3 + 3*j]    = 0.0;
		BBaseImgGrid[i*usedN3 + 3*j+1]  = 0.0;
		BBaseImgGrid[i*usedN3 + 3*j+2]  = 0.0;

		BBaseImgGrid3[i*usedN1 + j]     = 0.0; 

		OrthoBBImgGrid[i*usedN3 + 3*j]  = 0.0;
		OrthoBBImgGrid[i*usedN3 + 3*j+1]= 0.0;
		OrthoBBImgGrid[i*usedN3 + 3*j+2]= 0.0;
	  }
  
  vec  = new float[usedN1]; 
  vec1 = new float[usedN1];
  vec2 = new float[usedN1];
  vec3 = new float[usedN1];

// for (i = 0; i < ImgNx; i++)
  for (i = 0; i <= ImgNx; i++) 
	//for ( i = StartX; i <= FinishX; i++ )
	{
	  xx  = i * 1.0L;
	  iN3 = i * usedN3;
	  iN1 = i * usedN1;

	  //printf("\n");
	  for (j = 2; j <= N - 2; j++)
		//for ( j = StartX; j<= FinishX; j++ )
		{
		  j3 = j - 2 + j - 2 + j - 2;
		  
		  
		  x = invdelx*xx - j ;
		  //printf("\nBBase x=%Lf ", x);
		  Spline_N_Base(x, nx-1, values);
		  values[1] = values[1]*invdelx;
		  values[2] = values[2]*invdelx*invdelx;
		  
		  vec[j-2]  = (float)values[0];
		  vec1[j-2] = (float)values[1];
		  vec2[j-2] = (float)values[2];
		  vec3[j-2] = (float)values[3];

		  BBaseImgGrid[iN3 + j3]   = vec[j-2] ;
		  BBaseImgGrid[iN3 + j3+1] = vec1[j-2];
		  BBaseImgGrid[iN3 + j3+2] = vec2[j-2];
		  BBaseImgGrid3[iN1+ j-2 ] = vec3[j-2];

		  //printf("%f ", BernBaseGrid[iN3 + j3]);
		}

	for ( j = 0; j <= usedN; j++ )
	  {
		jN1 = j * usedN1;
		j3 = j + j +j;
		
		for ( k = 0; k <= usedN; k++ )
		  {
			OrthoBBImgGrid[iN3 + j3]   = OrthoBBImgGrid[iN3 + j3] + (float)SchmidtMat[jN1+k] * vec[k];
			OrthoBBImgGrid[iN3 + j3+1] = OrthoBBImgGrid[iN3 + j3+1] + (float)SchmidtMat[jN1+k] * vec1[k];
			OrthoBBImgGrid[iN3 + j3+2] = OrthoBBImgGrid[iN3 + j3+2] + (float)SchmidtMat[jN1+k] * vec2[k];
		  }
	  }

	}	   
 
  delete [] vec;
  delete [] vec1;
  delete [] vec2; 
  delete [] vec3;
}



void Bspline::Evaluate_BSpline_Basis_AtImgGrid_sub()
{
  long double  xx, x, values[6];
  float       *vec, *vec1, *vec2, *vec3, norm;
  int   i, j, k, iN3, iN1, jN1, j3, usedN, usedN3, usedN1, sub, subImgNx;
  long double  invdelx, subdelx;

  invdelx = 1.0L/delx;

  usedN  = N - 4;
  usedN1 = usedN + 1;
  usedN3 = 3 * (usedN + 1); 
 
  sub = 2;
  subImgNx = 2 * ImgNx;
  subdelx  = 1.0L/sub;
 

//  for ( i = 0; i < ImgNx; i++ )
  for ( i = 0; i <= subImgNx; i++ )
	for ( j = 0; j <= usedN; j++ )
	  {
		subBBaseImgGrid[i*usedN3 + 3*j]    = 0.0;
		subBBaseImgGrid[i*usedN3 + 3*j+1]  = 0.0;
		subBBaseImgGrid[i*usedN3 + 3*j+2]  = 0.0;

	  }
  
  vec  = new float[usedN1]; 
  vec1 = new float[usedN1];
  vec2 = new float[usedN1];
  vec3 = new float[usedN1];

// for (i = 0; i < ImgNx; i++)
  for (i = 0; i <= subImgNx; i++) 
	//for ( i = StartX; i <= FinishX; i++ )
	{
	  xx  = i * 1.0L;
	  iN3 = i * usedN3;
	  iN1 = i * usedN1;

	  //printf("\n");
	  for (j = 2; j <= N - 2; j++)
		//for ( j = StartX; j<= FinishX; j++ )
		{
		  j3 = j - 2 + j - 2 + j - 2;
		  
		  
		  //x = invdelx*xx*subdelx - j ;   //subdivide image.  07-12-09.
		    x =  invdelx * xx * 0.5L - j;
		  //printf("\nBBase x=%Lf ", x);
		  Spline_N_Base(x, nx-1, values);
		  values[1] = values[1]*invdelx;
		  values[2] = values[2]*invdelx*invdelx;
		  
		  vec[j-2]  = (float)values[0];
		  vec1[j-2] = (float)values[1];
		  vec2[j-2] = (float)values[2];
		  vec3[j-2] = (float)values[3];

		  subBBaseImgGrid[iN3 + j3]   = vec[j-2] ;
		  subBBaseImgGrid[iN3 + j3+1] = vec1[j-2];
		  subBBaseImgGrid[iN3 + j3+2] = vec2[j-2];

		  //printf("\ni j =%d %d %f ", i, j, subBBaseImgGrid[iN3 + j3]);
		}


	}	   
// getchar(); 
  delete [] vec;
  delete [] vec1;
  delete [] vec2; 
  delete [] vec3;
}



/*******************************************************************************
Descriptions:
    Evaluate BSpline Base Functions at Gauss nodes for Gauss Quadrature.
    BSpline Base Functions:    
    Knots : 0 = u0 = u1 = u2 = u3 < ...< u_(N-1) < u_(N+1) = u_(N+2) = u_(N+3) = u_(N+4)=1

********************************************************************************/
void Bspline::Evaluate_BSpline_Basis_AtGaussNodes()
{
  int         i, j, k, m;
  long double x, y, z, X, sx, sy, sz, tau, values[6];
  int         numb_in, ii, usedN, usedN3, iN3;
  long double xx,invdelx;

  invdelx = 1.0L/delx;
  usedN  = N - 4;
  usedN3 = 3 * (usedN + 1);

//delx = 4.0L/(M+1.0L);    // The length of the interval

// evaluate B-spline basis
  for (i = 0; i <= M; i++) 
	{
	  sx = 0.0L;
	  sy = 0.0L;
	  sz = 0.0L;
	 
	  numb_in = i/4;
	  ii = i - 4*numb_in;
	  xx = delx*numb_in + delx*Kot[ii];
	  
	  iN3 = i * usedN3;
	  
	  for (j = 2; j <= N - 2; j++) 
		{
		  X = invdelx * xx - j;
		  
		  Spline_N_Base(X, nx-1, values);
		  x = values[0];
		  y = values[1]*invdelx;
		  z = values[2]*invdelx*invdelx;
		  sx = sx + x;
		  sy = sy + y;
		  sz = sz + z;
		  
		  BBaseGN[iN3 + 3*(j-2)]     =(float)x;   
		  BBaseGN[iN3 + 3*(j-2) + 1] =(float)y;   
		  BBaseGN[iN3 + 3*(j-2) + 2] =(float)z;   
		  //printf("BBaseGN=%f ", BBaseGN[iN3 + 3*(j-2)+1]);
		}
	  //printf("%Lf, %Lf, %Lf\n", sx, sy, sz);

	}
  
}

/************************************************************************/
void Bspline::Evaluate_OrthoBSpline_Basis_AtVolGrid2()
{

  long double  xx, x, values[6];
  float       *vec, *vec1, *vec2, norm;
  int i, j, k, iN3, jN1, j3, usedN, usedN3, usedN1;
  long double  invdelx;

  invdelx = 1.0L/delx;

  usedN  = N - 4;
  usedN1 = usedN + 1;
  usedN3 = 3 * (usedN + 1); 

  for ( i = 0; i < nx; i++ )
	for ( j = 0; j <= usedN; j++ )
	  {
		OrthoBBaseGrid2[i*usedN3 + 3*j]    = 0.0;
		OrthoBBaseGrid2[i*usedN3 + 3*j+1]  = 0.0;
		OrthoBBaseGrid2[i*usedN3 + 3*j+2]  = 0.0;
	  }
  
  
  // delx = 1.0L/(nx-1.0L);
  //  printf("\northobbasegrid2 delx =%Lf",delx);getchar(); 
  vec  = new float[usedN1]; 
  vec1 = new float[usedN1];
  vec2 = new float[usedN1];

 for (i = 0; i < nx; i++) 
 //for ( i = StartX; i <= FinishX; i++ )
   {
	 xx = delx * i;
	 iN3 = i* usedN3;
	 
	 for (j = 2; j <= N - 2; j++)
	//for ( j = StartX; j<= FinishX; j++ )
	   {
		 j3 = j - 2 + j - 2 + j - 2;
		 
		 
		 x = invdelx*xx - j ;
		 //printf("\nBBase x=%Lf ", x);
		 Spline_N_Base(x, nx-1, values);
		 values[1] = values[1]*invdelx;
		 values[2] = values[2]*invdelx*invdelx;
		 

       vec[j-2]  = (float)values[0];
       vec1[j-2] = (float)values[1];
       vec2[j-2] = (float)values[2];
	  
	   BernBaseGrid[iN3 + j3]   = vec[j-2];
	   BernBaseGrid[iN3 + j3+1] = vec1[j-2];
	   BernBaseGrid[iN3 + j3+2] = vec2[j-2];
	  }	   
	for ( j = 0; j <= usedN; j++ )
	  {
		jN1 = j * usedN1;
		j3 = j + j +j;
		
		for ( k = 0; k <= usedN; k++ )
		  {
			OrthoBBaseGrid2[iN3 + j3]   = OrthoBBaseGrid2[iN3 + j3] + (float)SchmidtMat[jN1+k] * vec[k];
			
			OrthoBBaseGrid2[iN3 + j3+1] = OrthoBBaseGrid2[iN3 + j3+1] + (float)SchmidtMat[jN1+k] * vec1[k];
			OrthoBBaseGrid2[iN3 + j3+2] = OrthoBBaseGrid2[iN3 + j3+2] + (float)SchmidtMat[jN1+k] * vec2[k];
		  }
	  }
	
   }

 
 /*  
  
 for ( i = 0; i < nx; i++ )
   {printf("\n");
	   for ( j = 0; j <= usedN; j++ )
		 { 
		   vec[j] = OrthoBBaseGrid2[i*usedN3 + 3*j];
		   printf("%f  ", vec[j]);
		 }
	 
	   // norm = InnerProduct(vec, vec);
	   //  printf("\nnorm = %f ", norm);getchar();
   }

 
 printf("\nBernBaseGrid");
 for ( i = 0; i < nx; i++ )
   {printf("\n");
	   for ( j = 0; j <= usedN; j++ )
		 { 
		   vec[j] = BernBaseGrid[i*usedN3 + 3*j];
		   printf("%f  ", vec[j]);
		 }
   }
 */
 //getchar();
 

delete [] vec;
delete [] vec1;
delete [] vec2;

}




/************************************************************************/
void Bspline::Evaluate_OrthoBSpline_Basis_AtVolGrid()
{
int  i, j, k, m;
long double values[6];
 int numb_in, ii;
long double xx;
 float *vec,*vec1,*vec2;

 //delx = 1.0L/(N-2);
 
 vec  = new float[N1]; //(float *)malloc(N1*sizeof(float));
 vec1 = new float[N1];
 vec2 = new float[N1];

for ( i = 0; i < N1; i++ )   vec[i]  = 0.0;
for ( i = 0; i < N1; i++ )   vec1[i] = 0.0;
for ( i = 0; i < N1; i++ )   vec2[i] = 0.0;

for (i = 0; i < nx; i++) 
   {
    for (j = 0; j <= N; j++)
      {
		xx = delx * i;

       Spline_N_i(xx, j, nx-1, values);
       vec[j]  = (float)values[0];
       vec1[j] = (float)values[1];
       vec2[j] = (float)values[2];

	   //OrthoBBaseGrid[i*N3 + 3*j]   = vec[j];
	   //OrthoBBaseGrid[i*N3 + 3*j+1] = vec1[j];
	   //OrthoBBaseGrid[i*N3 + 3*j+2] = vec2[j];

 // printf("\nxx=%Lf, OrthoBBaseGrid = %f ", xx, OrthoBBaseGrid[i*N3 + 3*j]);

       //printf("\n i=%d j=%d  vec = %f ", i,j,vec[j]);
      }


	
    for (j = 0; j <= N; j++)
       {
        for ( k = 0; k <= N; k++ )
           {
            //printf(" SchmidtMat*vec = %f ", (float)SchmidtMat[j*N1+k]*vec[k]);

            OrthoBBaseGrid[i*N3 + 3*j] = OrthoBBaseGrid[i*N3 + 3*j] + (float)SchmidtMat[j*N1+k]* vec[k];
            OrthoBBaseGrid[i*N3 + 3*j+1] = OrthoBBaseGrid[i*N3 + 3*j+1] + (float)SchmidtMat[j*N1+k]* vec1[k];
            OrthoBBaseGrid[i*N3 + 3*j+2] = OrthoBBaseGrid[i*N3 + 3*j+2] + (float)SchmidtMat[j*N1+k]* vec2[k];

            //printf(" OrthoBBaseGrid[i*N1 + j] = %f ",OrthoBBaseGrid[i*N1 + j]);
 
           }
       }
 
   }

/*
printf("\n Ortho Bspline Base values at grids . OrthoBBaseGrid");

for ( i = 0; i < N - 1; i++ )
   {
    printf("\n");
    for ( j = 0; j <= N ; j++ )
       {
        printf( "%f   ",OrthoBBaseGrid[i*(N+1) + j]);
       }
   }

*/
//free(vec);
delete [] vec;
delete [] vec1;
delete [] vec2;
}

/*************************************************************************/
void Bspline::Evaluate_OrthoBSpline_Basis_AtGaussNodes()
{

  unsigned i,j,k;
  int      usedN, usedN1, usedN3, iN3, jN1;

  Evaluate_BSpline_Basis_AtGaussNodes();
  
  usedN  = N - 4;
  usedN1 = usedN + 1;
  usedN3 = 3 * (usedN + 1);

  for (i = 0; i <= M; i++) 
	{
	  iN3 = i * usedN3;
	  // printf("\n");
	  for (j = 0; j <= usedN; j++) 
		{
		  jN1 = j * usedN1;
 
		  for ( k = 0; k <= usedN; k++ )
			{
			  //printf(" BernBase = %f ",BernBase[i*N3 + 3*k]);
			  OrthoBBaseGN[iN3 + 3*j]     = OrthoBBaseGN[iN3 + 3*j] + (float)SchmidtMat[jN1+k]* BBaseGN[iN3 + 3*k];
			  OrthoBBaseGN[iN3 + 3*j + 1] = OrthoBBaseGN[iN3 + 3*j + 1] + (float)SchmidtMat[jN1+k]* BBaseGN[iN3 + 3*k + 1];
			  OrthoBBaseGN[iN3 + 3*j + 2] = OrthoBBaseGN[iN3 + 3*j + 2] + (float)SchmidtMat[jN1+k]* BBaseGN[iN3 + 3*k + 2];
			}
		  // printf(" OrthoBBaseGn = %f ",OrthoBBaseGN[iN3 + 3*j+2]);	  

		}
	}
}





/****************************************************************************
Author      : Prof. Xu.
Descriptions:
     Evaluate BSpline Base funcitions at Gauss nodes for Gauss Quadrature.
     Bspline Base Functions:   \beta^3(x-i). i \in (-\infinity, \infinity);
*****************************************************************************/
void Bspline::Evaluate_B_Spline_Basis()
{
int  i, j, k, m;
long double x, y, z, sx, sy, sz, tau, values[6];
int numb_in, ii ;
float xx;


// evaluate B-spline basis at Gauss nodes.
for (i = 0; i <= M; i++) {
   sx = 0.0L;
   sy = 0.0L;
   sz = 0.0L;

   numb_in = i/4;
   ii = i - 4*numb_in;
   xx = numb_in + Kot[ii];

   for (j = 0; j <= N; j++) {

      x = Cubic_Bspline_Interpo_kernel(xx, j);
     // x = values[0];
     // y = values[1];
     // z = values[2];
      sx = sx + x;
     // sy = sy + y;
     // sz = sz + z;

      BernBase[i*N3 + 3*j]     = x;   // = B^n_j(1/M)
      BernBase[i*N3 + 3*j + 1] = y;   // = B^n_j(1/M)'
      BernBase[i*N3 + 3*j + 2] = z;   // = B^n_j(1/M)''
   }
   //printf("%e, %e, %e\n", sx, sy, sz);
}

}


// Compute up to third order derivative
void Bspline::Phi_ijk_Partials_ImgGrid(int qx, int qy, int qz, int i, int j, int k, float *partials)
{
  int    ii, usedN, usedN3;
  float  Bx, Bx1, Bx2, Bx3, By, By1, By2, By3, Bz, Bz1, Bz2, Bz3;

  usedN  = N - 4;
  usedN3 = 3*(usedN + 1);

  ii  = qx*usedN3 + 3*i;
  Bx  = BBaseImgGrid[ii];
  Bx1 = BBaseImgGrid[ii + 1];
  Bx2 = BBaseImgGrid[ii + 2];
  Bx3 = BBaseImgGrid3[qx * (usedN+1) + i];
  
  ii  = qy*usedN3 + 3*j;
  By  = BBaseImgGrid[ii];
  By1 = BBaseImgGrid[ii + 1];
  By2 = BBaseImgGrid[ii + 2];
  By3 = BBaseImgGrid3[qy * (usedN+1) + j];
  //printf("By = %f ", By);
  
  ii  = qz*usedN3 + 3*k;
  Bz  = BBaseImgGrid[ii];
  Bz1 = BBaseImgGrid[ii + 1];
  Bz2 = BBaseImgGrid[ii + 2];
  Bz3 = BBaseImgGrid3[qz * (usedN+1) + k];

  partials[0] = Bx*By*Bz;
  
  partials[1] = Bx1*By*Bz;
  partials[2] = Bx*By1*Bz;
  partials[3] = Bx*By*Bz1;
  
  partials[4] = Bx2*By*Bz; 
  partials[5] = Bx1*By1*Bz;
  partials[6] = Bx1*By*Bz1;
  
  partials[7] = Bx*By2*Bz;
  partials[8] = Bx*By1*Bz1;
  partials[9] = Bx*By*Bz2;

  partials[10] = Bx3 * By  * Bz;
  partials[11] = Bx2 * By1 * Bz;
  partials[12] = Bx2 * By  * Bz1;

  partials[13] = Bx1 * By2 * Bz;
  partials[14] = Bx1 * By1 * Bz1;
  partials[15] = Bx1 * By  * Bz2;

  partials[16] = Bx  * By3 * Bz;
  partials[17] = Bx  * By2 * Bz1;
  partials[18] = Bx  * By1 * Bz2;
  partials[19] = Bx  * By  * Bz3;

}


// Compute the first order derivative only 
void Bspline::Phi_ijk_Partials_ImgGrid_3(int qx, int qy, int qz, int i, int j, int k, float *partials)
{
  int    ii, ii1, ii2, usedN, usedN1, usedN3;
  float  Bx, Bx1, By, By1, Bz, Bz1;

  usedN  = N - 4;
  usedN1  = usedN + 1;
  usedN3 = 3*usedN1;

  ii  = qx*usedN3 + 3*i;
  Bx  = BBaseImgGrid[ii];
  Bx1 = BBaseImgGrid[ii+1];
  
  ii  = qy*usedN3 + 3*j;
  By  = BBaseImgGrid[ii];
  By1 = BBaseImgGrid[ii+1];
  
  ii  = qz*usedN3 + 3*k;
  Bz  = BBaseImgGrid[ii];
  Bz1 = BBaseImgGrid[ii+1];

  partials[0] = Bx*By*Bz;
  partials[1] = Bx1*By*Bz;
  partials[2] = Bx*By1*Bz;
  partials[3] = Bx*By*Bz1;

}


// Compute up to third order derivative
void Bspline::Phi_ijk_Partials_ImgGrid_9(int qx, int qy, int qz, int i, int j, int k, float *partials)
{
  int    ii, usedN, usedN3;
  float  Bx, Bx1, Bx2, Bx3, By, By1, By2, By3, Bz, Bz1, Bz2, Bz3;

  usedN  = N - 4;
  usedN3 = 3*(usedN + 1);

  ii  = qx*usedN3 + 3*i;
  Bx  = BBaseImgGrid[ii];
  Bx1 = BBaseImgGrid[ii + 1];
  Bx2 = BBaseImgGrid[ii + 2];
  
  ii  = qy*usedN3 + 3*j;
  By  = BBaseImgGrid[ii];
  By1 = BBaseImgGrid[ii + 1];
  By2 = BBaseImgGrid[ii + 2];
  //printf("By = %f ", By);
  
  ii  = qz*usedN3 + 3*k;
  Bz  = BBaseImgGrid[ii];
  Bz1 = BBaseImgGrid[ii + 1];
  Bz2 = BBaseImgGrid[ii + 2];

  partials[0] = Bx*By*Bz;
  
  partials[1] = Bx1*By*Bz;
  partials[2] = Bx*By1*Bz;
  partials[3] = Bx*By*Bz1;
  
  partials[4] = Bx2*By*Bz; 
  partials[5] = Bx1*By1*Bz;
  partials[6] = Bx1*By*Bz1;
  
  partials[7] = Bx*By2*Bz;
  partials[8] = Bx*By1*Bz1;
  partials[9] = Bx*By*Bz2;
}


void Bspline::Ortho_Phi_ijk_Partials_ImgGrid(int qx, int qy, int qz, int i, int j, int k, float *partials)
{
  int    ii, usedN, usedN3;
  float  Bx, Bx1, Bx2, By, By1, By2, Bz, Bz1, Bz2;

  usedN  = N - 4;
  usedN3 = 3*(usedN + 1);

  ii  = qx*usedN3 + 3*i;
  Bx  = OrthoBBImgGrid[ii];
  Bx1 = OrthoBBImgGrid[ii + 1];
  Bx2 = OrthoBBImgGrid[ii + 2];
  
  
  ii  = qy*usedN3 + 3*j;
  By  = OrthoBBImgGrid[ii];
  By1 = OrthoBBImgGrid[ii + 1];
  By2 = OrthoBBImgGrid[ii + 2];
  
  //printf("By = %f ", By);
  
  ii  = qz*usedN3 + 3*k;
  Bz  = OrthoBBImgGrid[ii];
  Bz1 = OrthoBBImgGrid[ii + 1];
  Bz2 = OrthoBBImgGrid[ii + 2];
  
  partials[0] = Bx*By*Bz;
  
  partials[1] = Bx1*By*Bz;
  partials[2] = Bx*By1*Bz;
  partials[3] = Bx*By*Bz1;
  
  partials[4] = Bx2*By*Bz;  
  partials[5] = Bx1*By1*Bz;
  partials[6] = Bx1*By*Bz1;
  
  partials[7] = Bx*By2*Bz;
  partials[8] = Bx*By1*Bz1;
  partials[9] = Bx*By*Bz2;

}



/*************************************************************************
Author    : Prof. Xu.

**************************************************************************/
void Bspline::Phi_ijk_PartialsGN(int qx, int qy, int qz, int i, int j, int k, float *partials)
{
  int    ii, usedN, usedN3;
  float  Bx, Bx1, Bx2, By, By1, By2, Bz, Bz1, Bz2;

  usedN  = N - 4;
  usedN3 = 3*(usedN + 1);

  ii  = qx*usedN3 + 3*i;
  Bx  = BBaseGN[ii];
  Bx1 = BBaseGN[ii + 1];
  Bx2 = BBaseGN[ii + 2];
  
  
  ii  = qy*usedN3 + 3*j;
  By  = BBaseGN[ii];
  By1 = BBaseGN[ii + 1];
  By2 = BBaseGN[ii + 2];
  
  //printf("By = %f ", By);
  
  ii  = qz*usedN3 + 3*k;
  Bz  = BBaseGN[ii];
  Bz1 = BBaseGN[ii + 1];
  Bz2 = BBaseGN[ii + 2];
  
  partials[0] = Bx*By*Bz;
  
  partials[1] = Bx1*By*Bz;
  partials[2] = Bx*By1*Bz;
  partials[3] = Bx*By*Bz1;
  
  partials[4] = Bx2*By*Bz;
  partials[5] = Bx1*By1*Bz;
  partials[6] = Bx1*By*Bz1;
  
  partials[7] = Bx*By2*Bz;
  partials[8] = Bx*By1*Bz1;
  partials[9] = Bx*By*Bz2;

}

/************************************************************************
 Descriptions:
   
*************************************************************************/
void Bspline::Phi_ijk_Partials_at_Grids(int ix, int iy, int iz, int i,
											   int j, int k, float *partials)
{
  int    ii , N3;
float  Bx, Bx1, Bx2, By, By1, By2, Bz, Bz1, Bz2;

 N3 = 3*(N+1);

ii = ix*N3 + 3*i;
Bx  = OrthoBBaseGrid[ii];
//printf("\n Bx = %f ", Bx);

Bx1 = OrthoBBaseGrid[ii + 1];
Bx2 = OrthoBBaseGrid[ii + 2];


ii = iy*N3 + 3*j;
By  = OrthoBBaseGrid[ii];
By1 = OrthoBBaseGrid[ii + 1];
By2 = OrthoBBaseGrid[ii + 2];

//printf("By = %f ", By);

ii = iz*N3 + 3*k;
Bz  = OrthoBBaseGrid[ii];

//printf("Bz = %f ", Bz);

Bz1 = OrthoBBaseGrid[ii + 1];Bz2 = OrthoBBaseGrid[ii + 2];

partials[0] = Bx*By*Bz;

partials[1] = Bx1*By*Bz;
partials[2] = Bx*By1*Bz;
partials[3] = Bx*By*Bz1;

partials[4] = Bx2*By*Bz;
partials[5] = Bx1*By1*Bz;
partials[6] = Bx1*By*Bz1;

partials[7] = Bx*By2*Bz;
partials[8] = Bx*By1*Bz1;
partials[9] = Bx*By*Bz2;


}



/************************************************************************
 Descriptions:
     Compute the Hessian Matrix of Volume data f.

*************************************************************************/
/*void Bspline::ComputeHfGradientf(float *coef, int n, int *ijk,
                                     float *Hf, float *Gradientf, float *NormGradf)
{
  int   id, i, j, k, ii;
  int   ix, iy, iz, N2;
  float partials[10];

  N2 = (N+1)*(N+1);

  for ( id = 0; id < n; id++ )
	{
	ix = ijk[3*id + 0];
    iy = ijk[3*id + 1];
	iz = ijk[3*id + 2];
	for ( i = 0; i < N+1; i++ )
	  for ( j = 0; j < N+1; j++ )
		for ( k = 0; k < N+1; k++ )
          {
          ii = i*N2 + j*(N+1) + k;
          bspline->Phi_ijk_Partials_at_Grids(ix, iy, iz, i, j, k, partials);

          Gradientf[3*id+0] = Gradientf[3*id+0] + coef[ii]*partials[1];
          Gradientf[3*id+1] = Gradientf[3*id+1] + coef[ii]*partials[2];
          Gradientf[3*id+2] = Gradientf[3*id+2] + coef[ii]*partials[3];

          NormGradf[id]    = sqrt(Gradientf[3*id+0]*Gradientf[3*id+0] +
								Gradientf[3*id+1]*Gradientf[3*id+1] +
                                Gradientf[3*id+2]* Gradientf[3*id+2]);
 
 
		  Hf[9*id+0] = Hf[9*id+0] + coef[ii]*partials[4];
		  Hf[9*id+1] = Hf[9*id+1] + coef[ii]*partials[5];
		  Hf[9*id+2] = Hf[9*id+2] + coef[ii]*partials[6];
		  Hf[9*id+3] = Hf[9*id+3] + coef[ii]*partials[5];
		  Hf[9*id+4] = Hf[9*id+4] + coef[ii]*partials[7];
		  Hf[9*id+5] = Hf[9*id+5] + coef[ii]*partials[8];
		  Hf[9*id+6] = Hf[9*id+6] + coef[ii]*partials[6];
		  Hf[9*id+7] = Hf[9*id+7] + coef[ii]*partials[8];
		  Hf[9*id+8] = Hf[9*id+8] + coef[ii]*partials[9];


		  }


	}

}
*/


int Bspline::Reorder_BaseIndex_Volume(unsigned i,unsigned j, unsigned k, unsigned N)
{
int  ii, n0;
ii = i*(N+1)*(N+1) + j*(N+1) + k ;  // Equation index
if ((j >= 0) && (j <= N) && (i >= 0) && (i <= N) && (k >= 0) && (k <= N)) return(ii);



return(-1);
}



/********************************************************************************************
 Author : Prof. Xu.
 ********************************************************************************************/
int Bspline::Reorder_BaseIndex_Volume(int i,int j, int k, int nx, int ny, int nz)
{
int  ii, n0;
ii = (i-1)*(ny-2)*(nz-2) + (j-1)*(nz-2) + k - 1;  // Equation index
if ((j > 0) && (j < ny) && (i > 0) && (i < nx) && (k > 0) && (k < nz)) return(ii);



return(-1);
}







void Bspline::Support_Index(int i,int *min_index, int *max_index)
{

*min_index = 4*(i-2);
if (*min_index < 0) *min_index = 0;

*max_index = 4*(i+2);
if (*max_index > M) *max_index = M;
}




/***********************************************************************************************
Descriptions:
     Evaluate the Volume Image B_spline Interpolation function at artitray point [u,v,w].

Arguments:
     coeff    :  Coefficients of B_spline Interpolation function.
     nx,ny,nz :  Volume Image Dimensions nx,ny,nz.
     u,v,w    :  Voxel Coordinates. [u,v,w] \in [0,nx] X [0,ny] X [0,nz].

Notes :  one dimension B_spiine Base Function beta^3(x-i) correspondent to grid i in pixel image.         
*************************************************************************************************/
float Bspline::ComputeVolImgBsplineInpolatAtAnyPoint(float *coeff, int nx,int ny,int nz,float u, float v, float w)
{
 int i, j,k,shiftx, shifty, shiftz;
 int voli, volj, volk;
 float InterKerx,InterKery, InterKerz,coef,result = 0;
 float Delx = 1.0/(nx-1);

//Find the voxel coordinate where [u,v,w] lies in.

 voli = u/Delx;
 volj = v/Delx;
 volk = w/Delx;


// xd = u - voli*delx;
//yd = v - volj*delx;
// zd = w - volk*delx;


 for(i = 0; i <= 3; i++)    
    for(j = 0; j <= 3; j++)
       for(k = 0; k <= 3; k++)
          {
           shiftx = voli  -1 + i;   
           shifty = volj  -1 + j;
           shiftz = volk  -1 + k;

           InterKerx =  Cubic_Bspline_Interpo_kernel(u,shiftx);
           InterKery =  Cubic_Bspline_Interpo_kernel(v,shifty);
           InterKerz =  Cubic_Bspline_Interpo_kernel(w,shiftz);
          // printf("\n test %f %f %f %f  %f %f  ",u,v,w,  InterKerx,InterKery,InterKerz);

         

           if((voli+i-1) < 0  || (volj+j-1) < 0    || (volk+k-1) < 0 ||
              (voli+i-1) >= nx || (volj+j-1) >= ny || (volk+k-1) >= nz )
              coef = 0.0;
           else  
              coef = coeff[((voli+i-1)*ny + (volj+j-1))*nz + volk+k-1];


           result = result + coef * InterKerx * InterKery * InterKerz;

          // printf("\n coef = %f basex=%f basey=%f basez=%f", coef, InterKerx,InterKery,InterKerz);
           }

  return result;



}



/*****************************************************
Evaluate cubic B-spline beta^3(u-shift) at u.

u \in [-\infinity, \infinity], 
shift \in {..,-2,-1,0,1,2,3,4,5,...}
  

*****************************************************/
float Bspline::Cubic_Bspline_Interpo_kernel(float u, int shift)
{
float x, result;
x = u - shift;

if(fabs(x) >= 0 && fabs(x) < 1)
  result = 2*1.0/3 - x*x + 0.5 * fabs(x)*fabs(x)*fabs(x);
else if(fabs(x) >= 1 && fabs(x) <= 2)
  result = 1*1.0/6*(2-fabs(x))*(2-fabs(x))*(2-fabs(x)); 
else result = 0;

return result;


}

/*

float Bspline::Bspline3(float s, float shift, float scale)
{
  float x;
  x = -2 + 2*(s+






}
*/




// New conversion code by Xuguo
/*-----------------------------------------------------------------------------*/
void Bspline::convertToInterpolationCoefficients_1D(float *s, int DataLength, float EPSILON) 
//		float	*s,		/* input samples --> output coefficients */
//		int	DataLength,	/* number of samples or coefficients     */
//		float	EPSILON		/* admissible relative error             */

{
int   i, n, ni, ni1, K;
float sum, z1, w1, w2;

n = DataLength + 1;
z1 = sqrt(3.0) - 2.0;
K = log(EPSILON)/log(fabs(z1));
//printf("K = %i\n", K);

// compute initial value s(0)
sum = 0.0;
w2 = pow(z1, 2*n);
if (n < K) {
   for (i = 1; i < n; i++){
      w1 = pow(z1, i);
      sum = sum + s[i-1]*(w1 - w2/w1);
   }
} else {
   for (i = 1; i < n; i++){
      sum = (sum + s[n- i-1])*z1;
   }
}
sum = -sum/(1.0 - w2);

// compute c^{+}
n = DataLength;
s[0]  = s[0] + z1*sum;
for (i = 1; i < n; i++) {
   s[i]  = s[i] + z1*s[i-1];
   //printf("cp[%i] = %e, %f \n", i, cp[i], z1);
}

// compute c^- 
s[n-1] = -z1*s[n-1];
for (i = 1; i < n; i++) {
   ni = n - i; 
   ni1 = ni - 1;
   s[ni1]  = z1*(s[ni] - s[ni1]);
}

for (i = 0; i < n; i++) {
   s[i]  = 6.0*s[i];
}

}
/*Author : Prof. Xu.*/
/*-----------------------------------------------------------------------------*/
void Bspline::convertToInterpolationCoefficients_2D(float *s, int nx, int ny, float EPSILON) 
//		float	*s,     /* input samples --> output coefficients               */
//		int	nx,	/* number of samples or coefficients in x direction    */
//		int	ny,	/* number of samples or coefficients in y direction    */
//		float	EPSILON	/* admissible relative error                           */
{
float *d, z1;
int    i, l;

d =new float[nx]; // (float *) malloc (nx*sizeof (float)); 
for ( i = 0; i < nx; i++ ) d[i] = 0.0;
  
z1 = sqrt(3.0) - 2.0;
// x-direction interpolation
for (l = 0; l < ny; l++) {
   for (i = 0; i < nx; i++) {
      d[i] = s[i*ny + l];
   }
   convertToInterpolationCoefficients_1D(d, nx,CVC_DBL_EPSILON);
   //ConvertToInterpolationCoefficients_Qu(d, nx,&z1, 1, CVC_DBL_EPSILON);


   for (i = 0; i < nx; i++) {
      s[i*ny + l] = d[i];
   }
}

// y-direction interpolation
for (i = 0; i < nx; i++) {
   convertToInterpolationCoefficients_1D(s+i*ny, ny,CVC_DBL_EPSILON);
   //ConvertToInterpolationCoefficients_Qu(s+i*ny, ny, &z1, 1, CVC_DBL_EPSILON);

}

//free(d);
delete [] d;
}






/*Author: Prof. Xu.*/
/*-----------------------------------------------------------------------------*/
void Bspline::convertToInterpolationCoefficients_3D(float *s, int nx, int ny, 
                                                     int nz,   float EPSILON)
//              float   *s,    /* input samples --> output coefficients               */
//              int     nx,    /* number of samples or coefficients in x direction    */
//              int     ny,    /* number of samples or coefficients in y direction    */
//              int     nz,    /* number of samples or coefficients in z direction    */
//              float   EPSILON/* admissible relative error                           */
{
float *d, z1;
int    u, v, w, k, kk;

d = new float[nx]; // (float *) malloc (nx*sizeof (float));
for ( k = 0; k < nx; k++ ) d[k] = 0.0;
k = ny*nz;
z1 = sqrt(3.0) - 2.0;

// x-direction interpolation
for (v = 0; v < ny; v++) {
   for (w = 0; w < nz; w++) {
      kk = v*nz + w;
      for (u = 0; u < nx; u++) {
         d[u] = s[u*k + kk];
      }
      convertToInterpolationCoefficients_1D(d, nx,CVC_DBL_EPSILON);
      //ConvertToInterpolationCoefficients_Qu(d, nx,&z1, 1, CVC_DBL_EPSILON);


      for (u = 0; u < nx; u++) {
         s[u*k + kk] = d[u];
      }
   }
}

for (u = 0; u < nx; u++) {
   convertToInterpolationCoefficients_2D(s+u*k, ny, nz, CVC_DBL_EPSILON);
}

//free(d);
delete [] d;
}






/***************************************************************************
Description:
  Compute the value at u of ith B_spline Base Function 
  Spline  Knots: [u0,u1,..., u_(m+2k)]. 

Arguments:
  MM = m+k. the total number of B_spline Base Functions at the above interval.   
  i = the ith B_spline Base Function.
  k = The degree of B_spline Base Function.
****************************************************************************/
float Bspline::SplineBasePoly(float u,int MM,int i,int k,float *U)
{
 float u1,u2,u3,u4,Nik=0.0;
 if(k==0) {
    if(i<=MM-1 && u>=U[i] && u<U[i+1] ) {Nik=1.0;return(Nik);}
    if(i==MM-1 && u==U[MM])            {Nik=1.0;return(Nik);}
    return(Nik);
 }

 else {
    u1=(u-U[i])*SplineBasePoly(u,MM,i,k-1,U);
    u2=U[i+k]-U[i];
    u3=(U[i+k+1]-u)*SplineBasePoly(u,MM,i+1,k-1,U);
    u4=U[i+k+1]-U[i+1];
    u1=(u1==0.0 && u2==0.0)? 0.0:u1*1.0/u2;
    u3=(u3==0.0 && u4==0.0)? 0.0:u3*1.0/u4;
    Nik=u1+u3;

 }
 return(Nik);
}



/***************************************************************************
Description: 
  Compute the derivate of ith B_spline Base Function at point u.
  Test correct for u0=u1=...=uk, u_(m+k)=...=u_(m+2k.
Arguments:
  MM :      MM = m+k. the total number of B_spline Base Functions. 
  orderR : the order of Derivate.
  k:       The degree of B_spline Base Function.   
  U:       Define the Knots series.U = [u0,u1,...,u_(m+2k)].

***************************************************************************/
float Bspline::Dx_SplineBasePoly(float u,int orderR,int MM,int i,int k,float *U)

{
 float NN1,NN2,NN3,NN4,DerivateN=0.0;
 if(k==0) return(DerivateN);
 if(orderR==1) {
    NN1=SplineBasePoly(u,MM,i,k-1,U);
    NN2=U[i+k]-U[i];
    NN1=(NN1==0 && NN2==0)? 0.0:NN1/NN2;

   NN3=SplineBasePoly(u,MM,i+1,k-1,U);
    NN4=U[i+k+1]-U[i+1];
    NN3=(NN3==0 && NN4==0)? 0.0:NN3/NN4;
    DerivateN=k*(NN1-NN3);
    return(DerivateN);
  }
 else {
   NN1=Dx_SplineBasePoly(u,orderR-1,MM,i,k-1,U);
   NN2=U[i+k]-U[i];
   NN1=(NN1==0 && NN2==0)? 0.0:NN1/NN2;

   NN3=Dx_SplineBasePoly(u,orderR-1,MM,i+1,k-1,U);
   NN4=U[i+k+1]-U[i+1];
   NN3=(NN3==0 && NN4==0)? 0.0:NN3/NN4;
   DerivateN=k*(NN1-NN3);

  }
 return(DerivateN);
}




void Bspline::Spline_N_0(long double u, int m, long double *values)
{
long double mu, ww1, ww2, ww3;

mu = m*u;

values[0] = 0.0L;
values[1] = 0.0L;
values[2] = 0.0L;

ww1 = 1.0L - mu;
ww2 = ww1*ww1;
ww3 = ww2*ww1;
if (mu >=  0.0L && mu <= 1.0L) {
   values[0] = ww3;
   values[1] = -3*m*ww2;
   values[2] = 6*m*m*ww1;
}
}

// -----------------------------------------------------------------------------
// Cubic Spline N_{1,3}
void Bspline::Spline_N_1(long double u, int m, long double *values)
{
long double mu, ww;

mu = m*u;

values[0] = 0.0L;
values[1] = 0.0L;
values[2] = 0.0L;

if (mu >=  0.0L && mu < 1.0L) {
   values[0] = 3*mu*(1.0L - 1.5L*mu + 7.0L/12.0L*mu*mu); 
   values[1] = m*(3.0L - 9.0L*mu + 21.0L/4.0L*mu*mu);
   values[2] = m*m*(-9.0L + 10.5L*mu);
   return;
}

if (mu >=  1.0L && mu <= 2.0L) {
   ww = mu - 2.0L;
   values[0] = -0.25L*ww*ww*ww;
   values[1] = -0.75L*m*ww*ww;
   values[2] = -1.5L*m*m*ww;
}

}

/*Author : Prof. Xu.*/
// -----------------------------------------------------------------------------
// Cubic Spline N_{2,3}
void Bspline::Spline_N_2(long double u, int m, long double *values)
{
long double mu, ww1, ww2, ww3;

mu = m*u;

values[0] = 0.0L;
values[1] = 0.0L;
values[2] = 0.0L;

if (mu >=  0.0L && mu <= 1.0L) {
   ww1 = mu;
   ww2 = mu*mu;
   ww3 = mu*ww2;
   values[0] = 1.5L*ww2 - 11.0L/12.0L*ww3;
   values[1] = m*(3.0L*mu - 2.75L*ww2);
   values[2] = m*m*(3.0L - 5.5L*mu);
   return;
}

if (mu >=  1.0L && mu <= 2.0L) {
   ww1 = mu - 2.0L;
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 1.0L/6.0L - 0.5L*ww1 + 0.5L*ww2+ 7.0L/12.0L*ww3;
   values[1] = m*(-0.5L + ww1 + 1.75L*ww2);
   values[2] = m*m*(1.0L + 3.5L*ww1);
   return;
}

if (mu >=  2.0L && mu <= 3.0L) {
   ww1 = mu - 3.0L;
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = -1.0L/6.0L*ww3;
   values[1] = -0.5L*m*ww2;
   values[2] = -m*m*ww1;
}

}

// -----------------------------------------------------------------------------
// Cubic Spline Base_{m+2,3}
void Bspline::Spline_N_Base(long double u, int m, long double *values)
{
long double ww1, ww2, ww3, sign;

values[0] = 0.0L;
values[1] = 0.0L;
values[2] = 0.0L;
values[3] = 0.0L;

ww1 = fabsl(u);
if (ww1 >= 2.0L) {
   return;
}
ww2 = ww1*ww1;
ww3 = ww2*ww1;

if (u >= 0.0L) {
   sign = 1.0L;
}  else {
   sign = -1.0L;
}

if (ww1 >=  0.0L && ww1 < 1.0L) {
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 2.0L/3.0L - ww2 + 0.5L*ww3;
   values[1] = -2.0L*u + 1.5L*sign*ww2;
   values[2] = -2.0L + 3.0L*ww1;
   values[3] = sign * 3.0L;
   return;
}

if (ww1 >=  1.0L && ww1 <= 2.0L) {
   ww1 = 2.0L - ww1;
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 1.0L/6.0L*ww3;
   values[1] = -0.5L*sign*ww2;
   values[2] = ww1;
   values[3] = -sign;
}

}



/*
void Bspline::Spline_N_Base_3(float u, float *values)
{
float ww1, ww2, ww3, sign;

values[0] = 0.0;
values[1] = 0.0;
values[2] = 0.0;
values[3] = 0.0;

ww1 = fabs(u);
if (ww1 >= 2.0) {
   return;
}
ww2 = ww1*ww1;
ww3 = ww2*ww1;

if (u >= 0.0) {
   sign = 1.0;
}  else {
   sign = -1.0;
}

if (ww1 >=  0.0 && ww1 < 1.0) {
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 2.0/3.0 - ww2 + 0.5*ww3;
   values[1] = -2.0*u + 1.5*sign*ww2;
   values[2] = -2.0 + 3.0*ww1;
   values[3] = sign * 3.0;
   return;
}

if (ww1 >=  1.0 && ww1 <= 2.0) {
   ww1 = 2.0 - ww1;
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 1.0/6.0*ww3;
   values[1] = -0.5*sign*ww2;
   values[2] = ww1;
   values[3] = -sign;
}
}

*/
void Bspline::Spline_N_Base_3(float u, float *values)
{
float ww1, sign;

values[0] = 0.0;

ww1 = fabs(u);
if (ww1 >= 2.0) {
   return;
}

if (u >= 0.0) {
   sign = 1.0;
}  else {
   sign = -1.0;
}

if (ww1 >=  0.0 && ww1 < 1.0) {
   values[0] = sign * 3.0;
   return;
}

if (ww1 >=  1.0 && ww1 <= 2.0) {
   values[0] = -sign;
}
}


void Bspline::Spline_N_Base_1(float u, float *values)
{
float ww1, ww2, ww3, sign;

values[0] = 0.0;
values[1] = 0.0;

ww1 = fabs(u);
if (ww1 >= 2.0) {
   return;
}
ww2 = ww1*ww1;
ww3 = ww2*ww1;

if (u >= 0.0) {
   sign = 1.0;
}  else {
   sign = -1.0;
}

if (ww1 >=  0.0 && ww1 < 1.0) {
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 2.0/3.0 - ww2 + 0.5*ww3;
   values[1] = -2.0*u + 1.5*sign*ww2;
   return;
}

if (ww1 >=  1.0 && ww1 <= 2.0) {
   ww1 = 2.0 - ww1;
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 1.0/6.0*ww3;
   values[1] = -0.5*sign*ww2;
}
}

void Bspline::Spline_N_Base_2(float u, float *values)
{
float ww1, ww2, ww3, sign;

values[0] = 0.0;
values[1] = 0.0;
values[2] = 0.0;

ww1 = fabs(u);
if (ww1 >= 2.0) {
   return;
}
ww2 = ww1*ww1;
ww3 = ww2*ww1;

if (u >= 0.0) {
   sign = 1.0;
}  else {
   sign = -1.0;
}

if (ww1 >=  0.0 && ww1 < 1.0) {
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 2.0/3.0 - ww2 + 0.5*ww3;
   values[1] = -2.0*u + 1.5*sign*ww2;
   values[2] = -2.0 + 3.0*ww1;
   return;
}

if (ww1 >=  1.0 && ww1 <= 2.0) {
   ww1 = 2.0 - ww1;
   ww2 = ww1*ww1;
   ww3 = ww2*ww1;
   values[0] = 1.0/6.0*ww3;
   values[1] = -0.5*sign*ww2;
   values[2] = ww1;
}
}

void Bspline::Spline_N_Base(float u, float *value)
{

  float ww1, ww2, ww3, sign;
  *value = 0.0;

  ww1 = fabs(u);
  if (ww1 >= 2.0) {
	return;
  }
  ww2 = ww1*ww1;
  ww3 = ww2*ww1;
  
  if (u >= 0.0) {
	sign = 1.0;
  }  else {
	sign = -1.0;
  }
  
  if (ww1 >=  0.0 && ww1 < 1.0) {
	ww2 = ww1*ww1;
	ww3 = ww2*ww1;
   *value = 2.0/3.0 - ww2 + 0.5*ww3;
   return;
  }
  
  if (ww1 >=  1.0 && ww1 <= 2.0) {
	ww1 = 2.0 - ww1;
	ww2 = ww1*ww1;
	ww3 = ww2*ww1;
	*value = 1.0/6.0*ww3;

}




}

/*Author : Prof. Xu.*/
// -----------------------------------------------------------------------------
// Cubic Spline N_{i,3}
void Bspline::Spline_N_i(long double u, int i, int m, long double *values)
{
long double x, ww[3];

// m == 1 bezier
if (m == 1){
   if (i == 0) {
      Spline_N_0(u, 1, values);
      return;
   }

   if (i == 1) {
      values[0] = 3.0L*u*(1.0L - u)*(1.0L - u);
      values[1] = 3.0L*(1.0L - 4.0L*u + 3.0L*u*u);
      values[2] = - 12.0L + 18.0L*u;
      return;
   }

   if (i == 2) {
      values[0] = 3.0L*u*u*(1.0L - u);
      values[1] = 6.0L*u -9.0L*u*u;
      values[2] = 6.0L - 18.0L*u;
      return;
   }

   if (i == 3) {
      Spline_N_0(1.0-u, 1, values);
      values[1] = - values[1];
      return;
   }
}

// mm == 2
if (m == 2){
   if (i == 0) {
      Spline_N_0(u, 2, values);
      return;
   }

   if (i == 1) {
      Spline_N_1(u, 2, values);
      return;
   }

   if (i == 2) {
      values[0] = 1.0L;
      values[1] = 0.0L;
      values[2] = 0.0L;
      Spline_N_0(u, 2, ww);
      values[0] = values[0] - ww[0];
      values[1] = values[1] - ww[1];
      values[2] = values[2] - ww[2];

      Spline_N_1(u, 2, ww);
      values[0] = values[0] - ww[0];
      values[1] = values[1] - ww[1];
      values[2] = values[2] - ww[2];
      Spline_N_0(1.0L - u, 2, ww);
      values[0] = values[0] - ww[0];
      values[1] = values[1] + ww[1];
      values[2] = values[2] - ww[2];

      Spline_N_1(1.0L - u, 2, ww);
      values[0] = values[0] - ww[0];
      values[1] = values[1] + ww[1];
      values[2] = values[2] - ww[2];
      return;
   }

   if (i == 3) {
      Spline_N_1(1.0L-u, 2, values);
      values[1] = - values[1];
      return;
   }

   if (i == 4) {
      Spline_N_0(1.0L-u, 2, values);
      values[1] = - values[1];
      return;
   }

}

// other case

if (i == 0) {
  Spline_N_0(u, m, values);
  return;
} 

if (i == 1) {
  Spline_N_1(u, m, values);
  return;
}

if (i == 2) {
  Spline_N_2(u, m, values);
  return;
}

if (i > 2 && i < m) {
   x = m*u - i + 1.0L;
   Spline_N_Base(x, m, values);
   values[1] = values[1]*m;
   values[2] = values[2]*m*m;
   return;
}

if (i == m) {
   Spline_N_2(1.0L-u, m, values);
   values[1] = - values[1];
   return;
}

if (i == m+1) {
   Spline_N_1(1.0L-u, m, values);
   values[1] = - values[1];
   return;
}

if (i == m+2) {
   Spline_N_0(1.0L-u, m, values);
   values[1] = - values[1];
   return;
}

}


//bool Bspline::BsplineSetting(float StartXYZ, float FinishXYZ, long b Bscale)
void Bspline::BsplineSetting(int ImgDim[3], BGrids *bgrids)
{

  int i, usedN;
  float length;


  ImgNx = ImgDim[0]; 
  ImgNy = ImgDim[1]; 
  ImgNz = ImgDim[2];

  // M =  4*(dimx-1) -1;
  // N = dimx -1 ;     //N =  dimx + 1; for Type 1.

  //    length[0] = FinishXYZ[0] - StartXYZ[0];
  //length[1] = FinishXYZ[1] - StartXYZ[1];
  //    length[2] = FinishXYZ[2] - StartXYZ[2];
 

	//  if(length[0] != length[1] || length[0] != length[2] || length[1] != length[2] ) return false;


  delx = bgrids->scale;
  N    = bgrids->dim[0]-1 ; //finishX-startX; // N+1 is the toltal 1D cubic bspline number along each axis. 
  printf("\nN=%d scale=%Lf", N, delx);
  //getchar();


  usedN = N - 4;

  nx = N+1;  
  ny = N+1;
  nz = N+1;

  M = 4* (nx-1) -1;

  N3 = 3*(N+1);
  M1 = 3*(M+1);
  N1 = (N + 1);


 StartX  = bgrids->StartXYZ[0]; 
 FinishX = bgrids->FinishXYZ[0]; 
 printf("startx finishX=%d %d nx=%d ny=%d nz=%d", StartX, FinishX, nx, ny, nz);
/*

Kot[0]= (-0.8611363115940526L + 1.0L)/2.0L;
Kot[1] = (-0.3399810435848563L + 1.0L)/2.0L;
Kot[2] =  (0.3399810435848563L + 1.0L)/2.0L;
Kot[3] =  (0.8611363115940526L + 1.0L)/2.0L;

Weit[0] = 0.3478548451374539L/2.0L;
Weit[1] = 0.6521451548625461L/2.0L;
Weit[2] = 0.6521451548625461L/2.0L;
Weit[3] = 0.3478548451374539L/2.0L;

*/
 BernBase        = new float[3*(usedN+1)*M1];     
 BBaseGN         = new float[3*(usedN+1)*M1];
  
 BernBaseGrid    = new float[nx*3*(usedN+1)];
 OrthoBBaseGN    = new float[3*(usedN+1)*M1];       
 OrthoBBaseGrid  = new float[nx*N3];       
 OrthoBBaseGrid2 = new float[nx*3*(usedN+1)];

// OrthoBBImgGrid  = new float[ImgNx*3*(usedN+1)];
// BBaseImgGrid    = new float[ImgNx*3*(usedN+1)];

int subImgNx, sub;
sub = 2;
subImgNx = 2*ImgNx;

 OrthoBBImgGrid     = new float[(ImgNx+1)*3*(usedN+1)];  
 BBaseImgGrid       = new float[(ImgNx+1)*3*(usedN+1)];
 subBBaseImgGrid    = new float[(subImgNx+1)*3*(usedN+1)];


 BBaseImgGrid3   = new float[(ImgNx+1) * (usedN+1)];

 for ( i = 0; i <3*(usedN+1)*M1; i++ )
   OrthoBBaseGN[i] = 0.0;

 for ( i = 0; i < (usedN+1)*nx*3; i++ )
  {
   OrthoBBaseGrid[i] = 0.0;
   OrthoBBaseGrid2[i] = 0.0;
  }

 for ( i = 0; i < (usedN+1)*(ImgNx+1)*3; i++ )
  {
   OrthoBBImgGrid[i] = 0.0;
   BBaseImgGrid[i] = 0.0;
  }


 for ( i = 0; i < (usedN+1)*(subImgNx+1)*3; i++ )
  {
   subBBaseImgGrid[i] = 0.0;
  }


 for ( i = 0; i < (usedN+1)*(ImgNx+1); i++ )
   BBaseImgGrid3[i] = 0.0;
}



//By Prof. Xu
/************************************************************************/
void Bspline::ConvertToInterpolationCoefficients_1D(float *s, int DataLength, float EPSILON) 
//		float	*s,		/* input samples --> output coefficients */
//		int	DataLength,	/* number of samples or coefficients     */
//		float	EPSILON		/* admissible relative error             */

{
int   i, n, ni, ni1, K;
float sum, z1, w1, w2;

n = DataLength + 1;
z1 = sqrt(3.0) - 2.0;
K = log(EPSILON)/log(fabs(z1));
//printf("K = %i\n", K);

// compute initial value s(0)
sum = 0.0;
w2 = pow(z1, 2*n);
if (n < K) {
   for (i = 1; i < n; i++){
      w1 = pow(z1, i);
      sum = sum + s[i-1]*(w1 - w2/w1);
   }
} else {
   for (i = 1; i < n; i++){
      sum = (sum + s[n- i-1])*z1;
   }
}
sum = -sum/(1.0 - w2);


// compute c^{+}
n = DataLength;
s[0]  = s[0] + z1*sum;
for (i = 1; i < n; i++) {
   s[i]  = s[i] + z1*s[i-1];
   //printf("cp[%i] = %e, %f \n", i, cp[i], z1);
}

// compute c^- 
s[n-1] = -z1*s[n-1];
for (i = 1; i < n; i++) {
   ni = n - i; 
   ni1 = ni - 1;
   s[ni1]  = z1*(s[ni] - s[ni1]);
}

for (i = 0; i < n; i++) {
   s[i]  = 6.0*s[i];
}

}


void Bspline::ConvertToInterpolationCoefficients_2D(float *s, int nx, int ny, float EPSILON) 
//		float	*s,     /* input samples --> output coefficients               */
//		int	nx,	/* number of samples or coefficients in x direction    */
//		int	ny,	/* number of samples or coefficients in y direction    */
//		float	EPSILON	/* admissible relative error                           */
{
float *d, z1;
int    i, l;

d = (float *) malloc (nx*sizeof (float)); 

z1 = sqrt(3.0) - 2.0;
// x-direction interpolation
for (l = 0; l < ny; l++) {
   for (i = 0; i < nx; i++) {
      d[i] = s[i*ny + l];
   }
   ConvertToInterpolationCoefficients_1D(d, nx,CVC_DBL_EPSILON);
   //ConvertToInterpolationCoefficients_Qu(d, nx,&z1, 1, CVC_DBL_EPSILON);


   for (i = 0; i < nx; i++) {
      s[i*ny + l] = d[i];
   }
}

// y-direction interpolation
for (i = 0; i < nx; i++) {
   ConvertToInterpolationCoefficients_1D(s+i*ny, ny,CVC_DBL_EPSILON);
   //ConvertToInterpolationCoefficients_Qu(s+i*ny, ny, &z1, 1, CVC_DBL_EPSILON);

}

free(d);
}

void Bspline::ConvertToInterpolationCoefficients_3D(float *s, int nx, int ny, 
                                                     int nz,   float EPSILON)
//              float   *s,    /* input samples --> output coefficients               */
//              int     nx,    /* number of samples or coefficients in x direction    */
//              int     ny,    /* number of samples or coefficients in y direction    */
//              int     nz,    /* number of samples or coefficients in z direction    */
//              float   EPSILON/* admissible relative error                           */
{
float *d, z1;
int    u, v, w, k, kk;

d = (float *) malloc (nx*sizeof (float));

k = ny*nz;
z1 = sqrt(3.0) - 2.0;

// x-direction interpolation
for (v = 0; v < ny; v++) {
   for (w = 0; w < nz; w++) {
      kk = v*nz + w;
      for (u = 0; u < nx; u++) {
         d[u] = s[u*k + kk];
      }
      ConvertToInterpolationCoefficients_1D(d, nx,CVC_DBL_EPSILON);
      //ConvertToInterpolationCoefficients_Qu(d, nx,&z1, 1, CVC_DBL_EPSILON);


      for (u = 0; u < nx; u++) {
         s[u*k + kk] = d[u];
      }
   }
}

for (u = 0; u < nx; u++) {
   ConvertToInterpolationCoefficients_2D(s+u*k, ny, nz, CVC_DBL_EPSILON);
}

free(d);
}

/*
//old.
bool Bspline::Bspline_Projection(int p, int q, int r, int sub, float rotmat[9], int sample_num, float translate, float *prjimg, int *start_point)
{

  int i, j, k, s, t, lx, ly, rx, ry, half, size, partition_t, proj_length, proj_size;
  int gM, numb_in, ii, m, scale;
  float e1[3], e2[3],d[3];
  float center[3], a[3], point[3],  Bradius, h, R2, sum;
  float A, B, C, t1, t2, values_t, delt, temp;
  float xx, x, y, z, value, kot, weit;
  float tx, ty, step;

  //printf("\nbegin compute bspline projection.\n");
  partition_t = 100;
  gM          = 4 * partition_t;
  m           = FinishX - StartX;
 
  h      = (float)delx;
scale 	 = (int)h;
  //printf("\nh=%f ",h);

  half   = sample_num/2;  
  size   = sample_num*sample_num;

  if(sub == 1)
	{
	  proj_length = 8*scale;
	  proj_size   = proj_length*proj_length; 
	  step        = 1.0;
	}
  if (sub == 2)
	{
	  proj_length = sub*8*scale-1;
	  proj_size   = proj_length*proj_length; 
	  step        = 0.5;
	}



for ( i = 0; i < proj_size; i++ )
        prjimg[i] = 0.0;

  center[0] = h * p;
  center[1] = h * q;
  center[2] = h * r;
 
  point[0] = center[0];
  point[1] = center[1];
  point[2] = center[2];

  //The sphere radius which includes the cube [(p-2)delx, (p+2)delx]X[(q-2)delx, (q+2)delx]X[(r-2)delx, (r+2)delx];
  Bradius = sqrt(3.0)/2.0 * 4*h;  
  R2      = Bradius * Bradius; 

  //printf("\ncenter=%f %f %f Bradius=%f R2=%f half=%d ", center[0], center[1],center[2], Bradius, R2, half);

  for ( i = 0; i < 3; i++ )
	{
	  e1[i] = rotmat[i];
	  e2[i] = rotmat[3+i];
	  d[i]  = rotmat[6+i];

	  // printf("\ne1=%f e2=%f d=%f ", e1[i], e2[i], d[i]);

	}



  A = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];


  ProjectPoint2Plane(point, e1, e2);

  lx = point[0] - Bradius;
  ly = point[1] - Bradius;

  //printf("\npoint=%f %f %f", point[0], point[1], point[2]);

  lx = (-half <= lx)?lx:(-half); //image range: [-half, half]X[-half, half];

  ly = (-half <= ly)?ly:(-half);

  rx = point[0] + Bradius;
  ry = point[1] + Bradius;
 
  rx = (half >= rx)?rx:half;
  ry = (half >= ry)?ry:half;
  //printf("000 lx rx ly ry=%d %d %d %d ", lx, rx, ly, ry);

  start_point[0] = lx;
  start_point[1] = ly;
  if(rx-lx > 7 || ry - ly > 7) printf("\nproj_length is too small.lx rx ly ry=%d %d %d %d", lx, rx, ly, ry);//getchar();
  if(lx <= rx && ly <= ry)
	{
	  
	  //for ( i = lx; i <=rx; i++ )
	  //for ( j = ly; j <= ry; j++ )
	  for ( s = 0; s <= sub*(rx-lx); s++ )
		for ( t = 0; t <= sub*(ry-ly); t++ )
		  {
			//i = s + lx;
			//j = t + ly;
			tx  = lx + step * s;
			ty  = ly + step * t;

			a[0] = tx*e1[0] + ty*e2[0];
			a[1] = tx*e1[1] + ty*e2[1];
			a[2] = tx*e1[2] + ty*e2[2];

			B = 2 * d[0] * (a[0] - center[0]) + 
			    2 * d[1] * (a[1] - center[1]) +
			    2 * d[2] * (a[2] - center[2]);
			C = (a[0] - center[0]) * (a[0] - center[0]) + 
			    (a[1] - center[1]) * (a[1] - center[1]) +
			    (a[2] - center[2]) * (a[2] - center[2]) - R2;


			//printf("\nA=%f B=%f C=%f a=%f %f %f", A, B, C, a[0], a[1], a[2]);
			if(B*B-4*A*C <= 0.0 ) continue;

			C  = sqrt(B*B-4*A*C);
						
			t1 = 0.5*(-B-C)/A;
			t2 = 0.5*(-B+C)/A;
			//printf("\nt1=%f t2=%f ", t1, t2);
			if(t1 > t2 ) {temp = t1; t1 = t2; t2 = temp;}
			delt = (t2-t1)/partition_t;
			sum = 0.0;
			// evaluate numerical integral along line i*e1 + j*e2+t*d;

			// Gaussian 	
			//for ( k = 0; k < gM; k++ )
			  //{
			//	numb_in = k/4;
			//	ii      = k - 4 * numb_in; 
			//	kot     = (float)Kot[ii]; 
			//	xx      = t1 + delt * (numb_in + kot);
			//	x = a[0] + xx * d[0];
			//	y = a[1] + xx * d[1];
			//	z = a[2] + xx * d[2];
//
//				x = x/h - p;
//				y = y/h - q;
//				z = z/h - r;
//				//printf("\nx y z = %f %f %f p q r = %d %d %d ", x,y,z, p , q, r);
//				Spline_N_Base(x, m, &value);
//				x = value;
//
//				Spline_N_Base(y, m, &value);
//				y = value;
//
//				Spline_N_Base(z, m, &value);
//				z = value;
//
//				//if(i==0 && j==0) printf("\nx = %f y = %f z= %f ", x,y,z);
//
//				values_t =  x * y * z;
//				weit = (float)Weit[ii];
//				sum = sum + weit*values_t;   // wrong. here should multiply weit^3.
//			  }
//			sum = sum * delt ;
			

			for ( k = 0; k < partition_t; k++ )
			  {
				xx = t1+ delt * k;

				x = a[0] + xx * d[0];
				y = a[1] + xx * d[1];
				z = a[2] + xx * d[2];

				x = x/h - p;
				y = y/h - q;
				z = z/h - r;
				//printf("\nx y z = %f %f %f p q r = %d %d %d ", x,y,z, p , q, r);
				Spline_N_Base(x, m, &value);
				x = value;

				Spline_N_Base(y, m, &value);
				y = value;

				Spline_N_Base(z, m, &value);
				z = value;

				values_t =  x * y * z;

				sum += values_t * delt;
			  }



			//	if(i==0 && j==0 ) printf("\nsum=%f i=%d j=%d sample_num=%d, nx=%d", sum, i,j, sample_num, nx);
		
			//prjimg[(i+half)*sample_num + j+half] = sum;
			//prjimg[(i-lx)*proj_length + j-ly] = sum;

			prjimg[s*proj_length + t] = sum;



		  }
	}


  //getchar();

  return true;

}
*/



//Modified By Prof. Xu.
bool Bspline::Bspline_Projection(int p, int q, int r, int sub, float rotmat[9], int sample_num, 
                                              float translate, float *prjimg, int *start_point)
{

int i, j, k, s, t, lx, ly, rx, ry, half, size, partition_t, proj_size;
int gM, numb_in, ii, m, scale;
float e1[3], e2[3],d[3];
float center[3], a[3], point[3],  Bradius, h, R2, sum;
float A, B, C, t1, t2, values_t, delt, temp;
float xx, x, y, z, value, kot, weit;
float tx, ty, step;

partition_t = 100;
gM          = 4 * partition_t;
m           = FinishX - StartX;
 
h      = (float)delx;
//printf("\nbegin compute bspline projection. h = %f, sub = %d\n", h, sub);
scale 	 = (int)h;
//printf("\nh=%f ",h);

half   = sample_num/2;  
size   = sample_num*sample_num;

// Xu added these three lines
 proj_size   = PRO_LENGTH_SUB * PRO_LENGTH_SUB; //  proj_length*proj_length;
 step        = 1.0/sub;


// set initial values
for ( i = 0; i < proj_size; i++ ) {
   prjimg[i] = 0.0;
}

center[0] = h * p;
center[1] = h * q;
center[2] = h * r;
 
point[0] = center[0];
point[1] = center[1];
point[2] = center[2];

//The sphere radius which includes the cube [(p-2)delx, (p+2)delx]X[(q-2)delx, (q+2)delx]X[(r-2)delx, (r+2)delx];
Bradius = sqrt(3.0) * 2*h;  
R2      = Bradius * Bradius; 

//printf("\ncenter=%f %f %f Bradius=%f R2=%f half=%d ", center[0], center[1],center[2], Bradius, R2, half);

for ( i = 0; i < 3; i++ ) {
  e1[i] = rotmat[i];
  e2[i] = rotmat[3+i];
  d[i]  = rotmat[6+i];

  // printf("\ne1=%f e2=%f d=%f ", e1[i], e2[i], d[i]);
}

A = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];


// Xu project point to 2d plane
ProjectPoint2Plane(point, e1, e2);

lx = floor(point[0] - Bradius) + 1; // Xu Changed
ly = floor(point[1] - Bradius) + 1;

//printf("\npoint=%f %f %f", point[0], point[1], point[2]);

lx = (-half <= lx)?lx:(-half); //image range: [-half, half]X[-half, half];
ly = (-half <= ly)?ly:(-half);

rx = floor(point[0] + Bradius); // Xu Changed
ry = floor(point[1] + Bradius); 
/*
  rx = (half-1 >= rx)?rx:half-1;
  ry = (half-1 >= ry)?ry:half-1;
*/
rx = (half >= rx)?rx:half;
ry = (half >= ry)?ry:half;

//printf("\n 000 lx rx ly ry=%d %d %d %d, sub = %d , step = %f, proj_length = %d\n", lx, rx, ly, ry, sub, step,proj_length);

start_point[0] = lx;   
start_point[1] = ly;  
 //if(rx-lx > 7 || ry - ly > 7) printf("\nproj_length is too small.lx rx ly ry=%d %d %d %d", lx, rx, ly, ry);//getchar();
/*
   if(rx-lx+1 > proj_length || ry-ly+1>proj_length ) {
      printf("\nproj_length is too small.lx rx ly ry=%d %d %d %d", lx, rx, ly, ry);getchar();
}
*/
 
if (lx <= rx && ly <= ry) {
	  
   //for ( i = lx; i <=rx; i++ )
   //for ( j = ly; j <= ry; j++ )
   for ( s = 0; s <= sub*(rx-lx); s++ ) {
      for ( t = 0; t <= sub*(ry-ly); t++ ) {
 	 //i = s + lx;
	 //j = t + ly;
	 tx  = lx + step * s;
	 ty  = ly + step * t;

         // point in 2d plane
	 a[0] = tx*e1[0] + ty*e2[0];
	 a[1] = tx*e1[1] + ty*e2[1];
	 a[2] = tx*e1[2] + ty*e2[2];

	 B = 2 * d[0] * (a[0] - center[0]) + 
	     2 * d[1] * (a[1] - center[1]) +
	     2 * d[2] * (a[2] - center[2]);
	 C = (a[0] - center[0]) * (a[0] - center[0]) + 
	     (a[1] - center[1]) * (a[1] - center[1]) +
	     (a[2] - center[2]) * (a[2] - center[2]) - R2;


	 //printf("\nA=%f B=%f C=%f a=%f %f %f", A, B, C, a[0], a[1], a[2]);
	 if(B*B-4*A*C <= 0.0 ) continue;

	 C  = sqrt(B*B-4*A*C);
			
         // Intersect point of the project line with sphere
  	 t1 = 0.5*(-B-C)/A;
	 t2 = 0.5*(-B+C)/A;
	 //printf("\nt1=%f t2=%f ", t1, t2);

	 // if(t1 > t2 ) {temp = t1; t1 = t2; t2 = temp;} // Changed Commoned, since t1 < t2
	 delt = (t2-t1)/partition_t;

	 sum = 0.0;
	 // evaluate numerical integral along line i*e1 + j*e2+t*d, using Ju_Xing formula;
	 // could be easily changed to Ti_xing formula
	 for ( k = 0; k < partition_t; k++ ) {
	    xx = t1+ delt * k;

	    x = a[0] + xx * d[0];
	    y = a[1] + xx * d[1];
	    z = a[2] + xx * d[2];

	    x = x/h - p;
	    y = y/h - q;
	    z = z/h - r;
	    //printf("\nx y z = %f %f %f p q r = %d %d %d ", x,y,z, p , q, r);
	    //Spline_N_Base(x, m, &value);
            Spline_N_Base(x, &value);

            x = value;

	    Spline_N_Base(y, &value);
	    y = value;

	    Spline_N_Base(z, &value);
	    z = value;

	    values_t =  x * y * z;

	    sum += values_t * delt;
         }

         prjimg[s*PRO_LENGTH_SUB + t] = sum;
      }
   }
}

/*
for ( i = 0; i < sample_num; i++ ) { 
  printf("\n");

   for ( j = 0; j < sample_num; j++ )
      printf("%f ", prjimg[i * sample_num + j]);
}
*/

//getchar();

return true;
// Xu Checked of this code. Doubt about lx, ly, it is right for sub = 1. but for sub = 2?

}


bool Bspline::Bspline_GridProjection(int p, int q, int r, float rotmat[9], int sample_num, float translate, float *prjimg, int *start_point)
{
  
  int x, y, z, i, j, k, ii, lx, ly, lz, rx, ry, rz, ix, iy, iz, scale;
  int proj_length, proj_size, nx, ny, nz, plx, ply, prx, pry, half;
  float Trotmat[9], old[3], ox, oy, oz, X[3], xf, yf, h;
  float xx, yy, zz, point[3], Bradius, e1[3], e2[3], value;

  //printf("\np q r = %d %d %d ", p, q, r);
  half   = sample_num/2;  
  h     = (float)delx;
  scale = (int)delx;

  lx = (p - 2)*scale;
  ly = (q - 2)*scale;
  lz = (r - 2)*scale;

  rx = (p + 2)*scale;
  ry = (q + 2)*scale;
  rz = (r + 2)*scale;

  proj_length = 8*scale;
  proj_size = proj_length*proj_length; 

  point[0] = h * p;
  point[1] = h * q;
  point[2] = h * r;

  Bradius = sqrt(3.0)/2.0 * 4*h;  

for ( i = 0; i < 3; i++ )
	{
	  e1[i] = rotmat[i];
	  e2[i] = rotmat[3+i];
	}
  ProjectPoint2Plane(point, e1, e2);

  plx = point[0] - Bradius;
  ply = point[1] - Bradius;

  //printf("\npoint=%f %f %f", point[0], point[1], point[2]);

  plx = (-half <= plx)?plx:(-half); //image range: [-half, half]X[-half, half];

  ply = (-half <= ply)?ply:(-half);

  prx = point[0] + Bradius;
  pry = point[1] + Bradius;
/*
  rx = (half-1 >= rx)?rx:half-1;
  ry = (half-1 >= ry)?ry:half-1;
*/
  prx = (half >= prx)?prx:half;
  pry = (half >= pry)?pry:half;

  start_point[0] = plx;
  start_point[1] = ply;

  for ( i = 0; i < proj_size; i++ )
	prjimg[i] = 0.0;
  
  Matrix3Transpose(rotmat, Trotmat);  

  nx = ny = nz = sample_num;
  ox = oy = oz = (nx-1)*0.5;

  //printf("\nlx ly lz rx ry rz=%d %d %d %d %d %d", lx, ly,lz, rx, ry, rz); 
  for ( x = lx; x <= rx; x++ )
	{	  
	  //printf("\nxxxxxx=%d ", x);
	  old[0] = x;       // - ox;
	  for ( y = ly; y <= ry; y++ )
		{
		  old[1] = y;  /// - oy;
		  for ( z = lz; z <= rz; z++ )
			{
			  old[2] = z;   // - oz;

			  MatrixMultiply(Trotmat,3,3,old,3,1,X);
			  //X[0] = X[0]; // + ox;
			  //X[1] = X[1]; // + oy;
			  //X[2] = X[2]; // + oz;
			  
			  ix = floor(X[0]);
			  iy = floor(X[1]);
			  iz = floor(X[2]);

			  if( ix+half >= 0 && ix+half < nx-1 && iy+half >= 0 && iy+half < ny-1 )
				{
				  xf = X[0] - ix;
				  yf = X[1] - iy;
				  //i  = x*nynz+ y*nz+z;
				  
				  xx = x/h - p;
				  yy = y/h - q;
				  zz = z/h - r;

				  Spline_N_Base(xx, &value);
				  xx = value;
				  Spline_N_Base(yy, &value);
				  yy = value;
				  Spline_N_Base(zz, &value);
				  zz = value;
				  // printf("\nix iy iz = %d %d %d x y z = %d %d %d xx yy zz = %f %f %f ", ix, iy, iz, x, y, z, xx, yy, zz);
				  value = xx*yy*zz;

				  for (j = ix; j < ix+2; j++ )
					{
					  xf = 1 - xf;
					  for ( k = iy; k < iy+2; k++ )
						{
						  yf = 1 - yf;
						  if( j>=plx && j <= prx  && k >= ply && k <= pry)
							{
							  ii = (j-plx) *  proj_length  + k - ply;
							  prjimg[ii] += value*xf*yf;
							}
						}
					}

				}
			}
		}
	}

  //getchar();
}


/*************************************************************************
 Descriptions:
     Convert coefficients of Cubic Volume Bspline to Cubic Bspline Volume
     data f. 
 *************************************************************************/
void Bspline::ObtainObjectFromCoeffs(float *Rcoef,float *f)
{

  int   i, j, k, usedN, usedN2, usedN3;
  int    ii,jj, iii,jjj,kkk,xN3,id, jd, yN3,zN3,NN, N2;
 float    Phi_i, Phi_j, Phi_k, Phi_ij;

 usedN = N - 4;

 NN = ny*nz;
 usedN2 = (usedN+1)*(usedN+1);
 usedN3 = 3*(usedN+1);


 //3 = 3*(N+1);

 int x, y,z;

 for ( x = 0; x < nx; x++ ) 
   {
	 iii = x * NN;
     xN3 = x * usedN3;

    for ( y = 0; y < ny; y++ )
	  {
		jjj = y * nz;
		yN3 = y * usedN3;
 
       for ( z = 0; z < nz; z++ )
          {
           ii  = iii + jjj + z;
		   zN3 = z * usedN3;
		   //printf("\nii=%u ", ii);
          for ( i = 0; i <= usedN; i++ )
			{
			   Phi_i =  OrthoBBaseGrid2[xN3+i+i+i];
			  //Phi_i = bspline->BernBaseGrid[xN3+i+i+i]; 
			  id = i*usedN2;
			  //printf("\nx=%d,Phi_i = %f ", x, Phi_i);
             for ( j = 0; j <= usedN; j++ )
			   {
				 Phi_j  = OrthoBBaseGrid2[yN3+j+j+j];
				 // Phi_j = bspline->BernBaseGrid[yN3+j+j+j];  
                 Phi_ij = Phi_i * Phi_j;
				 // printf("y=%d, Phi_j %f ", y, Phi_j);
				 jd = j*(usedN+1);

                for ( k = 0; k <= usedN; k++ )
                   {
                    jj = id+ jd + k;

                    Phi_k = OrthoBBaseGrid2[zN3+k+k+k];
					//Phi_k = bspline->BernBaseGrid[zN3+k+k+k]; 

					//					printf("z=%d, Phi_k %f ", z, Phi_k);  
					//	printf("\nx = %d y = %d z = %d i = %d j=%d k=%d Phi_i=%f, Phi_j=%f, Phi_k=%f,Phi_ijk=%f ", x, y, z,i, j, k, Phi_i , Phi_j, Phi_k,Phi_ij*Phi_k);

                    f[ii] = f[ii] + Rcoef[jj] * Phi_ij * Phi_k;

                   }
				//printf("\nii = %d f=%f ", ii, f[ii]);
				//	getchar();
 
               }
			}
		  }
	  }
   }
}

void Bspline::ObtainObjectFromNonOrthoCoeffs(float *Rcoef,float *f)
{

  int   i, j, k, usedN, usedN2, usedN3;
  int  ii,jj, iii,jjj,kkk,xN3,id, jd, yN3,zN3,NN, N2;
 float    Phi_i, Phi_j, Phi_k, Phi_ij;

 usedN = N - 4;

 NN     = (ImgNy+1)*(ImgNz+1);
 usedN2 = (usedN+1)*(usedN+1);
 usedN3 = 3*(usedN+1);
 
 
 for ( i = 0 ; i< NN*(ImgNx+1); i++ )
        f[i] = 0.0;

 //3 = 3*(N+1);

 int x, y,z;

 for ( x = 1; x < ImgNx; x++ ) 
   {
	 iii = x * NN;
     xN3 = x * usedN3;

    for ( y = 1; y < ImgNy; y++ )
	  {
		jjj = y * (ImgNz+1);
		yN3 = y * usedN3;
 
       for ( z = 1; z < ImgNz; z++ )
          {
           ii  = iii + jjj + z;
		   zN3 = z * usedN3;
		   
		   //for ( i = 0; i <= usedN; i++ )
			 for ( i = x-3; i<= x-1; i++ )
			 {
			   if( i < 0 || i > N-4 ) continue;
			   Phi_i =  BBaseImgGrid[xN3+i+i+i];
			   //printf("\nx=%d phi_i=%f ", x, Phi_i);
			   id = i*usedN2;
			   //for ( j = 0; j <= usedN; j++ )
			   for ( j = y-3; j <= y-1; j++ )
				 {
                                  if ( j < 0 || j > N-4 ) continue;
				   Phi_j  = BBaseImgGrid[yN3+j+j+j];
				   //printf("y=% dphi_j=%f ", y, Phi_j);
				   Phi_ij = Phi_i * Phi_j;
			
				   jd = j*(usedN+1);
				   //for ( k = 0; k <= usedN; k++ )
				   for ( k = z-3; k <= z-1; k++ )
					 {
                                           if ( k < 0 || k > N-4 ) continue;
					   jj = id+ jd + k;
					   
					   Phi_k = BBaseImgGrid[zN3+k+k+k];
					   //printf("z=%d phi_k=%f ", z, Phi_k);
					   //printf("coef=%f ", Rcoef[jj]);
					   f[ii] = f[ii] + Rcoef[jj] * Phi_ij * Phi_k;
					   //if(x ==ImgNx/2 && y == ImgNx/2 && z == ImgNx/2-1 )  printf("\nxxx y z = %d %d %d f=%f Phi_i i k=%f %f %f ",x, y, z, f[ii],Phi_i, Phi_j,Phi_k);
					 }
			
				   
				 }
			 }
			 //if(x==ImgNx/2 && y == ImgNx/2 ) printf("\nx y z = %d %d %d f=%f ", x, y, z, f[ii]);
			 // printf("\nssii= %d f=%f " , ii, f[ii]);

		  }
	  }
   }
}



void Bspline::ObtainObjectFromNonOrthoCoeffs_sub(float *Rcoef,float *f, int sub)
{

  int      i, j, k, usedN, usedN2, usedN3, subImgNx, subNN;
  int      ii,jj, iii,jjj,kkk,xN3,id, jd, yN3,zN3,NN, N2, h, scale;
 float    Phi_i, Phi_j, Phi_k, Phi_ij,sum;

 h      = (int)delx;
 scale 	= h * sub;

 usedN = N - 4;
 
 NN     = (ImgNy+1)*(ImgNz+1);
 usedN2 = (usedN+1)*(usedN+1);
 usedN3 = 3*(usedN+1);
 
 
 //sub = 2;
 subImgNx = sub * ImgNx;
 subNN = (subImgNx+1) * (subImgNx+1);

 for ( i = 0 ; i< subNN*(subImgNx+1); i++ )
        f[i] = 0.0;

 //3 = 3*(N+1);

 int x, y,z;


// for ( i = 0; i < usedN2*(usedN+1); i++ ) printf("\nRcoef=%f   ", Rcoef[i]);

 for ( x = 1; x < subImgNx; x++ ) 
   {
	 iii = x * subNN;
     xN3 = x * usedN3;

    for ( y = 1; y < subImgNx; y++ )
	  {
		jjj = y * (subImgNx+1);
		yN3 = y * usedN3;
 
       for ( z = 1; z < subImgNx; z++ )
          {
           ii  = iii + jjj + z;
		   zN3 = z * usedN3;
		   //sum = 0.0; 
		   //for ( i = 0; i <= usedN; i++ )
		   //for ( i = x/2-3; i<= x/2; i++ )                //sub image.  07-12-09.
                     for ( i = x/scale-3; i<= x/scale; i++ )
			 {
			   if( i < 0 || i > N-4 ) continue;
			   Phi_i =  subBBaseImgGrid[xN3+i+i+i+0];
                           //sum = sum + Phi_i;
                           //printf("\nphi_i=%f ", Phi_i);
			   id = i*usedN2;
                           //for ( j = 0; j <= usedN; j++ )
			   //for ( j = y/2-3; j <= y/2; j++ )
			   for ( j = y/scale-3; j <= y/scale; j++ )
				 {
				   if ( j < 0 || j > N-4 ) continue;
				   Phi_j  = subBBaseImgGrid[yN3+j+j+j+0];
                                   //printf("phi_j=%f ", Phi_j);
				   Phi_ij = Phi_i * Phi_j;
			
				   jd = j*(usedN+1);
				   //for ( k = 0; k <= usedN; k++ )
				   //for ( k = z/2-3; k <= z/2; k++ )
				   for ( k = z/scale-3; k <= z/scale; k++ )
					 {
                                           if ( k < 0 || k > N-4 ) continue;
					   jj = id+ jd + k;
					   
					   Phi_k = subBBaseImgGrid[zN3+k+k+k+0];
                 			   //printf("phi_k=%f ", Phi_k);
                                           //printf("coef=%f ", Rcoef[jj]);
					   f[ii] = f[ii] + Rcoef[jj] * Phi_ij * Phi_k;
					   
					 }
				   
				 }
			 }
                  
		   //printf("\nsub=%d ii= %d f=%f ",  sub, ii, f[ii]);

		  }
	  }
   }
}



void Bspline::ObtainObjectFromNonOrthoCoeffs_FA(float *Rcoef,float *f)
{

  int   i, j, k, usedN, usedN2, usedN3;
  int   ii,jj, iii,jjj,kkk,xN3,id, jd, yN3,zN3,NN, N2;
 float    Phi_i, Phi_j, Phi_k, Phi_ij;

 usedN = N - 4;

 NN     = (ImgNy+1)*(ImgNz+1);
 usedN2 = (usedN+1)*(usedN+1);
 usedN3 = 3*(usedN+1);

 for (i = 0; i < NN * (ImgNx+1); i++ ) f[i] = 0.0;

 //3 = 3*(N+1);

 int x, y,z;

 for ( x = 1; x < ImgNx; x++ ) 
   {
	 iii = x * NN;
     xN3 = x * usedN3;

    for ( y = 1; y < ImgNy; y++ )
	  {
		jjj = y * (ImgNz+1);
		yN3 = y * usedN3;
 
       for ( z = 1; z < ImgNz; z++ )
          {
           ii  = iii + jjj + z;
		   zN3 = z * usedN3;
		   
		   for (i = x/2-3; i <= x/2; i++ )
			 {
			   if( i < 0 || i > N-4 ) continue;

			   Phi_i =  BBaseImgGrid[xN3+i+i+i];
			   id = i * usedN2;

			   for (j = y/2-3; j <= y/2; j++ )
				 {
				   if ( j < 0 || j > N-4 ) continue;

				   Phi_j  = BBaseImgGrid[yN3+j+j+j];
				   Phi_ij = Phi_i * Phi_j;
				   jd = j * (usedN+1);

				   for ( k = z/2-3; k <= z/2; k++ )
					 {
					   if ( k < 0 || k > N-4 ) continue;
 
					   jj = id+ jd + k;
					   
					   Phi_k = BBaseImgGrid[zN3+k+k+k];
					   
					   f[ii] = f[ii] + Rcoef[jj] * Phi_ij * Phi_k;
					   //printf("\ni j k =%d %d %d", i,j,k);getchar();
					   // printf("\nijk=%f %f %f ", Phi_i, Phi_j, Phi_k);
					 }
				 }
			 }
		   // printf("\nf=%f ", f[ii]);
		  }
	  }
   }

 /*
 printf("\nfast");
 for ( i = 0; i < ImgNx * ImgNy * ImgNz; i++ )
   printf("\nf=%f ", f[i]);
 getchar();
 */
}
