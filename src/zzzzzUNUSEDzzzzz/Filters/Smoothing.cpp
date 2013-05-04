//originally written by Sangmin Park, modified by Joe!

/* $Id: Smoothing.cpp 1498 2010-03-10 22:50:29Z transfix $ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <map>
#include <Filters/Smoothing.h>
#include <vector>

#define maxIndex 	20
#define	SHAREDT		15

//#define	FIX_BOUNDARY
//#define	PERTURB_1
#define GEOMETRIC_FLOW
#define	SMOOTHING
//#define PERTURB_2
//#define COMPUTING_MOLECULE_LOCATIONS

#ifdef __APPLE__
#define isnan(X) __inline_isnan((double)X)
#else if defined(__WINDOWS__)
#define isnan(X) false
#endif 

namespace Smoothing
{
  float  *Vertices_gf;
  float  *VerticesBackup_gf;
  float  *VertexNormal_gf;
  int    *FaceIndex_gi;
  //int    *SharedTriangleIndex_gi;	// Shared triangle indexes of a vertex
  typedef std::vector<int> TriangleIndexVec;
  typedef std::vector<TriangleIndexVec> SharedTriangleIndexVec;
  SharedTriangleIndexVec SharedTriangleIndex_gi;


#ifdef	COMPUTING_MOLECULE_LOCATIONS
  // Molecule locations
  int		NumMolecules_gi;
  int		*TriangleIndexes_gi;
  float	*MoleculeLocs_gf, *MoleculeNormal_gf;
#endif


  unsigned char	*FixedVertices_guc;

  bool initMem(int NumVertices_i, int ntet);
  void freeMem();

  // calculate normal at v0
  void crossproduct(float v0[3], float v1[3], float v2[3], float* normal) {

    float v01[3], v02[3], g;

    int i;
    for(i = 0; i < 3; i++) {
      v01[i] = v1[i] - v0[i];
      v02[i] = v2[i] - v0[i];
    }

    normal[0] = v01[1]*v02[2] - v02[1]*v01[2];
    normal[1] = v01[2]*v02[0] - v02[2]*v01[0];
    normal[2] = v01[0]*v02[1] - v02[0]*v01[1];


    g = normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2];
    if(g < 0.0) //avoid NaN
      for(i = 0; i < 3; i++) normal[i] = 0.0;
    else
      for(i = 0; i < 3; i++) normal[i] /= (float) sqrt(g);
  }

  // calculate the area for a triangle
  float area_tri(float v0[3], float v1[3], float v2[3]) {

    float a, b, c, p, area;
    int i;

    a = 0.0;		b = 0.0;		c = 0.0;
    for(i = 0; i < 3; i++) {
      a += (v1[i] - v0[i])*(v1[i] - v0[i]);
      b += (v2[i] - v1[i])*(v2[i] - v1[i]);
      c += (v0[i] - v2[i])*(v0[i] - v2[i]);
    }
    a = (float)sqrt(a);
    b = (float)sqrt(b);
    c = (float)sqrt(c);
	
    p = (a + b + c)*0.5;
    area = (float)sqrt(p * (p - a) * (p - b) * (p - c));

    return area;
  }

  // return length
  float Normalize(float *Vec3)
  {
    float	Tempf;
	
	
    Tempf = sqrt (Vec3[0]*Vec3[0] + Vec3[1]*Vec3[1] + Vec3[2]*Vec3[2]);
    if (Tempf<1e-6) {
      Vec3[0] = 0.0;
      Vec3[1] = 0.0;
      Vec3[2] = 0.0;
    }
    else {
      Vec3[0] /= Tempf;
      Vec3[1] /= Tempf;
      Vec3[2] /= Tempf;
    }
	
    return Tempf;
  }


  // Uniform Radom Number Generator from 0 to 1
  float ranf()
  {
    double	UniformRandomNum;

    UniformRandomNum = (double)rand();
    UniformRandomNum /= (double)RAND_MAX;
    return (float)UniformRandomNum; // from 0 to 1
  }


  // Uniform Radom Number Generator from Minf to Maxf
  float ranf(float Minf, float Maxf)
  {
    double	UniformRandomNum;

    UniformRandomNum = (double)rand();
    UniformRandomNum /= (double)RAND_MAX;
    UniformRandomNum *= (Maxf - Minf);
    UniformRandomNum += Minf;
	
    return (float)UniformRandomNum; // from 0 to 1
  }



  //
  // Normal Random Variate Generator
  // Mean=m, Standard deviation=S
  //
  float GaussianRandomNum(float m, float s)     
  {                                       
    float x1, x2, w, y1;
    static float y2;
    static int use_last = 0;

    // use value from previous call
    if (use_last) {
      y1 = y2;
      use_last = 0;
    }
    else {
      do {
	x1 = 2.0 * ranf() - 1.0;
	x2 = 2.0 * ranf() - 1.0;
	w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );

      w = sqrt( (-2.0 * log( w ) ) / w );
      y1 = x1 * w;
      y2 = x2 * w;
      use_last = 1;
    }

    return( m + y1 * s );
  }

  float Distance(float *Pt1, float *Pt2)
  {
    float	Dist_f;
    double	Dist_d;
	
    Dist_d = (Pt2[0] - Pt1[0])*(Pt2[0] - Pt1[0]) +
      (Pt2[1] - Pt1[1])*(Pt2[1] - Pt1[1]) +
      (Pt2[2] - Pt1[2])*(Pt2[2] - Pt1[2]);
    Dist_f = (float)sqrt(Dist_d);
    return	Dist_f;
  }


  void smoothGeometry(Geometry *geo, float delta, bool fix_boundary) {

    int 	NumVertices_i, NumTriangles_i, i, j, k, v0, v1, v2;
    int 	itri, index, NumNeighborVertices_i;
    float 	vx, vy, vz, Pt0[3], Pt1[3], Pt2[3], WeightedCenter[3];
    float 	sum_0, sum_1[3], Normal_f[3], delta_t, area, t;
    float	Min_f[3], Max_f[3], Half_f[3], Length_f;
    float	AveCenter_f[3];
    float	nx, ny, nz, r, g, b;

    nx = r; ny = g; nz = b;
    //delta_t = 0.1f;
    delta_t = delta;

    NumVertices_i = geo->m_NumTriVerts;
    NumTriangles_i = geo->m_NumTris;

    if(NumTriangles_i == 0)
      {
	printf("No triangles to smooth, returning.\n");
	return;
      }

    if(!initMem(NumVertices_i, NumTriangles_i)) 
      {
	printf("Failed to initialize memory, aborting...\n");
	//abort();
	return;
      }

    for (k=0; k<3; k++) Min_f[k] = 9999999.9999;
    for (k=0; k<3; k++) Max_f[k] = -9999999.9999;
	
    for (i = 0; i < NumVertices_i; i++) {
      
      vx = geo->m_TriVerts[i*3 + 0];
      vy = geo->m_TriVerts[i*3 + 1];
      vz = geo->m_TriVerts[i*3 + 2];

      Vertices_gf[i*3 + 0] = VerticesBackup_gf[i*3 + 0] = vx;
      Vertices_gf[i*3 + 1] = VerticesBackup_gf[i*3 + 1] = vy;
      Vertices_gf[i*3 + 2] = VerticesBackup_gf[i*3 + 2] = vz;
		
      for (k=0; k<3; k++) if (Min_f[k] > Vertices_gf[i*3 + k]) Min_f[k] = Vertices_gf[i*3 + k];
      for (k=0; k<3; k++) if (Max_f[k] < Vertices_gf[i*3 + k]) Max_f[k] = Vertices_gf[i*3 + k];
    }

    for (k=0; k<3; k++) Half_f[k] = (Min_f[k] + Max_f[k])/2;
	
    printf ("Min = %.4f %.4f %.4f\n", Min_f[0], Min_f[1], Min_f[2]);
    printf ("Max = %.4f %.4f %.4f\n", Max_f[0], Max_f[1], Max_f[2]);
    printf ("Half = %.4f %.4f %.4f\n", Half_f[0], Half_f[1], Half_f[2]);

    for (i = 0; i < NumTriangles_i; i++) {

      v0 = geo->m_Tris[i*3 + 0];
      v1 = geo->m_Tris[i*3 + 1];
      v2 = geo->m_Tris[i*3 + 2];

      FaceIndex_gi[i*3 + 0] = v0;
      FaceIndex_gi[i*3 + 1] = v1;
      FaceIndex_gi[i*3 + 2] = v2;

      SharedTriangleIndex_gi[v0].push_back(i);
      SharedTriangleIndex_gi[v1].push_back(i);
      SharedTriangleIndex_gi[v2].push_back(i);
    }


    if(fix_boundary)
      {
	std::map<int, unsigned char> NeighborVertices_m;
	std::map<int, unsigned char>::iterator NeighborVertices_it;
	int NumFixedVertices_i = 0, NumNeighborTriangles_i;
	
	printf ("Finding fixed vertices ...\n");
	for(i=0; i<NumVertices_i; i++) {
	  NumNeighborTriangles_i = 0;
	  NeighborVertices_m.clear();

	  for(TriangleIndexVec::const_iterator j = SharedTriangleIndex_gi[i].begin();
	      j != SharedTriangleIndex_gi[i].end();
	      j++)
	    {
	      itri = *j;
	      v0 = FaceIndex_gi[itri*3 + 0];	
	      v1 = FaceIndex_gi[itri*3 + 1];
	      v2 = FaceIndex_gi[itri*3 + 2];	
	      NeighborVertices_m[v0] = 1;
	      NeighborVertices_m[v1] = 1;
	      NeighborVertices_m[v2] = 1;
	      NumNeighborTriangles_i++;
	    }

	  if ((int)NeighborVertices_m.size()-1==NumNeighborTriangles_i) FixedVertices_guc[i] = 0;
	  else {
	    FixedVertices_guc[i] = 1;	// Fixed vertex
	    NumFixedVertices_i++;
	  }
	}
	printf ("Num Fixed Vertices = %d / %d ", NumFixedVertices_i, NumVertices_i);
	printf ("%10.4f %%\n", (float)NumFixedVertices_i/NumVertices_i*100.0);
      }
    else
      {
	for(i=0; i<NumVertices_i; i++) FixedVertices_guc[i] = 0;
      }

#ifdef	PERTURB_1
    printf ("Perturbing ...\n");
    // calculate the normal for each vertex
    for(i = 0; i < NumVertices_i; i++) {
      for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] = 0.0f;
    }
    for(i = 0; i < NumVertices_i; i++) {
		
      if (FixedVertices_guc[i]==1) continue;
		
      for(TriangleIndexVec::const_iterator j = SharedTriangleIndex_gi[i].begin();
	  j != SharedTriangleIndex_gi[i].end();
	  j++)
	{
	  itri = *j;
	  v0 = FaceIndex_gi[itri*3 + 0];	
	  v1 = FaceIndex_gi[itri*3 + 1];
	  v2 = FaceIndex_gi[itri*3 + 2];
	  for(k = 0; k < 3; k++)
	    {
	      Pt0[k] = Vertices_gf[v0*3 + k];	
	      Pt1[k] = Vertices_gf[v1*3 + k];
	      Pt2[k] = Vertices_gf[v2*3 + k];	
	    }
	  crossproduct(Pt0, Pt1, Pt2, Normal_f);
	  for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] += Normal_f[k];
	}

      Length_f = Normalize(&VertexNormal_gf[i*3]);
      if (fabs(Length_f)<1e-4) {
	printf ("Error! Length is equal to zero\n"); fflush (stdout);
      }
    }

    float	EdgeLength_f[3], MaxDist_f, Perturb_f;
    for(i = 0; i < NumVertices_i; i++) {
		
      for(k=0; k<3; k++) {
	Pt0[k] = Vertices_gf[v0*3 + k];	
	Pt1[k] = Vertices_gf[v1*3 + k];
	Pt2[k] = Vertices_gf[v2*3 + k];	
      }
      EdgeLength_f[0] = Distance(Pt0, Pt1);
      EdgeLength_f[1] = Distance(Pt0, Pt2);
      EdgeLength_f[2] = Distance(Pt1, Pt2);
      MaxDist_f = -1.0;
      for(k=0; k<3; k++) {
	if (MaxDist_f < EdgeLength_f[k]) MaxDist_f = EdgeLength_f[k];
      }
		
      Perturb_f = ranf(-MaxDist_f/2, MaxDist_f/2);
      for(k = 0; k < 3; k++) {
	Vertices_gf[i*3 + k] += Perturb_f*VertexNormal_gf[i*3 + k];
      }
    }
#endif


#ifdef	GEOMETRIC_FLOW

    // Rounding sharp edges
    printf ("Geometric flow ...\n");

    // Adjusting each triangle size
    for(index = 0; index < maxIndex; index++) {

      printf("%d ", index); fflush (stdout);

      // calculate the normal for each vertex
      for(i = 0; i < NumVertices_i; i++) {
	for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] = 0.0f;
      }
      for(i = 0; i < NumVertices_i; i++) {
	
	for(TriangleIndexVec::const_iterator j = SharedTriangleIndex_gi[i].begin();
	  j != SharedTriangleIndex_gi[i].end();
	  j++)
	{
	  itri = *j;
	  v0 = FaceIndex_gi[itri*3 + 0];	
	  v1 = FaceIndex_gi[itri*3 + 1];
	  v2 = FaceIndex_gi[itri*3 + 2];
	  for(k = 0; k < 3; k++)
	    {
	      Pt0[k] = Vertices_gf[v0*3 + k];	
	      Pt1[k] = Vertices_gf[v1*3 + k];
	      Pt2[k] = Vertices_gf[v2*3 + k];	
	    }
	  crossproduct(Pt0, Pt1, Pt2, Normal_f);
	  for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] += Normal_f[k];
	}

	Length_f = Normalize(&VertexNormal_gf[i*3]);
	if (fabs(Length_f)<1e-4) {
	  printf ("Error! Length is equal to zero\n"); fflush (stdout);
	}
				
      }


      for(i = 0; i < NumVertices_i; i++) {

	if (FixedVertices_guc[i]==1) continue;
			
	// calculate the mass center WeightedCenter[3]
	sum_0 = 0.0f;
	for(k = 0; k < 3; k++) { WeightedCenter[k] = 0.0f; sum_1[k] = 0.0f;}

	for(TriangleIndexVec::const_iterator j = SharedTriangleIndex_gi[i].begin();
	  j != SharedTriangleIndex_gi[i].end();
	  j++)
	{
	  itri = *j;
	  v0 = FaceIndex_gi[itri*3 + 0];	
	  v1 = FaceIndex_gi[itri*3 + 1];
	  v2 = FaceIndex_gi[itri*3 + 2];
	  for(k = 0; k < 3; k++)
	    {
	      Pt0[k] = Vertices_gf[v0*3 + k];	
	      Pt1[k] = Vertices_gf[v1*3 + k];
	      Pt2[k] = Vertices_gf[v2*3 + k];	
	    }

	  area = area_tri(Pt0, Pt1, Pt2);
	  sum_0 += area;
	  for(k = 0; k < 3; k++) {
	    WeightedCenter[k] += (Pt0[k] + Pt1[k] + Pt2[k])*area/3.0f;
	  }
	}

	for(k = 0; k < 3; k++) WeightedCenter[k] /= sum_0;

	// calculate the new position in tangent direction
	// xi+1 = xi + delta_t*((m-xi) - (n, m-xi)n))
	t = 0.0f;
	for(k = 0; k < 3; k++) {
	  WeightedCenter[k] -= Vertices_gf[i*3 + k];
	  t += WeightedCenter[k]*VertexNormal_gf[i*3 + k];
	}
	for(k = 0; k < 3; k++) {
	  Vertices_gf[i*3 + k] += delta_t*WeightedCenter[k];		// mass center
	  //Vertices_gf[i*3 + k] += delta_t*(WeightedCenter[k] - t*VertexNormal_gf[i*3 + k]);// tangent movement
	}
      }

    } // end of index loop
    printf ("\n"); fflush (stdout);
#endif	
	

#ifdef	SMOOTHING
    printf ("Smoothing ...\n");

    // Smoothing
    for(index = 0; index < maxIndex; index++) {

      printf("%d ", index); fflush (stdout);

      // calculate the normal for each vertex
      for(i = 0; i < NumVertices_i; i++) {
	for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] = 0.0f;
      }
      for(i=0; i<NumVertices_i; i++) {

	for(TriangleIndexVec::const_iterator j = SharedTriangleIndex_gi[i].begin();
	    j != SharedTriangleIndex_gi[i].end();
	    j++)
	  {
	    itri = *j;
	    v0 = FaceIndex_gi[itri*3 + 0];	
	    v1 = FaceIndex_gi[itri*3 + 1];
	    v2 = FaceIndex_gi[itri*3 + 2];
	    for(k = 0; k < 3; k++)
	      {
		Pt0[k] = Vertices_gf[v0*3 + k];	
		Pt1[k] = Vertices_gf[v1*3 + k];
		Pt2[k] = Vertices_gf[v2*3 + k];	
	      }
	    crossproduct(Pt0, Pt1, Pt2, Normal_f);
	    for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] += Normal_f[k];
	  }
	
	Length_f = Normalize(&VertexNormal_gf[i*3]);
	if (fabs(Length_f)<1e-4) {
	  printf ("Error! Length is equal to zero\n"); fflush (stdout);
	}
				
      }


      for(i = 0; i < NumVertices_i; i++) {
		
	if (FixedVertices_guc[i]==1) continue;
	NumNeighborVertices_i = 0;
	for(k = 0; k < 3; k++) AveCenter_f[k] = 0.0f;

	for(TriangleIndexVec::const_iterator j = SharedTriangleIndex_gi[i].begin();
	    j != SharedTriangleIndex_gi[i].end();
	    j++)
	  {
	    itri = *j;
	    v0 = FaceIndex_gi[itri*3 + 0];	
	    v1 = FaceIndex_gi[itri*3 + 1];
	    v2 = FaceIndex_gi[itri*3 + 2];
	    for(k = 0; k < 3; k++)
	      {
		Pt0[k] = Vertices_gf[v0*3 + k];	
		Pt1[k] = Vertices_gf[v1*3 + k];
		Pt2[k] = Vertices_gf[v2*3 + k];	
	      }
	    
	    for(k = 0; k < 3; k++) {
	      AveCenter_f[k] += (Pt0[k] + Pt1[k] + Pt2[k])/3.0f;
	    }
	    NumNeighborVertices_i++;
	  }

	for(k = 0; k < 3; k++) {
	  AveCenter_f[k] /= (float)NumNeighborVertices_i;
	}

	for(k = 0; k < 3; k++) {
	  Vertices_gf[i*3 + k] = AveCenter_f[k];
	}
      }

    } // end of index loop
    printf ("\n"); fflush (stdout);
#endif


#ifdef	PERTURB_2
    {
      printf ("Perturbing ...\n");
      // calculate the normal for each vertex
      for(i = 0; i < NumVertices_i; i++) {
	for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] = 0.0f;
      }
      for(i = 0; i < NumVertices_i; i++) {

	if (FixedVertices_guc[i]==1) continue;

	for(TriangleIndexVec::const_iterator j = SharedTriangleIndex_gi[i].begin();
	    j != SharedTriangleIndex_gi[i].end();
	    j++)
	  {
	    itri = *j;
	    v0 = FaceIndex_gi[itri*3 + 0];	
	    v1 = FaceIndex_gi[itri*3 + 1];
	    v2 = FaceIndex_gi[itri*3 + 2];
	    for(k = 0; k < 3; k++)
	      {
		Pt0[k] = Vertices_gf[v0*3 + k];	
		Pt1[k] = Vertices_gf[v1*3 + k];
		Pt2[k] = Vertices_gf[v2*3 + k];	
	      }
	    crossproduct(Pt0, Pt1, Pt2, Normal_f);
	    for(k = 0; k < 3; k++) VertexNormal_gf[i*3 + k] += Normal_f[k];
	  }

	Length_f = Normalize(&VertexNormal_gf[i*3]);
	if (fabs(Length_f)<1e-4) {
	  printf ("Error! Length is equal to zero\n"); fflush (stdout);
	}
      }

      float	EdgeLength_f[3], MaxDist_f, Perturb_f;
      for(i = 0; i < NumVertices_i; i++) {

	for(k=0; k<3; k++) {
	  Pt0[k] = Vertices_gf[v0*3 + k];	
	  Pt1[k] = Vertices_gf[v1*3 + k];
	  Pt2[k] = Vertices_gf[v2*3 + k];	
	}
	EdgeLength_f[0] = Distance(Pt0, Pt1);
	EdgeLength_f[1] = Distance(Pt0, Pt2);
	EdgeLength_f[2] = Distance(Pt1, Pt2);
	MaxDist_f = -1.0;
	for(k=0; k<3; k++) {
	  if (MaxDist_f < EdgeLength_f[k]) MaxDist_f = EdgeLength_f[k];
	}

	Perturb_f = ranf(-MaxDist_f/5, MaxDist_f/5);
	for(k = 0; k < 3; k++) {
	  Vertices_gf[i*3 + k] += Perturb_f*VertexNormal_gf[i*3 + k];
	}
      }
    }
#endif


    for(i = 0; i < NumVertices_i; i++)
      {
	//use old vertex if smoothing produced garbage
	if(isnan(Vertices_gf[i*3]) || isnan(Vertices_gf[i*3+1]) || isnan(Vertices_gf[i*3+2]))
	  {
	    geo->m_TriVerts[i*3+0] = VerticesBackup_gf[i*3+0];
	    geo->m_TriVerts[i*3+1] = VerticesBackup_gf[i*3+1];
	    geo->m_TriVerts[i*3+2] = VerticesBackup_gf[i*3+2];
	  }
	else
	  {
	    geo->m_TriVerts[i*3+0] = Vertices_gf[i*3+0];
	    geo->m_TriVerts[i*3+1] = Vertices_gf[i*3+1];
	    geo->m_TriVerts[i*3+2] = Vertices_gf[i*3+2];
	  }

	geo->m_TriVertNormals[i*3+0] = VertexNormal_gf[i*3+0];
	geo->m_TriVertNormals[i*3+1] = VertexNormal_gf[i*3+1];
	geo->m_TriVertNormals[i*3+2] = VertexNormal_gf[i*3+2];
      }

    freeMem();
  }

  bool initMem(int NumVertices_i, int NumTriangles_i)
  {
    int i, j;


    Vertices_gf = new float [NumVertices_i*3];
    VerticesBackup_gf = new float [NumVertices_i*3];
    if(!Vertices_gf) return false;
    for (i = 0; i < NumVertices_i; i++) {
      for (j = 0; j < 3; j++) Vertices_gf[i*3 + j] = VerticesBackup_gf[i*3 + j] = 0.0;
    }

    VertexNormal_gf = new float [NumVertices_i*3];
    if(!VertexNormal_gf) return false;
    for (i = 0; i < NumVertices_i; i++) {
      for (j = 0; j < 3; j++) 
	VertexNormal_gf[i*3 + j] = 0.0;
    }

    FixedVertices_guc = new unsigned char [NumVertices_i];
    if(!FixedVertices_guc) return false;
    for (i=0; i<NumVertices_i; i++) FixedVertices_guc[i] = 0;

    FaceIndex_gi = new int [NumTriangles_i*3];
    if(!FaceIndex_gi) return false;
    for (i = 0; i < NumTriangles_i; i++) {
      for (j = 0; j < 3; j++) 
	FaceIndex_gi[i*3 + j] = -1;
    }

    /*
    SharedTriangleIndex_gi = new int [NumVertices_i*SHAREDT];
    if(!SharedTriangleIndex_gi) return false;
    for (i = 0; i < NumVertices_i; i++) {
      for (j = 0; j < SHAREDT; j++) 
	SharedTriangleIndex_gi[i*SHAREDT + j] = -1;
	}*/

    SharedTriangleIndex_gi.resize(NumVertices_i);
    
    return true;
  }

  void freeMem()
  {
    delete [] Vertices_gf; Vertices_gf = NULL;
    delete [] VerticesBackup_gf; VerticesBackup_gf = NULL;
    delete [] VertexNormal_gf; VertexNormal_gf = NULL;
    delete [] FixedVertices_guc; FixedVertices_guc = NULL;
    delete [] FaceIndex_gi; FaceIndex_gi = NULL;
    //delete [] SharedTriangleIndex_gi; SharedTriangleIndex_gi = NULL;
  }

};
