#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <vector>

#include <ContourTiler/smfconvert.h>

using namespace std;

SMF_BEGIN_NAMESPACE

class Vertex
{
public:
  Vertex()
  {
  }
  double   Point[3];
  double   Normal[3];
  int	 NumFaces;
  double	 Color[3];
  double   weight;

  //      double   *Hessian;    /* the hansian at the vertices  */
  //      double   *SharpNormal;   /* the normal for sharp vertex */
};

class Face
{
public:
  Face()
  {
  }
  int Index[3];      		/* the indexs of a triangle-increase order*/
  int IndexInRAW[3]; 		/* the indices of a triangle in the origianl raw file */
  std::vector<Vertex *> Node;	/* Gaussian integration nodes             */ 
  int Orien;        		/* 1--orientaioned, othwise no            */
  int AdjTri[3];    		/* neighbor triangles index               */
  double Center[3];		/* the center of the triangle             */
  double Normal[3];		/* face normal                            */
  int whichnml[3];		/* Index[0] normal index                  */
  int Tou;			/* the begin index of the insided points  */
  int Wei;			/* the end index+1 of the insided points  */
  int State;			/* 1--removed; 2-- new, 0--old;           */
  std::vector<Face *> subFacets;
};

FILE *loadFile_read(const char* fileName)
{
  FILE* fp;
  fp=fopen(fileName, "r");
  if(fp == NULL)
  {
    printf("could not open file %s for read\n", fileName);
    exit(0);
  }
  return fp;
}

FILE *loadFile_write(const char* fileName)
{
  FILE* fp;
  fp=fopen(fileName, "w");
  if(fp == NULL)
  {
    printf("could not open file %s for read\n", fileName);
    exit(0);
  }
  return fp;
}

void smf2raw(string infile, string outfile)
// void smf2raw(int argc, char* argv[])
{
  FILE *fp1, *fp2;
  char line[256];
  int numbpts, numbtris;
  double x, y, z;
  // double type, x, y, z;
  char type;
  int ii, jj, kk;
  int min_index;
  int i;

  vector<Vertex*> m_Vertices;
  vector<Face*> m_Faces;

  fp1 = loadFile_read(infile.c_str());
  fp2 = loadFile_write(outfile.c_str());
  // fp1 = loadFile_read(argv[2]);
  // fp2 = loadFile_write(argv[3]);

  while( fgets( line, 256, fp1 ) != 0 )
  {
    if (line[0] == 'v')
    {
      sscanf(line, "%c %lf %lf %lf \n", &type, &x, &y, &z);
      Vertex *v = new Vertex();
      v->Point[0] = x;        v->Point[1] = y;        v->Point[2] = z;
      m_Vertices.push_back(v);
    }
    if (line[0] == 'f' || line[0] == 't')
    {
      sscanf(line, "%c %d %d %d \n", &type, &ii, &jj, &kk);
      Face *f = new Face();
      f->Index[0] = ii;       f->Index[1] = jj;       f->Index[2] = kk;
      if (ii < min_index)	min_index = ii;
      if (jj < min_index)     min_index = jj;
      if (kk < min_index)     min_index = kk;
      m_Faces.push_back(f);
    }
  }

  numbpts = m_Vertices.size();
  numbtris = m_Faces.size();

  fprintf(fp2, "%d %d\n", numbpts, numbtris);
  for (i = 0 ; i < numbpts; i++)
  {
    fprintf(fp2,"%f %f %f\n", m_Vertices[i]->Point[0], m_Vertices[i]->Point[1], m_Vertices[i]->Point[2]);
  }
  for (i = 0 ; i < numbtris; i++)
  {
    fprintf(fp2,"%d %d %d\n", m_Faces[i]->Index[0]-min_index, m_Faces[i]->Index[1]-min_index, m_Faces[i]->Index[2]-min_index);
  }
  fclose(fp1);
  fclose(fp2);
}

// void raw2smf(int argc, char* argv[])
void raw2smf(string infile, string outfile)
{
  FILE *fp1, *fp2;
  char line[256];
  int numbpts, numbtris;
  double x, y, z;
  // double type, x, y, z;
  char type;
  int ii, jj, kk;
  int i;

  vector<Vertex*> m_Vertices;
  vector<Face*> m_Faces;

  fp1 = loadFile_read(infile.c_str());
  fp2 = loadFile_write(outfile.c_str());
  // fp1 = loadFile_read(argv[2]);
  // fp2 = loadFile_write(argv[3]);

  int ret = fscanf(fp1, "%d %d\n", &numbpts, &numbtris);
  for (i = 0; i < numbpts; i++)
  {
    ret = fscanf(fp1, "%lf %lf %lf \n", &x, &y, &z);
    fprintf(fp2, "v %f %f %f\n", x, y, z);
  }
  for (i = 0; i < numbtris; i++)
  {
    ret = fscanf(fp1, "%d %d %d \n", &ii, &jj, &kk);
    fprintf(fp2, "f %d %d %d\n", ii+1, jj+1, kk+1);
  }
  fclose(fp1);
  fclose(fp2);

}

SMF_END_NAMESPACE

