#include <math.h>
#include <iostream>
#include "Mesh.h"

using namespace std;

void cross(float *dest, const float *v1, const float *v2)
{
  dest[0] = v1[1]*v2[2] - v1[2]*v2[1];
  dest[1] = v1[2]*v2[0] - v1[0]*v2[2];
  dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

void calculateTriangleNormal(float *norm, vector<Point> &verts, vector<vector<unsigned int> > &triangles, unsigned int c)
{
    
  float v1[3], v2[3];
  int vert;
  vert = triangles[c][0];
  v1[0] = v2[0] = -verts[vert].x;
  v1[1] = v2[1] = -verts[vert].y;
  v1[2] = v2[2] = -verts[vert].z;

  vert = triangles[c][1];
  v1[0] += verts[vert].x;
  v1[1] += verts[vert].y;
  v1[2] += verts[vert].z;

  vert = triangles[c][2];
  v2[0] += verts[vert].x;
  v2[1] += verts[vert].y;
  v2[2] += verts[vert].z;

  cross(norm, v1, v2);
  
}

void calculateNormals(Mesh &mesh,vector<vector<float> > &normals)
{
  vector<Point> &verts=mesh.vertices;
  vector<vector<unsigned int> > &triangles=mesh.triangles;
  
  vector<int> counts;

  unsigned int numverts=verts.size();
  unsigned int numtris=triangles.size();
  
  int c, vert;
  float normal[3];
  float len;

  for(int i=0;i<numverts;i++) { vector<float> n; n.push_back(0.0); n.push_back(0.0); n.push_back(0.0); normals.push_back(n); counts.push_back(0); }
    
    // for each triangle
  for (c=0; c<numtris; c++) {
    calculateTriangleNormal(normal,verts,triangles,c);

    for(int j=0;j<3;j++)
    {
        normals[triangles[c][j]][0] +=normal[0];
        normals[triangles[c][j]][1] +=normal[1];
        normals[triangles[c][j]][2] +=normal[2];
        counts[triangles[c][j]]++;
    }
  }

  for(c=0;c<numverts;c++)
    for(int j=0;j<3;j++)
      if(counts[c]>0)
        normals[c][j]/=counts[c];
   
    // normalize the vectors
  for (vert=0; vert<numverts; vert++) {
    len = (float) sqrt(
           normals[vert][0] * normals[vert][0] +
        normals[vert][1] * normals[vert][1] +
        normals[vert][2] * normals[vert][2]);
    if(len == 0.0)
    {
      normals[vert][0] = 1.0;
      normals[vert][1] = 0.0;
      normals[vert][2] = 0.0;
      printf("error\n"); continue;
    }
    normals[vert][0]/=len;
    normals[vert][1]/=len;
    normals[vert][2]/=len;
  }
    
}
// calculate the area for a triangle
float area_tri(float v0[3], float v1[3], float v2[3]) {

  float a, b, c, p, area;
  int i;

  a = 0.0;        b = 0.0;        c = 0.0;
  for(i = 0; i < 3; i++) {
    a += (v1[i] - v0[i])*(v1[i] - v0[i]);
    b += (v2[i] - v1[i])*(v2[i] - v1[i]);
    c += (v0[i] - v2[i])*(v0[i] - v2[i]);
  }
  a = (float)sqrt(a);     b = (float)sqrt(b);     c = (float)sqrt(c);
  p = (a + b + c) / 2.0f;
  area = (float)sqrt(p * (p - a) * (p - b) * (p - c));

  return area;

}

// quality improvement with geometric flow -- tri mesh
void geometricFlow(Mesh &mesh) {

  vector<Point> &verts=mesh.vertices;
  vector<vector<unsigned int> > &triangles=mesh.triangles;

  unsigned int numverts=verts.size();
  unsigned int numtris=triangles.size();

  vector<vector<float> > normals;

  calculateNormals(mesh,normals);

  int i, j, k, v0, v1, v2, itri, index, maxIndex;
  float mv0[3], mv1[3], mv2[3], p[3], sum_0, sum_1[3], delta_t, area, t;
  int **neighbor;

  delta_t = 0.01f;
  maxIndex = 100;

  neighbor = 0;
  neighbor = new int*[numverts];

  for(i = 0; i <  numverts; i++)    neighbor[i] = 0;
  for(i = 0; i <  numverts; i++)    neighbor[i] = new int[10];
  for(i = 0; i <  numverts; i++) {
    for (j = 0; j < 10; j++)    neighbor[i][j] = -1;
  }

  for (i = 0; i <  numtris; i++) {
    v0 =  triangles[i][0];
    v1 =  triangles[i][1];
    v2 =  triangles[i][2];

    for(j = 0; j < 10; j++) {
      if(neighbor[v0][j] == -1) {neighbor[v0][j] = i; break;}
    }
    for(j = 0; j < 10; j++) {
      if(neighbor[v1][j] == -1) {neighbor[v1][j] = i; break;}
    }
    for(j = 0; j < 10; j++) {
      if(neighbor[v2][j] == -1) {neighbor[v2][j] = i; break;}
    }
  }
  
  for(index = 0; index < maxIndex; index++) {

    for(i = 0; i <  numverts; i++) {

            // calculate the mass center p[3]
      sum_0 = 0.0f;
      for(j = 0; j < 3; j++) {p[j] = 0.0f; sum_1[j] = 0.0f;}
      for(j = 0; j < 10; j++) {
        itri = neighbor[i][j];
        if(itri == -1) break;
        v0 =  triangles[itri][0]; v1 =  triangles[itri][1];
        v2 =  triangles[itri][2];

        mv0[0]=verts[v0].x; mv1[0]=verts[v1].x; mv2[0]=verts[v2].x;
        mv0[1]=verts[v0].y; mv1[1]=verts[v1].y; mv2[1]=verts[v2].y;
        mv0[2]=verts[v0].z; mv1[2]=verts[v1].z; mv2[2]=verts[v2].z;
//         for(k = 0; k < 3; k++) {
//           mv0[k] =  verts[v0][k];   mv1[k] =  verts[v1][k];
//           mv2[k] =  verts[v2][k];
//         }

        area = fabs(area_tri(mv0, mv1, mv2));
        sum_0 += area;
        for(k = 0; k < 3; k++) {
          p[k] += (mv0[k] + mv1[k] + mv2[k])*area/3.0f;
        }

      }
      for(j = 0; j < 3; j++) p[j] /= sum_0;

            // calculate the new position in tangent direction
            // xi+1 = xi + delta_t*((m-xi) - (n, m-xi)n))
      t = 0.0f;
      p[0]-=verts[i].x;
      p[1]-=verts[i].y;
      p[2]-=verts[i].z;
      for(j = 0; j < 3; j++) {
       // p[j] -=  verts[i][j];
       t += p[j]* normals[i][j];
      }

      verts[i].x += delta_t*(p[0] - t* normals[i][0]);      // tangent movement
      verts[i].y += delta_t*(p[1] - t* normals[i][1]);
      verts[i].z += delta_t*(p[2] - t* normals[i][2]);

//       for(j = 0; j < 3; j++) {
//                 //verts[i][j] += delta_t*p[j];                               // mass center
//          verts[i][j] += delta_t*(p[j] - t* normals[i][j]);      // tangent movement
//       }
    }

  } // end of index loop

  for(i = 0; i <  numverts; i++)    delete [] neighbor[i];
  delete [] neighbor;

}

