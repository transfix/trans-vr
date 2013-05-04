#include "stdio.h"
#include "math.h"
#include "malloc.h"
#include "time.h"
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "Mesh.h"

using namespace std;

//change edgethresh to add more points

void subdionetriangleto4(vector<float> &vertArray, int &numbpts, float pt1[3], float pt2[3], float pt3[3], float edgethresh);
void generatePointCloud(MMHLS::Mesh &mesh, float edgethresh, MMHLS::PointCloud &pointCloud);

// void generatePointClouds(vector<MMHLS::Mesh> meshes, vector<MMHLS::PointCloud> &pointClouds)
// {
//   int sz=meshes.size();
//   for(int i=0;i<sz;i++)
//   {
//     MMHLS::PointCloud p;
//     generatePointCloud(meshes[i],p);
//     pointClouds.push_back(p);
//   }
// }

void generatePointCloud(MMHLS::Mesh &mesh, float edgethresh, MMHLS::PointCloud &pointCloud)
{

  float  pt1[3],pt2[3],pt3[3],edge1,edge2,edge3;
  int i,j,k,l,ii,jj,kk;
  int numbpts,numbtris;

  
  
 

  numbpts=mesh.vertexList.size();
  numbtris=mesh.faceList.size();

  vector<float> vertArray(numbpts*3);
  vector<int> faceArray(numbtris*3);
  
  for(i=0;i<numbpts;i++)
    for(j=0;j<3;j++)
      vertArray[3*i+j]=mesh.vertexList[i][j];

  for(i=0;i<numbtris;i++)
    for(j=0;j<3;j++)
      faceArray[3*i+j]=mesh.faceList[i][j];
    
  for(i = 0; i < numbtris; i++) {

    ii = faceArray[3*i+0];
    jj = faceArray[3*i+1];
    kk = faceArray[3*i+2];

    pt1[0]=vertArray[3*ii+0];
    pt1[1]=vertArray[3*ii+1];
    pt1[2]=vertArray[3*ii+2];

    pt2[0]=vertArray[3*jj+0];
    pt2[1]=vertArray[3*jj+1];
    pt2[2]=vertArray[3*jj+2];

    pt3[0]=vertArray[3*kk+0];
    pt3[1]=vertArray[3*kk+1];
    pt3[2]=vertArray[3*kk+2];


    edge1=sqrt((pt1[0]-pt2[0])*(pt1[0]-pt2[0])+(pt1[1]-pt2[1])*(pt1[1]-pt2[1])+(pt1[2]-pt2[2])*(pt1[2]-pt2[2]));
    edge2=sqrt((pt3[0]-pt2[0])*(pt3[0]-pt2[0])+(pt3[1]-pt2[1])*(pt3[1]-pt2[1])+(pt3[2]-pt2[2])*(pt3[2]-pt2[2]));
    edge3=sqrt((pt1[0]-pt3[0])*(pt1[0]-pt3[0])+(pt1[1]-pt3[1])*(pt1[1]-pt3[1])+(pt1[2]-pt3[2])*(pt1[2]-pt3[2]));
    if(edge1 > edgethresh || edge2 > edgethresh || edge3 > edgethresh)
      subdionetriangleto4(vertArray,numbpts, pt1, pt2, pt3, edgethresh);
  }

  for(i=0;i<numbpts;i++)
  {
    vector<float> pt;
    for(j=0;j<3;j++)
      pt.push_back(vertArray[3*i+j]);
    pointCloud.vertexList.push_back(pt);
  }
 

}

void subdionetriangleto4(vector<float> &vertArray, int &numbpts, float pt1[3], float pt2[3], float pt3[3], float edgethresh)
{
  float edg1,edg2,edg3;
  float pt4[3],pt5[3],pt6[3];

  pt4[0]=1/2.0*pt1[0]+1/2.0*pt2[0];
  pt4[1]=1/2.0*pt1[1]+1/2.0*pt2[1];
  pt4[2]=1/2.0*pt1[2]+1/2.0*pt2[2];

  pt5[0]=1/2.0*pt3[0]+1/2.0*pt2[0];
  pt5[1]=1/2.0*pt3[1]+1/2.0*pt2[1];
  pt5[2]=1/2.0*pt3[2]+1/2.0*pt2[2];

  pt6[0]=1/2.0*pt1[0]+1/2.0*pt3[0];
  pt6[1]=1/2.0*pt1[1]+1/2.0*pt3[1];
  pt6[2]=1/2.0*pt1[2]+1/2.0*pt3[2];
  
  vertArray.push_back(pt4[0]);
  vertArray.push_back(pt4[1]);
  vertArray.push_back(pt4[2]);
  
  vertArray.push_back(pt5[0]);
  vertArray.push_back(pt5[1]);
  vertArray.push_back(pt5[2]);

  vertArray.push_back(pt6[0]);
  vertArray.push_back(pt6[1]);
  vertArray.push_back(pt6[2]);

  numbpts=numbpts+3;


  edg1=sqrt((pt1[0]-pt4[0])*(pt1[0]-pt4[0])+(pt1[1]-pt4[1])*(pt1[1]-pt4[1])+(pt1[2]-pt4[2])*(pt1[2]-pt4[2]));
  edg2=sqrt((pt6[0]-pt4[0])*(pt6[0]-pt4[0])+(pt6[1]-pt4[1])*(pt6[1]-pt4[1])+(pt6[2]-pt4[2])*(pt6[2]-pt4[2]));
  edg3=sqrt((pt1[0]-pt6[0])*(pt1[0]-pt6[0])+(pt1[1]-pt6[1])*(pt1[1]-pt6[1])+(pt1[2]-pt6[2])*(pt1[2]-pt6[2]));


  if(edg1 > edgethresh || edg2 > edgethresh || edg3 > edgethresh) subdionetriangleto4(vertArray,numbpts,pt1,pt4,pt6,edgethresh);

  edg1=sqrt((pt2[0]-pt4[0])*(pt2[0]-pt4[0])+(pt2[1]-pt4[1])*(pt2[1]-pt4[1])+(pt2[2]-pt4[2])*(pt2[2]-pt4[2]));
  edg2=sqrt((pt5[0]-pt4[0])*(pt5[0]-pt4[0])+(pt5[1]-pt4[1])*(pt5[1]-pt4[1])+(pt5[2]-pt4[2])*(pt5[2]-pt4[2]));
  edg3=sqrt((pt2[0]-pt5[0])*(pt2[0]-pt5[0])+(pt2[1]-pt5[1])*(pt2[1]-pt5[1])+(pt2[2]-pt5[2])*(pt2[2]-pt5[2]));


  if(edg1 > edgethresh || edg2 > edgethresh || edg3 > edgethresh) subdionetriangleto4(vertArray,numbpts,pt2,pt4,pt5,edgethresh);

  edg1=sqrt((pt5[0]-pt4[0])*(pt5[0]-pt4[0])+(pt5[1]-pt4[1])*(pt5[1]-pt4[1])+(pt5[2]-pt4[2])*(pt5[2]-pt4[2]));
  edg2=sqrt((pt6[0]-pt4[0])*(pt6[0]-pt4[0])+(pt6[1]-pt4[1])*(pt6[1]-pt4[1])+(pt6[2]-pt4[2])*(pt6[2]-pt4[2]));
  edg3=sqrt((pt5[0]-pt6[0])*(pt5[0]-pt6[0])+(pt5[1]-pt6[1])*(pt5[1]-pt6[1])+(pt5[2]-pt6[2])*(pt5[2]-pt6[2]));


  if(edg1 > edgethresh || edg2 > edgethresh || edg3 > edgethresh) subdionetriangleto4(vertArray,numbpts,pt5,pt4,pt6,edgethresh);

  edg1=sqrt((pt5[0]-pt3[0])*(pt5[0]-pt3[0])+(pt5[1]-pt3[1])*(pt5[1]-pt3[1])+(pt5[2]-pt3[2])*(pt5[2]-pt3[2]));
  edg2=sqrt((pt6[0]-pt3[0])*(pt6[0]-pt3[0])+(pt6[1]-pt3[1])*(pt6[1]-pt3[1])+(pt6[2]-pt3[2])*(pt6[2]-pt3[2]));
  edg3=sqrt((pt5[0]-pt6[0])*(pt5[0]-pt6[0])+(pt5[1]-pt6[1])*(pt5[1]-pt6[1])+(pt5[2]-pt6[2])*(pt5[2]-pt6[2]));


  if(edg1 > edgethresh || edg2 > edgethresh || edg3 > edgethresh) subdionetriangleto4(vertArray,numbpts,pt3,pt5,pt6,edgethresh);

  return;


}