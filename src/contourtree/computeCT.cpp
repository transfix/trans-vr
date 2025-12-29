/*
  Copyright 2003 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of contourtree.

  contourtree is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  contourtree is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// john_contree.cpp : Defines the entry point for the console application.
//

// #include "stdafx.h"
#include <contourtree/HeightField.h>
#include <contourtree/computeCT.h>

// void computeCT(unsigned char* uchar_data, int* dim, int& no_vtx, int&
// no_edge, CTVTX** vtx_list, CTEDGE** edge_list);

/*
int main(int argc, char* argv[])
{
        unsigned char udata[9]={0,1,2,5,3,1,5,6};
        int dim[3]={2,2,2};
        int no_vtx,no_edge;
        CTVTX* vtx_list;
        CTEDGE* edge_list;

        computeCT(udata, dim, no_vtx,no_edge, &vtx_list, &edge_list);

        return 0;
}
*/

void computeCT(unsigned char *uchar_data, int *dim, int &no_vtx, int &no_edge,
               CTVTX **vtx_list, CTEDGE **edge_list) {
  int i;
  CTVTX *temp_vtx_list;
  CTEDGE *temp_edge_list;

  HeightField *hf = new HeightField(uchar_data, dim);
  hf->CreateJoinTree();
  hf->CreateSplitTree();
  hf->CombineTrees();
  hf->ReduceCT();
  hf->TreeDrawing();

  no_vtx = hf->getCTVtxNum();
  no_edge = hf->getCTEdgeNum();

  temp_vtx_list = (CTVTX *)malloc(sizeof(CTVTX) * no_vtx);
  temp_edge_list = (CTEDGE *)malloc(sizeof(CTEDGE) * no_edge);

  for (i = 0; i < no_vtx; i++) {
    temp_vtx_list[i].norm_x = hf->CT_array[i].x;
    temp_vtx_list[i].func_val = hf->CT_array[i].y;
  }

  for (i = 0; i < no_edge; i++) {
    temp_edge_list[i].v1 = hf->CTEdge_array[i].CT_vid1;
    temp_edge_list[i].v2 = hf->CTEdge_array[i].CT_vid2;
  }

  *vtx_list = temp_vtx_list;
  *edge_list = temp_edge_list;

  // hf1->writeConTree(fp_out,area_flag);
  delete hf;
}
