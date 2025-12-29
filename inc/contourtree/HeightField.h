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

#ifndef HEIGHTFIELD_H
#define HEIGHTFIELD_H 1

#include <assert.h>
#include <contourtree/cellQueue.h>
#include <contourtree/unionfind.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#define MAXVAL 100000000
#define MAX_NEIGHBOR 24
#define MAX_CT_EDGE 10000
#define M_PI 3.14159265358979323846

#define MIN_VAL -100000

#define MAX_EDGE_SET 10000
#define MAX_SIZE_EIDSET 2000000
#define MAX_CNODE 10000
#define MAX_CEDGE 10000
#define MAX_STACK_SIZE 100000
#define TRUE 1
#define FALSE 0
#define CAJT 1
#define CAST 2
#define JT_ID 1
#define ST_ID 2

using namespace std;

typedef struct _Vtx {
  int rank;
  float fval;

  int no_JTup;
  int *JTup;
  int JTdown;

  int no_STdown;
  int *STdown;
  int STup;

  int no_ACTup;
  int *ACTup;
  int no_ACTdown;
  int *ACTdown;
} Vtx;

typedef struct _order_vtx {
  int vid;
  float fval;
} order_vtx;

typedef struct _Edge {
  int CT_vid1;
  int CT_vid2;
} CTEdge;

typedef struct _CTVtx {
  int vid;
  float x;
  float y;
} CTVtx;
/*

typedef struct _CNode {
        float cval;
        int vtx;
        bool visit_flag;

        int no_up;
        int no_down;
        int* up;
        int* down;
} CNode;
*/

typedef struct _EdgeSeed {
  int no;
  int *array;
  int upper_vid;
  int next_upper_vid;
} EdgeSeed;

typedef struct _EdgeCellList {
  int *array;
  int len;
  int no;
  float *val;
  float *fx;
} EdgeCellList;

/*
typedef struct _CTEdge_CList {
        int up_vtx;
        int down_vtx;
        int no_cnode_list;
        int* cnode_list;
} CTEdge_CList;
*/

int compareHeightVoid(const void *e1, const void *e2);

const long maxNeighbours = 24;

class Stack {
  int top;
  int size;
  int *stack_array;

public:
  Stack(int Size) {
    stack_array = new int[Size];
    top = 0;
    size = Size;
  }

  ~Stack() { delete stack_array; }

  void push2(int n, int down_idx) {
    push(n);
    push(down_idx);
  }

  int pop(int &n, int &down_idx) {
    if (top == 0)
      return -1;
    assert(top >= 2);
    down_idx = pop();
    n = pop();
    return 1;
  }

  void push(int n) {
    int *temp_stack;
    int temp_size;

    if (top >= size) {
      temp_size = size;
      size *= 2;
      temp_stack = new int[size];
      memcpy(temp_stack, stack_array, sizeof(int) * temp_size);
      delete stack_array;
      stack_array = temp_stack;
    }

    stack_array[top++] = n;
  }

  int pop() {
    // assert (top>0);
    if (top == 0) {

      return -1;
    }
    return stack_array[--top];
  }
};

/*
class CNode_Stack
{
        int top;
        int size;
        int* ct_cnode;
        int* next_vtx;
        int* cajst_cnode;

public :
        CNode_Stack(int Size)
        {
                ct_cnode = new int[Size];
                next_vtx = new int[Size];
                cajst_cnode = new int[Size];

                top = 0;
                size = Size;
        }

        ~CNode_Stack() { delete ct_cnode; delete next_vtx; delete cajst_cnode;
}

        void push(int CT_CNode , int NextVtx , int CAJST_CNode)
        {
                int* temp_ct_cnode;
                int* temp_next_vtx;
                int* temp_cajst_cnode;
                int temp_size;

                if (top >= size) {
                        temp_size = size;
                        size *= 2;
                        temp_ct_cnode = new int[size];
                        temp_next_vtx = new int[size];
                        temp_cajst_cnode = new int[size];

                        memcpy(temp_ct_cnode , ct_cnode ,
sizeof(int)*temp_size); memcpy(temp_next_vtx , next_vtx ,
sizeof(int)*temp_size); memcpy(temp_cajst_cnode , cajst_cnode ,
sizeof(int)*temp_size); delete ct_cnode; delete next_vtx; delete cajst_cnode;

                        ct_cnode = temp_ct_cnode;
                        next_vtx = temp_next_vtx;
                        cajst_cnode = temp_cajst_cnode;
                }

                ct_cnode[top] = CT_CNode;
                next_vtx[top] = NextVtx;
                cajst_cnode[top] = CAJST_CNode;
                top++;
        }

        int pop(int& CT_CNode , int& NextVtx , int& CAJST_CNode)
        {
                //assert (top>0);
                if (top==0) {

                        return 0;
                }
                top--;
                CT_CNode = ct_cnode[top];
                NextVtx = next_vtx[top];
                CAJST_CNode = cajst_cnode[top];
                return 1;
        }
};
*/

class TreeNodeStack {
  int top;
  int size;
  int *vid_array;
  int *CT_vid_array;
  float *src_x;
  float *src_y;

public:
  TreeNodeStack(int Size) {
    vid_array = new int[Size];
    CT_vid_array = new int[Size];
    src_x = new float[Size];
    src_y = new float[Size];
    top = 0;
    size = Size;
  }

  ~TreeNodeStack() {
    delete vid_array;
    delete CT_vid_array;
    delete src_x;
    delete src_y;
  }

  void push(int n, int CT_vid, float s_x, float s_y) {
    assert(top < size);
    vid_array[top] = n;
    CT_vid_array[top] = CT_vid;
    src_x[top] = s_x;
    src_y[top] = s_y;
    top++;
  }

  int pop(int &vid, int &CT_vid, float &s_x, float &s_y) {
    // assert (top>0);
    if (top == 0) {

      return -1;
    }
    vid = vid_array[--top];
    CT_vid = CT_vid_array[top];
    s_x = src_x[top];
    s_y = src_y[top];
    return 1;
  }
};

class HeightField {
public:
  UnionFind *UF;
  order_vtx *vtxrank_array;
  Vtx *vtx_array;

  // HeightField(const char* fname);
  HeightField(unsigned char *uchar_data, int *dim);
  ~HeightField() {
    int i;
    for (i = 0; i < nverts; i++) {
      if (vtx_array[i].no_JTup > 0)
        free(vtx_array[i].JTup);
      if (vtx_array[i].no_STdown > 0)
        free(vtx_array[i].STdown);
      if (vtx_array[i].no_ACTup > 0)
        free(vtx_array[i].ACTup);
      if (vtx_array[i].no_ACTdown > 0)
        free(vtx_array[i].ACTdown);
    }
    free(vtxrank_array);
    free(vtx_array);
    free(CT_array);
    free(CTEdge_array);
    delete UF;
  }

  CTVtx *CT_array;
  CTEdge *CTEdge_array;

  int NoCTVtx;
  int NoCTEdge;

  float window_size[4];
  int CT_vsize, CT_esize;
  EdgeSeed *edge_seedset;
  EdgeCellList *edge_celllist;
  // EdgeSeed edge_seedset[MAX_EDGE_SET];
  int num_edge_seeds;

  bool CreateJoinTree();
  bool CreateSplitTree();
  bool CombineTrees();
  void genSeedSet();
  bool ReduceCT();
  void writeConTree(FILE *fp_out, int area_flag);

  int create_node(int flag, float cval, int v, int setId);
  void nodeConnect(int flag, int cnode_id1, int cnode_id2);
  void Visit(int vid) { visit_array[vid] = 1; }
  void clearVisit() { memset(visit_array, 0, nverts); }
  bool isVisit(int vid) {
    if (visit_array[vid])
      return true;
    else
      return false;
  }
  int GetNeighbor(int, int *);
  int getVidFromRank(int rank_idx);
  int getRankFromVid(int vid);

  bool isUpLeaf(int v) {
    if (vtx_array[v].no_ACTup == 0) {
      return true;
    } else
      return false;
  }
  bool isDownLeaf(int v) {
    if (vtx_array[v].no_ACTdown == 0) {
      return true;
    } else
      return false;
  }
  bool isRegular(int v) {
    if (vtx_array[v].no_ACTup == 1 && vtx_array[v].no_ACTdown == 1)
      return true;
    else
      return false;
  }
  bool isJoin(int v) {
    if (vtx_array[v].no_ACTup > 1)
      return true;
    else
      return false;
  }
  bool isSplit(int v) {
    if (vtx_array[v].no_ACTdown > 1)
      return true;
    else
      return false;
  }

  void TreeDrawing();
  void CleanUpConstruction();
  int getCTEdgeNum() { return CT_eidx; }
  int getCTVtxNum() { return CT_vidx; }
  int getNumSeedEdge() { return num_edge_seeds; }
  int getNVerts() { return nverts; }

  int getCell(int edge_v1, int edge_v2);
  float getValue(int vid) { return vtx_array[vid].fval; }
  float getValue(int i, int j, int k) {
    return vtx_array[idx2vtx(i, j, k)].fval;
  }
  int compareHeight(int vtx1, int vtx2);
  int GetSmallestNeighbor(int vtx);
  float lowval() { return lowest_val; }
  float highval() { return highest_val; }
  int getCTEdgeNo() { return CT_edge_no; }
  int hfCompareVtx(int vtx1, int vtx2, float f1, float f2);

  int nverts, ncells;
  void ClearTouchCell();
  inline void TouchCell(unsigned int id) {
    touched[id / 8] |= (1 << (id % 8));
  }
  inline int CellTouched(unsigned int id) {
    return (touched[id / 8] & (1 << (id % 8)));
  }
  inline void ClearTouched() {
    memset(touched, 0, sizeof(char) * (5 * ncells));
  }

  int getCellVtx(int tet_cell_id, int idx);
  int getNeighborCellfromVtx(int *nbr_array, int vtx);
  int getCellAdj(int cell, int neighbor_id);
  int idx2TetCell(int x, int y, int z, int case_id);
  int getCellCase(int tet_cell_id);
  void computeArea();
  int getRegionCellList(int *temp_list, int start_vtx2, int up_vtx,
                        int down_vtx);
  void computeRegionCellList();

  void ACTverify() {
    int i;
    for (i = 0; i < nverts; i++) {

      assert(vtx_array[i].no_JTup == vtx_array[i].no_ACTup);
      assert(vtx_array[i].no_STdown == vtx_array[i].no_ACTdown);

      assert(vtx_array[i].no_ACTup > 0 || vtx_array[i].no_ACTdown > 0);
      /*

      */
      if (!(vtx_array[i].no_JTup == vtx_array[i].no_ACTup)) {
        //				exit(0);
      }
      if (!(vtx_array[i].no_STdown == vtx_array[i].no_ACTdown)) {
        //				i++; i--;
        //				exit(0);
      }
    }
  }

  int mapEid2Seed(int up_vtx, int down_vtx);

private:
  long xDim, yDim, zDim;
  float minext[3], maxext[3];

  int dim[3];
  float orig[3], span[3];
  float highest_val, lowest_val;

  // long nVertices;

  char *visit_array;
  char *touched;

  int CT_edge_no;

  Stack *CAJT_Stack;
  Stack *CAST_Stack;

  void readHeader(FILE *vol_fp);

  int CT_vidx, CT_eidx;

  int cellContain(int cid, int vid);
  int getNext(int n, int flag);
  int getHighestNode(int flag);
  void getVtxEdgeNo();

  bool isNeighbor(int v1, int v2);
  int getEdgeSeedset(int upper_vid, int lower_idx, int *seeds_array,
                     int &next_upper_vid);
  void addEdgeSeedset(int upper_vid, int next_upper_vid, int *seeds_array,
                      int num_seeds);

  void addACTeid(int v_src);
  void addCTeid(int v_src, int v_des, int eid);
  int getCTEid(int v_src, int v_des);
  int compareLength(int n, int n1_next, int n2_next);

  void AddEdgeJT(int vtx1, int vtx2, int nbr_idx);
  void AddEdgeST(int vtx1, int vtx2, int nbr_idx);

  void Add_JT_upedge(int vtx1, int vtx2);
  void Add_JT_downedge(int vtx1, int vtx2, int nbr);
  void Add_ST_downedge(int vtx1, int vtx2, int nbr);
  void Add_ST_upedge(int vtx1, int vtx2);

  void ACT_link(int high_vtx, int low_vtx);
  bool is_updgr1_downdgr1_JT(int vid);
  bool is_updgr1_downdgr1_ST(int vid);
  void replace_edge_JT(int up_vid, int down_vid, int vid);
  void replace_edge_ST(int up_vid, int down_vid, int vid);

  void add_edge_JT(int vtx1, int vtx2);
  void add_edge_ST(int vtx1, int vtx2);

  bool is_leaf_JST(int v);
  bool upperleaf_JT(int v);
  bool lowerleaf_ST(int v);

  int find_nbr_JT(int v);
  int find_nbr_ST(int v);
  void add_arc(int v1, int v2);
  void del_JT(int v);
  void del_ST(int v);
  void del_ACT(int v);

  void vtx2idx(int v, int &i, int &j, int &k);
  void vtx2idx(int v, int &i, int &j);
  int idx2vtx(int i, int j, int k);
  int idx2vtx(int i, int j);
  void cell2index(int cid, int &i, int &j, int &k);
  int index2cell(int i, int j, int k);

  int add2DEdge(int CT_vid1, int CT_vid2);
  int add2DVtx(int vid, float pos_x, float pos_y);
  void compute_pos(float &tx, float &ty, float sx, float sy, float sf,
                   float tf, int n, int i);
  void updateWindow(float target_x, float target_y, float *pos) {
    // 0 : down
    // 1 : up
    // 2 : left
    // 3 : right
    if (pos[0] > target_y)
      pos[0] = target_y;
    if (pos[1] < target_y)
      pos[1] = target_y;
    if (pos[2] > target_x)
      pos[2] = target_x;
    if (pos[3] < target_x)
      pos[3] = target_x;
  }

  void processVtx(int vid, int CT_vid, float src_x, float src_y, float *pos);
  int getLowerVtx(int vid, int *downv_array);
  int getUpperVtx(int vid, int *upv_array);

  long XDim() { return xDim; }
  long YDim() { return yDim; }
  long ZDim() { return zDim; }

}; // end of class HeightField

void CompactArray(int *int_array, int &size) {
  int old_size, i, k;

  k = 0;
  old_size = size;

  for (i = 0; i < old_size; i++) {
    if (int_array[i] >= 0)
      int_array[k++] = int_array[i];
  }
  size = k;
}

int compareHeightVoid(const void *e1, const void *e2) {
  order_vtx *vtx1 = (order_vtx *)e1;
  order_vtx *vtx2 = (order_vtx *)e2;

  if (vtx1->fval == vtx2->fval) {
    if (vtx1->vid > vtx2->vid)
      return 1;
    else
      return -1;
  } else {
    if (vtx1->fval > vtx2->fval)
      return 1;
    else
      return -1;
  }
}

HeightField::HeightField(unsigned char *uchar_data, int *dim) {
  long i, j, k;

  CT_vidx = CT_eidx = 0;

  xDim = dim[0];
  yDim = dim[1];
  zDim = dim[2];

  nverts = dim[0] * dim[1] * dim[2];
  ncells = (dim[0] - 1) * (dim[1] - 1) * (dim[2] - 1);
  //	printf("nverts:%d , dim:(%d,%d,%d)\n",nverts,xDim,yDim,zDim);

  CT_edge_no = 0;
  edge_seedset = NULL;

  UF = new UnionFind(nverts);

  vtx_array = (Vtx *)malloc(sizeof(Vtx) * nverts);
  vtxrank_array = (order_vtx *)malloc(sizeof(order_vtx) * nverts);
  visit_array = (char *)malloc(sizeof(char) * nverts);
  touched = (char *)malloc(sizeof(char) * (ncells) * 5);
  ClearTouched();
  assert(touched);
  assert(vtx_array);
  assert(vtxrank_array);
  assert(visit_array);

  CT_vsize = 100;
  CT_esize = 100;
  CT_array = (CTVtx *)malloc(sizeof(CTVtx) * CT_vsize);
  CTEdge_array = (CTEdge *)malloc(sizeof(CTEdge) * CT_esize);

  num_edge_seeds = 0;

  float temp_fval;
  int vtx_idx;

  for (k = 0; k < zDim; k++) {
    for (j = 0; j < yDim; j++)
      for (i = 0; i < xDim; i++) {
        // getFloat(&temp_fval,1,vol_fp);
        temp_fval = (float)uchar_data[i + j * xDim + k * xDim * yDim];
        // fread(&temp_fval,4,1,vol_fp);
        vtx_idx = idx2vtx(i, j, k);
        vtx_array[vtx_idx].fval = temp_fval;
        vtxrank_array[vtx_idx].vid = vtx_idx;
        vtxrank_array[vtx_idx].fval = temp_fval;

        vtx_array[vtx_idx].no_JTup = 0;
        vtx_array[vtx_idx].JTdown = -1;
        vtx_array[vtx_idx].no_STdown = 0;
        vtx_array[vtx_idx].STup = -1;
        vtx_array[vtx_idx].no_ACTup = 0;
        vtx_array[vtx_idx].no_ACTdown = 0;
      }
  }

  cout << endl << "Sorting data" << endl;
  qsort(vtxrank_array, nverts, sizeof(order_vtx), compareHeightVoid);

  lowest_val = vtxrank_array[0].fval;
  highest_val = vtxrank_array[nverts - 1].fval;

  int rank_idx;
  for (rank_idx = 0; rank_idx < nverts; rank_idx++) {
    vtx_array[vtxrank_array[rank_idx].vid].rank = rank_idx;
  }

  cout << "Sorting finished." << endl;
}

bool HeightField::CreateJoinTree() {
  int i, j, vtx_id, NeighborNum, nbr_idx;
  int nbr_vtx[MAX_NEIGHBOR];
  UF->Clean(); // need to be REMOVED!

  for (i = 0; i < nverts; i++) {
    vtx_array[i].no_JTup = 0;
    vtx_array[i].JTdown = -1;
  }

  for (i = nverts - 1; i >= 0; i--) {

    vtx_id = getVidFromRank(i); // vtxrank_array[i].vid;
    UF->MakeSet(i);
    UF->LowestVertex(i, vtx_id);
    NeighborNum = GetNeighbor(vtx_id, nbr_vtx);

    for (nbr_idx = 0; nbr_idx < NeighborNum; nbr_idx++) {
      j = getRankFromVid(nbr_vtx[nbr_idx]);
      if ((j < i) || (UF->FindSet(j) == UF->FindSet(i)))
        continue;
      AddEdgeJT(vtx_id, UF->getLowestVertex(j), nbr_vtx[nbr_idx]);
      UF->Union(i, j);
      UF->LowestVertex(j, vtx_id);
    }
  }
  return true;
}

bool HeightField::CreateSplitTree() {

  int i, j, vtx_id, NeighborNum, nbr_idx;
  int nbr_vtx[MAX_NEIGHBOR];
  UF->Clean(); // need to be REMOVED!

  for (i = 0; i < nverts; i++) {
    vtx_array[i].no_STdown = 0;
    vtx_array[i].STup = -1;
  }

  for (i = 0; i < nverts; i++) {

    vtx_id = getVidFromRank(i); // vtxrank_array[i].vid;
    if (vtx_id == 167809) {
      j = 0;
    }
    UF->MakeSet(i);
    UF->HighestVertex(i, vtx_id);
    NeighborNum = GetNeighbor(vtx_id, nbr_vtx);
    for (nbr_idx = 0; nbr_idx < NeighborNum; nbr_idx++) {
      j = getRankFromVid(nbr_vtx[nbr_idx]);
      if ((j > i) || (UF->FindSet(j) == UF->FindSet(i)))
        continue;
      AddEdgeST(vtx_id, UF->getHighestVertex(j), nbr_vtx[nbr_idx]);
      UF->Union(i, j);
      UF->HighestVertex(j, vtx_id);
    }
  }
  return true;
}

bool HeightField::CombineTrees() {

  int vid, xi, yj;
  CellQueue1 leafq;
  // int j=0;
  int i;
  int tree_id;

  for (i = 0; i < nverts; i++) {
    vtx_array[i].no_ACTup = 0;
    vtx_array[i].no_ACTdown = 0;
  }

  for (vid = 0; vid < nverts; vid++) {
    if (upperleaf_JT(vid)) {
      leafq.Add(vid);
      leafq.Add(JT_ID);
    } else if (lowerleaf_ST(vid)) {
      leafq.Add(vid);
      leafq.Add(ST_ID);
    }
  }

  while (!leafq.Empty()) {
    leafq.Get(xi);
    leafq.Get(tree_id);

    if (vtx_array[xi].no_ACTdown == vtx_array[xi].no_STdown &&
        vtx_array[xi].no_ACTup == vtx_array[xi].no_JTup)
      continue;

    if (0 == vtx_array[xi].no_STdown && 0 == vtx_array[xi].no_JTup) {
      continue;
    }

    if (tree_id == JT_ID) {
      yj = find_nbr_JT(xi);
    } else {
      assert(tree_id == ST_ID);
      yj = find_nbr_ST(xi);
    }

    del_JT(xi);
    del_ST(xi);

    if (yj != -1) {
      add_arc(xi, yj);
    } else {
      continue;
      printf("yj wrong!!!!!!!!1\n");
      assert(0);
    }

    if (tree_id == JT_ID && upperleaf_JT(yj)) {
      leafq.Add(yj);
      leafq.Add(JT_ID);
    } else if (tree_id == ST_ID && lowerleaf_ST(yj)) {
      leafq.Add(yj);
      leafq.Add(ST_ID);
    }
  }
  getVtxEdgeNo();
  return true;

  return true;
}

bool HeightField::ReduceCT() {

  int i;

  for (i = 0; i < nverts; i++) {
    if ((vtx_array[i].no_ACTup == 1) && (vtx_array[i].no_ACTdown == 1))
      del_ACT(i);
  }

  return true;
}

void HeightField::TreeDrawing() {

  int CT_vid;
  int high_vid;

  high_vid = vtxrank_array[nverts - 1].vid;

  window_size[0] = window_size[1] = getValue(high_vid); // 0 : up , 1 : down
  window_size[2] = window_size[3] = 0; // 2 : left , 3 : right

  clearVisit();

  CT_vid = add2DVtx(high_vid, 0, getValue(high_vid));
  processVtx(high_vid, CT_vid, 0, getValue(high_vid), window_size);
}

int HeightField::idx2vtx(int i, int j, int k) {
  if (((i < 0) || (i >= xDim)) || ((j < 0) || (j >= yDim)) ||
      ((k < 0) || (k >= zDim)))
    return -1;

  return i + (j * xDim) + (k * xDim * yDim);
}

void HeightField::AddEdgeJT(int vtx1, int vtx2, int nbr_vtx) {
  assert(getRankFromVid(vtx1) < getRankFromVid(vtx2));

  Add_JT_upedge(vtx1, vtx2);
  Add_JT_downedge(vtx2, vtx1, nbr_vtx);
}

void HeightField::AddEdgeST(int vtx1, int vtx2, int nbr_vtx) {
  assert(getRankFromVid(vtx1) > getRankFromVid(vtx2));

  Add_ST_downedge(vtx1, vtx2, nbr_vtx);
  Add_ST_upedge(vtx2, vtx1);
}

int HeightField::getRankFromVid(int vid) { return vtx_array[vid].rank; }

int HeightField::GetNeighbor(int vtx, int *neighbor_vtx) {
  int i, j, k, idx;
  int nbr_idx = 0;
  int nidx = 0;
  int nbr_vtx[MAX_NEIGHBOR];

  vtx2idx(vtx, i, j, k);
  nbr_vtx[nbr_idx++] = idx2vtx(i - 1, j, k);
  nbr_vtx[nbr_idx++] = idx2vtx(i + 1, j, k);
  nbr_vtx[nbr_idx++] = idx2vtx(i, j - 1, k);
  nbr_vtx[nbr_idx++] = idx2vtx(i, j + 1, k);
  nbr_vtx[nbr_idx++] = idx2vtx(i, j, k - 1);
  nbr_vtx[nbr_idx++] = idx2vtx(i, j, k + 1);

  nbr_vtx[nbr_idx++] = idx2vtx(i + 1, j + 1, k);
  nbr_vtx[nbr_idx++] = idx2vtx(i - 1, j - 1, k);
  nbr_vtx[nbr_idx++] = idx2vtx(i, j + 1, k - 1);
  nbr_vtx[nbr_idx++] = idx2vtx(i, j - 1, k + 1);
  nbr_vtx[nbr_idx++] = idx2vtx(i + 1, j, k - 1);
  nbr_vtx[nbr_idx++] = idx2vtx(i - 1, j, k + 1);
  nbr_vtx[nbr_idx++] = idx2vtx(i + 1, j + 1, k - 1);
  nbr_vtx[nbr_idx++] = idx2vtx(i - 1, j - 1, k + 1);

  for (idx = 0; idx < nbr_idx; idx++) {
    if (nbr_vtx[idx] != -1)
      neighbor_vtx[nidx++] = nbr_vtx[idx];
  }

  return nidx;
}

int HeightField::getVidFromRank(int rank_idx) {
  return vtxrank_array[rank_idx].vid;
}

void HeightField::Add_JT_upedge(int vtx1, int vtx2) {
  int *temp;
  int i;

  if (vtx_array[vtx1].no_JTup == 0) {
    vtx_array[vtx1].JTup = (int *)malloc(sizeof(int));
    vtx_array[vtx1].JTup[0] = vtx2;
    vtx_array[vtx1].no_JTup++;
  } else {
    vtx_array[vtx1].no_JTup++;
    temp = (int *)malloc(sizeof(int) * vtx_array[vtx1].no_JTup);
    for (i = 0; i < vtx_array[vtx1].no_JTup - 1; i++) {
      temp[i] = vtx_array[vtx1].JTup[i];
    }
    temp[i] = vtx2;
    free(vtx_array[vtx1].JTup);
    vtx_array[vtx1].JTup = temp;
  }
}

void HeightField::Add_JT_downedge(int vtx1, int vtx2, int nbr_vtx) {
  assert(vtx_array[vtx1].JTdown == -1);

  vtx_array[vtx1].JTdown = vtx2;
  // vtx_array[vtx1].nbr_JTdown = nbr_vtx;
}

void HeightField::Add_ST_downedge(int vtx1, int vtx2, int nbr_vtx) {
  int *temp, *nbr_temp;
  int i;

  if (vtx_array[vtx1].no_STdown == 0) {
    vtx_array[vtx1].STdown = (int *)malloc(sizeof(int));
    // vtx_array[vtx1].nbr_STdown = (int*)malloc(sizeof(int));

    vtx_array[vtx1].STdown[0] = vtx2;
    // vtx_array[vtx1].nbr_STdown[0] = nbr_vtx;

    vtx_array[vtx1].no_STdown++;
  } else {
    vtx_array[vtx1].no_STdown++;
    temp = (int *)malloc(sizeof(int) * vtx_array[vtx1].no_STdown);
    nbr_temp = (int *)malloc(sizeof(int) * vtx_array[vtx1].no_STdown);

    for (i = 0; i < vtx_array[vtx1].no_STdown - 1; i++) {
      temp[i] = vtx_array[vtx1].STdown[i];
      // nbr_temp[i] = vtx_array[vtx1].nbr_STdown[i];
    }
    temp[i] = vtx2;
    nbr_temp[i] = nbr_vtx;
    free(vtx_array[vtx1].STdown);
    // free(vtx_array[vtx1].nbr_STdown);
    vtx_array[vtx1].STdown = temp;
    // vtx_array[vtx1].nbr_STdown = nbr_temp;
  }
}

void HeightField::Add_ST_upedge(int vtx1, int vtx2) {
  assert(vtx_array[vtx1].STup == -1);
  vtx_array[vtx1].STup = vtx2;
}

void HeightField::vtx2idx(int v, int &i, int &j, int &k) {
  i = v % xDim;
  j = (v / xDim) % yDim;
  k = (v / (xDim * yDim)) % zDim;
}

void HeightField::getVtxEdgeNo() {
  int i;
  NoCTVtx = 0;
  for (i = 0; i < nverts; i++) {
    if (!isRegular(i))
      NoCTVtx++;
  }
  NoCTEdge = NoCTVtx - 1;
}

void HeightField::add_arc(int v1, int v2) {
  int vtx1, vtx2;
  if (getRankFromVid(v1) > getRankFromVid(v2)) {
    vtx1 = v1;
    vtx2 = v2;
  } else {
    vtx1 = v2;
    vtx2 = v1;
  }

  ACT_link(vtx1, vtx2);
}

void HeightField::del_JT(int v) {
  int up_v, down_v;

  if (is_updgr1_downdgr1_JT(v)) {
    // vtx_array[v].vid = -1;

    down_v = vtx_array[v].JTdown;
    vtx_array[v].JTdown = -1;

    if (vtx_array[v].no_JTup > 0) {
      up_v = vtx_array[v].JTup[0];
      vtx_array[v].no_JTup = 0;
      free(vtx_array[v].JTup);
    } else
      up_v = -1;

    replace_edge_JT(up_v, down_v, v);
  }
}

void HeightField::del_ST(int v) {
  int up_v, down_v;

  if (is_updgr1_downdgr1_ST(v)) {
    // vtx_array[v].vtx = -1;

    up_v = vtx_array[v].STup;
    vtx_array[v].STup = -1;

    if (vtx_array[v].no_STdown > 0) {
      down_v = vtx_array[v].STdown[0];
      vtx_array[v].no_STdown = 0;
      free(vtx_array[v].STdown);
    } else
      down_v = -1;

    replace_edge_ST(up_v, down_v, v);
  }
}

int HeightField::find_nbr_JT(int v) { return vtx_array[v].JTdown; }

int HeightField::find_nbr_ST(int v) { return vtx_array[v].STup; }

bool HeightField::upperleaf_JT(int v) {
  if (vtx_array[v].no_JTup == 0)
    return true;
  return false;
}

bool HeightField::lowerleaf_ST(int v) {
  if (vtx_array[v].no_STdown == 0)
    return true;
  return false;
}

void HeightField::ACT_link(int high_vtx, int low_vtx) {
  int *temp;
  int i;

  if (high_vtx == 167809 || low_vtx == 167809) {

    i = 0;
  }

  assert(compareHeight(high_vtx, low_vtx) > 0);

  if (vtx_array[high_vtx].no_ACTdown == 0) {
    vtx_array[high_vtx].ACTdown = (int *)malloc(sizeof(int));
    vtx_array[high_vtx].ACTdown[0] = low_vtx;
    vtx_array[high_vtx].no_ACTdown = 1;
  } else {
    vtx_array[high_vtx].no_ACTdown++;
    temp = (int *)malloc(vtx_array[high_vtx].no_ACTdown * sizeof(int));
    for (i = 0; i < vtx_array[high_vtx].no_ACTdown - 1; i++) {
      temp[i] = vtx_array[high_vtx].ACTdown[i];
    }
    temp[i] = low_vtx;
    free(vtx_array[high_vtx].ACTdown);
    vtx_array[high_vtx].ACTdown = temp;
  }

  if (vtx_array[low_vtx].no_ACTup == 0) {
    vtx_array[low_vtx].ACTup = (int *)malloc(sizeof(int));
    vtx_array[low_vtx].ACTup[0] = high_vtx;
    vtx_array[low_vtx].no_ACTup = 1;
  } else {
    vtx_array[low_vtx].no_ACTup++;
    temp = (int *)malloc(vtx_array[low_vtx].no_ACTup * sizeof(int));
    for (i = 0; i < vtx_array[low_vtx].no_ACTup - 1; i++) {
      temp[i] = vtx_array[low_vtx].ACTup[i];
    }
    temp[i] = high_vtx;
    free(vtx_array[low_vtx].ACTup);
    vtx_array[low_vtx].ACTup = temp;
  }
}

void HeightField::replace_edge_JT(int up_vid, int down_vid, int vid) {
  int i, flag;
  flag = 0;

  if (up_vid == -1) {
    if (down_vid == -1)
      return;
    else {
      for (i = 0; i < vtx_array[down_vid].no_JTup; i++) {
        if (vtx_array[down_vid].JTup[i] == vid) {
          vtx_array[down_vid].JTup[i] = -1;
        }
      }
      CompactArray(vtx_array[down_vid].JTup, vtx_array[down_vid].no_JTup);
      return;
    }
  }
  if (down_vid == -1) {
    assert(up_vid != -1);
    vtx_array[up_vid].JTdown = -1;
    return;
  }

  assert(vtx_array[up_vid].JTdown == vid);
  vtx_array[up_vid].JTdown = down_vid;

  assert(vtx_array[down_vid].no_JTup > 0);
  for (i = 0; i < vtx_array[down_vid].no_JTup; i++) {
    if (vtx_array[down_vid].JTup[i] == vid) {
      vtx_array[down_vid].JTup[i] = up_vid;
      flag = 1;
    }
  }
  CompactArray(vtx_array[down_vid].JTup, vtx_array[down_vid].no_JTup);
  assert(flag == 1);
}

void HeightField::replace_edge_ST(int up_vid, int down_vid, int vid) {
  int i, flag;
  flag = 0;

  if (down_vid == -1) {
    if (up_vid == -1)
      return;
    else {
      for (i = 0; i < vtx_array[up_vid].no_STdown; i++) {
        if (vtx_array[up_vid].STdown[i] == vid) {
          vtx_array[up_vid].STdown[i] = -1;
        }
      }
      CompactArray(vtx_array[up_vid].STdown, vtx_array[up_vid].no_STdown);
      return;
    }
  }
  if (up_vid == -1) {
    assert(down_vid != -1);
    vtx_array[down_vid].STup = -1;
    return;
  }

  assert(vtx_array[down_vid].STup == vid);
  vtx_array[down_vid].STup = up_vid;

  assert(vtx_array[up_vid].no_STdown > 0);
  for (i = 0; i < vtx_array[up_vid].no_STdown; i++) {
    if (vtx_array[up_vid].STdown[i] == vid) {
      vtx_array[up_vid].STdown[i] = down_vid;
      flag = 1;
    }
  }
  CompactArray(vtx_array[up_vid].STdown, vtx_array[up_vid].no_STdown);
  assert(flag == 1);
}

bool HeightField::is_updgr1_downdgr1_JT(int vid) {
  if (vtx_array[vid].no_JTup <= 1)
    return true;
  return false;
}

bool HeightField::is_updgr1_downdgr1_ST(int vid) {
  if (vtx_array[vid].no_STdown <= 1)
    return true;
  return false;
}

int HeightField::compareHeight(int vtx1, int vtx2) {
  if (getValue(vtx1) == getValue(vtx2)) {
    if (vtx1 > vtx2)
      return 1;
    else
      return -1;
  } else {
    if (getValue(vtx1) > getValue(vtx2))
      return 1;
    else
      return -1;
  }
}

void HeightField::del_ACT(int v) {
  int i, up_vid, down_vid;
  int flag = 0;

  assert((vtx_array[v].no_ACTup == 1) && (vtx_array[v].no_ACTdown == 1));
  vtx_array[v].no_ACTup = 0;
  vtx_array[v].no_ACTdown = 0;
  up_vid = vtx_array[v].ACTup[0];
  down_vid = vtx_array[v].ACTdown[0];
  free(vtx_array[v].ACTup);
  free(vtx_array[v].ACTdown);

  assert((vtx_array[up_vid].no_ACTdown > 0) &&
         (vtx_array[down_vid].no_ACTup > 0));
  for (i = 0; i < vtx_array[up_vid].no_ACTdown; i++) {
    if (vtx_array[up_vid].ACTdown[i] == v) {
      vtx_array[up_vid].ACTdown[i] = down_vid;
      flag = 1;
    }
  }
  assert(flag == 1);
  flag = 0;

  for (i = 0; i < vtx_array[down_vid].no_ACTup; i++) {
    if (vtx_array[down_vid].ACTup[i] == v) {
      vtx_array[down_vid].ACTup[i] = up_vid;
      flag = 1;
    }
  }
  assert(flag == 1);
}

void HeightField::processVtx(int vid, int CT_vid, float src_x, float src_y,
                             float *pos) {
  int i, k;
  float target_x[MAX_NEIGHBOR], target_y[MAX_NEIGHBOR];
  int num_upvtx, num_downvtx;
  k = 0;
  int CT_nbr_array[MAX_NEIGHBOR];
  int upvtx_array[MAX_NEIGHBOR];
  int downvtx_array[MAX_NEIGHBOR];

  float s_x, s_y;

  TreeNodeStack st(1000000);

  st.push(vid, CT_vid, src_x, src_y);

  while ((st.pop(vid, CT_vid, s_x, s_y)) >= 0) {
    k = 0;
    Visit(vid);
    num_upvtx = getUpperVtx(vid, upvtx_array);
    num_downvtx = getLowerVtx(vid, downvtx_array);

    for (i = 0; i < num_upvtx; i++) {
      compute_pos(target_x[k], target_y[k], s_x, s_y, getValue(vid),
                  getValue(upvtx_array[i]), num_upvtx, i);
      CT_nbr_array[k] = add2DVtx(upvtx_array[i], target_x[k], target_y[k]);
      updateWindow(target_x[k], target_y[k], pos);
      k++;
    }

    for (i = 0; i < num_downvtx; i++) {
      compute_pos(target_x[k], target_y[k], s_x, s_y, getValue(vid),
                  getValue(downvtx_array[i]), num_downvtx, i);
      CT_nbr_array[k] = add2DVtx(downvtx_array[i], target_x[k], target_y[k]);
      updateWindow(target_x[k], target_y[k], pos);
      k++;
    }

    for (i = 0; i < k; i++) {
      assert(CT_nbr_array[i] >= 0 && CT_nbr_array[i] < nverts);
      add2DEdge(CT_vid, CT_nbr_array[i]);
    }

    k = 0;
    for (i = 0; i < num_upvtx; i++) {
      st.push(upvtx_array[i], CT_nbr_array[k], target_x[k], target_y[k]);
      k++;
    }

    for (i = 0; i < num_downvtx; i++) {
      st.push(downvtx_array[i], CT_nbr_array[k], target_x[k], target_y[k]);
      k++;
    }
  }
}

int HeightField::add2DVtx(int vid, float pos_x, float pos_y) {
  CTVtx *temp;
  int oldsize;

  if (CT_vidx >= CT_vsize) {
    oldsize = CT_vsize;
    CT_vsize *= 2;
    temp = (CTVtx *)malloc(sizeof(CTVtx) * CT_vsize);
    memcpy(temp, CT_array, oldsize * sizeof(CTVtx));
    free(CT_array);
    CT_array = temp;
  }
  CT_array[CT_vidx].vid = vid;
  CT_array[CT_vidx].x = pos_x;
  CT_array[CT_vidx].y = pos_y;

  return CT_vidx++;
}

int HeightField::add2DEdge(int CT_vid1, int CT_vid2) {
  CTEdge *temp;
  int oldsize;

  if (CT_eidx >= CT_esize) {
    oldsize = CT_esize;
    CT_esize *= 2;
    temp = (CTEdge *)malloc(sizeof(CTEdge) * CT_esize);
    memcpy(temp, CTEdge_array, oldsize * sizeof(CTEdge));
    free(CTEdge_array);
    CTEdge_array = temp;
  }
  CTEdge_array[CT_eidx].CT_vid1 = CT_vid1;
  CTEdge_array[CT_eidx].CT_vid2 = CT_vid2;

  return CT_eidx++;
}

void HeightField::compute_pos(float &tx, float &ty, float sx, float sy,
                              float sf, float tf, int n, int i) {
  float angle;
  ty = tf;
  int r;

  // srand(time(NULL));
  // r=rand()%13;
  // angle = (float)((r+3)*M_PI/18.0f);

  // arand, 8-25-2011
  //  I am just hacking this function because the results seem to
  //     look more accurate.
  //  I think there is a bug with the old formula that allows
  //     some tree branches to point the wrong direction.
  //  This possibly isn't changing anything important...
  r = rand() % 60;
  angle = (float)((r + 60) * M_PI / 180.0f);

  tx = (float)((tf - sf) / tan(angle) + sx);

  if (tf == sf)
    tx = sx;
}

int HeightField::getUpperVtx(int vid, int *upv_array) {
  int i, up_vid;
  int num = 0;
  // int j=0;

  for (i = 0; i < vtx_array[vid].no_ACTup; i++) {
    up_vid = vtx_array[vid].ACTup[i];
    assert(up_vid >= 0 && up_vid < nverts);
    if (!isVisit(up_vid))
      upv_array[num++] = up_vid;
  }
  return num;
}

int HeightField::getLowerVtx(int vid, int *downv_array) {
  int i, down_vid;
  int num = 0;
  // int j=0;

  for (i = 0; i < vtx_array[vid].no_ACTdown; i++) {
    down_vid = vtx_array[vid].ACTdown[i];
    assert(down_vid >= 0 && down_vid < nverts);
    if (!isVisit(down_vid))
      downv_array[num++] = down_vid;
  }
  return num;
}

#endif
