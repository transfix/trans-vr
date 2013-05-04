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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef UNIONFIND_H
#define UNIONFIND_H

#include<stdio.h>

class UnionFind
{
	int* p;
	int* rank;
	//float* val;
	int size;
	int* LowestVtxArray;

public :
	UnionFind(int Size) 
	{ 
		size = Size;
		p = new int[Size];
		rank = new int[Size];
		LowestVtxArray = new int[Size];
	}

	void Clean()
	{
		int i;
		for (i = 0 ; i<size ; i++) {
			p[i]=0; rank[i]=0; LowestVtxArray[i]=0;
		}
	}


	~UnionFind() 
	{
		delete p;
		delete rank;
		delete LowestVtxArray;
	}

	void MakeSet(int x) 
	{
		p[x] = x ;
		rank[x] = 0;
	}
	
	void Union(int x , int y)
	{
		Link(FindSet(x) , FindSet(y));
	}
	
	void Link(int x , int y)
	{
		if (rank[x] > rank[y]) p[y] = x;
		else p[x] = y;
		if (rank[x] == rank[y]) 
			rank[y]++;
	}

	int FindSet(int x)
	{
		if (x != p[x])
			p[x] = FindSet(p[x]);
		return p[x];
	}

	void LowestVertex(int v , int vid)
	{	
		LowestVtxArray[FindSet(v)] = vid; 
	}

	int getLowestVertex(int v)
	{
		return LowestVtxArray[FindSet(v)];
	}

	void HighestVertex(int v , int vid)
	{
		LowestVtxArray[FindSet(v)] = vid; 
	}

	int getHighestVertex(int v)
	{
		return LowestVtxArray[FindSet(v)];
	}
};
#endif

