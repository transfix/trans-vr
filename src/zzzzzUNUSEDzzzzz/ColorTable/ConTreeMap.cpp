/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
             Vinay Siddavanahalli <skvinay@cs.utexas.edu>
    Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <ColorTable/ConTreeMap.h>
#include <qfile.h>
#include <stdlib.h>

int ConTreeMap::ConTreeMapNode::ms_NextID = 0;

// ConTree ctors & dtor
// 
ConTreeMap::ConTreeMap()
: m_Size(0)
{
	m_pHead = 0;
	
	m_Edges = 0;
	m_Verts = 0;
	m_NumVerts = 0;
	m_NumEdges = 0;
}

ConTreeMap::ConTreeMap(const ConTreeMap& copy)
: m_Size(copy.m_Size)
{
	if (copy.m_pHead)
		m_pHead = new ConTreeMapNode(*(copy.m_pHead));
	else
		m_pHead = 0;

	// deep copy, otherwise the ConTreeMapNode list will be invalid
	if (copy.m_Verts && copy.m_Edges) {
		m_Verts = (CTVTX *)malloc(copy.m_NumVerts * sizeof(CTVTX));
		memcpy(m_Verts, copy.m_Verts, copy.m_NumVerts * sizeof(CTVTX));
		m_NumVerts = copy.m_NumVerts;
		
		m_Edges = (CTEDGE *)malloc(copy.m_NumEdges * sizeof(CTEDGE));
		memcpy(m_Edges, copy.m_Edges, copy.m_NumEdges * sizeof(CTEDGE));
		m_NumEdges = copy.m_NumEdges;
	}
	else {
		m_Edges = 0;
		m_Verts = 0;
		m_NumVerts = 0;
		m_NumEdges = 0;
	}
}

ConTreeMap::~ConTreeMap()
{
	delete m_pHead;

	if (m_Verts) {
		free(m_Verts);
		m_Verts = 0;
	}
	if (m_Edges) {
		free(m_Edges);
		m_Edges = 0;
	}
}

// ConTreeMapNode
// 
ConTreeMap::ConTreeMapNode::ConTreeMapNode(double isoval, int edge,
						ConTreeMap::ConTreeMapNode* Next) 
: m_Isoval(isoval),	m_Edge(edge), m_ID(getNextID()), m_pNext(Next)					
{
}

ConTreeMap::ConTreeMapNode::ConTreeMapNode(const ConTreeMap::ConTreeMapNode& copy)
: m_Edge(copy.m_Edge), m_Isoval(copy.m_Isoval), m_ID(copy.m_ID)
{
	if (copy.m_pNext)
		m_pNext = new ConTreeMapNode(*(copy.m_pNext));
	else 
		m_pNext = 0;
}

ConTreeMap::ConTreeMapNode::~ConTreeMapNode()
{
	delete m_pNext;
}

// node management
//
int ConTreeMap::AddNode(int edge, double isoval)
{
	if (edge>=0 && edge<m_NumEdges) {
		ConTreeMapNode* newNode;
		newNode = new ConTreeMapNode(isoval, edge, m_pHead);
		m_pHead = newNode;
		m_Size++;

		return newNode->m_ID;
	}
	return 0; //-1;
}

void ConTreeMap::DeleteIthNode(int index)
{
	ConTreeMapNode* pNode = m_pHead;
	ConTreeMapNode* toDelete;
	
	if( index == 0 )
	{
		m_pHead = m_pHead->m_pNext;
		pNode->m_pNext = 0;
		delete pNode;
		m_Size--;
		return;
	}

	int i = 1;
	while (pNode && i<index) {
		i++;
		pNode = pNode->m_pNext;
	}
	if (pNode && pNode->m_pNext) {
		toDelete = pNode->m_pNext;
		pNode->m_pNext = toDelete->m_pNext;
		toDelete->m_pNext = 0;
		delete toDelete;
		m_Size--;
	}

}

void ConTreeMap::MoveIthNode(int index, double isoval)
{
	if (index<m_Size) {
		ConTreeMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			// find the min and max possible isovalues for this node
			float iso1, iso2;
			CTEDGE edge = m_Edges[pNode->m_Edge];
			iso1 = m_Verts[edge.v1].func_val;
			iso2 = m_Verts[edge.v2].func_val;

			if (iso2 < iso1) {
				iso1 = iso2;
				iso2 = m_Verts[edge.v1].func_val;
			}
			
			// clamp to that range if needed
			if (isoval > iso2) {
				isoval = iso2;
			}
			else if (isoval < iso1) {
				isoval = iso1;
			}
			// assign
			pNode->m_Isoval = isoval;
		}
	}
}

int ConTreeMap::GetIDofIthNode(int index) const
{
	if (index<m_Size) {
		ConTreeMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_ID;
		}
	}

	// XXX: this is the only non-valid ID possible, right?
	// (between IsocontourMap and ConTreeMap)
	return 0;
}

int ConTreeMap::GetSize() const
{
	return m_Size;
}

double ConTreeMap::GetPosition(int id) const
{
	ConTreeMapNode* pNode;
	pNode = m_pHead;
	int i = 0;
	while (pNode && pNode->m_ID != id) {
		i++;
		pNode = pNode->m_pNext;
	}
	if (pNode) {
		return pNode->m_Isoval;
	}

	// no node found!
	return 0.0;
}

double ConTreeMap::GetPositionOfIthNode(int index) const
{
	if (index < m_Size && index >= 0) {
		ConTreeMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_Isoval;
		}
	}

	// no node found!
	return 0.0;
}

int ConTreeMap::GetEdge(int id) const
{
	ConTreeMapNode* pNode;
	pNode = m_pHead;
	int i = 0;
	while (pNode && pNode->m_ID != id) {
		i++;
		pNode = pNode->m_pNext;
	}
	if (pNode) {
		return pNode->m_Edge;
	}

	// no node found!
	return -1;
}

int ConTreeMap::GetEdgeOfIthNode(int index) const
{
	if (index < m_Size && index >= 0) {
		ConTreeMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_Edge;
		}
	}

	// no node found!
	return -1;
}

// Contour Tree data management
//
void ConTreeMap::SetCTData(CTVTX *verts, CTEDGE *edges, int numverts,
													int numedges)
{
	// free any memory for existing data
	if (m_Verts) {
		free(m_Verts);
		m_Verts = 0;
		m_NumVerts = 0;
	}
	if (m_Edges) {
		free(m_Edges);
		m_Edges = 0;
		m_NumEdges = 0;
	}

	// clear out all the map nodes (because their data is now invalid)
	delete m_pHead;
	m_pHead = 0;

	// reassign pointers
	m_Verts = verts;
	m_NumVerts = numverts;
	m_Edges = edges;
	m_NumEdges = numedges;
}

CTVTX * ConTreeMap::GetCTVerts()
{
	return m_Verts;
}

CTEDGE * ConTreeMap::GetCTEdges()
{
	return m_Edges;
}

int ConTreeMap::GetVertNum() const
{
	return m_NumVerts;
}

int ConTreeMap::GetEdgeNum() const
{
	return m_NumEdges;
}

