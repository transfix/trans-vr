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
#ifndef CONTREEMAP_H
#define CONTREEMAP_H

#include <contourtree/computeCT.h>
#include <qtextstream.h>

///\class ConTreeMap ConTreeMap.h
///\author John Wiggins
///\brief A ConTreeMap is like an IsocontourMap for contour trees.
class ConTreeMap {
public:
	// node management
///\fn int GetEdge(int id) const
///\brief Returns the edge for a specific node
///\param id The id of the node
///\return An edge index
	int GetEdge(int id) const;
///\fn int GetEdgeOfIthNode(int index) const
///\brief Returns the edge for a specific node
///\param index The index of the node
///\return An edge index
	int GetEdgeOfIthNode(int index) const;
///\fn double GetPosition(int id) const
///\brief Returns the position of a specific node
///\param id The id of the node
///\return A position
	double GetPosition(int id) const;
///\fn double GetPositionOfIthNode(int index) const
///\brief Returns the position of a specific node
///\param index The index of the node
///\return A position
	double GetPositionOfIthNode(int index) const;
///\fn int GetSize() const
///\brief Returns the number of nodes
///\return The number of nodes
	int GetSize() const;
///\fn int AddNode(int edge, double isoval)
///\brief Adds a node to the map
///\param edge The edge that the node resides on
///\param isoval The isovalue of the node
///\return The id of the new node
	int AddNode(int edge, double isoval);
///\fn void DeleteIthNode(int index)
///\brief Deletes a node
///\param index The index of the node to be deleted
	void DeleteIthNode(int index);
///\fn void MoveIthNode(int index, double isoval)
///\brief Moves a node along its edge
///\param index The index of the node being moved
///\param isoval The new isovalue for the node
	void MoveIthNode(int index, double isoval);
///\fn int GetIDofIthNode(int index) const
///\brief Returns the id of a node given its index
///\param index The index of a node
///\return A node id
	int GetIDofIthNode(int index) const;
	// contour tree management
///\fn void SetCTData(CTVTX *verts, CTEDGE *edges, int numverts, int numedges)
///\brief Assigns a contour tree to the map
///\param verts The vertices of the tree
///\param edges The edges of the tree
///\param numverts The number of vertices
///\param numedges The number of edges
	void SetCTData(CTVTX *verts, CTEDGE *edges, int numverts, int numedges);
///\fn CTVTX* GetCTVerts()
///\brief Returns the vertices of the tree
///\return A pointer to some vertices
	CTVTX* GetCTVerts();
///\fn int GetVertNum() const
///\brief Returns the number of vertices
///\return The number of vertices
	int GetVertNum() const;
///\fn CTEDGE* GetCTEdges()
///\brief Returns the edges of the tree
///\return A pointer to some edges
	CTEDGE* GetCTEdges();
///\fn int GetEdgeNum() const
///\brief Returns the number of edges
///\return The number of edges
	int GetEdgeNum() const;
	// ctors & dtors
	ConTreeMap();
	ConTreeMap(const ConTreeMap& copy);
	virtual ~ConTreeMap();

private:
///\class ConTreeMapNode
///\brief Holds the information for a single node in a ConTreeMap
	class ConTreeMapNode
	{
	public:
		ConTreeMapNode(double isoval, int edge, ConTreeMapNode* Next);
		ConTreeMapNode(const ConTreeMapNode& copy);
		virtual ~ConTreeMapNode();

		ConTreeMapNode* m_pNext;
		double m_Isoval;
		int m_Edge;
		int m_ID;

		// give each node a unique ID
		static int ms_NextID;
		static int getNextID() { return --ms_NextID; }
	};

protected:
	ConTreeMapNode* m_pHead;

	CTEDGE *m_Edges;
	CTVTX *m_Verts;
	int m_NumVerts;
	int m_NumEdges;
	int m_Size;
};

#endif

