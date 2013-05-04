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
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef ALPHAMAP_H
#define ALPHAMAP_H

#include <qtextstream.h>

///\class AlphaMap
///\author Anthony Thane
///\author Vinay Siddavanahalli
///\brief The AlphaMap class stores the opacity component of a 1D transfer
/// function
class AlphaMap {
public:
///\fn void ChangeAlpha(int index, double fAlpha)
///\brief Changes the alpha value of a node
///\param index The index of the node being changed
///\param fAlpha The new alpha value
	void ChangeAlpha(int index, double fAlpha);
///\fn double GetAlpha(int index) const
///\brief Returns the alpha value of a node
///\param index The index of the node
///\return An alpha value
	double GetAlpha(int index) const;
///\fn double GetPosition(int index) const
///\brief Returns the position of a node
///\param index The index of the node
///\return The position of the node (a number between 0 and 1)
	double GetPosition(int index) const;
///\fn double GetAlpha(double fPosition) const
///\brief Returns the alpha value at some arbitrary position using linear
/// interpolation
///\param fPosition The position
///\return An alpha value
	double GetAlpha(double fPosition) const;
///\fn int GetSize() const
///\brief Returns the number of nodes in the map
///\return The number of nodes
	int GetSize() const;
///\fn void MoveNode(int index, double fPosition)
///\brief Moves a node to a new position
///\param index The index of the node to be moved
///\param fPosition The new position
	void MoveNode(int index, double fPosition);
///\fn void AddNode(double fPosition, double fAlpha)
///\brief Adds a new node to the map
///\param fPosition The position of the node
///\param fAlpha The alpha value of the node
	void AddNode(double fPosition, double fAlpha);
///\fn void AddNode(double fPosition)
///\brief Adds a new node to the map with the default alpha value. The default
/// alpha value is the value that GetAlpha() would return for the given
/// position.
///\param fPosition The position of the node
	void AddNode(double fPosition);
///\fn void DeleteNode(int index)
///\brief Deletes a specific node in the map
///\param index The index of the node to be deleted
	void DeleteNode(int index);
	AlphaMap();
///\fn AlphaMap(const AlphaMap& copy)
///\brief The copy constructor
///\param copy The object to be copied
	AlphaMap(const AlphaMap& copy);
///\fn AlphaMap& operator=(const AlphaMap& copy)
///\brief The assignment operator
///\param copy The object to copy
	AlphaMap& operator=(const AlphaMap& copy);
	virtual ~AlphaMap();
///\fn void saveMap( QTextStream& stream )
///\brief Saves the nodes in the map to a text file
///\param stream A QTextStream object opened for writing
	void saveMap( QTextStream& stream );
///\fn void loadMap( QTextStream& stream )
///\brief Reads a map from a text file
///\param stream A QTextStream object opened for reading
	void loadMap( QTextStream& stream );
///\fn void removeSandwichedNodes()
///\brief Removes nodes that are between two other nodes with the same position
	void removeSandwichedNodes();
private:
///\class AlphaMapNode
///\brief AlphaMapNode objects hold the position and alpha information for each
/// node in an AlphaMap
	class AlphaMapNode
	{
	public:
///\fn AlphaMapNode(double fPosition, double fAlpha, AlphaMapNode* Prev, AlphaMapNode* Next);
///\brief The constructor
///\param fPosition The position of the node
///\param fAlpha The alpha value at the node
///\param Prev The node preceeding this node
///\param Next The node following this node
		AlphaMapNode(double fPosition, double fAlpha, AlphaMapNode* Prev, AlphaMapNode* Next);
///\fn AlphaMapNode(const AlphaMapNode& copy, AlphaMapNode* prev);
///\brief The 'copy' constructor
///\param copy The object to be copied
///\param prev The node preceeding this node
		AlphaMapNode(const AlphaMapNode& copy, AlphaMapNode* prev);
		virtual ~AlphaMapNode();
		AlphaMapNode* m_pNext;
		AlphaMapNode* m_pPrev;
		double m_fAlpha;
		double m_fPosition;
	};
protected:
	int m_Size;
	AlphaMapNode* m_pTail;
	AlphaMapNode* m_pHead;
};

#endif
