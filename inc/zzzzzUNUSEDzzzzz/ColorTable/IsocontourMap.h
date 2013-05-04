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
#ifndef ISOCONTOURMAP_H
#define ISOCONTOURMAP_H

#include <qtextstream.h>

///\class IsocontourMap IsocontourMap.h
///\author Anthony Thane
///\author Vinay Siddavanahalli
///\author John Wiggins
///\brief The IsocontourMap class stores the isocontours for a ColorTable
/// object.
class IsocontourMap {
public:
///\fn double GetPosition(int id) const
///\brief Returns the position of a specific node
///\param id The id of the node
///\return The position of the node
	double GetPosition(int index) const;
///\fn double GetPositionOfIthNode(int index) const
///\brief Returns the position of a specific node
///\param index The index of the node
///\return The position of the node
	double GetPositionOfIthNode(int index) const;
///\fn int GetSize() const
///\brief Returns the number of nodes in the map
///\return The number of nodes
	int GetSize() const;
///\fn int AddNode(double fPosition, double fR, double fG, double fB)
///\brief Adds a new node to the map
///\param fPosition The position of the new node
///\param fR The red component of the new isocontour's color
///\param fG The green component of the new isocontour's color
///\param fB The blue component of the new isocontour's color
///\return The id of the new node
	int AddNode(double fPosition, double fR, double fG, double fB);
///\fn void DeleteIthNode(int index)
///\brief Deletes a specific node in the map
///\param index The index of the node
	void DeleteIthNode(int index);
///\fn void MoveIthNode(int index, double fPosition)
///\brief Moves a specific node in the map
///\param index The index of the node to be moved
///\param fPosition The new position of the node
	void MoveIthNode(int index, double fPosition);
///\fn int GetIDofIthNode(int index) const
///\brief Returns the id of a specific node
///\param index The index of the node you want an id for
///\return The id of the node
	int GetIDofIthNode(int index) const;
	
///\fn void ChangeColor(int index, double fR, double fG, double fB)
///\brief Changes the color of a specific node
///\param index The index of the node
///\param fR The red component of the node's new color
///\param fG The green component of the node's new color
///\param fB The blue component of the node's new color
	void ChangeColor(int index, double fR, double fG, double fB);
///\fn double GetRed(int index) const
///\brief Returns the red component of a specific node's color
///\param index The index of the node
///\return A number between 0 and 1
	double GetRed(int index) const;
///\fn double GetGreen(int index) const
///\brief Returns the green component of a specific node's color
///\param index The index of the node
///\return A number between 0 and 1
	double GetGreen(int index) const;
///\fn double GetBlue(int index) const
///\brief Returns the blue component of a specific node's color
///\param index The index of the node
///\return A number between 0 and 1
	double GetBlue(int index) const;

	IsocontourMap();
	IsocontourMap(const IsocontourMap& copy);
	virtual ~IsocontourMap();
	
///\fn void saveMap( QTextStream& stream )
///\brief Writes the nodes in the map to an open text file
///\param stream A QTextStream that is opened for writing
	void saveMap( QTextStream& stream );
///\fn void loadMap( QTextStream& stream )
///\brief Reads isocontour nodes from an open text file
///\param stream A QTextStream that is opened for reading
	void loadMap( QTextStream& stream );

private:
///\class IsocontourMapNode
///\brief IsocontourMapNode holds the information for individual
/// nodes in an IsocontourMap
	class IsocontourMapNode
	{
	public:
		IsocontourMapNode(double fPosition, double fR, double fG, double fB,
						IsocontourMapNode* Next);
		IsocontourMapNode(const IsocontourMapNode& copy);
		virtual ~IsocontourMapNode();
		IsocontourMapNode* m_pNext;
		double m_fPosition;
		double m_R,m_G,m_B;
		int m_ID;

		// give each node a unique ID
		static int ms_NextID;
		static int getNextID() { return ++ms_NextID; }
	};
protected:
	int m_Size;
	IsocontourMapNode* m_pHead;
};

#endif
