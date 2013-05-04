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
#ifndef COLORMAP_H
#define COLORMAP_H

#include <qtextstream.h>

///\class ColorMap ColorMap.h
///\author Anthony Thane
///\author Vinay Siddavanahalli
///\brief The ColorMap class stores the color component of a 1D transfer
/// function.
class ColorMap
{
public:
///\fn void ChangeColor(int index, double fRed,  double fGreen,  double fBlue)
///\brief Changes the color of a node
///\param index The index of the node to be changed
///\param fRed The red component of the new color
///\param fGreen The green component of the new color
///\param fBlue The blue component of the new color
	void ChangeColor(int index, double fRed,  double fGreen,  double fBlue);
///\fn double GetRed(int index) const
///\brief Returns the red value for a specific node
///\param index The index of the node
///\return The red value (0 to 1)
	double GetRed(int index) const;
///\fn double GetGreen(int index) const
///\brief Returns the green value for a specific node
///\param index The index of the node
///\return The green value (0 to 1)
	double GetGreen(int index) const;
///\fn double GetBlue(int index) const
///\brief Returns the blue value for a specific node
///\param index The index of the node
///\return The blue value (0 to 1)
	double GetBlue(int index) const;
///\fn double GetPosition(int index) const
///\brief Returns the position of a specific node
///\param index The index of the node
///\return The position
	double GetPosition(int index) const;
///\fn double GetRed(double fPosition) const
///\brief Returns the linearly interpolated red value at a specific position
///\param fPosition The position
///\return The red value (0 to 1)
	double GetRed(double fPosition) const;
///\fn double GetGreen(double fPosition) const
///\brief Returns the linearly interpolated green value at a specific position
///\param fPosition The position
///\return The green value (0 to 1)
	double GetGreen(double fPosition) const;
///\fn double GetBlue(double fPosition) const
///\brief Returns the linearly interpolated blue value at a specific position
///\param fPosition The position
///\return The blue value (0 to 1)
	double GetBlue(double fPosition) const;
///\fn int GetSize() const
///\brief Returns the number of nodes in the map
///\return The number of nodes
	int GetSize() const;
///\fn void MoveNode(int index, double fPosition)
///\brief Changes the position of a node
///\param index The index of the node to be moved
///\param fPosition The new position of the node
	void MoveNode(int index, double fPosition);
///\fn void AddNode(double fPosition, double fRed, double fGreen, double fBlue)
///\brief Adds a new node to the map
///\param fPosition The position of the node
///\param fRed The red component of the node's color
///\param fGreen The green component of the node's color
///\param fBlue The blue component of the node's color
	void AddNode(double fPosition, double fRed, double fGreen, double fBlue);
///\fn void AddNode(double fPosition)
///\brief Adds a new node to the map with the default color. The default color
/// is the linearly
/// interpolated color for the given position.
///\param fPosition The position of the new node
	void AddNode(double fPosition);
///\fn void DeleteNode(int index)
///\brief Deletes a specific node
///\param index The index of the node to be deleted
	void DeleteNode(int index);
	ColorMap();
	ColorMap(const ColorMap& copy);
	ColorMap& operator=(const ColorMap& copy);
	virtual ~ColorMap();
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
///\class ColorMapNode
///\brief A ColorMapNode holds the position and color information for each of
/// the nodes in a ColorMap
	class ColorMapNode
	{
	public:
///\fn ColorMapNode(double fPosition, double fRed, double fGreen, double fBlue, ColorMapNode* Prev, ColorMapNode* Next)
///\brief The constructor
///\param fPosition The position of the node
///\param fRed The red component of the node's color
///\param fGreen The green component of the node's color
///\param fBlue The blue component of the node's color
///\param Prev The node preceeding this node
///\param Next The node following this node
		ColorMapNode(double fPosition, double fRed, double fGreen, double fBlue, ColorMapNode* Prev, ColorMapNode* Next);
///\fn ColorMapNode(const ColorMapNode& copy, ColorMapNode* prev)
///\brief The 'copy' constructor
///\param copy The node to make a copy of
///\param prev The node preceeding this node
		ColorMapNode(const ColorMapNode& copy, ColorMapNode* prev);
		virtual ~ColorMapNode();
		ColorMapNode* m_pNext;
		ColorMapNode* m_pPrev;
		double m_fRed;
		double m_fGreen;
		double m_fBlue;
		double m_fPosition;
	};
protected:
	int m_Size;
	ColorMapNode* m_pTail;
	ColorMapNode* m_pHead;
};
#endif
