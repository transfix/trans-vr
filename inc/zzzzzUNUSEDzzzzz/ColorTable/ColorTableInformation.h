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

//////////////////////////////////////////////////////////////////////
//
// ColorTableInformation.h: interface for the ColorTableInformation class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_COLORTABLEINFORMATION_H__890F2A7C_3D2F_4296_A3AA_C15CDB1217E3__INCLUDED_)
#define AFX_COLORTABLEINFORMATION_H__890F2A7C_3D2F_4296_A3AA_C15CDB1217E3__INCLUDED_

#include <ColorTable/AlphaMap.h>
#include <ColorTable/ColorMap.h>
#include <ColorTable/IsocontourMap.h>
#include <ColorTable/ConTreeMap.h>

///\class ColorTableInformation ColorTableInformation.h
///\author Anthony Thane
///\author Vinay Siddavanahalli
///\author John Wiggins
///\brief ColorTableInformation holds all the information in a ColorTable,
/// including: the transfer function, the isovalues, and the contour tree
/// segments.
class ColorTableInformation  
{
public:
	ColorTableInformation();
///\fn ColorTableInformation(const AlphaMap& alphaMap, const ColorMap& colorMap, const IsocontourMap& isocontourMap, const ConTreeMap& contreeMap)
///\brief The constructor
///\param alphaMap An AlphaMap
///\param colorMap A ColorMap
///\param isocontourMap An IsocontourMap
///\param contreeMap A ConTreeMap
	ColorTableInformation(const AlphaMap& alphaMap, const ColorMap& colorMap, const IsocontourMap& isocontourMap, const ConTreeMap& contreeMap);
///\fn ColorTableInformation(const ColorTableInformation& copy)
///\brief The copy constructor
///\param copy The object to be copied
	ColorTableInformation(const ColorTableInformation& copy);
	virtual ~ColorTableInformation();

///\fn void removeSandwichedNodes()
///\brief Removes any nodes from the AlphaMap or ColorMap that are sandwiched
/// between two nodes with the same position.
	void removeSandwichedNodes();

///\fn const AlphaMap& getAlphaMap() const
///\brief Returns the AlphaMap
///\return An AlphaMap
	const AlphaMap& getAlphaMap() const;
///\fn const ColorMap& getColorMap() const
///\brief Returns the ColorMap
///\return A ColorMap
	const ColorMap& getColorMap() const;
///\fn const IsocontourMap& getIsocontourMap() const
///\brief Returns the IsocontourMap
///\return An IsocontourMap
	const IsocontourMap& getIsocontourMap() const;
///\fn const ConTreeMap& getConTreeMap() const
///\brief Returns the ConTreeMap
///\return A ConTreeMap
	const ConTreeMap& getConTreeMap() const;

///\fn AlphaMap& getAlphaMap()
///\brief Returns the AlphaMap
///\return An AlphaMap
	AlphaMap& getAlphaMap();
///\fn ColorMap& getColorMap()
///\brief Returns the ColorMap
///\return An ColorMap
	ColorMap& getColorMap();
///\fn IsocontourMap& getIsocontourMap()
///\brief Returns the IsocontourMap
///\return An IsocontourMap
	IsocontourMap& getIsocontourMap();
///\fn ConTreeMap& getConTreeMap()
///\brief Returns the ConTreeMap
///\return An ConTreeMap
	ConTreeMap& getConTreeMap();

///\fn bool saveColorTable(const QString& fileName)
///\brief Writes a .vinay file (minus the ConTreeMap)
///\param fileName A QString containing a path to a file
///\return A bool indicating success or failure
	bool saveColorTable(const QString& fileName);
///\fn bool loadColorTable(const QString& fileName)
///\brief Reads a .vinay file (minus the ConTreeMap)
///\param fileName A QString containing a path to a file
///\return A bool indicating success or failure
	bool loadColorTable(const QString& fileName);

protected:
	void removeSandwichedAlphaNodes();
	void removeSandwichedColorNodes();

	AlphaMap m_AlphaMap;
	ColorMap m_ColorMap;
	IsocontourMap m_IsocontourMap;
	ConTreeMap m_ConTreeMap;
};

#endif // !defined(AFX_COLORTABLEINFORMATION_H__890F2A7C_3D2F_4296_A3AA_C15CDB1217E3__INCLUDED_)

