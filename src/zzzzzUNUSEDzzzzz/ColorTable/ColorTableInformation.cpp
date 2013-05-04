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

// ColorTableInformation.cpp: implementation of the ColorTableInformation class.
//
//////////////////////////////////////////////////////////////////////

#include <ColorTable/ColorTableInformation.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qstring.h>
#include <qmessagebox.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ColorTableInformation::ColorTableInformation()
{
	// just uses the default constructors for color, alpha, isocontour and contree maps
}

ColorTableInformation::ColorTableInformation(const AlphaMap& alphaMap, const ColorMap& colorMap, const IsocontourMap& isocontourMap, const ConTreeMap& contreeMap)
: m_AlphaMap(alphaMap), m_ColorMap(colorMap), m_IsocontourMap(isocontourMap), m_ConTreeMap(contreeMap)
{
	removeSandwichedNodes();
}

ColorTableInformation::ColorTableInformation(const ColorTableInformation& copy)
: m_AlphaMap(copy.getAlphaMap()), m_ColorMap(copy.getColorMap()), m_IsocontourMap(copy.getIsocontourMap()), m_ConTreeMap(copy.getConTreeMap())
{

}

ColorTableInformation::~ColorTableInformation()
{

}

void ColorTableInformation::removeSandwichedNodes()
{
	removeSandwichedAlphaNodes();
	removeSandwichedColorNodes();
}

const AlphaMap& ColorTableInformation::getAlphaMap() const
{
	return m_AlphaMap;
}

const ColorMap& ColorTableInformation::getColorMap() const
{
	return m_ColorMap;
}

const IsocontourMap& ColorTableInformation::getIsocontourMap() const
{
	return m_IsocontourMap;
}

const ConTreeMap& ColorTableInformation::getConTreeMap() const
{
	return m_ConTreeMap;
}

AlphaMap& ColorTableInformation::getAlphaMap()
{
	return m_AlphaMap;
}

ColorMap& ColorTableInformation::getColorMap()
{
	return m_ColorMap;
}

IsocontourMap& ColorTableInformation::getIsocontourMap()
{
	return m_IsocontourMap;
}

ConTreeMap& ColorTableInformation::getConTreeMap()
{
	return m_ConTreeMap;
}

bool ColorTableInformation::saveColorTable(const QString& fileName)
{
	QFileInfo fileInfo(fileName);
	QString extension = fileInfo.extension();
	QString longName;

	// if no extension, add one
	if (extension.isEmpty()) {
		longName = fileName + ".vinay";
	}
	else {
		longName = fileName;
	}

	QFile file(longName);
	if (!file.open(IO_WriteOnly)) {
		return false;
	}
	QTextStream stream(&file);
	stream << "Anthony and Vinay are Great.\n";
	m_AlphaMap.saveMap(stream);
	m_ColorMap.saveMap(stream);
	m_IsocontourMap.saveMap(stream);
	//m_ConTreeMap.saveMap(stream); // not implemented
	file.close();

	return true;
}

bool ColorTableInformation::loadColorTable(const QString& fileName)
{
	QFile file(fileName);
	if (!file.open(IO_ReadOnly)) {
		return false;
	}
	QTextStream stream(&file);
	stream.skipWhiteSpace();
	QString verificationString = stream.readLine(); // Anthony and Vinay are Great.
	if (verificationString != "Anthony and Vinay are Great.") {
		file.close();
		return false;
	}
	else {
		m_AlphaMap.loadMap(stream);
		m_ColorMap.loadMap(stream);
		m_IsocontourMap.loadMap(stream);
		//m_ConTreeMap.loadMap(stream); // not implemented
		file.close();
		return true;
	}
}

void ColorTableInformation::removeSandwichedAlphaNodes()
{
	m_AlphaMap.removeSandwichedNodes();
}

void ColorTableInformation::removeSandwichedColorNodes()
{
	m_ColorMap.removeSandwichedNodes();
}

