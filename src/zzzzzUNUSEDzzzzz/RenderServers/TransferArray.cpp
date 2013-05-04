/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
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

// TransferArray.cpp: implementation of the TransferArray class.
//
//////////////////////////////////////////////////////////////////////

#include <RenderServers/TransferArray.h>
#include <ColorTable/ColorTableInformation.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

TransferArray::TransferArray(unsigned int arraySize)
{
	setDefaults();
	allocateArray(arraySize);
}

TransferArray::~TransferArray()
{
	destroyArray();
}

void TransferArray::addElement(const TransferElement& transferElement)
{
	if (m_NumElements==m_ArraySize) {
		doubleArray();
	}
	m_Array[m_NumElements] = transferElement;
	m_NumElements++;
}

void TransferArray::buildFromColorTable(const ColorTableInformation& colorTableInformation)
{
	destroyArray();
	allocateArray(16);

	const ColorMap& colorMap = colorTableInformation.getColorMap();
	const AlphaMap& alphaMap = colorTableInformation.getAlphaMap();

	// check that the tables have at least 2 nodes each
	// This assert will never fail normally
	Q_ASSERT(colorMap.GetSize() >= 2 && 
		alphaMap.GetSize() >= 2);

	// Make sure that the first elements are at the same position
	// This assert will never fail normally
	Q_ASSERT(colorMap.GetPosition(0) == alphaMap.GetPosition(0));

	int indexAlpha=1, indexColor=1;
	double posAlpha, posColor;

	// Do the first node:
	addElement(colorMap.GetPosition(0), alphaMap.GetAlpha(0),
		colorMap.GetRed(0), colorMap.GetGreen(0), colorMap.GetBlue(0));

	// loop while each array still has elements
	while (indexAlpha<alphaMap.GetSize() || indexColor<colorMap.GetSize()) {
		posAlpha = alphaMap.GetPosition(indexAlpha);
		posColor = colorMap.GetPosition(indexColor);
		if (posAlpha==posColor) { // combine into one node
			addElement(posAlpha, alphaMap.GetAlpha(indexAlpha),
				colorMap.GetRed(indexColor), colorMap.GetGreen(indexColor), colorMap.GetBlue(indexColor));
			indexAlpha++;
			indexColor++;
		}
		else if (posAlpha < posColor) { // add the next alpha node to the array
			addElement(posAlpha, alphaMap.GetAlpha(indexAlpha),
				colorMap.GetRed(posAlpha), colorMap.GetGreen(posAlpha), colorMap.GetBlue(posAlpha));
			indexAlpha++;
		}
		else { // add the next color node to the array
			addElement(posColor, alphaMap.GetAlpha(posColor),
				colorMap.GetRed(indexColor), colorMap.GetGreen(indexColor), colorMap.GetBlue(indexColor));
			indexColor++;
		}
	}

	// the loop should end with indexAlpha==alphaMap.GetSize()-1 && indexColor==colorMap.GetSize()-1
	Q_ASSERT(indexAlpha==alphaMap.GetSize() && indexColor==colorMap.GetSize());
}

void TransferArray::addElement(double position, double alpha, double red, double green, double blue)
{
	addElement(TransferElement(position, alpha, red, green, blue));
}

unsigned int TransferArray::getNumElements() const
{
	return m_NumElements;
}

TransferArray::TransferElement& TransferArray::operator[](unsigned int index)
{
	return m_Array[index];
}

const TransferArray::TransferElement& TransferArray::operator[](unsigned int index) const
{
	return m_Array[index];
}

void TransferArray::setDefaults()
{
	m_Array = 0;
	m_ArraySize = 0;
	m_NumElements = 0;
}

void TransferArray::destroyArray()
{
	delete [] m_Array;
	setDefaults();
}

void TransferArray::allocateArray(unsigned int arraySize)
{
	destroyArray();
	m_Array = new TransferElement[arraySize];
	m_ArraySize = arraySize;
}

void TransferArray::doubleArray()
{
	TransferElement* newArray = new TransferElement[m_ArraySize*2];
	unsigned int c;
	for (c=0; c<m_NumElements; c++) {
		newArray[c] = m_Array[c];
	}
	delete [] m_Array;
	m_Array = newArray;
	m_ArraySize*=2;
}

