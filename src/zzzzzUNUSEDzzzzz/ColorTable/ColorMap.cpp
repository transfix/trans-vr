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

#include <ColorTable/ColorMap.h>
#include <qfile.h>


ColorMap::ColorMap()
: m_Size(2)
{
	m_pHead = new ColorMapNode(0.0, 0.0, 0.0, 0.0, 0, 0);
	m_pTail = new ColorMapNode(1.0, 0.0, 1.0, 0.0, m_pHead, 0);
	m_pHead->m_pNext = m_pTail;
	AddNode(0.5, 1.0, 0.0, 0.0);
	AddNode(0.75, 1.0, 1.0, 0.0);
}

ColorMap::ColorMap(const ColorMap& copy) : m_Size(copy.m_Size)
{
	if (copy.m_pHead) {
		m_pHead = new ColorMapNode(*(copy.m_pHead), 0);
		ColorMapNode* current = m_pHead;
		while (current->m_pNext)
			current = current->m_pNext;
		m_pTail = current;
	}
	else {
		m_pHead = 0;
		m_pTail = 0;
	}
}

ColorMap& ColorMap::operator=(const ColorMap& copy)
{
	if (this!=&copy) {
		if (copy.m_pHead) {
			m_pHead = new ColorMapNode(*(copy.m_pHead), 0);
			ColorMapNode* current = m_pHead;
			while (current->m_pNext)
				current = current->m_pNext;
			m_pTail = current;
		}
		else {
			m_pHead = 0;
			m_pTail = 0;
		}
	}
	return *this;
}

ColorMap::~ColorMap()
{
	delete m_pHead;
}

ColorMap::ColorMapNode::ColorMapNode(double fPosition, double fRed, double fGreen, double fBlue, ColorMap::ColorMapNode* Prev, 
										ColorMap::ColorMapNode* Next) 
										: m_fPosition(fPosition),	m_fRed(fRed), m_fGreen(fGreen), m_fBlue(fBlue), m_pPrev(Prev), 
										m_pNext(Next)					
{

	if (m_pNext)
		m_pNext->m_pPrev = this;
	if (m_pPrev)
		m_pPrev->m_pNext = this;
}

ColorMap::ColorMapNode::ColorMapNode(const ColorMap::ColorMapNode& copy, ColorMap::ColorMapNode* prev)
: m_pPrev (prev), m_fRed(copy.m_fRed), m_fGreen(copy.m_fGreen), m_fBlue(copy.m_fBlue), m_fPosition(copy.m_fPosition)
{
	if (copy.m_pNext)
		m_pNext = new ColorMapNode(*(copy.m_pNext), this);
	else 
		m_pNext = 0;
}

ColorMap::ColorMapNode::~ColorMapNode()
{
	delete m_pNext;
}



void ColorMap::AddNode(double fPosition, double fRed, double fGreen, double fBlue)
{
	if (fPosition>=0.0 && fPosition<=1.0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode) {
			new ColorMapNode(fPosition, fRed, fGreen, fBlue, pNode->m_pPrev, pNode);
			m_Size++;
		}
	}
}

void ColorMap::AddNode(double fPosition)
{
	if (fPosition==0)  {
		ColorMapNode* newNode = new ColorMapNode(fPosition, m_pHead->m_fRed, m_pHead->m_fGreen, m_pHead->m_fBlue, m_pHead, m_pHead->m_pNext);
		m_Size++;
	}
	 else if (fPosition>0.0 && fPosition<=1.0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode) {
			if (fPosition<pNode->m_fPosition)
			{
				double f1, f2, a;
				double cr1, cr2, cg1, cg2, cb1, cb2;
				f1 = pNode->m_pPrev->m_fPosition;
				f2 = pNode->m_fPosition;
				cr1 = pNode->m_pPrev->m_fRed;
				cg1 = pNode->m_pPrev->m_fGreen;
				cb1 = pNode->m_pPrev->m_fBlue;
				cr2 = pNode->m_fRed;
				cg2 = pNode->m_fGreen;
				cb2 = pNode->m_fBlue;

				a = (fPosition-f1)/(f2-f1);
				double fRed, fGreen, fBlue;
				fRed = cr1 * (1.0f-a) + cr2 * a;
				fGreen = cg1 * (1.0f-a) + cg2 * a;
				fBlue = cb1 * (1.0f-a) + cb2 * a;
				new ColorMapNode(fPosition, fRed, fGreen, fBlue, pNode->m_pPrev, pNode);
				m_Size++;
			}
		}
	}
}

void ColorMap::DeleteNode(int index)
{
	if (index < m_Size-1 && index > 0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			pNode->m_pPrev->m_pNext = pNode->m_pNext;
			pNode->m_pNext->m_pPrev = pNode->m_pPrev;
			pNode->m_pPrev = 0;
			pNode->m_pNext = 0;
			delete pNode;
			m_Size--;
		}
	}
}

void ColorMap::MoveNode(int index, double fPosition)
{
	if (index<m_Size-1 && index>0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			pNode->m_fPosition = fPosition;
			if (pNode->m_fPosition < pNode->m_pPrev->m_fPosition) {
				pNode->m_fPosition = pNode->m_pPrev->m_fPosition;
			}
			if (pNode->m_fPosition > pNode->m_pNext->m_fPosition) {
				pNode->m_fPosition = pNode->m_pNext->m_fPosition;
			}
		}
	}
}

int ColorMap::GetSize() const
{
	return m_Size;
}

double ColorMap::GetPosition(int index) const
{
	if (index < m_Size) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_fPosition;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0.0;
	}
}

double ColorMap::GetRed(int index) const
{
	if (index < m_Size) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_fRed;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0;
	}
}

double ColorMap::GetGreen(int index) const
{
	if (index < m_Size) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_fGreen;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0;
	}
}

double ColorMap::GetBlue(int index) const
{
	if (index < m_Size) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_fBlue;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0;
	}
}

void ColorMap::ChangeColor(int index, double fRed,  double fGreen,  double fBlue)
{
	if (index <= m_Size-1 && index >= 0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			if (fRed >= 1.0) fRed = 1.0;
			if (fRed <= 0.0) fRed = 0.0;
			pNode->m_fRed = fRed;
			if (fGreen >= 1.0) fGreen = 1.0;
			if (fGreen <= 0.0) fGreen = 0.0;
			pNode->m_fGreen = fGreen;
			if (fBlue >= 1.0) fBlue = 1.0;
			if (fBlue <= 0.0) fBlue = 0.0;
			pNode->m_fBlue = fBlue;
		}
	}

}

double ColorMap::GetRed(double fPosition) const
{
	if (fPosition==0)  {
		return m_pHead->m_fRed;
	}
	else if (fPosition>0.0 && fPosition<=1.0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode) {
			double f1, f2, a;
			double cr1, cr2;
			f1 = pNode->m_pPrev->m_fPosition;
			f2 = pNode->m_fPosition;
			cr1 = pNode->m_pPrev->m_fRed;
			cr2 = pNode->m_fRed;

			a = (fPosition-f1)/(f2-f1);
			double fRed;
			fRed = cr1 * (1.0f-a) + cr2 * a;
			return fRed;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0.0;
	}
}

double ColorMap::GetGreen(double fPosition) const
{
	if (fPosition==0)  {
		return m_pHead->m_fGreen;
	}
	else if (fPosition>0.0 && fPosition<=1.0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode) {
			double f1, f2, a;
			double cr1, cr2;
			f1 = pNode->m_pPrev->m_fPosition;
			f2 = pNode->m_fPosition;
			cr1 = pNode->m_pPrev->m_fGreen;
			cr2 = pNode->m_fGreen;

			a = (fPosition-f1)/(f2-f1);
			double fGreen;
			fGreen = cr1 * (1.0f-a) + cr2 * a;
			return fGreen;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0.0;
	}
}

double ColorMap::GetBlue(double fPosition) const
{
	if (fPosition==0)  {
		return m_pHead->m_fBlue;
	}
	else if (fPosition>0.0 && fPosition<=1.0) {
		ColorMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode) {
			double f1, f2, a;
			double cr1, cr2;
			f1 = pNode->m_pPrev->m_fPosition;
			f2 = pNode->m_fPosition;
			cr1 = pNode->m_pPrev->m_fBlue;
			cr2 = pNode->m_fBlue;

			a = (fPosition-f1)/(f2-f1);
			double fBlue;
			fBlue = cr1 * (1.0f-a) + cr2 * a;
			return fBlue;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0.0;
	}
}

/********************

Alphamap
No of nodes
< no of nodes >
Position and opacity
< Position A1 >
< Position AN >
< Position A2 >
< Position A3 >
...
< Position AN-1 >

*********************/
void ColorMap::saveMap( QTextStream& stream )
{
	ColorMapNode* colorMapNode;

	stream << "ColorMap\n";
	stream << "Number of nodes\n";
	stream << m_Size << "\n";
	
	stream << "Position and RGB\n";

	stream << m_pHead->m_fPosition << " "  << m_pHead->m_fRed << " "  << m_pHead->m_fGreen << " "  << m_pHead->m_fBlue<< "\n";
	stream << m_pTail->m_fPosition << " "  << m_pTail->m_fRed << " "  << m_pTail->m_fGreen << " "  << m_pTail->m_fBlue<< "\n";

	colorMapNode = m_pHead->m_pNext;

	for( int i=1;i<m_Size-1; i++ )
	{
		stream << colorMapNode->m_fPosition << " "  << colorMapNode->m_fRed << " "  << colorMapNode->m_fGreen << " "  << colorMapNode->m_fBlue << "\n";
		colorMapNode = colorMapNode->m_pNext;
	}
}

void ColorMap::loadMap( QTextStream& stream )
{
	ColorMapNode* pHead, *pTail, *node;
	double position, red, green, blue;
	int size;
	QString junk;



	stream.skipWhiteSpace();
	junk = stream.readLine(); // ColorMap

	stream.skipWhiteSpace();
	junk = stream.readLine(); // Number of nodes
	stream >> size;

	stream.skipWhiteSpace();
	junk = stream.readLine(); // Position and R G B
	stream >> position >> red >> green >> blue;
	pHead = new ColorMapNode(position, red, green, blue, 0, 0);

	stream >> position >> red >> green >> blue;
	pTail = new ColorMapNode(position, red, green, blue, 0, 0);

	node = pHead;

	for( int i=1;i<size-1; i++ ) {
		stream >> position >> red >> green >> blue;

		node->m_pNext = new ColorMapNode(position, red, green, blue, node, 0);
		node = node->m_pNext;

	}
	node->m_pNext = pTail;
	pTail->m_pPrev = node;


	// if successful, replace lists
	delete m_pHead;
	m_pHead = pHead;
	m_pTail = pTail;
	m_Size = size;

}

void ColorMap::removeSandwichedNodes()
{
	// when there are 3 or more nodes with the same position
	// remove the ones in the middle

	if (m_Size<3) { // with size < 3, cant have sandwiched nodes
		return;
	}

	ColorMapNode* pNode = m_pHead->m_pNext;
	ColorMapNode* pToDelete;

	// go to the second to last node
	while (pNode->m_pNext) {
		// if current position is equal to the previous and last
		if ((pNode->m_pPrev->m_fPosition==pNode->m_fPosition) &&
			(pNode->m_pNext->m_fPosition==pNode->m_fPosition)) {
			// remove the node
			pToDelete = pNode;
			pNode = pNode->m_pNext;
			pToDelete->m_pPrev->m_pNext = pToDelete->m_pNext;
			pToDelete->m_pNext->m_pPrev = pToDelete->m_pPrev;
			pToDelete->m_pPrev = 0;
			pToDelete->m_pNext = 0;
			delete pToDelete;
			m_Size--;
		}
		else { // next node
			pNode = pNode->m_pNext;
		}
	}
}

