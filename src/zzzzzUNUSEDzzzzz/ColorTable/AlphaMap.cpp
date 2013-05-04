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

#include <ColorTable/AlphaMap.h>
#include <qfile.h>



AlphaMap::AlphaMap()
: m_Size(2)
{
	m_pHead = new AlphaMapNode(0.0, 0.0, 0, 0);
	m_pTail = new AlphaMapNode(1.0, 1.0, m_pHead, 0);
	m_pHead->m_pNext = m_pTail;
	AddNode(0.25, 0.75);
	AddNode(0.75, 0.25);
}

AlphaMap::AlphaMap(const AlphaMap& copy) : m_Size(copy.m_Size)
{
	if (copy.m_pHead) {
		m_pHead = new AlphaMapNode(*(copy.m_pHead), 0);
		AlphaMapNode* current = m_pHead;
		while (current->m_pNext)
			current = current->m_pNext;
		m_pTail = current;
	}
	else {
		m_pHead = 0;
		m_pTail = 0;
	}
}

AlphaMap& AlphaMap::operator=(const AlphaMap& copy)
{
	if (this!=&copy) {
		if (copy.m_pHead) {
			m_pHead = new AlphaMapNode(*(copy.m_pHead), 0);
			AlphaMapNode* current = m_pHead;
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

AlphaMap::~AlphaMap()
{
	delete m_pHead;
}

AlphaMap::AlphaMapNode::AlphaMapNode(double fPosition, double fAlpha, AlphaMap::AlphaMapNode* Prev, 
										AlphaMap::AlphaMapNode* Next) 
										: m_fPosition(fPosition),	m_fAlpha(fAlpha), m_pPrev(Prev), 
										m_pNext(Next)					
{

	if (m_pNext)
		m_pNext->m_pPrev = this;
	if (m_pPrev)
		m_pPrev->m_pNext = this;
}

AlphaMap::AlphaMapNode::AlphaMapNode(const AlphaMap::AlphaMapNode& copy, AlphaMap::AlphaMapNode* prev)
: m_pPrev (prev), m_fAlpha(copy.m_fAlpha), m_fPosition(copy.m_fPosition)
{
	if (copy.m_pNext)
		m_pNext = new AlphaMapNode(*(copy.m_pNext), this);
	else 
		m_pNext = 0;
}

AlphaMap::AlphaMapNode::~AlphaMapNode()
{
	delete m_pNext;
}



void AlphaMap::AddNode(double fPosition, double fAlpha)
{
	if (fPosition>=0.0 && fPosition<=1.0) {
		AlphaMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode) {
			new AlphaMapNode(fPosition, fAlpha, pNode->m_pPrev, pNode);
			m_Size++;
		}
	}
}

void AlphaMap::AddNode(double fPosition)
{
	if (fPosition>=0.0 && fPosition<=1.0) {
		AlphaMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode) {
			if (fPosition<pNode->m_fPosition)
			{
				double f1, f2, a;
				double cr1, cr2;
				f1 = pNode->m_pPrev->m_fPosition;
				f2 = pNode->m_fPosition;
				cr1 = pNode->m_pPrev->m_fAlpha;
				cr2 = pNode->m_fAlpha;

				a = (fPosition-f1)/(f2-f1);
				double fAlpha;
				fAlpha = cr1 * (1.0f-a) + cr2 * a;
				new AlphaMapNode(fPosition, fAlpha, pNode->m_pPrev, pNode);
				m_Size++;
			}
		}
	}
}

void AlphaMap::DeleteNode(int index)
{
	if (index < m_Size-1 && index > 0) {
		AlphaMapNode* pNode;
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

void AlphaMap::MoveNode(int index, double fPosition)
{
	if (index<m_Size-1 && index>0) {
		AlphaMapNode* pNode;
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

int AlphaMap::GetSize() const
{
	return m_Size;
}

double AlphaMap::GetPosition(int index) const
{
	if (index < m_Size) {
		AlphaMapNode* pNode;
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

double AlphaMap::GetAlpha(int index) const
{
	if (index < m_Size) {
		AlphaMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_fAlpha;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0;
	}
}

void AlphaMap::ChangeAlpha(int index, double fAlpha)
{
	if (index <= m_Size-1 && index >= 0) {
		AlphaMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			if (fAlpha >= 1.0) fAlpha = 1.0;
			if (fAlpha <= 0.0) fAlpha = 0.0;
			pNode->m_fAlpha = fAlpha;
		}
	}

}

double AlphaMap::GetAlpha(double fPosition) const
{
	if (fPosition>=0.0 && fPosition<=1.0) {
		AlphaMapNode* pNode;
		pNode = m_pHead;
		while (pNode && fPosition>pNode->m_fPosition) {
			pNode = pNode->m_pNext;
		}

		if (pNode == m_pHead) {
			return pNode->m_fAlpha;
		}
		else if (pNode) {
			double f1, f2, a;
			double cr1, cr2;
			f1 = pNode->m_pPrev->m_fPosition;
			f2 = pNode->m_fPosition;
			cr1 = pNode->m_pPrev->m_fAlpha;
			cr2 = pNode->m_fAlpha;

			a = (fPosition-f1)/(f2-f1);
			double fAlpha;
			fAlpha = cr1 * (1.0f-a) + cr2 * a;
			return fAlpha;
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
void AlphaMap::saveMap( QTextStream& stream )
{
	AlphaMapNode* alphaMapNode;

	stream << "Alphamap\n";
	stream << "Number of nodes\n";
	stream << m_Size << "\n";
	
	stream << "Position and opacity\n";

	stream << m_pHead->m_fPosition << " " << m_pHead->m_fAlpha << "\n";
	stream << m_pTail->m_fPosition << " "  << m_pTail->m_fAlpha << "\n";

	alphaMapNode = m_pHead->m_pNext;

	for( int i=1;i<m_Size-1; i++ )
	{
		stream << alphaMapNode->m_fPosition << " "  << alphaMapNode->m_fAlpha << "\n";
		alphaMapNode = alphaMapNode->m_pNext;
	}
}

void AlphaMap::loadMap( QTextStream& stream )
{
	AlphaMapNode* pHead, *pTail, *node;
	double position, alpha;
	int size;
	QString junk;



	stream.skipWhiteSpace();
	junk = stream.readLine(); // Alphamap

	stream.skipWhiteSpace();
	junk = stream.readLine(); // Number of nodes
	stream >> size;

	stream.skipWhiteSpace(); // end of size line
	junk = stream.readLine(); // Position and opacity
	stream >> position >> alpha;
	pHead = new AlphaMapNode(position, alpha, 0, 0);

	stream >> position >> alpha;
	pTail = new AlphaMapNode(position, alpha, 0, 0);

	node = pHead;

	for( int i=1;i<size-1; i++ ) {
		stream >> position >> alpha;

		node->m_pNext = new AlphaMapNode(position, alpha, node, 0);
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

void AlphaMap::removeSandwichedNodes()
{
	// when there are 3 or more nodes with the same position
	// remove the ones in the middle

	if (m_Size<3) { // with size < 3, cant have sandwiched nodes
		return;
	}

	AlphaMapNode* pNode = m_pHead->m_pNext;
	AlphaMapNode* pToDelete;

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

