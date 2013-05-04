/*
  Copyright 2002-2005 The University of Texas at Austin
  
    Authors: John Wiggins <prok@ices.utexas.edu>
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


/*
#include <ColorTable/ContourSpectrumInfo.h>
#include <q3frame.h>

ContourSpectrumInfo::ContourSpectrumInfo(QWidget *widget)
: QToolTip(widget)
//ContourSpectrumInfo::ContourSpectrumInfo(QWidget *widget)
{
	m_Head = NULL;
	m_RangeMin = 0.0;
	m_RangeMax = 1.0;
}

ContourSpectrumInfo::~ContourSpectrumInfo()
{
	if (m_Head) delete m_Head;
}

ContourSpectrumInfo::ConSpecInfoNode::ConSpecInfoNode(int id, double position, float isoval, float area, float minvol, float maxvol, ConSpecInfoNode* Next)
: m_Position(position), m_IsoVal(isoval), m_Area(area), m_MinVol(minvol),
	m_MaxVol(maxvol), m_pNext(Next), m_ID(id)
{
}

ContourSpectrumInfo::ConSpecInfoNode::ConSpecInfoNode(const ConSpecInfoNode& copy)
: m_Position(copy.m_Position), m_IsoVal(copy.m_IsoVal), m_Area(copy.m_Area),
	m_MinVol(copy.m_MinVol), m_MaxVol(copy.m_MaxVol), m_ID(copy.m_ID)
{
	if (copy.m_pNext)
		m_pNext = new ConSpecInfoNode(*(copy.m_pNext));
	else
		m_pNext = NULL;
}

ContourSpectrumInfo::ConSpecInfoNode::~ConSpecInfoNode()
{
	if (m_pNext) delete m_pNext;
}


int ContourSpectrumInfo::addNode(int id, double position, float isoval, float area, float minvol, float maxvol)
{
	if (position >= 0.0 && position <= 1.0)
	{
		// create the new node
		ConSpecInfoNode *node = new ConSpecInfoNode(id, position, isoval, area, minvol, maxvol, m_Head);
		// update the head of the list
		m_Head = node;
		return node->m_ID;
	}
	return -1;
}

void ContourSpectrumInfo::moveNode(int id, double position, float isoval,
float area, float minvol, float maxvol)
{
	// don't allow invalid positions
	if (position >= 0.0 && position <= 1.0)
	{
		ConSpecInfoNode *node = m_Head;
		while (node)
		{
			// find the right node
			if (node->m_ID == id)
			{
				// update
				node->m_Position = position;
				node->m_IsoVal = isoval;
				node->m_Area = area;
				node->m_MinVol = minvol;
				node->m_MaxVol = maxvol;
				// we're done
				break;
			}
			
			node = node->m_pNext;
		}
	}
}

void ContourSpectrumInfo::removeNode(int id)
{
	ConSpecInfoNode *node = m_Head, *prev = m_Head;
	while (node)
	{
		// find the right node
		if (node->m_ID == id)
		{
			// remove the node
			if (node == m_Head)
				// the new head is node->m_pNext
				m_Head = node->m_pNext;
			else
				prev->m_pNext = node->m_pNext;

			// don't delete the whole chain
			node->m_pNext = NULL;
			delete node;
			
			break;
		}
		
		prev = node;
		node = node->m_pNext;
	}
}

void ContourSpectrumInfo::setMin(double min)
{
	m_RangeMin = min;
}

void ContourSpectrumInfo::setMax(double max)
{
	m_RangeMax = max;
}

void ContourSpectrumInfo::maybeTip(const QPoint &point)
{
	if (m_Head)
	{

	  
		Q3Frame *parent = dynamic_cast<Q3Frame *>(parentWidget());

		if (parent)
		{
			ConSpecInfoNode *node = m_Head;
			QRect rect = parent->contentsRect();
			QRect myRect(rect.x()+10, rect.y()+5, rect.width()-20, rect.height()-10);
			int width = myRect.right()-myRect.left();
			int offset = point.x()-myRect.left();
			double normalizedPos;
			const double delta = 0.005;

			// get the normalized horizontal position of point
			// XoomedIn.cpp has this code spread between several functions
			normalizedPos = ((double)offset/(double)width)*(m_RangeMax-m_RangeMin) + m_RangeMin;
			
			// try to find a node with that position
			while (node)
			{
				if (node->m_Position > normalizedPos-delta
						&& node->m_Position < normalizedPos+delta)
				{
					// found it. call tip()
					tip(QRect(point.x()-1, point.y()-5, 3, 10),
							QString("Isovalue: %1\nSurface Area: %2\n"
											"Min. Volume: %3\nMax. Volume: %4\n")
											.arg(node->m_IsoVal)
											.arg(node->m_Area)
											.arg(node->m_MinVol)
											.arg(node->m_MaxVol));

					break;
				}
				node = node->m_pNext;
			}
		}
	  
	}
}
*/
