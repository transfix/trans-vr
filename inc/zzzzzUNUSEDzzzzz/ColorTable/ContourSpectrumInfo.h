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

#ifndef CONTOUR_SPECTRUM_INFO
#define CONTOUR_SPECTRUM_INFO

//#include <qtooltip.h>


/*

#include <QToolTip>

///\class ContourSpectrumInfo ContourSpectrumInfo.h
///\author John Wiggins
///\brief ContourSpectrumInfo handles the display of tool-tips whenever the
/// contour spectrum is displayed.
class ContourSpectrumInfo : public QToolTip
{
	public:
///\fn ContourSpectrumInfo(QWidget *widget);
///\brief The constructor
///\param widget A QWidget to attach to and monitor for tool-tip events
		ContourSpectrumInfo(QWidget *widget);
		virtual ~ContourSpectrumInfo();

///\fn int addNode(int id, double position, float isoval, float area, float minvol, float maxvol);
///\brief Called whenever an isocontour node is added to the transfer function
///\param id The id of the node
///\param position The position of the node
///\param isoval The isovalue of the node
///\param area The surface area of the isosurface
///\param minvol The minimum volume of the isosurface
///\param maxvol The maximum volume of the isosurface
///\return The id of the node or -1 if there was an error
		int addNode(int id, double position, float isoval, float area, float
minvol, float maxvol);
///\fn void moveNode(int id, double position, float isoval, float area, float minvol, float maxvol);
///\brief Called whenever an isocontour node is moved
///\param id The id of the node
///\param position The new position of the node
///\param isoval The new isovalue of the node
///\param area The new surface area of the isosurface
///\param minvol The new minimum volume of the isosurface
///\param maxvol The new maximum volume of the isosurface
		void moveNode(int id, double position, float isoval, float area, float
minvol, float maxvol);
///\fn void removeNode(int id);
///\brief Called when an isocontour node is removed
///\param id The id of the node being removed
		void removeNode(int id);

///\fn void setMin(double min);
///\brief Sets the minimum function value for the current dataset. This is used
/// to display the correct isovalue.
///\param min The minimum value
		void setMin(double min);
///\fn void setMax(double max);
///\brief Sets the maximum function value for the current dataset. This is used
/// to display the correct isovalue.
///\param max The maximum value
		void setMax(double max);

		virtual void maybeTip(const QPoint &);

	private:

///\class ConSpecInfoNode
///\brief ConSpecInfoNode holds the contour spectrum info for each isocontour
/// node in the ColorTable widget.
		class ConSpecInfoNode {
			public:
				ConSpecInfoNode(int id, double position, float isoval, float area, float minvol, float maxvol, ConSpecInfoNode* Next);
				ConSpecInfoNode(const ConSpecInfoNode& copy);
				virtual ~ConSpecInfoNode();
				ConSpecInfoNode* m_pNext;
				double m_Position;
				float m_IsoVal;
				float m_Area;
				float m_MinVol;
				float m_MaxVol;
				int m_ID;
		};

		ConSpecInfoNode *m_Head;
		double m_RangeMin, m_RangeMax;
};

*/

#endif

