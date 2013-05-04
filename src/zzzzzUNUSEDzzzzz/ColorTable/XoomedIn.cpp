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

#include <ColorTable/XoomedIn.h>
#include <stdlib.h>
#include <math.h>
#include <q3popupmenu.h>
#include <qfile.h>
#include <q3filedialog.h>
#include <qmessagebox.h>
#include <qcolordialog.h>
//Added by qt3to4:
#include <QContextMenuEvent>
#include <Q3PointArray>
#include <QPixmap>
#include <Q3Frame>
#include <QMouseEvent>


const bool HorizontalMap = true;
const bool InvertDensity = false;
const bool InvertAlpha = true;

/*
const bool HorizontalMap = false;
const bool InvertDensity = false;
const bool InvertAlpha = false;
*/

const int CLICK_RANGE = 5;
const int ALPHA_NODE_SIZE = 5;

XoomedIn::XoomedIn( ColorTableInformation* colorTableInformation, QWidget *parent, const char *name )
: QFrame( parent, name )
{
	setFrameStyle( QFrame::Panel | QFrame::Sunken );
	m_ColorTableInformation = colorTableInformation;
	//m_ConSpecInfo = new ContourSpectrumInfo(this);
	m_RangeMin = 0.0;
	m_RangeMax = 1.0;
	m_IsoMin = 0.0;
	m_IsoMax = 1.0;
	m_MoveMode = NO_MOVE;
	
	for (int i=0; i < 5; i++)
	  m_SpecFuncs[i] = 0;
	
	m_SelectedCTEdge = -1;
	m_SelectedIsovalue = 0.;

	m_DrawContourSpec = false;
	m_DrawContourTree = false;
	m_DrawAlphaMap = true;

	setBackgroundMode( Qt::NoBackground );
}

XoomedIn::~XoomedIn()
{
  // clean up any contour spectrum data
  for (int i=0; i < 5; i++) {
    if (m_SpecFuncs[i]) {
      free(m_SpecFuncs[i]);
      m_SpecFuncs[i] = 0;
    }
  }

  //delete m_ConSpecInfo;
}

QRect XoomedIn::getMyRect() const
{
	QRect rect = contentsRect();
	//return rect;
	return QRect(rect.x()+10, rect.y()+5, rect.width()-20, rect.height()-10);
	
}

/*double XoomedIn::transformFromScreenX(const QRect& rect, int x) const
{
	return m_RangeMin + ((x / (double)rect.width()) * (m_RangeMax-m_RangeMin));
}

double XoomedIn::transformFromScreenY(const QRect& rect, int y) const
{
	return (y / (double)rect.height());
}*/

QPoint XoomedIn::transformToScreen(const QRect& rect, double density, double alpha) const
{
	return transformToScreen(rect,
		getIntDensity(rect, density),
		getIntAlpha(rect, alpha)
		);
}

QPoint XoomedIn::transformToScreen(const QRect& rect, int density, int alpha) const
{
	if (HorizontalMap) {
		if (InvertDensity) density = possibleFlip(density, rect.left(), rect.right());
		if (InvertAlpha) alpha = possibleFlip(alpha, rect.top(), rect.bottom());
		return QPoint(
			density,
			alpha
			);
	}
	else {
		if (InvertDensity) density = possibleFlip(density, rect.top(), rect.bottom());
		if (InvertAlpha) alpha = possibleFlip(alpha, rect.left(), rect.right());
		return QPoint(
			alpha,
			density
			);
	}
}

int XoomedIn::getIntDensity(const QRect& rect, double density) const
{
	if (HorizontalMap) {
		return mapToPixel(density, rect.left(), rect.right());
	}
	else {
		return mapToPixel(density, rect.top(), rect.bottom());
	}
}

int XoomedIn::getIntAlpha(const QRect& rect, double alpha) const
{
	if (HorizontalMap) {
		return mapToPixel(alpha, rect.top(), rect.bottom());
	}
	else {
		return mapToPixel(alpha, rect.left(), rect.right());
	}
}

double XoomedIn::getNormalizedDensity(const QRect& rect, const QPoint& source) const
{
	if (HorizontalMap) {
		return mapToDouble(getIntDensity(rect, source), rect.left(), rect.right(), m_RangeMin, m_RangeMax);
	}
	else {
		return mapToDouble(getIntDensity(rect, source), rect.top(), rect.bottom(), m_RangeMin, m_RangeMax);
	}
}

double XoomedIn::getNormalizedAlpha(const QRect& rect, const QPoint& source) const
{
	if (HorizontalMap) {
		return mapToDouble(getIntAlpha(rect, source), rect.top(), rect.bottom(), 0.0, 1.0);
	}
	else {
		return mapToDouble(getIntAlpha(rect, source), rect.left(), rect.right(), 0.0, 1.0);
	}
}

int XoomedIn::getIntDensity(const QRect& rect, const QPoint& source) const
{
	int density;
	int min, max;
	if (HorizontalMap) {
		density = source.x();
		min = rect.left(); max = rect.right();
	}
	else {
		density = source.y();
		min = rect.top(); max = rect.bottom();
	}
	// flip if necessary
	density = (InvertDensity?possibleFlip(density,min, max) :density);
	return density;
}

int XoomedIn::getIntAlpha(const QRect& rect, const QPoint& source) const
{
	int alpha;
	int min, max;
	if (HorizontalMap) {
		alpha = source.y();
		min = rect.top(); max = rect.bottom();
	}
	else {
		alpha = source.x();
		min = rect.left(); max = rect.right();
	}
	// flip if necessary
	alpha = (InvertAlpha?possibleFlip(alpha,min, max) :alpha);
	return alpha;
}

inline int XoomedIn::mapToPixel(double input, int start, int end) const
{
	if (input <= 1.0 && input >= 0.0) {
		int width = end-start;
		int offset = (int)(input * width);
		return start + offset;
	}
	else if (input > 1.0) {
		int width = end-start;
		int offset = (int)(1.0f * width);
		return start + offset;
	}
	else {
		int width = end-start;
		int offset = (int)(0.0f * width);
		return start + offset;
	}
}

inline double XoomedIn::mapToDouble(int input, int start, int end, double min, double max) const
{
	int width = end-start;
	int offset = input-start;

	return ((double)offset/(double)width)*(max-min) + min;
}

inline int XoomedIn::possibleFlip(int input, int start, int end) const
{
	return end -  input + start;
}

inline void XoomedIn::normalizeFunc( float *func, int len,
                                     float & min, float & max)
{
	//float min=func[0], max=func[0];
	int i;

	min = max = func[0];

	// find the min and max
	for (i=0; i<len; i++) {
		if (func[i] < min) min = func[i];
		if (func[i] > max) max = func[i];
	}

	// normalize the values
	for (i=0; i<len; i++)
		func[i] = (func[i]-min) / (max-min);
}

inline float XoomedIn::getSpecFuncValue( int func, double density )
{
	float val = 0.0;

	if (func == 0) // special case for isovalue
		val = density*(m_IsoMax-m_IsoMin)+m_IsoMin;
	else if (func > 0 && func < 5)
		val = m_SpecFuncs[func][(int)(255*density)]*(m_SpecFuncsMax[func]-m_SpecFuncsMin[func])+m_SpecFuncsMin[func];

	return val;
}

void XoomedIn::drawContents( QPainter* painter )
{
	//drawContentsHorizontal( painter );

	QRect rect = contentsRect();
	QRect myRect = getMyRect();

	QPixmap  pix( rect.size() );       // Pixmap for double-buffering

	QColor test = paletteBackgroundColor();

	pix.fill( test ); //this, rect.topLeft() );  // fill with widget background

	QPainter p( &pix );
	
	int i;
	drawColorMap(&p, myRect);
	if (m_DrawContourSpec)
		drawContourSpectrum(&p, myRect);
	if (m_DrawContourTree) {
		drawContourTree(&p, myRect);
		for (i=0; i<m_ColorTableInformation->getConTreeMap().GetSize(); i++) {
			drawConTreeNode(&p, myRect, i);
		}
	}
	
	for (i=0; i<m_ColorTableInformation->getColorMap().GetSize(); i++) {
		drawColorBar(&p, myRect, i);
	}

	for (i=0; i<m_ColorTableInformation->getIsocontourMap().GetSize(); i++) {
		drawIsocontourBar(&p, myRect, i);
	}

	if (m_DrawAlphaMap) {
		drawAlphaMap(&p, myRect);
		for (i=0; i<m_ColorTableInformation->getAlphaMap().GetSize(); i++) {
			drawAlphaNode(&p, myRect, i);
		}
	}

	bitBlt( this, rect.topLeft(), &pix );
	p.end();
	
}

void XoomedIn::drawColorMap( QPainter* painter, QRect rect )
{
	double mag;
	int pos1, pos2;
	double doublePos1, doublePos2, clampedDoublePos1, clampedDoublePos2;
	double cr1, cr2;
	double cg1, cg2;
	double cb1, cb2;

	int minAlpha = getIntAlpha(rect, 0.0);
	int maxAlpha = getIntAlpha(rect, 1.0);

	int mapSize = m_ColorTableInformation->getColorMap().GetSize();
	for (int i=0; i<mapSize-1; i++) {
		doublePos1 = m_ColorTableInformation->getColorMap().GetPosition(i+0);
		doublePos2 = m_ColorTableInformation->getColorMap().GetPosition(i+1);

		if( doublePos2 < m_RangeMin ) continue;
		if( doublePos1 > m_RangeMax ) break;

		if( doublePos1 < m_RangeMin ) {
			mag = (m_RangeMin-doublePos1)/(doublePos2-doublePos1);
			cr1 = m_ColorTableInformation->getColorMap().GetRed(i) * (1.0-mag) + m_ColorTableInformation->getColorMap().GetRed(i+1) * (mag);
			cg1 = m_ColorTableInformation->getColorMap().GetGreen(i) * (1.0-mag) + m_ColorTableInformation->getColorMap().GetGreen(i+1) * (mag);
			cb1 = m_ColorTableInformation->getColorMap().GetBlue(i) * (1.0-mag) + m_ColorTableInformation->getColorMap().GetBlue(i+1) * (mag);
			clampedDoublePos1 = m_RangeMin;
		}
		else {
			cr1 = m_ColorTableInformation->getColorMap().GetRed(i);
			cg1 = m_ColorTableInformation->getColorMap().GetGreen(i);
			cb1 = m_ColorTableInformation->getColorMap().GetBlue(i);
			clampedDoublePos1 = doublePos1;
		}
		if( doublePos2 > m_RangeMax ) {
			mag = (m_RangeMax-doublePos1)/(doublePos2-doublePos1);
			cr2 = m_ColorTableInformation->getColorMap().GetRed(i) * (1.0-mag) + m_ColorTableInformation->getColorMap().GetRed(i+1) * (mag);
			cg2 = m_ColorTableInformation->getColorMap().GetGreen(i) * (1.0-mag) + m_ColorTableInformation->getColorMap().GetGreen(i+1) * (mag);
			cb2 = m_ColorTableInformation->getColorMap().GetBlue(i) * (1.0-mag) + m_ColorTableInformation->getColorMap().GetBlue(i+1) * (mag);
			clampedDoublePos2 = m_RangeMax;
		}
		else {
			cr2 = m_ColorTableInformation->getColorMap().GetRed(i+1);
			cg2 = m_ColorTableInformation->getColorMap().GetGreen(i+1);
			cb2 = m_ColorTableInformation->getColorMap().GetBlue(i+1);
			clampedDoublePos2 = doublePos2;
		}

		pos1 = getIntDensity(rect, (float)((clampedDoublePos1-m_RangeMin)/(m_RangeMax-m_RangeMin)));
		pos2 = getIntDensity(rect, (float)((clampedDoublePos2-m_RangeMin)/(m_RangeMax-m_RangeMin)));
		for (int y=pos1; y<=pos2; y++) {
			mag = mapToDouble(y, pos1, pos2);
			QColor color( mapToPixel(cr1, 0, 255)*(1.0-mag) + mapToPixel(cr2, 0, 255)*(mag),
				mapToPixel(cg1, 0, 255)*(1.0-mag) + mapToPixel(cg2, 0, 255)*(mag),
				mapToPixel(cb1, 0, 255)*(1.0-mag) + mapToPixel(cb2, 0, 255)*(mag));

			painter->setPen(color);
			QPoint p1 = transformToScreen(rect,y,minAlpha);
			QPoint p2 = transformToScreen(rect,y,maxAlpha);
			painter->drawLine(p1, p2);
		}
	}
}

void XoomedIn::drawColorBar( QPainter* painter, QRect rect, int index )
{
	double doublePos = m_ColorTableInformation->getColorMap().GetPosition(index);
	QColor color(mapToPixel(1.0-m_ColorTableInformation->getColorMap().GetRed(index), 0, 255),
		mapToPixel(1.0-m_ColorTableInformation->getColorMap().GetGreen(index), 0, 255),
		mapToPixel(1.0-m_ColorTableInformation->getColorMap().GetBlue(index), 0, 255));		
		
	int pos;
	int minAlpha = getIntAlpha(rect, 0.0);
	int maxAlpha = getIntAlpha(rect, 1.0);
	if (doublePos<m_RangeMin || doublePos>m_RangeMax) return;

	pos = getIntDensity(rect, (doublePos-m_RangeMin)/(m_RangeMax-m_RangeMin));

	painter->setPen(color);

	QPoint p1 = transformToScreen(rect,pos-1,minAlpha);
	QPoint p2 = transformToScreen(rect,pos-1,maxAlpha);
	painter->drawLine(p1.x(), p1.y(), p2.x(), p2.y());
	p1 = transformToScreen(rect,pos+1,minAlpha);
	p2 = transformToScreen(rect,pos+1,maxAlpha);
	painter->drawLine(p1.x(), p1.y(), p2.x(), p2.y());

	// get the screen point
	QPoint screenPoint = transformToScreen(rect, (doublePos-m_RangeMin)/(m_RangeMax-m_RangeMin), 0.0);

	// draw the rectangle
	QPoint topLeft(screenPoint); topLeft+=QPoint(-ALPHA_NODE_SIZE, -ALPHA_NODE_SIZE);
	painter->fillRect( topLeft.x(), topLeft.y(), ALPHA_NODE_SIZE*2+1, ALPHA_NODE_SIZE*2+1, Qt::red );
	painter->drawRect( topLeft.x(), topLeft.y(), ALPHA_NODE_SIZE*2+1, ALPHA_NODE_SIZE*2+1);

}

void XoomedIn::drawIsocontourBar( QPainter* painter, QRect rect, int index )
{
	double doublePos = m_ColorTableInformation->getIsocontourMap().GetPositionOfIthNode(index);
	QColor color(mapToPixel(1.0-m_ColorTableInformation->getColorMap().GetRed(doublePos), 0, 255),
		mapToPixel(1.0-m_ColorTableInformation->getColorMap().GetGreen(doublePos), 0, 255),
		mapToPixel(1.0-m_ColorTableInformation->getColorMap().GetBlue(doublePos), 0, 255));		
		
	int pos;
	int minAlpha = getIntAlpha(rect, 0.0);
	int maxAlpha = getIntAlpha(rect, 1.0);
	if (doublePos<m_RangeMin || doublePos>m_RangeMax) return;

	pos = getIntDensity(rect, (doublePos-m_RangeMin)/(m_RangeMax-m_RangeMin));

	painter->setPen(color);

	QPoint p1 = transformToScreen(rect,pos-1,minAlpha);
	QPoint p2 = transformToScreen(rect,pos-1,maxAlpha);
	painter->drawLine(p1.x(), p1.y(), p2.x(), p2.y());
	p1 = transformToScreen(rect,pos+1,minAlpha);
	p2 = transformToScreen(rect,pos+1,maxAlpha);
	painter->drawLine(p1.x(), p1.y(), p2.x(), p2.y());

	// get the screen point
	QPoint screenPoint = transformToScreen(rect, (doublePos-m_RangeMin)/(m_RangeMax-m_RangeMin), 1.0);

	// draw the rectangle
	QPoint topLeft(screenPoint); topLeft+=QPoint(-ALPHA_NODE_SIZE, -ALPHA_NODE_SIZE);
	painter->fillRect( topLeft.x(), topLeft.y(), ALPHA_NODE_SIZE*2+1, ALPHA_NODE_SIZE*2+1, Qt::green );
	painter->drawRect( topLeft.x(), topLeft.y(), ALPHA_NODE_SIZE*2+1, ALPHA_NODE_SIZE*2+1);
}

void XoomedIn::drawAlphaNode( QPainter* painter, QRect rect, int index )
{
	double doubleDensity = m_ColorTableInformation->getAlphaMap().GetPosition(index);
	double doubleAlpha = m_ColorTableInformation->getAlphaMap().GetAlpha(index);

	// throw away if off the visible range
	if (doubleDensity<m_RangeMin || doubleDensity>m_RangeMax) return;

	// get the screen point
	QPoint screenPoint = transformToScreen(rect, (doubleDensity-m_RangeMin)/(m_RangeMax-m_RangeMin), doubleAlpha);

	// draw the rectangle
	QPoint topLeft(screenPoint); topLeft+=QPoint(-ALPHA_NODE_SIZE, -ALPHA_NODE_SIZE);
	painter->fillRect( topLeft.x(), topLeft.y(), ALPHA_NODE_SIZE*2+1, ALPHA_NODE_SIZE*2+1, Qt::blue );
}

void XoomedIn::drawAlphaMap( QPainter* painter, QRect rect )
{
	int minAlpha = getIntAlpha(rect, 0.0);
	int maxAlpha = getIntAlpha(rect, 1.0);
	int minDensity = getIntDensity(rect, 0.0);
	int maxDensity = getIntDensity(rect, 1.0);

	double mag;
	double doublePos1, doublePos2, clampedDoublePos1, clampedDoublePos2;
	double alpha1, alpha2;

	int mapSize = m_ColorTableInformation->getAlphaMap().GetSize();
	for (int i=0; i<mapSize-1; i++) {
		doublePos1 = m_ColorTableInformation->getAlphaMap().GetPosition(i+0);
		doublePos2 = m_ColorTableInformation->getAlphaMap().GetPosition(i+1);

		// totally off screen, get next node
		if( doublePos2 < m_RangeMin ) continue;
		// we went past the end of the range, break
		if( doublePos1 > m_RangeMax ) break;

		// if needed find the intersection of the line with the min and max
		if( doublePos1 < m_RangeMin ) {
			mag = (m_RangeMin-doublePos1)/(doublePos2-doublePos1);
			alpha1 = m_ColorTableInformation->getAlphaMap().GetAlpha(i) * (1.0-mag) + m_ColorTableInformation->getAlphaMap().GetAlpha(i+1) * (mag);
			clampedDoublePos1 = m_RangeMin;
		}
		else {
			alpha1 = m_ColorTableInformation->getAlphaMap().GetAlpha(i);
			clampedDoublePos1 = doublePos1;
		}
		if( doublePos2 > m_RangeMax ) {
			mag = (m_RangeMax-doublePos1)/(doublePos2-doublePos1);
			alpha2 = m_ColorTableInformation->getAlphaMap().GetAlpha(i) * (1.0-mag) + m_ColorTableInformation->getAlphaMap().GetAlpha(i+1) * (mag);
			clampedDoublePos2 = m_RangeMax;
		}
		else {
			alpha2 = m_ColorTableInformation->getAlphaMap().GetAlpha(i+1);
			clampedDoublePos2 = doublePos2;
		}

		if( alpha1 > alpha2 )
		{
			double a1 = m_ColorTableInformation->getAlphaMap().GetAlpha(i);
			double a2 = m_ColorTableInformation->getAlphaMap().GetAlpha(i+1);			
		}
		QPoint screenPoint1 = transformToScreen(rect,(clampedDoublePos1-m_RangeMin)/(m_RangeMax-m_RangeMin), alpha1);
		QPoint screenPoint2 = transformToScreen(rect,(clampedDoublePos2-m_RangeMin)/(m_RangeMax-m_RangeMin), alpha2);

		// set the color to black
		painter->setPen(QColor(0,0,0));

		// draw the line
		painter->drawLine(screenPoint1.x(), screenPoint1.y(), screenPoint2.x(), screenPoint2.y());
	}
}

void XoomedIn::drawContourSpectrum( QPainter* painter, QRect rect )
{
	// m_RangeMin, m_RangeMax are in [0,1] (normalized X coords)
	//
	// only render if the contour spectrum functions have been initialized
	if (m_SpecFuncs[0] && m_SpecFuncs[1] && m_SpecFuncs[2]
									&& m_SpecFuncs[3] && m_SpecFuncs[4]) {
		double curPos,nextPos, clampedCurPos,clampedNextPos,mag;
		double func1, func2;
		float *specFunc;
		int i,j;
		QColor funcColors[4];

		funcColors[0].setRgb(255,0,0); // area = dark red
		funcColors[1].setRgb(0,255,0); // min volume = dark green
		funcColors[2].setRgb(0,0,255); // max volume = dark blue
		funcColors[3].setRgb(255,255,0); // gradient = dark yellow


		for (j=0; j < 4; j++) {
			// pick a function to plot
			specFunc = m_SpecFuncs[j+1];
			
			// set the pen color to the current function's color
			painter->setPen(funcColors[j]);
			
			// loop through the function table
			for (i=0; i < 255; i++) {
				// current and next index normalized X coords
				curPos = i / 255.0;
				nextPos = (i+1) / 255.0;
	
				// make sure the points are within range
				if (nextPos < m_RangeMin) continue;
				if (curPos > m_RangeMax) break;
	
				// if needed find the intersection of the line with the min and max
				if( curPos < m_RangeMin ) {
					mag = (m_RangeMin-curPos)/(nextPos-curPos);
					func1 = specFunc[i] * (1.0-mag) + specFunc[i+1] * (mag);
					clampedCurPos = m_RangeMin;
				}
				else {
					func1 = specFunc[i];
					clampedCurPos = curPos;
				}
				if( nextPos > m_RangeMax ) {
					mag = (m_RangeMax-curPos)/(nextPos-curPos);
					func2 = specFunc[i] * (1.0-mag) + specFunc[i+1] * (mag);
					clampedNextPos = m_RangeMax;
				}
				else {
					func2 = specFunc[i+1];
					clampedNextPos = nextPos;
				}
	
				// create some endpoints
				QPoint screenPoint1 = transformToScreen(rect,(clampedCurPos-m_RangeMin)/(m_RangeMax-m_RangeMin), func1);
				QPoint screenPoint2 = transformToScreen(rect,(clampedNextPos-m_RangeMin)/(m_RangeMax-m_RangeMin), func2);
	
	
				// draw the line
				painter->drawLine(screenPoint1.x(), screenPoint1.y(), screenPoint2.x(), screenPoint2.y());
	
			}
		}
	}
}

void XoomedIn::drawConTreeNode( QPainter* painter, QRect rect, int index )
{
	CTEDGE *CTEdges=0;
	CTVTX *CTVerts=0;
	double p1x,p1y,p2x,p2y, mag,yPos=0.;
	double doubleDensity = m_ColorTableInformation->getConTreeMap().GetPositionOfIthNode(index);
	int edgeIndex = m_ColorTableInformation->getConTreeMap().GetEdgeOfIthNode(index);

	// throw away if off the visible range
	if (doubleDensity < m_RangeMin || doubleDensity > m_RangeMax) return;

	// get pointers to the contour tree
	CTEdges = m_ColorTableInformation->getConTreeMap().GetCTEdges();
	CTVerts = m_ColorTableInformation->getConTreeMap().GetCTVerts();

	// bail if there is no contour tree
	if (CTEdges && CTVerts) {

		// get the endpoints of the edge
		p1x = CTVerts[CTEdges[edgeIndex].v1].func_val;
		p1y = CTVerts[CTEdges[edgeIndex].v1].norm_x;
		p2x = CTVerts[CTEdges[edgeIndex].v2].func_val;
		p2y = CTVerts[CTEdges[edgeIndex].v2].norm_x;
		
		// ensure that p1 is to the left of p2 (assuming horizontal drawing)
		if (p1x > p2x) {
			p2x = p1x;
			p2y = p1y;
			p1x = CTVerts[CTEdges[edgeIndex].v2].func_val;
			p1y = CTVerts[CTEdges[edgeIndex].v2].norm_x;
		}

		// calculate the y coord of the node
		mag = (doubleDensity-p1x) / (p2x-p1x);
		yPos = p1y * (1.0-mag) + p2y * mag;
	
		// get the screen point
		QPoint screenPoint = transformToScreen(rect, (doubleDensity-m_RangeMin)/(m_RangeMax-m_RangeMin), yPos);

		//qDebug("edge: (%f,%f)->(%f,%f)", p1x,p1y,p2x,p2y);
		//qDebug("isoval: %f, mag: %f, yPos: %f", doubleDensity,mag,yPos);

		// draw the rectangle
		QPoint topLeft(screenPoint); topLeft+=QPoint(-ALPHA_NODE_SIZE, -ALPHA_NODE_SIZE);
		painter->fillRect( topLeft.x(), topLeft.y(), ALPHA_NODE_SIZE*2+1, ALPHA_NODE_SIZE*2+1, Qt::white );

	}
}

void XoomedIn::drawContourTree( QPainter* painter, QRect rect )
{
	// m_RangeMin, m_RangeMax are in [0,1] (normalized X coords)
	CTEDGE *CTEdges = m_ColorTableInformation->getConTreeMap().GetCTEdges();
	CTVTX *CTVerts = m_ColorTableInformation->getConTreeMap().GetCTVerts();
	
	// don't draw unless there is data to be drawn
	if (CTVerts && CTEdges) {
		double p1x,p1y, p2x,p2y;
		double clampedP1x,clampedP1y,clampedP2x,clampedP2y,mag;
		int i,numPoints=0,numCTEdges=0;
		Q3PointArray points;

		numCTEdges = m_ColorTableInformation->getConTreeMap().GetEdgeNum();

		// build up an array of line segments
		// float m_CTVerts.norm_x , m_CTVerts.func_val (isoval)
		// int m_CTEdges.v1 , m_CTEdges.v2
		for (i=0; i < numCTEdges; i++) {

			// get the vertex coords
			p1x = CTVerts[CTEdges[i].v1].func_val;
			p1y = CTVerts[CTEdges[i].v1].norm_x;
			p2x = CTVerts[CTEdges[i].v2].func_val;
			p2y = CTVerts[CTEdges[i].v2].norm_x;

			//qDebug("(%f,%f) (%f,%f)", p1x,p1y,p2x,p2y);

			// don't add edges that are outside of the zoomed in region
			if (p1x < m_RangeMin && p2x < m_RangeMin) continue;
			if (p1x > m_RangeMax && p2x > m_RangeMax) continue;

			// find the cases where edges span the edge of our bounding box
			if (p1x < p2x) {
				if (p1x < m_RangeMin) {
					mag = (m_RangeMin-p1x)/(p2x-p1x);
					clampedP1y = p1y * (1.0-mag) + p2y * mag;
					clampedP1x = m_RangeMin;
				}
				else {
					clampedP1x = p1x;
					clampedP1y = p1y;
				}
				if (p2x > m_RangeMax) {
					mag = (m_RangeMax-p1x)/(p2x-p1x);
					clampedP2y = p1y * (1.0-mag) + p2y * mag;
					clampedP2x = m_RangeMax;
				}
				else {
					clampedP2x = p2x;
					clampedP2y = p2y;
				}
			}
			else { // p2x <= p1x (but probably p2x < p1x)
				if (p2x < m_RangeMin) {
					mag = (m_RangeMin-p2x)/(p1x-p2x);
					clampedP1y = p2y * (1.0-mag) + p1y * mag;
					clampedP1x = m_RangeMin;
				}
				else {
					clampedP1x = p2x;
					clampedP1y = p2y;
				}
				if (p1x > m_RangeMax) {
					mag = (m_RangeMax-p2x)/(p1x-p2x);
					clampedP2y = p2y * (1.0-mag) + p1y * mag;
					clampedP2x = m_RangeMax;
				}
				else {
					clampedP2x = p1x;
					clampedP2y = p1y;
				}
			}

			//qDebug("-> (%f,%f) (%f,%f)",clampedP1x,clampedP1y,clampedP2x,clampedP2y);
			
			QPoint P1 = transformToScreen(rect, (clampedP1x-m_RangeMin)/(m_RangeMax-m_RangeMin), clampedP1y),
						 P2 = transformToScreen(rect, (clampedP2x-m_RangeMin)/(m_RangeMax-m_RangeMin), clampedP2y);
			//if (m_SelectedCTEdge == i) {
			//	painter->setPen(QColor(255,255,255));
			//	painter->drawLine(P1,P2);
			//}
			//else {
				points.putPoints(numPoints, 2, P1.x(),P1.y(), P2.x(),P2.y());
				//qDebug("putPoints: (%f,%f), (%f,%f)", P1.x(),P1.y(), P2.x(),P2.y());
				numPoints += 2;
			//}
		}
		
		// set the pen color to black
		painter->setPen(QColor(0,0,0));
	
		// draw the Contour Tree
		painter->drawLineSegments(points);
	}
}

XoomedIn::POPUPSELECTION XoomedIn::showPopup(QPoint point)
{
	Q3PopupMenu popup;
	Q3PopupMenu add;
	Q3PopupMenu display;

	add.insertItem( "Alpha Node", ADD_ALPHA);
	add.insertItem( "Color Node", ADD_COLOR);
	add.insertItem( "Isocontour Node", ADD_ISOCONTOUR);

	display.insertItem( "Contour Spectrum", DISP_CONTOUR_SPECTRUM);
	display.setItemChecked(DISP_CONTOUR_SPECTRUM, m_DrawContourSpec);
	display.insertItem( "Contour Tree", DISP_CONTOUR_TREE);
	display.setItemChecked(DISP_CONTOUR_TREE, m_DrawContourTree);
	display.insertItem( "Opacity Function", DISP_ALPHA_MAP);
	display.setItemChecked(DISP_ALPHA_MAP, m_DrawAlphaMap);

	popup.insertItem( "Open", LOAD_MAP );
	popup.insertItem( "Save", SAVE_MAP );
	popup.insertItem( "Add", &add, ADD_MENU );
	popup.insertItem( "Display", &display, DISP_MENU);
	popup.insertItem( "Edit", EDIT_SELECTION);
	popup.insertItem( "Delete", DELETE_SELECTION );

	return (POPUPSELECTION)popup.exec(point);
}

void XoomedIn::setMin( double min )
{
	m_RangeMin = min;
	//m_ConSpecInfo->setMin(min);
	update();
}

void XoomedIn::setMax( double max )
{
	m_RangeMax = max;
	//m_ConSpecInfo->setMax(max);
	update();
}

void XoomedIn::setSpectrumFunctions( float *isoval, float *area, float *min_vol, float *max_vol, float *gradient )
{
	// clean up previous functions
  for (int i=0; i < 5; i++) {
    if (m_SpecFuncs[i]) {
      free(m_SpecFuncs[i]);
      m_SpecFuncs[i] = 0;
    }
  }

	// assign the new functions
  m_SpecFuncs[0] = isoval;
  m_SpecFuncs[1] = area;
  m_SpecFuncs[2] = min_vol;
  m_SpecFuncs[3] = max_vol;
  m_SpecFuncs[4] = gradient;
	
	// those pointers might be NULL
	if (isoval && area && min_vol && max_vol && area) {
		// normalize all the functions
		normalizeFunc( m_SpecFuncs[0], 256, m_SpecFuncsMin[0], m_SpecFuncsMax[0] );
		normalizeFunc( m_SpecFuncs[1], 256, m_SpecFuncsMin[1], m_SpecFuncsMax[1] );
		normalizeFunc( m_SpecFuncs[2], 256, m_SpecFuncsMin[2], m_SpecFuncsMax[2] );
		normalizeFunc( m_SpecFuncs[3], 256, m_SpecFuncsMin[3], m_SpecFuncsMax[3] );
		normalizeFunc( m_SpecFuncs[4], 256, m_SpecFuncsMin[4], m_SpecFuncsMax[4] );
	}

	// repaint if needed
	if (m_DrawContourSpec)
		update();
}

void XoomedIn::setCTGraph(int num_vtx, int num_edge, CTVTX* vtx_list, CTEDGE* edge_list)
{
	// normalize the vertices
	if (vtx_list) {			
		float xmin = vtx_list[0].norm_x, xmax=xmin,
					ymin = vtx_list[0].func_val, ymax=ymin;
		int i;
		
		for (i=0; i<num_vtx; i++) {
			if (vtx_list[i].func_val < ymin) ymin = vtx_list[i].func_val;
			else if (vtx_list[i].func_val > ymax) ymax = vtx_list[i].func_val;
			
			if (vtx_list[i].norm_x < xmin) xmin = vtx_list[i].norm_x;
			else if (vtx_list[i].norm_x > xmax) xmax = vtx_list[i].norm_x;
		}
		for (i=0; i<num_vtx; i++) {
			vtx_list[i].func_val = (vtx_list[i].func_val-ymin) / (ymax-ymin);
			vtx_list[i].norm_x = (vtx_list[i].norm_x-xmin) / (xmax-xmin);
		}
	}

	// hand it off the the ConTreeMap
	m_ColorTableInformation->getConTreeMap().SetCTData(vtx_list, edge_list, num_vtx, num_edge);
	
	// repaint if needed
	if (m_DrawContourTree)
		update();
}

void XoomedIn::setDataMinMax( double min, double max )
{
	m_IsoMin = min;
	m_IsoMax = max;
}

void XoomedIn::moveIsocontourNode(int node, double value)
{
  double clipped_value = std::min(std::max(value,m_RangeMin),m_RangeMax);
  m_ColorTableInformation->getIsocontourMap().MoveIthNode(node,clipped_value);
  if (m_SpecFuncs[0] && m_SpecFuncs[1] && m_SpecFuncs[2] && m_SpecFuncs[3])
    
    /*
    m_ConSpecInfo->moveNode(m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(node), 
			    clipped_value, 
			    getSpecFuncValue(0, clipped_value), 
			    getSpecFuncValue(1, clipped_value), 
			    getSpecFuncValue(2, clipped_value), 
			    getSpecFuncValue(3, clipped_value));
    */
  update();
  emit isocontourNodeChanged(m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(clipped_value), 
			     clipped_value);
}

void XoomedIn::mouseMoveEvent( QMouseEvent* q )
{
	//mouseMoveEventHorizontal(q);
	double doubleAlpha, doubleDensity;

	QRect myRect = getMyRect();

	QPoint point = q->pos();

	if (m_MoveMode == DRAG_MOVE) {

		switch (m_SelectedType) {
		case COLOR_NODE:
			doubleDensity = getNormalizedDensity(myRect,point);
			if (doubleDensity<m_RangeMin) {
				doubleDensity=m_RangeMin;
			}
			else if (doubleDensity>m_RangeMax) {
				doubleDensity=m_RangeMax;
			}
			m_ColorTableInformation->getColorMap().MoveNode(m_SelectedNode, doubleDensity);
			emit functionExploring();
			break;
		case ISOCONTOUR_NODE:
			doubleDensity = getNormalizedDensity(myRect,point);
			if (doubleDensity<m_RangeMin) {
				doubleDensity=m_RangeMin;
			}
			else if (doubleDensity>m_RangeMax) {
				doubleDensity=m_RangeMax;
			}
			m_ColorTableInformation->getIsocontourMap().MoveIthNode(m_SelectedNode, doubleDensity);
			// only update the ContourSpectrumInfo if the data is there
			/*
			if (m_SpecFuncs[0] && m_SpecFuncs[1] && m_SpecFuncs[2] && m_SpecFuncs[3])
				m_ConSpecInfo->moveNode(m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(m_SelectedNode), doubleDensity, getSpecFuncValue(0, doubleDensity), getSpecFuncValue(1, doubleDensity), getSpecFuncValue(2, doubleDensity), getSpecFuncValue(3, doubleDensity));
			emit isocontourNodeExploring(m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(m_SelectedNode), doubleDensity);
			*/
			break;
		case ALPHA_NODE:
			doubleDensity = getNormalizedDensity(myRect,point);
			doubleAlpha = getNormalizedAlpha(myRect,point);

			if (doubleDensity<m_RangeMin) {
				doubleDensity=m_RangeMin;
			}
			else if (doubleDensity>m_RangeMax) {
				doubleDensity=m_RangeMax;
			}
			m_ColorTableInformation->getAlphaMap().MoveNode(m_SelectedNode, doubleDensity);
			m_ColorTableInformation->getAlphaMap().ChangeAlpha( m_SelectedNode, doubleAlpha);
			emit functionExploring();
			break;
		case CONTOURTREE_NODE:
			doubleDensity = getNormalizedDensity(myRect,point);
			if (doubleDensity<m_RangeMin) {
				doubleDensity=m_RangeMin;
			}
			else if (doubleDensity>m_RangeMax) {
				doubleDensity=m_RangeMax;
			}
			m_ColorTableInformation->getConTreeMap().MoveIthNode(m_SelectedNode, doubleDensity);
			emit contourTreeNodeExploring(m_ColorTableInformation->getConTreeMap().GetIDofIthNode(m_SelectedNode), doubleDensity);
			break;
		default:
			break;
		};
	}

	repaint();
	q->accept();
}

void XoomedIn::mousePressEvent( QMouseEvent* q )
{
	//mousePressEventHorizontal(q);

	int x = q->x();
	int y = q->y();

	if (q->button() == Qt::LeftButton ) {
		m_SelectedType = NO_NODE;
		m_SelectedNode = -1;
		if (selectNodes(x,y)) {
			m_MoveMode = DRAG_MOVE;
		}
	}
	else if (q->button() == Qt::MidButton && m_DrawContourTree) {
		// implement contour tree selection here
		m_SelectedCTEdge = -1;
		if (selectCTEdge(x,y)) {
			// m_SelectedCTEdge => an index into the CTEdges array
			// m_SelectedIsovalue => the x coord of the selection

			// make sure there is not already a node on this edge
			bool hasNode = false;
			for (int i=0; i<m_ColorTableInformation->getConTreeMap().GetSize(); i++) {
				if (m_ColorTableInformation->getConTreeMap().GetEdgeOfIthNode(i) 
							== m_SelectedCTEdge) {
					hasNode = true;
					break;
				}
			}

			// add the node
			if (!hasNode)
			{
				int index = m_ColorTableInformation->getConTreeMap().AddNode(m_SelectedCTEdge, m_SelectedIsovalue);
				
				if (index != 0)
					emit contourTreeNodeAdded(index, m_SelectedCTEdge, m_SelectedIsovalue);
			}
		}
	}

	update();
	q->accept();
}

void XoomedIn::mouseReleaseEvent( QMouseEvent* q )
{
	//mouseReleaseEventHorizontal(q);

	QPoint point = q->pos();

	QString verificationString;

	QRect myRect = getMyRect();

	double doubleDensity;

	if( q->button() == Qt::LeftButton ) {
		if (m_MoveMode == DRAG_MOVE) {

			switch (m_SelectedType) {
			case COLOR_NODE:
				emit functionChanged();
				break;
			case ISOCONTOUR_NODE:
				emit isocontourNodeChanged(m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(m_SelectedNode), m_ColorTableInformation->getIsocontourMap().GetPositionOfIthNode(m_SelectedNode));
				break;
			case ALPHA_NODE:
				emit functionChanged();
				break;
			case CONTOURTREE_NODE:
				emit contourTreeNodeChanged(m_ColorTableInformation->getConTreeMap().GetIDofIthNode(m_SelectedNode), m_ColorTableInformation->getConTreeMap().GetPositionOfIthNode(m_SelectedNode));
				break;
			default:
				break;
			};
		}

		m_MoveMode = NO_MOVE;
	}
	if( q->button() == Qt::RightButton ) {
		POPUPSELECTION selection = showPopup(q->globalPos());
		QString fileName;
		switch (selection) {
		case LOAD_MAP:
			fileName = Q3FileDialog::getOpenFileName( "", "Vinay files (*.vinay)", 
				this, "open file dialog", "Choose a file");
			if ( !(fileName.isNull()) ) {				
				if (!m_ColorTableInformation->loadColorTable(fileName)) {
					QMessageBox::warning( this, "Color Table",
							"Error loading file: incorrect file format",
							"Ok");
				}
				else {
					emit everythingChanged();
					emit everythingChanged(m_ColorTableInformation);
				}
			}
			break;
		case SAVE_MAP:
			fileName = Q3FileDialog::getSaveFileName( "", "Vinay files (*.vinay)", 
				this, "save file dialog", "Choose a file");
			if ( !(fileName.isNull()) ) {
				if (!m_ColorTableInformation->saveColorTable(fileName)) {
					QMessageBox::warning( this, "Color Table",
							"Error saving file",
							"Ok");
				}
			}
			break;
		case ADD_COLOR:
			doubleDensity = getNormalizedDensity(myRect, point);
			if (doubleDensity<m_RangeMin) {
				doubleDensity=m_RangeMin;
			}
			else if (doubleDensity>m_RangeMax) {
				doubleDensity=m_RangeMax;
			}
			m_ColorTableInformation->getColorMap().AddNode(doubleDensity);
			emit functionChanged();
			break;
		case ADD_ISOCONTOUR:
			{
				doubleDensity = getNormalizedDensity(myRect, point);
				if (doubleDensity<m_RangeMin) {
					doubleDensity=m_RangeMin;
				}
				else if (doubleDensity>m_RangeMax) {
					doubleDensity=m_RangeMax;
				}

				double R = m_ColorTableInformation->getColorMap().GetRed(doubleDensity),
				G = m_ColorTableInformation->getColorMap().GetGreen(doubleDensity),
				B = m_ColorTableInformation->getColorMap().GetBlue(doubleDensity);
				
				int index = m_ColorTableInformation->getIsocontourMap().AddNode(doubleDensity, R,G,B);

				/*
				if (index != -1 && m_SpecFuncs[0] && m_SpecFuncs[1] && m_SpecFuncs[2]
						&& m_SpecFuncs[3])

					m_ConSpecInfo->addNode(index, doubleDensity, 
						getSpecFuncValue(0, doubleDensity),
						getSpecFuncValue(1, doubleDensity),
						getSpecFuncValue(2, doubleDensity),
						getSpecFuncValue(3, doubleDensity));
				*/
						
				if( index != -1 )
					emit isocontourNodeAdded(index, doubleDensity, R,G,B);
			}
			break;
		case ADD_ALPHA:
			doubleDensity = getNormalizedDensity(myRect, point);

			if (doubleDensity<m_RangeMin) {
				doubleDensity=m_RangeMin;
			}
			else if (doubleDensity>m_RangeMax) {
				doubleDensity=m_RangeMax;
			}
			m_ColorTableInformation->getAlphaMap().AddNode(doubleDensity);
			emit functionChanged();
			break;
		case EDIT_SELECTION:
			m_SelectedType = NO_NODE;
			m_SelectedNode = -1;
			if (selectColorNodes(q->x(), q->y())) {
				// edit a color node
				m_MoveMode = NO_MOVE;
				QColor color(mapToPixel(m_ColorTableInformation->getColorMap().GetRed(m_SelectedNode), 0, 255),
					mapToPixel(m_ColorTableInformation->getColorMap().GetGreen(m_SelectedNode), 0, 255),
					mapToPixel(m_ColorTableInformation->getColorMap().GetBlue(m_SelectedNode), 0, 255));		
				QColor c = QColorDialog::getColor( color );
				if ( c.isValid() ) {
					m_ColorTableInformation->getColorMap().ChangeColor(m_SelectedNode,
						mapToDouble(c.red(), 0, 255),
						mapToDouble(c.green(), 0, 255),
						mapToDouble(c.blue(), 0, 255));
					emit functionChanged();
				}
				else {

				}
			}
			else if (selectIsocontourNodes(q->x(), q->y())) {
				// edit an isocontour node (set its color)
				m_MoveMode = NO_MOVE;
#if 0 //changing the color of an isocontour node doesn't have an effect after it was created
				QColor color(mapToPixel(m_ColorTableInformation->getIsocontourMap().GetRed(m_SelectedNode), 0, 255),
					mapToPixel(m_ColorTableInformation->getIsocontourMap().GetGreen(m_SelectedNode), 0, 255),
					mapToPixel(m_ColorTableInformation->getIsocontourMap().GetBlue(m_SelectedNode), 0, 255));
				
				QColor c = QColorDialog::getColor( color );
				if ( c.isValid() ) {
					double R,G,B;
					R = mapToDouble(c.Qt::red(), 0, 255);
					G = mapToDouble(c.Qt::green(), 0, 255);
					B = mapToDouble(c.Qt::blue(), 0, 255);
					
					m_ColorTableInformation->getIsocontourMap().ChangeColor(m_SelectedNode, R, G, B);
					
					emit isocontourNodeColorChanged(m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(m_SelectedNode), R,G,B);
				}
				else {

				}
#endif

				emit isocontourNodeEditRequest(m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(m_SelectedNode));
			}
			break;
		case DISP_CONTOUR_SPECTRUM:
		{
		  qDebug("XoomedIn::mouseReleaseEvent: DISP_CONTOUR_SPECTRUM");

			// toggle the flag
			m_DrawContourSpec = !m_DrawContourSpec;
			// change the display
			if (m_DrawContourSpec) {
			  qDebug("XoomedIn::mouseReleaseEvent: drawing the contour spectrum");
				// we need to draw the contour spectrum, make sure we have the data
				if ( !(m_SpecFuncs[1] && m_SpecFuncs[2]	&& m_SpecFuncs[3] && m_SpecFuncs[4]) ) {
				  qDebug("XoomedIn::mouseReleaseEvent: emitting acquireContourSpectrum()");
					// request the spectrum data
					emit acquireContourSpectrum();
				}
			}
			qDebug("XoomedIn::mouseReleaseEvent: not drawing the contour spectrum");
			break;
		}
		case DISP_CONTOUR_TREE:
		{
			// toggle the flag
			m_DrawContourTree = !m_DrawContourTree ;
			// change the display
			if (m_DrawContourTree ) {
				// we need to draw the contour tree, make sure we have the data
				CTVTX *CTVerts = m_ColorTableInformation->getConTreeMap().GetCTVerts();
				CTEDGE *CTEdges = m_ColorTableInformation->getConTreeMap().GetCTEdges();
				if (!(CTVerts && CTEdges)) {
					// request the contour tree data
					emit acquireContourTree();
				}
			}
			break;
		}
		case DISP_ALPHA_MAP:
		{
			// toggle the flag
			m_DrawAlphaMap = !m_DrawAlphaMap;
			// update is called after the switch()
			break;
		}
		case DELETE_SELECTION:
		{
			int ID=0;
			m_SelectedType = NO_NODE;
			m_SelectedNode = -1;
			if (selectNodes(q->x(), q->y())) {
				m_MoveMode = NO_MOVE;
				switch (m_SelectedType) {
				case COLOR_NODE:
					m_ColorTableInformation->getColorMap().DeleteNode(m_SelectedNode);
					emit functionChanged();
					break;
				case ISOCONTOUR_NODE:
					ID = m_ColorTableInformation->getIsocontourMap().GetIDofIthNode(m_SelectedNode);
					m_ColorTableInformation->getIsocontourMap().DeleteIthNode(m_SelectedNode);
					//m_ConSpecInfo->removeNode(ID);
					emit isocontourNodeDeleted(ID);
					break;
				case ALPHA_NODE:
					m_ColorTableInformation->getAlphaMap().DeleteNode(m_SelectedNode);
					emit functionChanged();
					break;
				case CONTOURTREE_NODE:
					ID = m_ColorTableInformation->getConTreeMap().GetIDofIthNode(m_SelectedNode);
					m_ColorTableInformation->getConTreeMap().DeleteIthNode(m_SelectedNode);
					emit contourTreeNodeDeleted(ID);
					break;
				default:
					break;
				};
			}
			m_SelectedType = NO_NODE;
			m_SelectedNode = -1;
			break;
		}
		default:
			break;
		}
	}
	update();


	q->accept();
}

void XoomedIn::contextMenuEvent( QContextMenuEvent* e)
{
	// block these messages for now
	e->accept();
}

bool XoomedIn::selectNodes( int x, int y )
{
	return selectAlphaNodes(x, y) || selectIsocontourNodes(x, y) || selectColorNodes(x, y) || selectCTNodes(x, y);
}

bool XoomedIn::selectColorNodes( int x, int y )
{
	int curDistance = 1000000;
	int intNodeDensity, intQueryDensity, intDistance;
	int i;

	QRect myRect = getMyRect();
	if (y<myRect.y()-2 || y>myRect.bottom()+2 || x<myRect.x() || x>myRect.right() ) {
		return false;
	}

	intQueryDensity = getIntDensity(myRect, QPoint(x,y));

	for(i=0;i<this->m_ColorTableInformation->getColorMap().GetSize();i++)
	{
		double doubleNodeDensity = m_ColorTableInformation->getColorMap().GetPosition(i);
		intNodeDensity = getIntDensity(myRect, (doubleNodeDensity-m_RangeMin)/(m_RangeMax-m_RangeMin));
		intDistance = abs(intNodeDensity-intQueryDensity);
		if( intDistance <= CLICK_RANGE && intDistance <curDistance) {
			curDistance = intDistance;
			m_SelectedNode = i;
			m_SelectedType = COLOR_NODE;
		}
	}
	if (m_SelectedType == COLOR_NODE) {
		return true;
	}
	else {
		return false;
	}
}

bool XoomedIn::selectIsocontourNodes( int x, int y )
{
	int curDistance = 1000000;
	int intNodeDensity, intQueryDensity, intDistance;
	int i;

	QRect myRect = getMyRect();
	if (y<myRect.y()-2 || y>myRect.bottom()+2 || x<myRect.x() || x>myRect.right() ) {
		return false;
	}

	intQueryDensity = getIntDensity(myRect, QPoint(x,y));

	for(i=0;i<this->m_ColorTableInformation->getIsocontourMap().GetSize();i++)
	{
		double doubleNodeDensity = m_ColorTableInformation->getIsocontourMap().GetPositionOfIthNode(i);
		intNodeDensity = getIntDensity(myRect, (doubleNodeDensity-m_RangeMin)/(m_RangeMax-m_RangeMin));
		intDistance = abs(intNodeDensity-intQueryDensity);
		if( intDistance <= CLICK_RANGE && intDistance <curDistance) {
			curDistance = intDistance;
			m_SelectedNode = i;
			m_SelectedType = ISOCONTOUR_NODE;
		}
	}
	if (m_SelectedType == ISOCONTOUR_NODE) {
		return true;
	}
	else {
		return false;
	}
}

bool XoomedIn::selectAlphaNodes( int x, int y )
{
	int curDistance = 1000000;
	int intManhattanDistance, intDistance;
	int i;
	int intQueryAlpha, intQueryDensity;
	int intNodeAlpha, intNodeDensity;

	QRect myRect = getMyRect();
	if (y<myRect.y()-2 || y>myRect.bottom()+2 ) {
		return false;
	}

	intQueryDensity = getIntDensity(myRect, QPoint(x,y));
	intQueryAlpha = getIntAlpha(myRect, QPoint(x,y));

	for(i=0;i<this->m_ColorTableInformation->getAlphaMap().GetSize();i++)
	{
		double doubleNodeDensity = m_ColorTableInformation->getAlphaMap().GetPosition(i);
		intNodeDensity = getIntDensity(myRect, (doubleNodeDensity-m_RangeMin)/(m_RangeMax-m_RangeMin));
		intNodeAlpha = getIntAlpha(myRect, m_ColorTableInformation->getAlphaMap().GetAlpha(i));
		intManhattanDistance = abs(intNodeDensity-intQueryDensity) + abs(intNodeAlpha-intQueryAlpha);
		intDistance = (abs(intNodeDensity-intQueryDensity) > abs(intNodeAlpha-intQueryAlpha) ? 
			abs(intNodeDensity-intQueryDensity) : abs(intNodeAlpha-intQueryAlpha));
		if( intDistance <= CLICK_RANGE && intManhattanDistance <curDistance) {
			curDistance = intManhattanDistance;
			m_SelectedNode = i;
			m_SelectedType = ALPHA_NODE;
		}
	}
	if (m_SelectedType == ALPHA_NODE) {
		return true;
	}
	else {
		return false;
	}
}

bool XoomedIn::selectCTNodes( int x, int y )
{
	CTEDGE *CTEdges=0;
	CTVTX *CTVerts=0;
	double p1x,p1y,p2x,p2y, mag,xPos=0.,yPos=0.;
	int curDistance = 1000000;
	int intManhattanDistance,intDistance;
	int intXPos,intYPos;
	int intQueryX,intQueryY;
	
	// bail early if we can
	QRect myRect = getMyRect();
	if (!m_DrawContourTree || y<myRect.y()-2 || y>myRect.bottom()+2 ) {
		return false;
	}
	
	intQueryX = getIntDensity(myRect, QPoint(x,y));
	intQueryY = getIntAlpha(myRect, QPoint(x,y));

	// get pointers to the contour tree
	CTEdges = m_ColorTableInformation->getConTreeMap().GetCTEdges();
	CTVerts = m_ColorTableInformation->getConTreeMap().GetCTVerts();

	// make sure the CT data is there
	if (CTEdges && CTVerts) {
		// loop through the nodes
		for (int i=0; i<m_ColorTableInformation->getConTreeMap().GetSize(); i++) {
			// get the position of the node
			double doubleDensity = m_ColorTableInformation->getConTreeMap().GetPositionOfIthNode(i);
			int edgeIndex = m_ColorTableInformation->getConTreeMap().GetEdgeOfIthNode(i);
			// skip this node if off the visible range
			if (doubleDensity < m_RangeMin || doubleDensity > m_RangeMax) continue;

			// get the endpoints of the edge
			p1x = CTVerts[CTEdges[edgeIndex].v1].func_val;
			p1y = CTVerts[CTEdges[edgeIndex].v1].norm_x;
			p2x = CTVerts[CTEdges[edgeIndex].v2].func_val;
			p2y = CTVerts[CTEdges[edgeIndex].v2].norm_x;
		
			// ensure that p1 is to the left of p2 (assuming horizontal drawing)
			if (p1x > p2x) {
				p2x = p1x;
				p2y = p1y;
				p1x = CTVerts[CTEdges[edgeIndex].v2].func_val;
				p1y = CTVerts[CTEdges[edgeIndex].v2].norm_x;
			}

			// calculate the coords of the node
			xPos = (doubleDensity-m_RangeMin)/(m_RangeMax-m_RangeMin);
			mag = (doubleDensity-p1x) / (p2x-p1x);
			yPos = p1y * (1.0-mag) + p2y * mag;

			// convert the coords to screen space
			intXPos = getIntDensity(myRect, xPos);
			intYPos = getIntAlpha(myRect, yPos);

			// determine the manhattan distance from click to node
			intManhattanDistance = abs(intXPos-intQueryX) + abs(intYPos-intQueryY);
			intDistance = (abs(intXPos-intQueryX)	> abs(intYPos-intQueryY) ? 
											abs(intXPos-intQueryX) : abs(intYPos-intQueryY));

			// test for proximity to the mouse click
			if (intDistance <= CLICK_RANGE && intManhattanDistance < curDistance) {
				curDistance = intManhattanDistance;
				m_SelectedNode = i;
				m_SelectedType = CONTOURTREE_NODE;
				//qDebug("a CT node was clicked");
			}
		}
	}
	if (m_SelectedType == CONTOURTREE_NODE) {
		return true;
	}
	else {
		return false;
	}
}

bool XoomedIn::selectCTEdge( int x, int y )
{
	// m_RangeMin, m_RangeMax are in [0,1] (normalized X coords)
	bool ret = false;
	CTEDGE *CTEdges = m_ColorTableInformation->getConTreeMap().GetCTEdges();
	CTVTX *CTVerts = m_ColorTableInformation->getConTreeMap().GetCTVerts();
	
	// don't search unless there is data to search & the CT is being drawn
	if (CTVerts && CTEdges && m_DrawContourTree) {
		double p1x,p1y, p2x,p2y, p0x,p0y, mag,mindist=0.01;
		double v1[2],v2[2],r1[2],r2[2], dist=0.,vDotR1=0.,vDotR2=0.;
		int i;
		QRect rect = getMyRect();
		QPoint point(x,y);

		// find the edge in the contour tree which corresponds to the
		// coordinate (x,y)

		// first, convert (x,y) to a normalized point
		p0x = getNormalizedDensity(rect, point);
		p0y = getNormalizedAlpha(rect, point);

		//qDebug("(%d,%d) -> (%f,%f)", x,y, p0x,p0y);
		
		// float CTVerts.norm_x , CTVerts.func_val (isoval)
		// int CTEdges.v1 , CTEdges.v2
		for(i=0;i<m_ColorTableInformation->getConTreeMap().GetEdgeNum();i++) {

			// get the vertex coords
			p1x = CTVerts[CTEdges[i].v1].func_val;
			p1y = CTVerts[CTEdges[i].v1].norm_x;
			p2x = CTVerts[CTEdges[i].v2].func_val;
			p2y = CTVerts[CTEdges[i].v2].norm_x;

			//qDebug("(%f,%f) (%f,%f)", p1x,p1y,p2x,p2y);

			// don't consider edges that are outside of the zoomed in region
			if (p1x < m_RangeMin && p2x < m_RangeMin) continue;
			if (p1x > m_RangeMax && p2x > m_RangeMax) continue;

			// v1
			v1[0] = p0x - p1x;
			v1[1] = p0y - p1y;
			// v2
			v2[0] = p0x - p2x;
			v2[1] = p0y - p2y;
			// r1 (normalized)
			r1[0] = p2x - p1x;
			r1[1] = p2y - p1y;
			mag = sqrt(r1[0]*r1[0] + r1[1]*r1[1]);
			r1[0] /= mag;
			r1[1] /= mag;
			// r2 (normalized)
			r2[0] = p1x - p2x;
			r2[1] = p1y - p2y;
			mag = sqrt(r2[0]*r2[0] + r2[1]*r2[1]);
			r2[0] /= mag;
			r2[1] /= mag;
			// vDotR1 = v1 projected onto r1
			vDotR1 = v1[0]*r1[0] + v1[1]*r1[1];
			// vDotR2 = v2 projected onto r2
			vDotR2 = v2[0]*r2[0] + v2[1]*r2[1];
			// r2 becomes p0 projected onto the line
			r2[0] = p1x + r1[0]*vDotR1;
			r2[1] = p1y + r1[1]*vDotR1;
			// v2 becomes the vector from r2 to p0
			v2[0] = r2[0] - p0x;
			v2[1] = r2[1] - p0y;
			// dist = distance from p0 to the line
			dist = sqrt(v2[0]*v2[0] + v2[1]*v2[1]);

			// if vDotR1 and vDotR2 have the same sign and dist is very small
			// (dist is at most 0.01 and we search for the smallest dist overall)
			if (((vDotR1 > 0. && vDotR2 > 0.) || (vDotR1 < 0. && vDotR2 < 0.))
					&& dist < mindist) {
				m_SelectedCTEdge = i;
				m_SelectedIsovalue = p0x;
				mindist = dist;

				ret = true;
				//qDebug("distance to the line = %f", dist);
			}
		}
	}

	return ret;
}

