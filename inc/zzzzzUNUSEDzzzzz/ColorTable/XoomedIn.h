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
#ifndef XOOMEDIN_H
#define XOOMEDIN_H
#include <qframe.h>
#include <qpainter.h>
#include <ColorTable/ColorTableInformation.h>
#include <ColorTable/ColorMap.h>
#include <ColorTable/AlphaMap.h>
#include <ColorTable/IsocontourMap.h>
#include <ColorTable/ConTreeMap.h>
#include <ColorTable/ContourSpectrumInfo.h>
//#include <computeCT.h>

///\class XoomedIn XoomedIn.h
///\author Anthony Thane
///\author Vinay Siddavanahalli
///\author John Wiggins
///\brief This class does the bulk of the drawing and mouse handling for
/// the ColorTable widget. It draws the transfer function, the contour
/// spectrum, the contour tree, and all the isocontour nodes. It can be
/// scaled along the X and Y axes and zoomed into along the X axis. These
/// scaling properties make for some very interesting drawing code. If you
/// ever need to modify it to draw something new... Good Luck. There are
/// plenty of examples to follow, so it's not impossible.
class XoomedIn : public QFrame 
{
	Q_OBJECT
public:
///\fn XoomedIn( ColorTableInformation* colorTableInformation, QWidget *parent, const char *name )
///\brief The constructor
///\param colorTableInformation The ColorTableInformation object owned by the parent ColorTable object
///\param parent The parent widget (ColorTable)
///\param name A name for this widget
	XoomedIn( ColorTableInformation* colorTableInformation, QWidget *parent, const char *name );
	virtual ~XoomedIn();

protected:
///\fn QRect getMyRect() const
///\brief Returns the rectangle that the widget draws inside of
///\return A QRect
	QRect getMyRect() const;

///\fn QPoint transformToScreen(const QRect& rect, double density, double alpha) const
///\brief Transforms a density value (x coord in the range [0,1]) and an alpha
/// value (y coord in the range [0,1]) to a coordinate in the widget's
/// coordinate system
///\param rect The rectangle that the coordinate should be inside of
///\param density A density value
///\param alpha An alpha value
///\return A QPoint that is inside rect
	QPoint transformToScreen(const QRect& rect, double density, double alpha) const;
///\fn QPoint transformToScreen(const QRect& rect, int density, int alpha) const
///\brief Transforms a density value (x coord) and an alpha value (y coord) to
/// a coordinate in the widget's coordinate system.
///\param rect The rectangle that the output point should be inside of
///\param density A density value
///\param alpha An alpha value
///\return A QPoint that is inside rect
	QPoint transformToScreen(const QRect& rect, int density, int alpha) const;

///\fn int getIntDensity(const QRect& rect, double density) const
///\brief Converts a floating point density to an integer density
///\param rect The rectangle that defines the drawing area
///\param density The floating point density
///\return The integer version of density
	int getIntDensity(const QRect& rect, double density) const;
///\fn int getIntAlpha(const QRect& rect, double alpha) const
///\brief Converts a floating point alpha to an integer alpha
///\param rect The rectangle that defines the drawing area
///\param alpha The floating point alpha
///\return The integer version of alpha
	int getIntAlpha(const QRect& rect, double alpha) const;

///\fn double getNormalizedDensity(const QRect& rect, const QPoint& source) const
///\brief Returns the normalized density value for a point inside of a
/// rectangle.
///\param rect The rectangle
///\param source The point
///\return A double in the range [0,1]
	double getNormalizedDensity(const QRect& rect, const QPoint& source) const;
///\fn double getNormalizedAlpha(const QRect& rect, const QPoint& source) const
///\brief Returns the normalized alpha value for a point inside of a rectangle.
///\param rect The rectangle
///\param source The point
///\return A double in the range [0,1]
	double getNormalizedAlpha(const QRect& rect, const QPoint& source) const;

///\fn int getIntDensity(const QRect& rect, const QPoint& source) const;
///\brief Returns the integer density value for a point inside of a rectangle.
///\param rect The rectangle
///\param source The point
///\return An int
	int getIntDensity(const QRect& rect, const QPoint& source) const;
///\fn int getIntAlpha(const QRect& rect, const QPoint& source) const;
///\brief Returns the integer alpha value for a point inside of a rectangle.
///\param rect The rectangle
///\param source The point
///\return An int
	int getIntAlpha(const QRect& rect, const QPoint& source) const;

///\fn inline int mapToPixel(double input, int start, int end) const;
///\brief Converts from a double in the range [0,1] to an int in the range
/// [start, end].
///\param input The value being converted
///\param start The minimum output value
///\param end The maximum output value
///\return An int in the range [start,end]
	inline int mapToPixel(double input, int start, int end) const;
///\fn inline double mapToDouble(int input, int start, int end, double min=0.0, double max=1.0) const;
///\brief Does the inverse of mapToPixel
	inline double mapToDouble(int input, int start, int end, double min=0.0, double max=1.0) const;
///\fn inline int possibleFlip(int input, int start, int end) const;
///\brief Hard to explain. This just returns (end - input + start). I think
/// that it's supposed to give a kind of 'mirror image' of the input value.
/// Yeah... the drawing code is kinda convoluted.
///\param input The value to be 'flipped'
///\param start The minimum value of the range
///\param end The maximum value of the range
///\return A 'flipped' number. Whatever that is.
	inline int possibleFlip(int input, int start, int end) const;
	
	// inherited from QFrame
	virtual void drawContents( QPainter* painter );

///\fn void drawColorMap( QPainter* painter, QRect rect );
///\brief Draws the ColorMap
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
	void drawColorMap( QPainter* painter, QRect rect );
///\fn void drawColorBar( QPainter* painter, QRect rect, int index );
///\brief Draws a node from the ColorMap
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
///\param index The index of the node in the ColorMap object
	void drawColorBar( QPainter* painter, QRect rect, int index );

///\fn void drawIsocontourBar( QPainter* painter, QRect rect, int index );
///\brief Draws a node from the IsocontourMap
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
///\param index The index of the node in the IsocontourMap object
	void drawIsocontourBar( QPainter* painter, QRect rect, int index );

///\fn void drawAlphaNode( QPainter* painter, QRect rect, int index );
///\brief Draws a node from the AlphaMap
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
///\param index The index of the node in the AlphaMap object
	void drawAlphaNode( QPainter* painter, QRect rect, int index );
///\fn void drawAlphaMap( QPainter* painter, QRect rect );
///\brief Draws the AlphaMap
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
	void drawAlphaMap( QPainter* painter, QRect rect );

///\fn void drawContourSpectrum( QPainter* painter, QRect rect );
///\brief Draws the contour spectrum
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
	void drawContourSpectrum( QPainter* painter, QRect rect );
///\fn inline void normalizeFunc( float *func, int len, float &minval, float &maxval );
///\brief Normalizes a function of the contour spectrum. (could be used on any
/// function represented by an array)
///\param func The function to normalize (an array)
///\param len The number of entries in the array
///\param minval The minimum function value before normalization
///\param maxval The maximum function value before normalization
	inline void normalizeFunc( float *func, int len, float &, float & );
///\fn inline float getSpecFuncValue( int func, double density );
///\brief Returns the un-normalized function value for a contour spectrum
/// function.
///\param func The index of the function
///\param density A density to get a function value for
///\return A function value
	inline float getSpecFuncValue( int func, double density );
	
///\fn void drawConTreeNode( QPainter* painter, QRect rect, int index );
///\brief Draws a node from the ConTreeMap
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
///\param index The index of the node in the ConTreeMap
	void drawConTreeNode( QPainter* painter, QRect rect, int index );
///\fn void drawContourTree( QPainter* painter, QRect rect );
///\brief Draws the contour tree
///\param painter A QPainter object to draw with
///\param rect The bounding rectangle
	void drawContourTree( QPainter* painter, QRect rect );

	ColorTableInformation* m_ColorTableInformation;

	// hack: Qt4 doesn't like this...
	//ContourSpectrumInfo *m_ConSpecInfo;

	double m_RangeMin;
	double m_RangeMax;

	double m_IsoMin;
	double m_IsoMax;

	float m_SpecFuncsMin[5];
	float m_SpecFuncsMax[5];

	enum MOVEMODE {DRAG_MOVE, NO_MOVE};

	enum POPUPSELECTION {ADD_ALPHA, ADD_COLOR, ADD_ISOCONTOUR,
				DISP_CONTOUR_SPECTRUM, DISP_CONTOUR_TREE, DISP_ALPHA_MAP,
				DELETE_SELECTION,EDIT_SELECTION, SAVE_MAP, LOAD_MAP,
				ADD_MENU, DISP_MENU};

	enum SELECTEDTYPE {ALPHA_NODE, COLOR_NODE, ISOCONTOUR_NODE, CONTOURTREE_NODE,
					NO_NODE};

///\fn POPUPSELECTION showPopup(QPoint point)
///\brief Shows a context menu when the widget is right-clicked
///\param point The point where the mouse was clicked
///\return A POPUPSELECTION
	POPUPSELECTION showPopup(QPoint point);

	MOVEMODE m_MoveMode ;
	SELECTEDTYPE m_SelectedType;
	int m_SelectedNode ;
	int m_SelectedCTEdge;
	double m_SelectedIsovalue;
	
	float *m_SpecFuncs[5];
	bool m_DrawContourSpec;
	bool m_DrawContourTree;
	bool m_DrawAlphaMap;

	// these functions are inherited from QWidget
	virtual void mouseMoveEvent( QMouseEvent* q );
	virtual void mousePressEvent( QMouseEvent* q );
	virtual void mouseReleaseEvent( QMouseEvent* q );
	virtual void contextMenuEvent( QContextMenuEvent* e);

///\fn bool selectNodes( int x, int y )
///\brief Tries to find a node that was clicked on. This is just a proxy for
/// selectColorNodes, selectIsocontourNodes, selectAlphaNodes, selectCTEdge, and
/// selectCTNodes.
///\param x The X coordinate of the mouseclick
///\param y The Y coordinate of the mouseclick
///\return A bool indicating whether or not a node was clicked
	bool selectNodes( int x, int y );
///\fn bool selectColorNodes( int x, int y )
///\brief Tries to find a ColorMap node where a mouseclick has occurred
///\param x The X coordinate of the mouseclick
///\param y The Y coordinate of the mouseclick
///\return A bool indicating whether or not a node was clicked
	bool selectColorNodes( int x, int y );
///\fn bool selectIsocontourNodes( int x, int y )
///\brief Tries to find an IsocontourMap node where a mouseclick has occurred
///\param x The X coordinate of the mouseclick
///\param y The Y coordinate of the mouseclick
///\return A bool indicating whether or not a node was clicked
	bool selectIsocontourNodes( int x, int y );
///\fn bool selectAlphaNodes( int x, int y )
///\brief Tries to find an AlphaMap node where a mouseclick has occurred
///\param x The X coordinate of the mouseclick
///\param y The Y coordinate of the mouseclick
///\return A bool indicating whether or not a node was clicked
	bool selectAlphaNodes( int x, int y );
///\fn bool selectCTEdge( int x, int y )
///\brief Tries to find a contour tree edge where a mouseclick has occurred
///\param x The X coordinate of the mouseclick
///\param y The Y coordinate of the mouseclick
///\return A bool indicating whether or not a node was clicked
	bool selectCTEdge( int x, int y );
///\fn bool selectCTNodes( int x, int y )
///\brief Tries to find a ConTreeMap node where a mouseclick has occurred
///\param x The X coordinate of the mouseclick
///\param y The Y coordinate of the mouseclick
///\return A bool indicating whether or not a node was clicked
	bool selectCTNodes( int x, int y );

public slots:
///\fn void setMin( double min )
///\brief Called when the minimum displayed range value is changed by XoomeOut
	void setMin( double min );
///\fn void setMax( double max )
///\brief Called when the maximum displayed range value is changed by XoomedOut
	void setMax( double max );

///\fn void setSpectrumFunctions( float *isoval, float *area, float *min_vol, float *max_vol, float *gradient )
///\brief Called when the contour spectrum changes (ie- the variable or dataset
/// changes)
	void setSpectrumFunctions( float *isoval, float *area, float *min_vol, float *max_vol, float *gradient );
///\fn void setCTGraph( int numVerts, int numEdges, CTVTX* verts, CTEDGE* edges )
///\brief Called when the contour tree changes (ie- the variable or dataset
/// changes)
	void setCTGraph( int numVerts, int numEdges, CTVTX* verts, CTEDGE* edges );
///\fn void setDataMinMax( double min, double max )
///\brief Called when the dataset changes and there are new min/max function
/// values.
	void setDataMinMax( double min, double max );

	void moveIsocontourNode(int node, double value);

signals:
/*	void alphaNodeChanged( int index, double value, double position );
	void alphaNodeExploring( int index, double value, double position );
	void alphaNodeAdded( int index, double value, double position );
	void alphaNodeDeleted( int index, double value, double position );

	void colorNodeChanged( int index, double red, double green, double blue, double position );
	void colorNodeExploring( int index, double red, double green, double blue, double position );
	void colorNodeAdded( int index, double red, double green, double blue, double position );
	void colorNodeDeleted( int index, double red, double green, double blue, double position );
*/
///\fn void isocontourNodeChanged( int index, double isovalue )
///\brief Signals when an isocontour node has moved
	void isocontourNodeChanged( int index, double isovalue );
///\fn void isocontourNodeColorChanged( int index, double R, double G, double B )
///\brief Signals when an isocontour node's color has changed
	void isocontourNodeColorChanged( int index, double R, double G, double B );
///\fn void isocontourNodeExploring( int index, double isovalue )
///\brief Signals that an isocontour node is currently moving
	void isocontourNodeExploring( int index, double isovalue );
///\fn void isocontourNodeAdded( int index, double isovalue, double R, double G, double B)
///\brief Signals when a new isocontour node is added
	void isocontourNodeAdded( int index, double isovalue, double R, double G, double B);
///\fn void isocontourNodeDeleted( int index )
///\brief Signals when an isocontour node is deleted
	void isocontourNodeDeleted( int index );
///\fn void isocontourNodeEditRequest( int index )
///\brief Signals when the edit menu was selected for a particular isocontour node
	void isocontourNodeEditRequest( int index );

///\fn void contourTreeNodeChanged( int index, double isovalue )
///\brief Signals when a contour tree node has moved
	void contourTreeNodeChanged( int index, double isovalue );
	//void contourTreeNodeColorChanged( int index, double R, double G, double B );
///\fn void contourTreeNodeExploring( int index, double isovalue )
///\brief Signals that a contour tree node is currently moving
	void contourTreeNodeExploring( int index, double isovalue );
///\fn void contourTreeNodeAdded( int index, int edge, double isovalue)
///\brief Signals when a new contour tree node is added
	void contourTreeNodeAdded( int index, int edge, double isovalue);
///\fn void contourTreeNodeDeleted( int index )
///\brief Signals when a contour tree node is deleted
	void contourTreeNodeDeleted( int index );

///\fn void acquireContourSpectrum()
///\brief Signals when the user has requested that the contour spectrum be
/// displayed
	void acquireContourSpectrum();
///\fn void acquireContourTree()
///\brief Signals when the user has requested that the contour tree be displayed
	void acquireContourTree();

///\fn void functionChanged( )
///\brief Signals when the transfer function has changed
	void functionChanged( );
///\fn void functionExploring( )
///\brief Signals that the transfer function is currently changing
	void functionExploring( );

///\fn void everythingChanged()
///\brief Signals that a .vinay file has been loaded
	void everythingChanged();

	void everythingChanged(ColorTableInformation*);
};

#endif

