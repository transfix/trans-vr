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
#ifndef COLORTABLE_H
#define COLORTABLE_H
#include <qframe.h>
#include <ColorTable/ColorTableInformation.h>
#include <ColorTable/XoomedIn.h>
#include <ColorTable/XoomedOut.h>

#include <contourtree/computeCT.h>

namespace CVC {

///\class ColorTable ColorTable.h
///\author Anthony Thane
///\author Vinay Siddavanahalli
///\author John Wiggins
///\brief The ColorTable class is a QFrame-derived widget that holds the 
/// transfer function UI for Volume Rover, TexMol, Volume Video, and any other
/// software that needs a 1D transfer function. It contains a XoomedIn widget 
/// and a XoomedOut widget, which are the guts of the UI. It also has a 
/// ColorTableInformation object that holds the actual transfer function.
class ColorTable : public QFrame 
{
	Q_OBJECT
public:
///\fn ColorTable( QWidget *parent, const char *name )
///\brief The constructor
///\param parent The widget that will contain this ColorTable
///\param name The name of this widget
	ColorTable( QWidget *parent = 0, const char *name = 0 );
	~ColorTable();

///\fn int MapToPixel(float input, int start, int end)
///\brief Maps from a float in the range [0,1] to the given integer range
///\param input The value to be mapped
///\param start The value mapped to 0
///\param end The value mapped to 1
///\return The mapped version of input
	int MapToPixel(float input, int start, int end);
///\fn double MapToDouble(int input, int start, int end)
///\brief Maps from an int to a double in the range [0,1].
///\param input The value to be mapped
///\param start The value mapped to 0
///\param end The value mapped to 1
///\return The mapped version of input
	double MapToDouble(int input, int start, int end);
///\fn void GetTransferFunction(double *pMap, int size)
///\brief This function converts the transfer function to a flat array of 
/// doubles. Each 'item' in the array is a tuple of 4 doubles like so: 
/// (red, green, blue, alpha).
///\param pMap The array of tuples
///\param size The number of tuples in the array (the size of the array is actually size*4 if you're counting doubles)
	void GetTransferFunction(double *pMap, int size);
///\fn QSize sizeHint() const
///\brief Returns the preferred size of the widget
///\return A QSize object
	QSize sizeHint() const;
///\fn ColorTableInformation getColorTableInformation()
///\brief Returns the ColorTableInformation for the widget
///\return A ColorTableInformation object
	ColorTableInformation getColorTableInformation();

///\fn const IsocontourMap& getIsocontourMap() const
///\brief Returns the IsocontourMap for the widget
///\return An IsocontourMap object
	const IsocontourMap& getIsocontourMap() const;

///\fn void setSpectrumFunctions(float *isoval, float *area, float *min_vol, float *max_vol, float *gradient)
///\brief This function is called by NewVolumeMainWindow to pass a contour
/// spectrum to the ColorTable
///\param isoval The isovalues for the contour spectrum
///\param area The area function
///\param min_vol The minimum volume function
///\param max_vol The maximum volume function
///\param gradient The gradient function
	void setSpectrumFunctions(float *isoval, float *area, float *min_vol, float *max_vol, float *gradient);
///\fn void setContourTree(int numVerts, int numEdges, CTVTX *verts, CTEDGE *edges)
///\brief This function is called by NewVolumeMainWindow to pass a contour tree
/// to the ColorTable
///\param numVerts The number of vertices in the contour tree
///\param numEdges The number of edges in the contour tree
///\param verts The array of vertices
///\param edges The array of edges
	void setContourTree(int numVerts, int numEdges, CTVTX *verts, CTEDGE *edges);
///\fn void setDataMinMax( double min, double max )
///\brief Sets the min and max values for the current dataset. These values are
/// used to display
/// the isovalues of isocontour nodes
///\param min The minimum function value for the current dataset.
///\param max The maximum function value for the current dataset.
	void setDataMinMax( double min, double max );

public slots:
	void moveIsocontourNode(int node, double value);

protected:

	ColorTableInformation m_ColorTableInformation;

	XoomedIn* m_XoomedIn;
	XoomedOut* m_XoomedOut;

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
///\brief Signals when an isocontour node has been moved
	void isocontourNodeChanged( int index, double isovalue );
///\fn void isocontourNodeColorChanged( int index, double R, double G, double B )
///\brief Signals when an isocontour node's color has been changed
	void isocontourNodeColorChanged( int index, double R, double G, double B );
///\fn void isocontourNodeExploring( int index, double isovalue )
///\brief Signals while an isocontour node is being moved
	void isocontourNodeExploring( int index, double isovalue );
///\fn void isocontourNodeAdded( int index, double isovalue, double R, double G, double B )
///\brief Signals when a new isocontour has been added
	void isocontourNodeAdded( int index, double isovalue, double R, double G, double B );
///\fn void isocontourNodeDeleted( int index )
///\brief Signals when an isocontour node has been deleted
	void isocontourNodeDeleted( int index );
///\fn void isocontourNodeEditRequest( int index )
///\brief Signals when the edit menu was selected for a particular isocontour node
	void isocontourNodeEditRequest( int index );
	
///\fn void contourTreeNodeChanged( int index, double isovalue )
///\brief Signals when a contour tree node has been moved
	void contourTreeNodeChanged( int index, double isovalue );
	//void contourTreeNodeColorChanged( int index, double R, double G, double B );
///\fn void contourTreeNodeExploring( int index, double isovalue )
///\brief Signals while a contour tree node is being moved
	void contourTreeNodeExploring( int index, double isovalue );
///\fn void contourTreeNodeAdded( int index, int edge, double isovalue)
///\brief Signals when a new contour tree node has been added
	void contourTreeNodeAdded( int index, int edge, double isovalue);
///\fn void contourTreeNodeDeleted( int index )
///\brief Signals when a contour tree node has been deleted
	void contourTreeNodeDeleted( int index );

///\fn void functionChanged( )
///\brief Signals when a color or alpha node has been moved
	void functionChanged( );
///\fn void functionExploring( )
///\brief Signals while a color or alpha node is being moved
	void functionExploring( );
	
///\fn void acquireContourSpectrum()
///\brief Signals when the user has requested the contour spectrum to be displayed
	void acquireContourSpectrum();
///\fn void acquireContourTree()
///\brief Signals when the user has requested the contour tree to be displayed
	void acquireContourTree();
///\fn void spectrumFunctionsChanged( float *isoval, float *area, float *min_vol, float *max_vol, float *gradient)
///\brief Signals when a contour spectrum has been received
	void spectrumFunctionsChanged( float *isoval, float *area, float *min_vol, float *max_vol, float *gradient);
///\fn void contourTreeChanged( int numVerts, int numEdges, CTVTX *verts, CTEDGE *edges)
///\brief Signals when a contour tree has been received
	void contourTreeChanged( int numVerts, int numEdges, CTVTX *verts, CTEDGE *edges);
///\fn void dataMinMaxChanged(double min, double max)
///\brief Signals when the setDataMinMax function has been called
	void dataMinMaxChanged(double min, double max);

///\fn void everythingChanged()
///\brief Signals when a transfer function has been loaded from a file
	void everythingChanged();
	
	void everythingChanged(ColorTableInformation*);

protected slots:
/*	void relayAlphaNodeChanged( int index, double value, double position );
	void relayAlphaNodeExploring( int index, double value, double position );
	void relayAlphaNodeAdded( int index, double value, double position );
	void relayAlphaNodeDeleted( int index, double value, double position );

	void relayColorNodeChanged( int index, double red, double green, double blue, double position );
	void relayColorNodeExploring( int index, double red, double green, double blue, double position );
	void relayColorNodeAdded( int index, double red, double green, double blue, double position );
	void relayColorNodeDeleted( int index, double red, double green, double blue, double position );
*/
///\fn void relayIsocontourNodeChanged( int index, double isovalue )
///\brief Relay for the XoomedIn widget
	void relayIsocontourNodeChanged( int index, double isovalue );
///\fn void relayIsocontourNodeColorChanged( int index, double R, double G, double B )
///\brief Relay for the XoomedIn widget
	void relayIsocontourNodeColorChanged( int index, double R, double G, double B );
///\fn void relayIsocontourNodeExploring( int index, double isovalue )
///\brief Relay for the XoomedIn widget
	void relayIsocontourNodeExploring( int index, double isovalue );
///\fn void relayIsocontourNodeAdded( int index, double isovalue, double R, double G, double B )
///\brief Relay for the XoomedIn widget
	void relayIsocontourNodeAdded( int index, double isovalue, double R, double G, double B );
///\fn void relayIsocontourNodeDeleted( int index )
///\brief Relay for the XoomedIn widget
	void relayIsocontourNodeDeleted( int index );
///\fn void relayIsocontourNodeEditRequest( int index )
///\brief Relay for the XoomedIn widget
	void relayIsocontourNodeEditRequest( int index );
	
///\fn void relayContourTreeNodeChanged( int index, double isovalue )
///\brief Relay for the XoomedIn widget
	void relayContourTreeNodeChanged( int index, double isovalue );
	//void relayContourTreeNodeColorChanged( int index, double R, double G, double B )
///\fn void relayContourTreeNodeExploring( int index, double isovalue )
///\brief Relay for the XoomedIn widget
	void relayContourTreeNodeExploring( int index, double isovalue );
///\fn void relayContourTreeNodeAdded( int index, int edge, double isovalue )
///\brief Relay for the XoomedIn widget
	void relayContourTreeNodeAdded( int index, int edge, double isovalue );
///\fn void relayContourTreeNodeDeleted( int index )
///\brief Relay for the XoomedIn widget
	void relayContourTreeNodeDeleted( int index );

///\fn void relayFunctionChanged( )
///\brief Relay for the XoomedIn widget
	void relayFunctionChanged( );
///\fn void relayFunctionExploring( )
///\brief Relay for the XoomedIn widget
	void relayFunctionExploring( );

///\fn void relayAcquireContourSpectrum()
///\brief Relay for the XoomedIn widget
	void relayAcquireContourSpectrum();
///\fn void relayAcquireContourTree()
///\brief Relay for the XoomedIn widget
	void relayAcquireContourTree();

///\fn void relayEverythingChanged()
///\brief Relay for the XoomedIn widget
	void relayEverythingChanged();

	void relayEverythingChanged(ColorTableInformation*);
};

};

#endif

/********************* transfer function format **********************

Anthony and Vinay are Great.

Alphamap
No of nodes
< no of nodes >
Position and opacity
< Position A >
< Position A >
< Position A >
< Position A >
< Position A >

Colormap
No of nodes
< no of nodes >
Position and colors R, G, B
< Position R G B >
< Position R G B >
< Position R G B >
< Position R G B >
< Position R G B >

Isocontours
No of isovalues
< no of isovalues >
Isovalues
< iso 1 >
< iso 2 >
< iso 3 >
< iso 4 >

********************************************************************/
