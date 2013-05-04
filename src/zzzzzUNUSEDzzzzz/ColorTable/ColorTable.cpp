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

#include <ColorTable/ColorTable.h>
#include <qlayout.h>

CVC::ColorTable::ColorTable( QWidget *parent, const char *name )
: QFrame( parent, name )
{
	setFrameStyle( QFrame::Panel | QFrame::Raised );

	QBoxLayout *layout = new QBoxLayout( this, QBoxLayout::Down );
	layout->setMargin( 3 );
	layout->setSpacing( 3 );

	m_XoomedIn = new XoomedIn( &m_ColorTableInformation, this, "Xoomed In" );
	m_XoomedOut = new XoomedOut( &m_ColorTableInformation, m_XoomedIn, this, "Xoomed Out" );

	layout->addWidget( m_XoomedOut );
	layout->addWidget( m_XoomedIn );
	m_XoomedOut->setFixedHeight( 20 );

	// connect the signals and slots
	connect(m_XoomedIn, SIGNAL(isocontourNodeChanged(int, double)), this, SLOT(relayIsocontourNodeChanged(int, double)));
	connect(m_XoomedIn, SIGNAL(isocontourNodeColorChanged(int, double,double,double)), this, SLOT(relayIsocontourNodeColorChanged(int, double,double,double)));
	connect(m_XoomedIn, SIGNAL(isocontourNodeExploring(int, double)), this, SLOT(relayIsocontourNodeExploring(int, double)));
	connect(m_XoomedIn, SIGNAL(isocontourNodeAdded(int, double,double,double,double)), this, SLOT(relayIsocontourNodeAdded(int, double,double,double,double)));
	connect(m_XoomedIn, SIGNAL(isocontourNodeDeleted(int)),  this, SLOT(relayIsocontourNodeDeleted(int)));
	connect(m_XoomedIn, SIGNAL(isocontourNodeEditRequest(int)),  this, SLOT(relayIsocontourNodeEditRequest(int)));

	connect(m_XoomedIn, SIGNAL(contourTreeNodeChanged(int, double)), this, SLOT(relayContourTreeNodeChanged(int, double)));
	//connect(m_XoomedIn, SIGNAL(contourTreeNodeColorChanged(int, double,double,double)), this, SLOT(relayContourTreeNodeColorChanged(int, double,double,double)));
	connect(m_XoomedIn, SIGNAL(contourTreeNodeExploring(int, double)), this, SLOT(relayContourTreeNodeExploring(int, double)));
	connect(m_XoomedIn, SIGNAL(contourTreeNodeAdded(int,int,double)), this, SLOT(relayContourTreeNodeAdded(int,int,double)));
	connect(m_XoomedIn, SIGNAL(contourTreeNodeDeleted(int)),  this, SLOT(relayContourTreeNodeDeleted(int)));

	connect(m_XoomedIn, SIGNAL(functionChanged()),  this, SLOT(relayFunctionChanged()));
	connect(m_XoomedIn, SIGNAL(functionExploring()),  this, SLOT(relayFunctionExploring()));

	connect(m_XoomedIn, SIGNAL(everythingChanged()), this, SLOT(relayEverythingChanged()));
	connect(m_XoomedIn, SIGNAL(everythingChanged(ColorTableInformation*)), this, SLOT(relayEverythingChanged(ColorTableInformation*)));
	
	connect(m_XoomedIn, SIGNAL(acquireContourSpectrum()), this, SLOT(relayAcquireContourSpectrum()));
	connect(m_XoomedIn, SIGNAL(acquireContourTree()), this, SLOT(relayAcquireContourTree()));
	connect(this, SIGNAL(spectrumFunctionsChanged(float*,float*,float*,float*,float*)), m_XoomedIn, SLOT(setSpectrumFunctions(float*,float*,float*,float*,float*)));
	connect(this, SIGNAL(contourTreeChanged(int,int,CTVTX*,CTEDGE*)), m_XoomedIn, SLOT(setCTGraph(int,int,CTVTX*,CTEDGE*)));
	connect(this, SIGNAL(dataMinMaxChanged(double,double)), m_XoomedIn,SLOT(setDataMinMax(double,double)));
}

CVC::ColorTable::~ColorTable()
{

}

int CVC::ColorTable::MapToPixel(float input, int start, int end)
{
	if (input <= 1.0 && input >= 0.0) {
		int width = end-start;
		int offset = (int)(input * width);
		return start + offset;
	}
	else if (input > 1.0) {
		return end;
	}
	else {
		return start;
	}
}

double CVC::ColorTable::MapToDouble(int input, int start, int end)
{
	int width = end-start;
	int offset = input-start;

	return (double)offset/(double)width;
}

void CVC::ColorTable::GetTransferFunction(double *pMap, int size)
{
	double mag;
	int pos1, pos2;
	double cr1, cr2;
	double cg1, cg2;
	double cb1, cb2;
	double alpha1, alpha2;

	int mapSize = m_ColorTableInformation.getColorMap().GetSize();
	int i;
	for (i=0; i<mapSize-1; i++) {
		pos1 = MapToPixel(m_ColorTableInformation.getColorMap().GetPosition(i), 0, size-1);
		pos2 = MapToPixel(m_ColorTableInformation.getColorMap().GetPosition(i+1), 0, size-1);
		cr1 = m_ColorTableInformation.getColorMap().GetRed(i);
		cr2 = m_ColorTableInformation.getColorMap().GetRed(i+1);
		cg1 = m_ColorTableInformation.getColorMap().GetGreen(i);
		cg2 = m_ColorTableInformation.getColorMap().GetGreen(i+1);
		cb1 = m_ColorTableInformation.getColorMap().GetBlue(i);
		cb2 = m_ColorTableInformation.getColorMap().GetBlue(i+1);
		for (int y=pos1; y<=pos2; y++) {
			mag = MapToDouble(y, pos1, pos2);
			pMap[y*4] = cr1*(1.0f-mag) + cr2*(mag);
			pMap[y*4+1] = cg1*(1.0f-mag) + cg2*(mag);
			pMap[y*4+2] = cb1*(1.0f-mag) + cb2*(mag);
		}
	}
	mapSize = m_ColorTableInformation.getAlphaMap().GetSize();
	for (i=0; i<mapSize-1; i++) {
		pos1 = MapToPixel(m_ColorTableInformation.getAlphaMap().GetPosition(i), 0, size-1);
		pos2 = MapToPixel(m_ColorTableInformation.getAlphaMap().GetPosition(i+1), 0, size-1);
		alpha1 = m_ColorTableInformation.getAlphaMap().GetAlpha(i+0);
		alpha2 = m_ColorTableInformation.getAlphaMap().GetAlpha(i+1);
		//alpha1 = MapToPixel(m_ColorTableInformation.getAlphaMap().GetAlpha(i), 0, 255);
		//alpha2 = MapToPixel(m_ColorTableInformation.getAlphaMap().GetAlpha(i+1), 0, 255);
		for (int y=pos1; y<=pos2; y++) {
			mag = MapToDouble(y, pos1, pos2);
			pMap[y*4+3] = alpha1*(1.0f-mag) + alpha2*(mag);
		}
	}
}

QSize CVC::ColorTable::sizeHint() const
{
	return QSize(150, 150);
}

ColorTableInformation CVC::ColorTable::getColorTableInformation()
{
	return ColorTableInformation(m_ColorTableInformation);
}

const IsocontourMap& CVC::ColorTable::getIsocontourMap() const
{
	return m_ColorTableInformation.getIsocontourMap();
}

void CVC::ColorTable::setSpectrumFunctions(float *isoval, float *area, float *min_vol, float *max_vol, float *gradient)
{
	emit spectrumFunctionsChanged(isoval,area,min_vol,max_vol,gradient);
}

void CVC::ColorTable::setContourTree(int numVerts, int numEdges, CTVTX *verts, CTEDGE *edges)
{
	emit contourTreeChanged(numVerts,numEdges,verts,edges);
}

void CVC::ColorTable::setDataMinMax( double min, double max )
{
	emit dataMinMaxChanged(min,max);
}

void CVC::ColorTable::moveIsocontourNode(int node, double value)
{
  m_XoomedIn->moveIsocontourNode(node,value);
}

void CVC::ColorTable::relayIsocontourNodeChanged( int index, double isovalue )
{
	emit isocontourNodeChanged(index, isovalue);
}

void CVC::ColorTable::relayIsocontourNodeColorChanged( int index, double R, double G, double B )
{
	emit isocontourNodeColorChanged(index, R,G,B);
}

void CVC::ColorTable::relayIsocontourNodeExploring( int index, double isovalue )
{
	emit isocontourNodeExploring(index, isovalue);
}

void CVC::ColorTable::relayIsocontourNodeAdded( int index, double isovalue, double R, double G, double B )
{
	emit isocontourNodeAdded(index, isovalue, R,G,B);
}

void CVC::ColorTable::relayIsocontourNodeDeleted( int index )
{
	emit isocontourNodeDeleted(index);
}

void CVC::ColorTable::relayIsocontourNodeEditRequest( int index )
{
	emit isocontourNodeEditRequest(index);
}

void CVC::ColorTable::relayContourTreeNodeChanged( int index, double isovalue )
{
	emit contourTreeNodeChanged(index, isovalue);
}

/*void ColorTable::relayContourTreeNodeColorChanged( int index, double R, double G, double B )
{
	emit socontourNodeColorChanged(index, R,G,B);
}*/

void CVC::ColorTable::relayContourTreeNodeExploring( int index, double isovalue )
{
	emit contourTreeNodeExploring(index, isovalue);
}

void CVC::ColorTable::relayContourTreeNodeAdded( int index, int edge, double isovalue )
{
	emit contourTreeNodeAdded(index, edge, isovalue);
}

void CVC::ColorTable::relayContourTreeNodeDeleted( int index )
{
	emit contourTreeNodeDeleted(index);
}

void CVC::ColorTable::relayFunctionChanged( )
{
	m_XoomedOut->repaint();
	emit functionChanged();
}

void CVC::ColorTable::relayFunctionExploring( )
{
	m_XoomedOut->repaint();
	emit functionExploring();
}


void CVC::ColorTable::relayAcquireContourSpectrum()
{
	emit acquireContourSpectrum();
}

void CVC::ColorTable::relayAcquireContourTree()
{
	emit acquireContourTree();
}


void CVC::ColorTable::relayEverythingChanged()
{
	m_XoomedOut->repaint();
	emit everythingChanged();
}

void CVC::ColorTable::relayEverythingChanged(ColorTableInformation *cti)
{
	m_XoomedOut->repaint();
	emit everythingChanged(cti);
}
