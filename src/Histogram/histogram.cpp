/*
  Copyright 2011 The University of Texas at Austin

	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TexMol.

  TexMol is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TexMol is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
#include <cmath>

#if defined(__APPLE__)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif


#include <Histogram/histogram.h>
//Added by qt3to4:
#include <QWheelEvent>
#include <QMouseEvent>

const float Histogram::BOTTOM_BORDER_PERCENT = 0.2f;
const float Histogram::DEFAULT_START_WIDTH = 2.5 - .5;
const float Histogram::DEFAULT_END_WIDTH = 2.5 + .5;
const int Histogram::FONT_BORDER_X = 4;
const int Histogram::FONT_BORDER_Y = 8;
const int Histogram::FONT_SIZE = 12;
const int Histogram::SMOOTHING_EXTENT = 4;
const QColor Histogram::BKGND_INACTIVE =  QColor(0, 0, 0);
const QColor Histogram::BKGND_ACTIVE = QColor(128, 128, 128);
const QColor Histogram::LINE_INACTIVE = QColor(0, 128, 0);
const QColor Histogram::LINE_ACTIVE = QColor(0, 255, 0);

Histogram::Histogram(QWidget* parent, const char* name, Qt::WFlags f, HistogramData data)
	: GLControlWidget(parent, name, 0, f)
{
	_data = data;
	_width = 0;
	_height = 0;
	_dragging = false;
	_cursorWidth = 0;
	_startWidth = DEFAULT_START_WIDTH;
	_endWidth = DEFAULT_END_WIDTH;
	// this widget needs to get events whenever a mouse motion occurs (even without a button down)
	setMouseTracking(true);
}

void Histogram::draw()
{
	glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glLineWidth(1.0);
	float histogram_bottom = _height * BOTTOM_BORDER_PERCENT;
	float histogram_height = _height - histogram_bottom;
	glBegin(GL_LINES);
	{
		for (int i=0; i<_data.width(); i++)
		{
			// draw background
			if (binSelected(i))
			{
				qglColor(BKGND_ACTIVE);
			}
			else
			{
				qglColor(BKGND_INACTIVE);
			}
			glVertex2f(i, histogram_bottom);
			glVertex2f(i, _height);
			// draw graph line
			if (binSelected(i))
			{
				qglColor(LINE_ACTIVE);
			}
			else
			{
				qglColor(LINE_INACTIVE);
			}
			// average over 4 bins (smoothing)
			float histogram_top_average = 0;
			for (int q=0; q<SMOOTHING_EXTENT; q++)
			{
				int index = i + 1 + q - SMOOTHING_EXTENT/2;
				if (index < 0)
				{
					index = 0;
				}
				if (index >= _data.width())
				{
					index = (_data.width() - 1);
				}
				float histogram_top = histogram_bottom + (_data.getBinNormalized(index) * histogram_height);
				histogram_top_average += histogram_top * (1.0f/SMOOTHING_EXTENT);
			}
			glVertex2f(i, histogram_bottom);
			glVertex2f(i, histogram_top_average);
		}
	}
	glEnd();
	// draw the text for the cursor pos
	{
		qglColor(Qt::green);
		glLineWidth(1.0);
		QString cursorString = QString::number(_cursorWidth, 'f', 5);
		renderText(FONT_BORDER_X, FONT_BORDER_Y, 0.0, cursorString, QFont("courier", FONT_SIZE, QFont::Bold, FALSE));
	}
	glPopAttrib();
}

void Histogram::initializeGL()
{
	glEnable(GL_NORMALIZE);
}

void Histogram::resizeGL(int width, int height)
{
	_width = width;
	_height = height;
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	// 2d-pixel-mode
	glOrtho(0,width,0,height,-1,1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	// tell data to resize itself
	_data.rebin(_width);
}

void Histogram::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
	draw();
	glPopMatrix();
}

void Histogram::animate()
{
	updateGL();
}

void Histogram::mousePressEvent(QMouseEvent* e)
{
	e->accept();
	// start drag gesture
	_dragging = true;
	float width = _data.binToWidth(e->x());
	_startWidth = _endWidth = width;
}

void Histogram::mouseReleaseEvent(QMouseEvent* e)
{
	e->accept();
	// end drag gesture
	if (_dragging)
	{
		_dragging = false;
		float width = _data.binToWidth(e->x());
		_endWidth = width;
	}
	emit valueChanged(getMinWidth(),getMaxWidth());
}

void Histogram::mouseMoveEvent(QMouseEvent* e)
{
	e->accept();
	_cursorWidth = _data.binToWidth(e->x());
	if (_dragging)
	{
		_endWidth = _cursorWidth;
	}
}

void Histogram::wheelEvent(QWheelEvent* e)
{
	e->accept();
}

void Histogram::mouseDoubleClickEvent(QMouseEvent* e)
{
	e->accept();
}

bool Histogram::binSelected(unsigned int bin)
{
	float width = _data.binToWidth(bin);
	return ((getMinWidth() <= width) && (width <= getMaxWidth()));
}

float Histogram::getMinWidth()
{
	if (_startWidth<_endWidth)
	{
		return _startWidth;
	}
	else
	{
		return _endWidth;
	}
}

float Histogram::getMaxWidth()
{
	if (_startWidth<_endWidth)
	{
		return _endWidth;
	}
	else
	{
		return _startWidth;
	}
}

void Histogram::setData(const HistogramData& newData)
{
	_data = newData;
	_data.rebin(_width);
}
