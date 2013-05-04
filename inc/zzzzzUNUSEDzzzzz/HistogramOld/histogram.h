#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include <Histogram/glcontrolwidget.h>
#include <Histogram/histogram_data.h>

//Added by qt3to4:
#include <QWheelEvent>

class Histogram : public GLControlWidget
{
		Q_OBJECT
	public:
		Histogram(QWidget* parent = 0, const char* name = 0, WFlags f = 0, HistogramData data = HistogramData());
		~Histogram() {}
		void setData(const HistogramData& newData);
		float getMinWidth();
		float getMaxWidth();

	protected:
		void draw();
		void animate();
		void initializeGL();
		void resizeGL(int, int);
		void paintGL();
		void mousePressEvent(QMouseEvent* e);
		void mouseReleaseEvent(QMouseEvent* e);
		void mouseMoveEvent(QMouseEvent*);
		void mouseDoubleClickEvent(QMouseEvent*);
		void wheelEvent(QWheelEvent*);

	signals:
		void valueChanged(float, float);

	private:
		bool binSelected(unsigned int bin);
		bool _dragging;
		float _startWidth, _endWidth;
		float _cursorWidth;
		static const float DEFAULT_START_WIDTH;
		static const float DEFAULT_END_WIDTH;
		int _width, _height;
		HistogramData _data;
		static const float BOTTOM_BORDER_PERCENT;
		static const int FONT_BORDER_X;
		static const int FONT_BORDER_Y;
		static const int FONT_SIZE;
		static const int SMOOTHING_EXTENT;
		static const QColor BKGND_INACTIVE;
		static const QColor BKGND_ACTIVE;
		static const QColor LINE_INACTIVE;
		static const QColor LINE_ACTIVE;
};

#endif
