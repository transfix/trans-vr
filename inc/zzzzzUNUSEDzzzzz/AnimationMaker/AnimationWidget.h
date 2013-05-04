// AnimationWidget.h: interface for the AnimationWidget class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ANIMATIONWIDGET_H__704B3F03_7E83_4AC2_9F36_3D0F8B29E57B__INCLUDED_)
#define AFX_ANIMATIONWIDGET_H__704B3F03_7E83_4AC2_9F36_3D0F8B29E57B__INCLUDED_

#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <stdio.h>
#include <AnimationMaker/Animation.h>
#include <qdatetime.h>

#include <VolumeWidget/RenderableArray.h>
#include <VolumeFileTypes/SourceManager.h>
#include <VolumeWidget/Extents.h>
#include <AnimationMaker/RoverRenderable.h>
#include <qdir.h>

class AnimationWidget : public SimpleOpenGLWidget  
{
public:
	AnimationWidget();
	virtual ~AnimationWidget();

	void recordTo(FILE* fp);
	void playBackFrom(FILE* fp);
	double m_NearPlane;

protected:
	virtual void paintGL();
	virtual QSize sizeHint() const;
	void saveImage();

	void timerEvent( QTimerEvent * );

	virtual void keyPressEvent( QKeyEvent* e );

	virtual void initializeGL();
	virtual void initRawV();
	virtual void initRawIV();
	virtual void initScene();
	virtual void initGeometryScene();

	virtual void mousePressEvent(QMouseEvent* e);
	virtual void mouseReleaseEvent(QMouseEvent* e);
	virtual void mouseDoubleClickEvent(QMouseEvent* e);
	//virtual void mouseMoveEvent(QMouseEvent* e);
	virtual void wheelEvent(QWheelEvent* e);

	void recordFrame();

	bool m_WireFrame;

	bool m_SaveImage;

	unsigned int m_FrameNumber;
	
	FILE* m_WriteToFile;
	FILE* m_ReadFromFile;
	Animation* m_Animation;

	QTime m_Time;
	bool m_Initialized;



	unsigned char* m_UploadBuffer;
	RenderableArray m_Geometries;
	SourceManager m_SourceManager;
	Extents m_Extents;
	RoverRenderable m_RoverRenderable;
	QDir m_CacheDir;


};

#endif // !defined(AFX_ANIMATIONWIDGET_H__704B3F03_7E83_4AC2_9F36_3D0F8B29E57B__INCLUDED_)
