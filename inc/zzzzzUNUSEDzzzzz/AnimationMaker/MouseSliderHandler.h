// MouseSliderHandler.h: interface for the MouseSliderHandler class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MOUSESLIDERHANDLER_H__73797BD9_A8E4_4A40_8E28_8DC1482FE50C__INCLUDED_)
#define AFX_MOUSESLIDERHANDLER_H__73797BD9_A8E4_4A40_8E28_8DC1482FE50C__INCLUDED_

#include <VolumeWidget/MouseHandler.h>

class MouseSliderHandler : public MouseHandler  
{
public:
	MouseSliderHandler(double* target);
	virtual ~MouseSliderHandler();
	//MouseSliderHandler(const MouseSliderHandler& copy);

	virtual MouseHandler* clone() const;

	virtual bool mousePressEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool mouseReleaseEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool mouseDoubleClickEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool mouseMoveEvent(SimpleOpenGLWidget*, QMouseEvent* e);
	virtual bool wheelEvent(SimpleOpenGLWidget*, QWheelEvent* e);
protected:
	virtual void startDrag(SimpleOpenGLWidget*, int x, int y);
	virtual void mouseMove(SimpleOpenGLWidget*, int x, int y);
	virtual void endDrag(SimpleOpenGLWidget*, int x, int y);
	virtual void doInteraction(SimpleOpenGLWidget*, int startX, int startY, int endX, int endY);

	void setDefaults();
	inline int invertY(int y, int height) const;

	bool m_ButtonDown;
	int m_OldX, m_OldY;

	double* m_Target;
};

#endif // !defined(AFX_MOUSESLIDERHANDLER_H__73797BD9_A8E4_4A40_8E28_8DC1482FE50C__INCLUDED_)
