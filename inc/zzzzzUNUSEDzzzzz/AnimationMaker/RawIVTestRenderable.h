// RawIVTestRenderable.h: interface for the RawIVTestRenderable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAWIVTESTRENDERABLE_H__DDDEEF24_F38D_4B7B_AF96_1588F8F96344__INCLUDED_)
#define AFX_RAWIVTESTRENDERABLE_H__DDDEEF24_F38D_4B7B_AF96_1588F8F96344__INCLUDED_

#include <VolumeWidget/VolumeRenderable.h>
#include <qstring.h>

class RawIVTestRenderable : public VolumeRenderable  
{
public:
	RawIVTestRenderable();
	virtual ~RawIVTestRenderable();

	bool loadFile(const char* fileName);

	bool setFileName(const char* fileName);

	virtual bool initForContext();

	QString m_FileName;

};

#endif // !defined(AFX_RAWIVTESTRENDERABLE_H__DDDEEF24_F38D_4B7B_AF96_1588F8F96344__INCLUDED_)
