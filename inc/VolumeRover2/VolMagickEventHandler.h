#ifndef __VOLMAGICKEVENTHANDLER_H__
#define __VOLMAGICKEVENTHANDLER_H__

#include <qapplication.h>
#include <qprogressdialog.h>
#include <qevent.h>
#include <VolMagick/VolMagick.h>

class VolMagickOpStartEvent : public QCustomEvent
{
 public:
  static const int id = 60000;

  VolMagickOpStartEvent(const VolMagick::Voxels *vox, 
			VolMagick::VoxelOperationStatusMessenger::Operation op, 
			VolMagick::uint64 numSteps) : 
    QCustomEvent(id), voxels(vox), operation(op), steps(numSteps) {}
  const VolMagick::Voxels *voxels;
  VolMagick::VoxelOperationStatusMessenger::Operation operation;
  VolMagick::uint64 steps;
};

class VolMagickOpStepEvent : public QCustomEvent
{
 public:
  static const int id = 60001;

  VolMagickOpStepEvent(const VolMagick::Voxels *vox, 
			VolMagick::VoxelOperationStatusMessenger::Operation op, 
			VolMagick::uint64 curStep) : 
    QCustomEvent(id), voxels(vox), operation(op), currentStep(curStep) {}
  const VolMagick::Voxels *voxels;
  VolMagick::VoxelOperationStatusMessenger::Operation operation;
  VolMagick::uint64 currentStep;
};

class VolMagickOpEndEvent : public QCustomEvent
{
 public:
  static const int id = 60002;

  VolMagickOpEndEvent(const VolMagick::Voxels *vox, 
		       VolMagick::VoxelOperationStatusMessenger::Operation op) : 
    QCustomEvent(id), voxels(vox), operation(op) {}
  const VolMagick::Voxels *voxels;
  VolMagick::VoxelOperationStatusMessenger::Operation operation;
};

class VolMagickEventBasedOpStatus : public VolMagick::VoxelOperationStatusMessenger
{
public:
  void start(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 numSteps) const
  {
    QApplication::postEvent(qApp->mainWidget(),new VolMagickOpStartEvent(vox,op,numSteps));
  }
  
  void step(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 curStep) const
  {
    QApplication::postEvent(qApp->mainWidget(),new VolMagickOpStepEvent(vox,op,curStep));
  }
  
  void end(const VolMagick::Voxels *vox, Operation op) const
  {
    QApplication::postEvent(qApp->mainWidget(),new VolMagickOpEndEvent(vox,op));
  }
};

class VolMagickOpStatus : public VolMagick::VoxelOperationStatusMessenger
{
public:
  VOLMAGICK_DEF_EXCEPTION(OperationCancelled);

  VolMagickOpStatus() : pd(NULL) {}
  
  ~VolMagickOpStatus() { if(pd) delete pd; }
  
  void start(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 numSteps) const
  {
    const char *opStrings[] = { "Calculating Min/Max", "Calculating Min", "Calculating Max",
                                "Subvolume Extraction", "Fill", "Map", "Resize", "Composite",
                                "Bilateral Filter", "Contrast Enhancement", "Anisotropic Diffusion"};
    
    _numSteps = numSteps;
    
#if QT_VERSION < 0x040000
    pd = new QProgressDialog(opStrings[op],"Abort",numSteps,NULL,NULL,TRUE);
#else
    pd = new QProgressDialog(opStrings[op],"Abort",0,numSteps);
    pd->setWindowModality(Qt::WindowModal);
#endif
  }
  
  void step(const VolMagick::Voxels *vox, Operation op, VolMagick::uint64 curStep) const
  {
    //fprintf(stderr,"%s: %5.2f %%\r",opStrings[op],(((float)curStep)/((float)((int)(_numSteps-1))))*100.0);
#if QT_VERSION < 0x040000
    pd->setProgress(curStep);
#else
    pd->setValue(curStep);
#endif
    
    qApp->processEvents();
    if(pd->wasCancelled())
      throw OperationCancelled();
  }
  
  void end(const VolMagick::Voxels *vox, Operation op) const
  {
#if QT_VERSION < 0x040000
    pd->setProgress(_numSteps);
#else
    pd->setValue(_numSteps);
#endif
  }
  
private:
  mutable QProgressDialog *pd;
  mutable VolMagick::uint64 _numSteps;
};

#endif
