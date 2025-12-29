#ifndef __VOLMAGICKEVENTHANDLER_H__
#define __VOLMAGICKEVENTHANDLER_H__

#include <QApplication>
#include <QEvent>
#include <QProgressDialog>
#include <VolMagick/VolMagick.h>

class VolMagickOpStartEvent : public QEvent {
public:
  static const int id = QEvent::User + 60000;

  VolMagickOpStartEvent(
      const VolMagick::Voxels *vox,
      VolMagick::VoxelOperationStatusMessenger::Operation op,
      VolMagick::uint64 numSteps)
      : QEvent(static_cast<QEvent::Type>(id)), voxels(vox), operation(op),
        steps(numSteps) {}
  const VolMagick::Voxels *voxels;
  VolMagick::VoxelOperationStatusMessenger::Operation operation;
  VolMagick::uint64 steps;
};

class VolMagickOpStepEvent : public QEvent {
public:
  static const int id = QEvent::User + 60001;

  VolMagickOpStepEvent(const VolMagick::Voxels *vox,
                       VolMagick::VoxelOperationStatusMessenger::Operation op,
                       VolMagick::uint64 curStep)
      : QEvent(static_cast<QEvent::Type>(id)), voxels(vox), operation(op),
        currentStep(curStep) {}
  const VolMagick::Voxels *voxels;
  VolMagick::VoxelOperationStatusMessenger::Operation operation;
  VolMagick::uint64 currentStep;
};

class VolMagickOpEndEvent : public QEvent {
public:
  static const int id = QEvent::User + 60002;

  VolMagickOpEndEvent(const VolMagick::Voxels *vox,
                      VolMagick::VoxelOperationStatusMessenger::Operation op)
      : QEvent(static_cast<QEvent::Type>(id)), voxels(vox), operation(op) {}
  const VolMagick::Voxels *voxels;
  VolMagick::VoxelOperationStatusMessenger::Operation operation;
};

class VolMagickEventBasedOpStatus
    : public VolMagick::VoxelOperationStatusMessenger {
public:
  void start(const VolMagick::Voxels *vox, Operation op,
             VolMagick::uint64 numSteps) const {
    QApplication::postEvent(qApp,
                            new VolMagickOpStartEvent(vox, op, numSteps));
  }

  void step(const VolMagick::Voxels *vox, Operation op,
            VolMagick::uint64 curStep) const {
    QApplication::postEvent(qApp, new VolMagickOpStepEvent(vox, op, curStep));
  }

  void end(const VolMagick::Voxels *vox, Operation op) const {
    QApplication::postEvent(qApp, new VolMagickOpEndEvent(vox, op));
  }
};

class VolMagickOpStatus : public VolMagick::VoxelOperationStatusMessenger {
public:
  VOLMAGICK_DEF_EXCEPTION(OperationCancelled);

  VolMagickOpStatus() : pd(NULL) {}

  ~VolMagickOpStatus() {
    if (pd)
      delete pd;
  }

  void start(const VolMagick::Voxels *vox, Operation op,
             VolMagick::uint64 numSteps) const {
    const char *opStrings[] = {"Calculating Min/Max",
                               "Calculating Min",
                               "Calculating Max",
                               "Subvolume Extraction",
                               "Fill",
                               "Map",
                               "Resize",
                               "Composite",
                               "Bilateral Filter",
                               "Contrast Enhancement",
                               "Anisotropic Diffusion"};

    _numSteps = numSteps;

    pd = new QProgressDialog(opStrings[op], "Abort", 0, numSteps);
    pd->setWindowModality(Qt::WindowModal);
  }

  void step(const VolMagick::Voxels *vox, Operation op,
            VolMagick::uint64 curStep) const {
    pd->setValue(curStep);

    qApp->processEvents();
    if (pd->wasCancelled())
      throw OperationCancelled();
  }

  void end(const VolMagick::Voxels *vox, Operation op) const {
    pd->setValue(_numSteps);
  }

private:
  mutable QProgressDialog *pd;
  mutable VolMagick::uint64 _numSteps;
};

#endif
