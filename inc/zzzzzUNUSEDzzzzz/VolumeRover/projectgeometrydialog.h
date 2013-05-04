#ifndef __PROJECTGEOMETRYDIALOG_H__
#define __PROJECTGEOMETRYDIALOG_H__

#include "projectgeometrydialogbase.Qt3.h"

class ProjectGeometryDialog : public ProjectGeometryDialogBase
{
Q_OBJECT

 public:
  ProjectGeometryDialog();
  ~ProjectGeometryDialog();

 protected:
  void openFileDialog();

};

#endif
