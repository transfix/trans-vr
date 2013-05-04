#include <q3filedialog.h>
#include <qlineedit.h>
#include <VolumeRover/projectgeometrydialog.h>

ProjectGeometryDialog::ProjectGeometryDialog() {}
ProjectGeometryDialog::~ProjectGeometryDialog() {}

void ProjectGeometryDialog::openFileDialog()
{
  m_FileName->setText(Q3FileDialog::getOpenFileName(QString::null, 
						   "raw files (*.raw*)", 
						   this, "open file dialog", "Choose a file"));
}
