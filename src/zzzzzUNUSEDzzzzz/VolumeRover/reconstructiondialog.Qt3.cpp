#include "../../inc/VolumeRover/reconstructiondialog.Qt3.h"

#include <qvariant.h>
/*
 *  Constructs a ReconstructionDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  true to construct a modal dialog.
 */
ReconstructionDialog::ReconstructionDialog(QWidget* parent, const char* name, bool modal, Qt::WindowFlags fl)
    : QDialog(parent, name, modal, fl)
{
    setupUi(this);

}

/*
 *  Destroys the object and frees any allocated resources
 */
ReconstructionDialog::~ReconstructionDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void ReconstructionDialog::languageChange()
{
    retranslateUi(this);
}

void ReconstructionDialog::changeParamsValuesSlot(int)
{
    qWarning("ReconstructionDialog::changeParamsValuesSlot(int): Not implemented yet");
}

void ReconstructionDialog::changeStatusSlot()
{
    qWarning("ReconstructionDialog::changeStatusSlot(): Not implemented yet");
}

void ReconstructionDialog::changeReconMannerSlot(int)
{
    qWarning("ReconstructionDialog::changeReconMannerSlot(int): Not implemented yet");
}

void ReconstructionDialog::browseSlot()
{
    qWarning("ReconstructionDialog::browseSlot(): Not implemented yet");
}

void ReconstructionDialog::browseloadinitfSlot()
{
    qWarning("ReconstructionDialog::browseloadinitfSlot(): Not implemented yet");
}

