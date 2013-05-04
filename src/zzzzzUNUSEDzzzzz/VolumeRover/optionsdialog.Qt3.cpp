#include "../../inc/VolumeRover/optionsdialog.Qt3.h"

#include <qvariant.h>
/*
 *  Constructs a OptionsDialogBase as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  true to construct a modal dialog.
 */
OptionsDialogBase::OptionsDialogBase(QWidget* parent, const char* name, bool modal, Qt::WindowFlags fl)
    : QDialog(parent, name, modal, fl)
{
    setupUi(this);

}

/*
 *  Destroys the object and frees any allocated resources
 */
OptionsDialogBase::~OptionsDialogBase()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void OptionsDialogBase::languageChange()
{
    retranslateUi(this);
}

void OptionsDialogBase::browseSlot()
{
    qWarning("OptionsDialogBase::browseSlot(): Not implemented yet");
}

void OptionsDialogBase::colorSlot()
{
    qWarning("OptionsDialogBase::colorSlot(): Not implemented yet");
}

