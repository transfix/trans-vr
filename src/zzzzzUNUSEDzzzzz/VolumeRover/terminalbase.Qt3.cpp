#include "../../inc/VolumeRover/terminalbase.Qt3.h"

#include <qvariant.h>
#include <qimage.h>
#include <qpixmap.h>

/*
 *  Constructs a TerminalBase as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
TerminalBase::TerminalBase(QWidget* parent, const char* name, Qt::WindowFlags fl)
    : QWidget(parent, name, fl)
{
    setupUi(this);

}

/*
 *  Destroys the object and frees any allocated resources
 */
TerminalBase::~TerminalBase()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void TerminalBase::languageChange()
{
    retranslateUi(this);
}

