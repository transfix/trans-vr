#include "../../inc/RenderServers/textureserversettingsdialog.Qt3.h"

#include <qvariant.h>
/*
 *  Constructs a TextureServerSettingsDialogBase as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  true to construct a modal dialog.
 */
TextureServerSettingsDialogBase::TextureServerSettingsDialogBase(QWidget* parent, const char* name, bool modal, Qt::WindowFlags fl)
    : QDialog(parent, name, modal, fl)
{
    setupUi(this);

}

/*
 *  Destroys the object and frees any allocated resources
 */
TextureServerSettingsDialogBase::~TextureServerSettingsDialogBase()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void TextureServerSettingsDialogBase::languageChange()
{
    retranslateUi(this);
}

