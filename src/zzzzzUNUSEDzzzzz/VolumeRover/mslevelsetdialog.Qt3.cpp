#include "../../inc/VolumeRover/mslevelsetdialog.Qt3.h"

#include <qvariant.h>
#include <qvalidator.h>
/*
 *  Constructs a MSLevelSetDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  true to construct a modal dialog.
 */
MSLevelSetDialog::MSLevelSetDialog(QWidget* parent, const char* name, bool modal, Qt::WindowFlags fl)
    : QDialog(parent, name, modal, fl)
{
    setupUi(this);

    init();
}

/*
 *  Destroys the object and frees any allocated resources
 */
MSLevelSetDialog::~MSLevelSetDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void MSLevelSetDialog::languageChange()
{
    retranslateUi(this);
}

void MSLevelSetDialog::on_DTInitComboBox_activated(int)
{
    qWarning("MSLevelSetDialog::on_DTInitComboBox_activated(int): Not implemented yet");
}

void MSLevelSetDialog::on_BlockDimEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_BlockDimEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_DTWidth_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_DTWidth_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_DeltaTEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_DeltaTEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_EllipsoidPowerEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_EllipsoidPowerEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_EpsilonEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_EpsilonEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_Lambda1Edit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_Lambda1Edit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_Lambda2Edit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_Lambda2Edit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_MaxMedianIterEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_MaxMedianIterEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_MaxSolverIterEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_MaxSolverIterEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_MedianTolEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_MedianTolEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_MuEdit_textChangedSLot()
{
    qWarning("MSLevelSetDialog::on_MuEdit_textChangedSLot(): Not implemented yet");
}

void MSLevelSetDialog::on_NuEdit_textChangedSLot()
{
    qWarning("MSLevelSetDialog::on_NuEdit_textChangedSLot(): Not implemented yet");
}

void MSLevelSetDialog::on_SubvolDimEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_SubvolDimEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::on_BlockDimComboBox_activatedSlot(int)
{
    qWarning("MSLevelSetDialog::on_BlockDimComboBox_activatedSlot(int): Not implemented yet");
}

void MSLevelSetDialog::on_BBoxOffsetEdit_textChangedSlot()
{
    qWarning("MSLevelSetDialog::on_BBoxOffsetEdit_textChangedSlot(): Not implemented yet");
}

void MSLevelSetDialog::paramReference( MSLevelSetParams *)
{
    qWarning("MSLevelSetDialog::paramReference( MSLevelSetParams *): Not implemented yet");
}

void MSLevelSetDialog::init()
{
}

