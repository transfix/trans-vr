#include "../../inc/VolumeGridRover/VolumeGridRoverBase.Qt3.h"

#include <qvariant.h>
#include <qimage.h>
#include <qpixmap.h>

/*
 *  Constructs a VolumeGridRoverBase as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 */
VolumeGridRoverBase::VolumeGridRoverBase(QWidget* parent, const char* name, Qt::WindowFlags fl)
    : QWidget(parent, name, fl)
{
    setupUi(this);

}

/*
 *  Destroys the object and frees any allocated resources
 */
VolumeGridRoverBase::~VolumeGridRoverBase()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void VolumeGridRoverBase::languageChange()
{
    retranslateUi(this);
}

void VolumeGridRoverBase::colorSlot()
{
    qWarning("VolumeGridRoverBase::colorSlot(): Not implemented yet");
}

void VolumeGridRoverBase::addPointClassSlot()
{
    qWarning("VolumeGridRoverBase::addPointClassSlot(): Not implemented yet");
}

void VolumeGridRoverBase::deletePointClassSlot()
{
    qWarning("VolumeGridRoverBase::deletePointClassSlot(): Not implemented yet");
}

void VolumeGridRoverBase::getLocalOutputFileSlot()
{
    qWarning("VolumeGridRoverBase::getLocalOutputFileSlot(): Not implemented yet");
}

void VolumeGridRoverBase::localSegmentationRunSlot()
{
    qWarning("VolumeGridRoverBase::localSegmentationRunSlot(): Not implemented yet");
}

void VolumeGridRoverBase::remoteSegmentationRunSlot()
{
    qWarning("VolumeGridRoverBase::remoteSegmentationRunSlot(): Not implemented yet");
}

void VolumeGridRoverBase::getRemoteFileSlot()
{
    qWarning("VolumeGridRoverBase::getRemoteFileSlot(): Not implemented yet");
}

void VolumeGridRoverBase::sliceAxisChangedSlot()
{
    qWarning("VolumeGridRoverBase::sliceAxisChangedSlot(): Not implemented yet");
}

void VolumeGridRoverBase::zChangedSlot()
{
    qWarning("VolumeGridRoverBase::zChangedSlot(): Not implemented yet");
}

void VolumeGridRoverBase::yChangedSlot()
{
    qWarning("VolumeGridRoverBase::yChangedSlot(): Not implemented yet");
}

void VolumeGridRoverBase::xChangedSlot()
{
    qWarning("VolumeGridRoverBase::xChangedSlot(): Not implemented yet");
}

void VolumeGridRoverBase::EMClusteringRunSlot()
{
    qWarning("VolumeGridRoverBase::EMClusteringRunSlot(): Not implemented yet");
}

void VolumeGridRoverBase::backgroundColorSlot()
{
    qWarning("VolumeGridRoverBase::backgroundColorSlot(): Not implemented yet");
}

void VolumeGridRoverBase::savePointClassesSlot()
{
    qWarning("VolumeGridRoverBase::savePointClassesSlot(): Not implemented yet");
}

void VolumeGridRoverBase::loadPointClassesSlot()
{
    qWarning("VolumeGridRoverBase::loadPointClassesSlot(): Not implemented yet");
}

void VolumeGridRoverBase::addContourSlot()
{
    qWarning("VolumeGridRoverBase::addContourSlot(): Not implemented yet");
}

void VolumeGridRoverBase::deleteContourSlot()
{
    qWarning("VolumeGridRoverBase::deleteContourSlot(): Not implemented yet");
}

void VolumeGridRoverBase::contourColorSlot()
{
    qWarning("VolumeGridRoverBase::contourColorSlot(): Not implemented yet");
}

void VolumeGridRoverBase::tilingRunSlot()
{
    qWarning("VolumeGridRoverBase::tilingRunSlot(): Not implemented yet");
}

void VolumeGridRoverBase::cellMarkingModeTabChangedSlot(QWidget *)
{
    qWarning("VolumeGridRoverBase::cellMarkingModeTabChangedSlot(QWidget *): Not implemented yet");
}

void VolumeGridRoverBase::loadContoursSlot()
{
    qWarning("VolumeGridRoverBase::loadContoursSlot(): Not implemented yet");
}

void VolumeGridRoverBase::setInterpolationTypeSlot(int)
{
    qWarning("VolumeGridRoverBase::setInterpolationTypeSlot(int): Not implemented yet");
}

void VolumeGridRoverBase::setInterpolationSamplingSlot(int)
{
    qWarning("VolumeGridRoverBase::setInterpolationSamplingSlot(int): Not implemented yet");
}

void VolumeGridRoverBase::getTilingOutputDirectorySlot()
{
    qWarning("VolumeGridRoverBase::getTilingOutputDirectorySlot(): Not implemented yet");
}

void VolumeGridRoverBase::handleTilingOutputDestinationSelectionSlot(int)
{
    qWarning("VolumeGridRoverBase::handleTilingOutputDestinationSelectionSlot(int): Not implemented yet");
}

void VolumeGridRoverBase::saveContoursSlot()
{
    qWarning("VolumeGridRoverBase::saveContoursSlot(): Not implemented yet");
}

void VolumeGridRoverBase::sdfCurationSlot()
{
    qWarning("VolumeGridRoverBase::sdfCurationSlot(): Not implemented yet");
}

void VolumeGridRoverBase::sdfOptionsSlot()
{
    qWarning("VolumeGridRoverBase::sdfOptionsSlot(): Not implemented yet");
}

void VolumeGridRoverBase::medialAxisSlot()
{
    qWarning("VolumeGridRoverBase::medialAxisSlot(): Not implemented yet");
}

void VolumeGridRoverBase::curateContoursSlot()
{
    qWarning("VolumeGridRoverBase::curateContoursSlot(): Not implemented yet");
}

