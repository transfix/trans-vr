#include "../../inc/VolumeRover/newvolumemainwindowbase.Qt3.h"

#include <qvariant.h>
#include "VolumeWidget/SimpleOpenGLWidget.h"
/*
 *  Constructs a NewVolumeMainWindowBase as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
NewVolumeMainWindowBase::NewVolumeMainWindowBase(QWidget* parent, const char* name, Qt::WindowFlags fl)
    : Q3MainWindow(parent, name, fl)
{
    setupUi(this);

    (void)statusBar();
}

/*
 *  Destroys the object and frees any allocated resources
 */
NewVolumeMainWindowBase::~NewVolumeMainWindowBase()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void NewVolumeMainWindowBase::languageChange()
{
    retranslateUi(this);
}

void NewVolumeMainWindowBase::actionSlot()
{
    qWarning("NewVolumeMainWindowBase::actionSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::functionChangedSlot()
{
    qWarning("NewVolumeMainWindowBase::functionChangedSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::setExplorerQualitySlot(int)
{
    qWarning("NewVolumeMainWindowBase::setExplorerQualitySlot(int): Not implemented yet");
}

void NewVolumeMainWindowBase::mouseReleasedMain()
{
    qWarning("NewVolumeMainWindowBase::mouseReleasedMain(): Not implemented yet");
}

void NewVolumeMainWindowBase::mouseReleasedPreview()
{
    qWarning("NewVolumeMainWindowBase::mouseReleasedPreview(): Not implemented yet");
}

void NewVolumeMainWindowBase::explorerChangedSlot()
{
    qWarning("NewVolumeMainWindowBase::explorerChangedSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::connectToDCSlot()
{
    qWarning("NewVolumeMainWindowBase::connectToDCSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::setMainQualitySlot(int)
{
    qWarning("NewVolumeMainWindowBase::setMainQualitySlot(int): Not implemented yet");
}

void NewVolumeMainWindowBase::optionsSlot()
{
    qWarning("NewVolumeMainWindowBase::optionsSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::explorerMoveSlot()
{
    qWarning("NewVolumeMainWindowBase::explorerMoveSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::explorerReleaseSlot()
{
    qWarning("NewVolumeMainWindowBase::explorerReleaseSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::connectServerSlot()
{
    qWarning("NewVolumeMainWindowBase::connectServerSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::serverSettingsSlot()
{
    qWarning("NewVolumeMainWindowBase::serverSettingsSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::disconnectServerSlot()
{
    qWarning("NewVolumeMainWindowBase::disconnectServerSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::renderAnimationSlot()
{
    qWarning("NewVolumeMainWindowBase::renderAnimationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::renderFrameSlot()
{
    qWarning("NewVolumeMainWindowBase::renderFrameSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::isocontourNodeColorChangedSlot(int,double,double,double)
{
    qWarning("NewVolumeMainWindowBase::isocontourNodeColorChangedSlot(int,double,double,double): Not implemented yet");
}

void NewVolumeMainWindowBase::isocontourNodeChangedSlot(int,double)
{
    qWarning("NewVolumeMainWindowBase::isocontourNodeChangedSlot(int,double): Not implemented yet");
}

void NewVolumeMainWindowBase::isocontourNodeAddedSlot(int,double,double,double,double)
{
    qWarning("NewVolumeMainWindowBase::isocontourNodeAddedSlot(int,double,double,double,double): Not implemented yet");
}

void NewVolumeMainWindowBase::isocontourNodeDeletedSlot(int)
{
    qWarning("NewVolumeMainWindowBase::isocontourNodeDeletedSlot(int): Not implemented yet");
}

void NewVolumeMainWindowBase::isocontourNodesAllChangedSlot()
{
    qWarning("NewVolumeMainWindowBase::isocontourNodesAllChangedSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::bilateralFilterSlot()
{
    qWarning("NewVolumeMainWindowBase::bilateralFilterSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::saveSubvolumeSlot()
{
    qWarning("NewVolumeMainWindowBase::saveSubvolumeSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::zoomedOutClipSlot(int)
{
    qWarning("NewVolumeMainWindowBase::zoomedOutClipSlot(int): Not implemented yet");
}

void NewVolumeMainWindowBase::zoomedInClipSlot(int)
{
    qWarning("NewVolumeMainWindowBase::zoomedInClipSlot(int): Not implemented yet");
}

void NewVolumeMainWindowBase::centerSlot()
{
    qWarning("NewVolumeMainWindowBase::centerSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::loadGeometrySlot()
{
    qWarning("NewVolumeMainWindowBase::loadGeometrySlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::clearGeometrySlot()
{
    qWarning("NewVolumeMainWindowBase::clearGeometrySlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::saveGeometrySlot()
{
    qWarning("NewVolumeMainWindowBase::saveGeometrySlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::exportZoomedInIsosurfaceSlot()
{
    qWarning("NewVolumeMainWindowBase::exportZoomedInIsosurfaceSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::exportZoomedOutIsosurfaceSlot()
{
    qWarning("NewVolumeMainWindowBase::exportZoomedOutIsosurfaceSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::variableOrTimeChangeSlot()
{
    qWarning("NewVolumeMainWindowBase::variableOrTimeChangeSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::contrastEnhancementSlot()
{
    qWarning("NewVolumeMainWindowBase::contrastEnhancementSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::PEDetectionSlot()
{
    qWarning("NewVolumeMainWindowBase::PEDetectionSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::pocketTunnelSlot()
{
    qWarning("NewVolumeMainWindowBase::pocketTunnelSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::smoothGeometrySlot()
{
    qWarning("NewVolumeMainWindowBase::smoothGeometrySlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::anisotropicDiffusionSlot()
{
    qWarning("NewVolumeMainWindowBase::anisotropicDiffusionSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::sliceRenderingSlot()
{
    qWarning("NewVolumeMainWindowBase::sliceRenderingSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::acquireConTreeSlot()
{
    qWarning("NewVolumeMainWindowBase::acquireConTreeSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::acquireConSpecSlot()
{
    qWarning("NewVolumeMainWindowBase::acquireConSpecSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::toggleWireframeRenderingSlot(bool)
{
    qWarning("NewVolumeMainWindowBase::toggleWireframeRenderingSlot(bool): Not implemented yet");
}

void NewVolumeMainWindowBase::toggleRenderSurfaceWithWireframeSlot(bool)
{
    qWarning("NewVolumeMainWindowBase::toggleRenderSurfaceWithWireframeSlot(bool): Not implemented yet");
}

void NewVolumeMainWindowBase::toggleWireCubeSlot(bool)
{
    qWarning("NewVolumeMainWindowBase::toggleWireCubeSlot(bool): Not implemented yet");
}

void NewVolumeMainWindowBase::toggleDepthCueSlot(bool)
{
    qWarning("NewVolumeMainWindowBase::toggleDepthCueSlot(bool): Not implemented yet");
}

void NewVolumeMainWindowBase::saveImageSlot()
{
    qWarning("NewVolumeMainWindowBase::saveImageSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::contourTreeNodeAddedSlot(int,int,double)
{
    qWarning("NewVolumeMainWindowBase::contourTreeNodeAddedSlot(int,int,double): Not implemented yet");
}

void NewVolumeMainWindowBase::contourTreeNodeDeletedSlot(int)
{
    qWarning("NewVolumeMainWindowBase::contourTreeNodeDeletedSlot(int): Not implemented yet");
}

void NewVolumeMainWindowBase::contourTreeNodeChangedSlot(int,double)
{
    qWarning("NewVolumeMainWindowBase::contourTreeNodeChangedSlot(int,double): Not implemented yet");
}

void NewVolumeMainWindowBase::contourTreeNodeExploringSlot(int,double)
{
    qWarning("NewVolumeMainWindowBase::contourTreeNodeExploringSlot(int,double): Not implemented yet");
}

void NewVolumeMainWindowBase::startRecordingAnimationSlot()
{
    qWarning("NewVolumeMainWindowBase::startRecordingAnimationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::stopRecordingAnimationSlot()
{
    qWarning("NewVolumeMainWindowBase::stopRecordingAnimationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::playAnimationSlot()
{
    qWarning("NewVolumeMainWindowBase::playAnimationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::saveAnimationSlot()
{
    qWarning("NewVolumeMainWindowBase::saveAnimationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::loadAnimationSlot()
{
    qWarning("NewVolumeMainWindowBase::loadAnimationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::stopAnimationSlot()
{
    qWarning("NewVolumeMainWindowBase::stopAnimationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::renderSequenceSlot()
{
    qWarning("NewVolumeMainWindowBase::renderSequenceSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::virusSegmentationSlot()
{
    qWarning("NewVolumeMainWindowBase::virusSegmentationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::toggleGeometryTransformationSlot()
{
    qWarning("NewVolumeMainWindowBase::toggleGeometryTransformationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::resetGeometryTransformationSlot()
{
    qWarning("NewVolumeMainWindowBase::resetGeometryTransformationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::toggleTerminalSlot(bool)
{
    qWarning("NewVolumeMainWindowBase::toggleTerminalSlot(bool): Not implemented yet");
}

void NewVolumeMainWindowBase::boundaryPointCloudSlot()
{
    qWarning("NewVolumeMainWindowBase::boundaryPointCloudSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::tightCoconeSlot()
{
    qWarning("NewVolumeMainWindowBase::tightCoconeSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::curationSlot()
{
    qWarning("NewVolumeMainWindowBase::curationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::skeletonizationSlot()
{
    qWarning("NewVolumeMainWindowBase::skeletonizationSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::clipGeometryToVolumeBoxSlot(bool)
{
    qWarning("NewVolumeMainWindowBase::clipGeometryToVolumeBoxSlot(bool): Not implemented yet");
}

void NewVolumeMainWindowBase::saveSkeletonSlot()
{
    qWarning("NewVolumeMainWindowBase::saveSkeletonSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::clearSkeletonSlot()
{
    qWarning("NewVolumeMainWindowBase::clearSkeletonSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::signedDistanceFunctionSlot()
{
    qWarning("NewVolumeMainWindowBase::signedDistanceFunctionSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::mergeGeometrySlot()
{
    qWarning("NewVolumeMainWindowBase::mergeGeometrySlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::convertIsosurfaceToGeometrySlot()
{
    qWarning("NewVolumeMainWindowBase::convertIsosurfaceToGeometrySlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::highLevelSetReconSlot()
{
    qWarning("NewVolumeMainWindowBase::highLevelSetReconSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::highLevelSetSlot()
{
    qWarning("NewVolumeMainWindowBase::highLevelSetSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::LBIEMeshingSlot()
{
    qWarning("NewVolumeMainWindowBase::LBIEMeshingSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::LBIEQualityImprovementSlot()
{
    qWarning("NewVolumeMainWindowBase::LBIEQualityImprovementSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::openImageFileSlot()
{
    qWarning("NewVolumeMainWindowBase::openImageFileSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::projectGeometrySlot()
{
    qWarning("NewVolumeMainWindowBase::projectGeometrySlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::gdtvFilterSlot()
{
    qWarning("NewVolumeMainWindowBase::gdtvFilterSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::MSLevelSetSlot()
{
    qWarning("NewVolumeMainWindowBase::MSLevelSetSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::isocontourAskIsovalueSlot(int)
{
    qWarning("NewVolumeMainWindowBase::isocontourAskIsovalueSlot(int): Not implemented yet");
}

void NewVolumeMainWindowBase::colorGeometryByVolumeSlot()
{
    qWarning("NewVolumeMainWindowBase::colorGeometryByVolumeSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::cullGeometryWithSubvolumeBoxSlot()
{
    qWarning("NewVolumeMainWindowBase::cullGeometryWithSubvolumeBoxSlot(): Not implemented yet");
}

void NewVolumeMainWindowBase::secondaryStructureElucidationSlot()
{
    qWarning("NewVolumeMainWindowBase::secondaryStructureElucidationSlot(): Not implemented yet");
}

