#ifndef NEWVOLUMEMAINWINDOWBASE_H
#define NEWVOLUMEMAINWINDOWBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3MainWindow>
#include <Qt3Support/Q3WidgetStack>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QTabWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "VolumeWidget/SimpleOpenGLWidget.h"

QT_BEGIN_NAMESPACE

class Ui_NewVolumeMainWindowBase
{
public:
    QAction *ConnectToFile;
    QAction *explorerChangedAction;
    QAction *ConnectToDC;
    QAction *ExitAction;
    QAction *WhatsThisAction;
    QAction *CenterAction;
    QAction *OptionsAction;
    QAction *ConnectServerAction;
    QAction *DisconnectServerAction;
    QAction *ServerSettingAction;
    QAction *RenderFrameAction;
    QAction *RenderAnimationAction;
    QAction *BilateralFilterAction;
    QAction *SaveSubVolumeAction;
    QAction *LoadGeometryAction;
    QAction *ClearGeometryAction;
    QAction *ExportZoomedInIsosurfaceAction;
    QAction *ExportZoomedOutIsosurfaceAction;
    QAction *WireCubeToggleAction;
    QAction *WireframeRenderToggleAction;
    QAction *viewUnnamedAction;
    QAction *viewUnnamedAction_2;
    QAction *DepthCueToggleAction;
    QAction *SaveImageAction;
    QAction *StartRecordingAnimationAction;
    QAction *StopRecordingAnimationAction;
    QAction *PlayAnimationAction;
    QAction *SaveAnimationAction;
    QAction *LoadAnimationAction;
    QAction *StopAnimationAction;
    QAction *SaveFrameSequenceAction;
    QAction *SegmentVirusMapAction;
    QAction *EditTransformationAction;
    QAction *ClearTransformationAction;
    QAction *ShowTerminalAction;
    QAction *ShowCorrelationAction;
    QAction *ContrastEnhancementAction;
    QAction *PEDetectionAction;
    QAction *PocketTunnelAction;
    QAction *SaveGeometryAction;
    QAction *geometrySmoothingAction;
    QAction *geometrynew_itemAction;
    QAction *SmoothGeometryAction;
    QAction *AnisotropicDiffusionAction;
    QAction *SliceRenderingAction;
    QAction *TightCoconeAction;
    QAction *CurationAction;
    QAction *SkeletonizationAction;
    QAction *ClipGeometryToVolumeBoxAction;
    QAction *BoundaryPointCloudAction;
    QAction *SaveSkeletonAction;
    QAction *ClearSkeletonAction;
    QAction *SignedDistanceFunctionAction;
    QAction *MergeGeometryObjectsAction;
    QAction *ConvertCurrentIsosurfaceToGeometryAction;
    QAction *HighLevelSetAction;
    QAction *HighLevelSetReconAction;
    QAction *LBIEAction;
    QAction *LBIEMeshingAction;
    QAction *LBIEQualityImprovementAction;
    QAction *RenderSurfaceWithWireframeAction;
    QAction *OpenImageFileAction;
    QAction *ProjectToSurfaceAction;
    QAction *GDTVFilterAction;
    QAction *MumfordShahLevelSetAction;
    QAction *ColorGeometryByVolumeAction;
    QAction *CullGeometryWithSubvolumeBoxAction;
    QAction *toolsnew_itemAction;
    QAction *SecondaryStructureAction;
    QActionGroup *RecentFilesPopup;
    QAction *RecentFile1;
    QWidget *widget;
    QGridLayout *gridLayout;
    QTabWidget *m_ViewTabs;
    QWidget *ThreeDRover;
    QGridLayout *gridLayout1;
    QVBoxLayout *vboxLayout;
    SimpleOpenGLWidget *m_ZoomedIn;
    QGridLayout *gridLayout2;
    QLabel *TextLabel1;
    QLabel *TextLabel1_3;
    QSlider *Slider2;
    QSlider *m_MainNearPlane;
    QVBoxLayout *vboxLayout1;
    SimpleOpenGLWidget *m_ZoomedOut;
    QGridLayout *gridLayout3;
    QSlider *m_ExplorerNearPlane;
    QLabel *TextLabel2;
    QSlider *Slider1;
    QLabel *TextLabel1_2;
    Q3WidgetStack *m_VariableSelectionStack;
    QWidget *Single;
    QHBoxLayout *hboxLayout;
    QLabel *m_VariableLabel;
    QComboBox *m_VariableBox;
    QLabel *m_TimeStepLevel;
    QSpinBox *m_TimeStep;
    QSpacerItem *spacer1;
    QWidget *RGBA;
    QHBoxLayout *hboxLayout1;
    QLabel *m_RedLabel;
    QComboBox *m_RedBox;
    QLabel *m_GreenLabel;
    QComboBox *m_GreenBox;
    QLabel *m_BlueLabel;
    QComboBox *m_BlueBox;
    QLabel *m_AlphaLabel;
    QComboBox *m_AlphaBox;
    QLabel *m_RGBATimeLabel;
    QSpinBox *m_RGBATimeStep;
    QMenuBar *menubar;
    QMenu *m_FileMenu;
    QMenu *m_ViewMenu;
    QMenu *m_GeometryMenu;
    QMenu *popupMenu_39;
    QMenu *m_ServersMenu;
    QMenu *m_AnimationMenu;
    QMenu *Tools;
    QMenu *m_HelpMenu;

    void setupUi(Q3MainWindow *NewVolumeMainWindowBase)
    {
        if (NewVolumeMainWindowBase->objectName().isEmpty())
            NewVolumeMainWindowBase->setObjectName(QString::fromUtf8("NewVolumeMainWindowBase"));
        NewVolumeMainWindowBase->resize(747, 642);
        ConnectToFile = new QAction(NewVolumeMainWindowBase);
        ConnectToFile->setObjectName(QString::fromUtf8("ConnectToFile"));
        ConnectToFile->setName("ConnectToFile");
        explorerChangedAction = new QAction(NewVolumeMainWindowBase);
        explorerChangedAction->setObjectName(QString::fromUtf8("explorerChangedAction"));
        explorerChangedAction->setName("explorerChangedAction");
        ConnectToDC = new QAction(NewVolumeMainWindowBase);
        ConnectToDC->setObjectName(QString::fromUtf8("ConnectToDC"));
        ConnectToDC->setName("ConnectToDC");
        ConnectToDC->setEnabled(true);
        ExitAction = new QAction(NewVolumeMainWindowBase);
        ExitAction->setObjectName(QString::fromUtf8("ExitAction"));
        ExitAction->setName("ExitAction");
        WhatsThisAction = new QAction(NewVolumeMainWindowBase);
        WhatsThisAction->setObjectName(QString::fromUtf8("WhatsThisAction"));
        WhatsThisAction->setName("WhatsThisAction");
        CenterAction = new QAction(NewVolumeMainWindowBase);
        CenterAction->setObjectName(QString::fromUtf8("CenterAction"));
        CenterAction->setName("CenterAction");
        OptionsAction = new QAction(NewVolumeMainWindowBase);
        OptionsAction->setObjectName(QString::fromUtf8("OptionsAction"));
        OptionsAction->setName("OptionsAction");
        OptionsAction->setEnabled(true);
        ConnectServerAction = new QAction(NewVolumeMainWindowBase);
        ConnectServerAction->setObjectName(QString::fromUtf8("ConnectServerAction"));
        ConnectServerAction->setName("ConnectServerAction");
        DisconnectServerAction = new QAction(NewVolumeMainWindowBase);
        DisconnectServerAction->setObjectName(QString::fromUtf8("DisconnectServerAction"));
        DisconnectServerAction->setName("DisconnectServerAction");
        DisconnectServerAction->setEnabled(false);
        ServerSettingAction = new QAction(NewVolumeMainWindowBase);
        ServerSettingAction->setObjectName(QString::fromUtf8("ServerSettingAction"));
        ServerSettingAction->setName("ServerSettingAction");
        ServerSettingAction->setEnabled(false);
        RenderFrameAction = new QAction(NewVolumeMainWindowBase);
        RenderFrameAction->setObjectName(QString::fromUtf8("RenderFrameAction"));
        RenderFrameAction->setName("RenderFrameAction");
        RenderFrameAction->setEnabled(false);
        RenderAnimationAction = new QAction(NewVolumeMainWindowBase);
        RenderAnimationAction->setObjectName(QString::fromUtf8("RenderAnimationAction"));
        RenderAnimationAction->setName("RenderAnimationAction");
        BilateralFilterAction = new QAction(NewVolumeMainWindowBase);
        BilateralFilterAction->setObjectName(QString::fromUtf8("BilateralFilterAction"));
        BilateralFilterAction->setName("BilateralFilterAction");
        SaveSubVolumeAction = new QAction(NewVolumeMainWindowBase);
        SaveSubVolumeAction->setObjectName(QString::fromUtf8("SaveSubVolumeAction"));
        SaveSubVolumeAction->setName("SaveSubVolumeAction");
        SaveSubVolumeAction->setEnabled(true);
        LoadGeometryAction = new QAction(NewVolumeMainWindowBase);
        LoadGeometryAction->setObjectName(QString::fromUtf8("LoadGeometryAction"));
        LoadGeometryAction->setName("LoadGeometryAction");
        ClearGeometryAction = new QAction(NewVolumeMainWindowBase);
        ClearGeometryAction->setObjectName(QString::fromUtf8("ClearGeometryAction"));
        ClearGeometryAction->setName("ClearGeometryAction");
        ExportZoomedInIsosurfaceAction = new QAction(NewVolumeMainWindowBase);
        ExportZoomedInIsosurfaceAction->setObjectName(QString::fromUtf8("ExportZoomedInIsosurfaceAction"));
        ExportZoomedInIsosurfaceAction->setName("ExportZoomedInIsosurfaceAction");
        ExportZoomedOutIsosurfaceAction = new QAction(NewVolumeMainWindowBase);
        ExportZoomedOutIsosurfaceAction->setObjectName(QString::fromUtf8("ExportZoomedOutIsosurfaceAction"));
        ExportZoomedOutIsosurfaceAction->setName("ExportZoomedOutIsosurfaceAction");
        WireCubeToggleAction = new QAction(NewVolumeMainWindowBase);
        WireCubeToggleAction->setObjectName(QString::fromUtf8("WireCubeToggleAction"));
        WireCubeToggleAction->setName("WireCubeToggleAction");
        WireCubeToggleAction->setCheckable(true);
        WireCubeToggleAction->setChecked(true);
        WireframeRenderToggleAction = new QAction(NewVolumeMainWindowBase);
        WireframeRenderToggleAction->setObjectName(QString::fromUtf8("WireframeRenderToggleAction"));
        WireframeRenderToggleAction->setName("WireframeRenderToggleAction");
        WireframeRenderToggleAction->setCheckable(true);
        WireframeRenderToggleAction->setChecked(false);
        viewUnnamedAction = new QAction(NewVolumeMainWindowBase);
        viewUnnamedAction->setObjectName(QString::fromUtf8("viewUnnamedAction"));
        viewUnnamedAction->setName("viewUnnamedAction");
        viewUnnamedAction_2 = new QAction(NewVolumeMainWindowBase);
        viewUnnamedAction_2->setObjectName(QString::fromUtf8("viewUnnamedAction_2"));
        viewUnnamedAction_2->setName("viewUnnamedAction_2");
        DepthCueToggleAction = new QAction(NewVolumeMainWindowBase);
        DepthCueToggleAction->setObjectName(QString::fromUtf8("DepthCueToggleAction"));
        DepthCueToggleAction->setName("DepthCueToggleAction");
        DepthCueToggleAction->setCheckable(true);
        SaveImageAction = new QAction(NewVolumeMainWindowBase);
        SaveImageAction->setObjectName(QString::fromUtf8("SaveImageAction"));
        SaveImageAction->setName("SaveImageAction");
        StartRecordingAnimationAction = new QAction(NewVolumeMainWindowBase);
        StartRecordingAnimationAction->setObjectName(QString::fromUtf8("StartRecordingAnimationAction"));
        StartRecordingAnimationAction->setName("StartRecordingAnimationAction");
        StopRecordingAnimationAction = new QAction(NewVolumeMainWindowBase);
        StopRecordingAnimationAction->setObjectName(QString::fromUtf8("StopRecordingAnimationAction"));
        StopRecordingAnimationAction->setName("StopRecordingAnimationAction");
        PlayAnimationAction = new QAction(NewVolumeMainWindowBase);
        PlayAnimationAction->setObjectName(QString::fromUtf8("PlayAnimationAction"));
        PlayAnimationAction->setName("PlayAnimationAction");
        SaveAnimationAction = new QAction(NewVolumeMainWindowBase);
        SaveAnimationAction->setObjectName(QString::fromUtf8("SaveAnimationAction"));
        SaveAnimationAction->setName("SaveAnimationAction");
        SaveAnimationAction->setEnabled(true);
        LoadAnimationAction = new QAction(NewVolumeMainWindowBase);
        LoadAnimationAction->setObjectName(QString::fromUtf8("LoadAnimationAction"));
        LoadAnimationAction->setName("LoadAnimationAction");
        StopAnimationAction = new QAction(NewVolumeMainWindowBase);
        StopAnimationAction->setObjectName(QString::fromUtf8("StopAnimationAction"));
        StopAnimationAction->setName("StopAnimationAction");
        SaveFrameSequenceAction = new QAction(NewVolumeMainWindowBase);
        SaveFrameSequenceAction->setObjectName(QString::fromUtf8("SaveFrameSequenceAction"));
        SaveFrameSequenceAction->setName("SaveFrameSequenceAction");
        SegmentVirusMapAction = new QAction(NewVolumeMainWindowBase);
        SegmentVirusMapAction->setObjectName(QString::fromUtf8("SegmentVirusMapAction"));
        SegmentVirusMapAction->setName("SegmentVirusMapAction");
        EditTransformationAction = new QAction(NewVolumeMainWindowBase);
        EditTransformationAction->setObjectName(QString::fromUtf8("EditTransformationAction"));
        EditTransformationAction->setName("EditTransformationAction");
        ClearTransformationAction = new QAction(NewVolumeMainWindowBase);
        ClearTransformationAction->setObjectName(QString::fromUtf8("ClearTransformationAction"));
        ClearTransformationAction->setName("ClearTransformationAction");
        ShowTerminalAction = new QAction(NewVolumeMainWindowBase);
        ShowTerminalAction->setObjectName(QString::fromUtf8("ShowTerminalAction"));
        ShowTerminalAction->setName("ShowTerminalAction");
        ShowTerminalAction->setCheckable(true);
        ShowCorrelationAction = new QAction(NewVolumeMainWindowBase);
        ShowCorrelationAction->setObjectName(QString::fromUtf8("ShowCorrelationAction"));
        ShowCorrelationAction->setName("ShowCorrelationAction");
        ContrastEnhancementAction = new QAction(NewVolumeMainWindowBase);
        ContrastEnhancementAction->setObjectName(QString::fromUtf8("ContrastEnhancementAction"));
        ContrastEnhancementAction->setName("ContrastEnhancementAction");
        PEDetectionAction = new QAction(NewVolumeMainWindowBase);
        PEDetectionAction->setObjectName(QString::fromUtf8("PEDetectionAction"));
        PEDetectionAction->setName("PEDetectionAction");
        PocketTunnelAction = new QAction(NewVolumeMainWindowBase);
        PocketTunnelAction->setObjectName(QString::fromUtf8("PocketTunnelAction"));
        PocketTunnelAction->setName("PocketTunnelAction");
        SaveGeometryAction = new QAction(NewVolumeMainWindowBase);
        SaveGeometryAction->setObjectName(QString::fromUtf8("SaveGeometryAction"));
        SaveGeometryAction->setName("SaveGeometryAction");
        geometrySmoothingAction = new QAction(NewVolumeMainWindowBase);
        geometrySmoothingAction->setObjectName(QString::fromUtf8("geometrySmoothingAction"));
        geometrySmoothingAction->setName("geometrySmoothingAction");
        geometrynew_itemAction = new QAction(NewVolumeMainWindowBase);
        geometrynew_itemAction->setObjectName(QString::fromUtf8("geometrynew_itemAction"));
        geometrynew_itemAction->setName("geometrynew_itemAction");
        SmoothGeometryAction = new QAction(NewVolumeMainWindowBase);
        SmoothGeometryAction->setObjectName(QString::fromUtf8("SmoothGeometryAction"));
        SmoothGeometryAction->setName("SmoothGeometryAction");
        AnisotropicDiffusionAction = new QAction(NewVolumeMainWindowBase);
        AnisotropicDiffusionAction->setObjectName(QString::fromUtf8("AnisotropicDiffusionAction"));
        AnisotropicDiffusionAction->setName("AnisotropicDiffusionAction");
        SliceRenderingAction = new QAction(NewVolumeMainWindowBase);
        SliceRenderingAction->setObjectName(QString::fromUtf8("SliceRenderingAction"));
        SliceRenderingAction->setName("SliceRenderingAction");
        TightCoconeAction = new QAction(NewVolumeMainWindowBase);
        TightCoconeAction->setObjectName(QString::fromUtf8("TightCoconeAction"));
        TightCoconeAction->setName("TightCoconeAction");
        CurationAction = new QAction(NewVolumeMainWindowBase);
        CurationAction->setObjectName(QString::fromUtf8("CurationAction"));
        CurationAction->setName("CurationAction");
        SkeletonizationAction = new QAction(NewVolumeMainWindowBase);
        SkeletonizationAction->setObjectName(QString::fromUtf8("SkeletonizationAction"));
        SkeletonizationAction->setName("SkeletonizationAction");
        ClipGeometryToVolumeBoxAction = new QAction(NewVolumeMainWindowBase);
        ClipGeometryToVolumeBoxAction->setObjectName(QString::fromUtf8("ClipGeometryToVolumeBoxAction"));
        ClipGeometryToVolumeBoxAction->setName("ClipGeometryToVolumeBoxAction");
        ClipGeometryToVolumeBoxAction->setCheckable(true);
        ClipGeometryToVolumeBoxAction->setChecked(true);
        BoundaryPointCloudAction = new QAction(NewVolumeMainWindowBase);
        BoundaryPointCloudAction->setObjectName(QString::fromUtf8("BoundaryPointCloudAction"));
        BoundaryPointCloudAction->setName("BoundaryPointCloudAction");
        SaveSkeletonAction = new QAction(NewVolumeMainWindowBase);
        SaveSkeletonAction->setObjectName(QString::fromUtf8("SaveSkeletonAction"));
        SaveSkeletonAction->setName("SaveSkeletonAction");
        ClearSkeletonAction = new QAction(NewVolumeMainWindowBase);
        ClearSkeletonAction->setObjectName(QString::fromUtf8("ClearSkeletonAction"));
        ClearSkeletonAction->setName("ClearSkeletonAction");
        SignedDistanceFunctionAction = new QAction(NewVolumeMainWindowBase);
        SignedDistanceFunctionAction->setObjectName(QString::fromUtf8("SignedDistanceFunctionAction"));
        SignedDistanceFunctionAction->setName("SignedDistanceFunctionAction");
        MergeGeometryObjectsAction = new QAction(NewVolumeMainWindowBase);
        MergeGeometryObjectsAction->setObjectName(QString::fromUtf8("MergeGeometryObjectsAction"));
        MergeGeometryObjectsAction->setName("MergeGeometryObjectsAction");
        ConvertCurrentIsosurfaceToGeometryAction = new QAction(NewVolumeMainWindowBase);
        ConvertCurrentIsosurfaceToGeometryAction->setObjectName(QString::fromUtf8("ConvertCurrentIsosurfaceToGeometryAction"));
        ConvertCurrentIsosurfaceToGeometryAction->setName("ConvertCurrentIsosurfaceToGeometryAction");
        HighLevelSetAction = new QAction(NewVolumeMainWindowBase);
        HighLevelSetAction->setObjectName(QString::fromUtf8("HighLevelSetAction"));
        HighLevelSetAction->setName("HighLevelSetAction");
        HighLevelSetReconAction = new QAction(NewVolumeMainWindowBase);
        HighLevelSetReconAction->setObjectName(QString::fromUtf8("HighLevelSetReconAction"));
        HighLevelSetReconAction->setName("HighLevelSetReconAction");
        LBIEAction = new QAction(NewVolumeMainWindowBase);
        LBIEAction->setObjectName(QString::fromUtf8("LBIEAction"));
        LBIEAction->setName("LBIEAction");
        LBIEMeshingAction = new QAction(NewVolumeMainWindowBase);
        LBIEMeshingAction->setObjectName(QString::fromUtf8("LBIEMeshingAction"));
        LBIEMeshingAction->setName("LBIEMeshingAction");
        LBIEQualityImprovementAction = new QAction(NewVolumeMainWindowBase);
        LBIEQualityImprovementAction->setObjectName(QString::fromUtf8("LBIEQualityImprovementAction"));
        LBIEQualityImprovementAction->setName("LBIEQualityImprovementAction");
        RenderSurfaceWithWireframeAction = new QAction(NewVolumeMainWindowBase);
        RenderSurfaceWithWireframeAction->setObjectName(QString::fromUtf8("RenderSurfaceWithWireframeAction"));
        RenderSurfaceWithWireframeAction->setName("RenderSurfaceWithWireframeAction");
        RenderSurfaceWithWireframeAction->setCheckable(true);
        OpenImageFileAction = new QAction(NewVolumeMainWindowBase);
        OpenImageFileAction->setObjectName(QString::fromUtf8("OpenImageFileAction"));
        OpenImageFileAction->setName("OpenImageFileAction");
        ProjectToSurfaceAction = new QAction(NewVolumeMainWindowBase);
        ProjectToSurfaceAction->setObjectName(QString::fromUtf8("ProjectToSurfaceAction"));
        ProjectToSurfaceAction->setName("ProjectToSurfaceAction");
        GDTVFilterAction = new QAction(NewVolumeMainWindowBase);
        GDTVFilterAction->setObjectName(QString::fromUtf8("GDTVFilterAction"));
        GDTVFilterAction->setName("GDTVFilterAction");
        MumfordShahLevelSetAction = new QAction(NewVolumeMainWindowBase);
        MumfordShahLevelSetAction->setObjectName(QString::fromUtf8("MumfordShahLevelSetAction"));
        MumfordShahLevelSetAction->setName("MumfordShahLevelSetAction");
        ColorGeometryByVolumeAction = new QAction(NewVolumeMainWindowBase);
        ColorGeometryByVolumeAction->setObjectName(QString::fromUtf8("ColorGeometryByVolumeAction"));
        ColorGeometryByVolumeAction->setName("ColorGeometryByVolumeAction");
        CullGeometryWithSubvolumeBoxAction = new QAction(NewVolumeMainWindowBase);
        CullGeometryWithSubvolumeBoxAction->setObjectName(QString::fromUtf8("CullGeometryWithSubvolumeBoxAction"));
        CullGeometryWithSubvolumeBoxAction->setName("CullGeometryWithSubvolumeBoxAction");
        toolsnew_itemAction = new QAction(NewVolumeMainWindowBase);
        toolsnew_itemAction->setObjectName(QString::fromUtf8("toolsnew_itemAction"));
        toolsnew_itemAction->setName("toolsnew_itemAction");
        SecondaryStructureAction = new QAction(NewVolumeMainWindowBase);
        SecondaryStructureAction->setObjectName(QString::fromUtf8("SecondaryStructureAction"));
        SecondaryStructureAction->setName("SecondaryStructureAction");
        RecentFilesPopup = new QActionGroup(NewVolumeMainWindowBase);
        RecentFilesPopup->setObjectName(QString::fromUtf8("RecentFilesPopup"));
        RecentFilesPopup->setName("RecentFilesPopup");
        RecentFile1 = new QAction(RecentFilesPopup);
        RecentFile1->setObjectName(QString::fromUtf8("RecentFile1"));
        RecentFile1->setName("RecentFile1");
        RecentFile1->setCheckable(false);
        widget = new QWidget(NewVolumeMainWindowBase);
        widget->setObjectName(QString::fromUtf8("widget"));
        gridLayout = new QGridLayout(widget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        m_ViewTabs = new QTabWidget(widget);
        m_ViewTabs->setObjectName(QString::fromUtf8("m_ViewTabs"));
        ThreeDRover = new QWidget();
        ThreeDRover->setObjectName(QString::fromUtf8("ThreeDRover"));
        gridLayout1 = new QGridLayout(ThreeDRover);
        gridLayout1->setSpacing(6);
        gridLayout1->setContentsMargins(11, 11, 11, 11);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        vboxLayout = new QVBoxLayout();
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(0, 0, 0, 0);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        m_ZoomedIn = new SimpleOpenGLWidget(ThreeDRover);
        m_ZoomedIn->setObjectName(QString::fromUtf8("m_ZoomedIn"));

        vboxLayout->addWidget(m_ZoomedIn);

        gridLayout2 = new QGridLayout();
        gridLayout2->setSpacing(6);
        gridLayout2->setContentsMargins(0, 0, 0, 0);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        TextLabel1 = new QLabel(ThreeDRover);
        TextLabel1->setObjectName(QString::fromUtf8("TextLabel1"));
        TextLabel1->setWordWrap(false);

        gridLayout2->addWidget(TextLabel1, 0, 0, 1, 1);

        TextLabel1_3 = new QLabel(ThreeDRover);
        TextLabel1_3->setObjectName(QString::fromUtf8("TextLabel1_3"));
        TextLabel1_3->setWordWrap(false);

        gridLayout2->addWidget(TextLabel1_3, 1, 0, 1, 1);

        Slider2 = new QSlider(ThreeDRover);
        Slider2->setObjectName(QString::fromUtf8("Slider2"));
        Slider2->setValue(99);
        Slider2->setOrientation(Qt::Horizontal);

        gridLayout2->addWidget(Slider2, 0, 1, 1, 1);

        m_MainNearPlane = new QSlider(ThreeDRover);
        m_MainNearPlane->setObjectName(QString::fromUtf8("m_MainNearPlane"));
        m_MainNearPlane->setOrientation(Qt::Horizontal);

        gridLayout2->addWidget(m_MainNearPlane, 1, 1, 1, 1);


        vboxLayout->addLayout(gridLayout2);


        gridLayout1->addLayout(vboxLayout, 1, 0, 1, 1);

        vboxLayout1 = new QVBoxLayout();
        vboxLayout1->setSpacing(6);
        vboxLayout1->setContentsMargins(0, 0, 0, 0);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        m_ZoomedOut = new SimpleOpenGLWidget(ThreeDRover);
        m_ZoomedOut->setObjectName(QString::fromUtf8("m_ZoomedOut"));

        vboxLayout1->addWidget(m_ZoomedOut);

        gridLayout3 = new QGridLayout();
        gridLayout3->setSpacing(6);
        gridLayout3->setContentsMargins(0, 0, 0, 0);
        gridLayout3->setObjectName(QString::fromUtf8("gridLayout3"));
        m_ExplorerNearPlane = new QSlider(ThreeDRover);
        m_ExplorerNearPlane->setObjectName(QString::fromUtf8("m_ExplorerNearPlane"));
        m_ExplorerNearPlane->setOrientation(Qt::Horizontal);

        gridLayout3->addWidget(m_ExplorerNearPlane, 1, 1, 1, 1);

        TextLabel2 = new QLabel(ThreeDRover);
        TextLabel2->setObjectName(QString::fromUtf8("TextLabel2"));
        TextLabel2->setWordWrap(false);

        gridLayout3->addWidget(TextLabel2, 1, 0, 1, 1);

        Slider1 = new QSlider(ThreeDRover);
        Slider1->setObjectName(QString::fromUtf8("Slider1"));
        Slider1->setValue(99);
        Slider1->setOrientation(Qt::Horizontal);

        gridLayout3->addWidget(Slider1, 0, 1, 1, 1);

        TextLabel1_2 = new QLabel(ThreeDRover);
        TextLabel1_2->setObjectName(QString::fromUtf8("TextLabel1_2"));
        TextLabel1_2->setWordWrap(false);

        gridLayout3->addWidget(TextLabel1_2, 0, 0, 1, 1);


        vboxLayout1->addLayout(gridLayout3);


        gridLayout1->addLayout(vboxLayout1, 1, 1, 1, 1);

        m_VariableSelectionStack = new Q3WidgetStack(ThreeDRover);
        m_VariableSelectionStack->setObjectName(QString::fromUtf8("m_VariableSelectionStack"));
        Single = new QWidget(m_VariableSelectionStack);
        Single->setObjectName(QString::fromUtf8("Single"));
        hboxLayout = new QHBoxLayout(Single);
        hboxLayout->setSpacing(6);
        hboxLayout->setContentsMargins(11, 11, 11, 11);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        hboxLayout->setContentsMargins(0, 0, 0, 0);
        m_VariableLabel = new QLabel(Single);
        m_VariableLabel->setObjectName(QString::fromUtf8("m_VariableLabel"));
        m_VariableLabel->setEnabled(true);
        m_VariableLabel->setWordWrap(false);

        hboxLayout->addWidget(m_VariableLabel);

        m_VariableBox = new QComboBox(Single);
        m_VariableBox->setObjectName(QString::fromUtf8("m_VariableBox"));
        m_VariableBox->setEnabled(true);
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(0));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_VariableBox->sizePolicy().hasHeightForWidth());
        m_VariableBox->setSizePolicy(sizePolicy);
        m_VariableBox->setMinimumSize(QSize(0, 0));

        hboxLayout->addWidget(m_VariableBox);

        m_TimeStepLevel = new QLabel(Single);
        m_TimeStepLevel->setObjectName(QString::fromUtf8("m_TimeStepLevel"));
        m_TimeStepLevel->setEnabled(true);
        m_TimeStepLevel->setWordWrap(false);

        hboxLayout->addWidget(m_TimeStepLevel);

        m_TimeStep = new QSpinBox(Single);
        m_TimeStep->setObjectName(QString::fromUtf8("m_TimeStep"));
        m_TimeStep->setEnabled(true);

        hboxLayout->addWidget(m_TimeStep);

        spacer1 = new QSpacerItem(321, 21, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout->addItem(spacer1);

        m_VariableSelectionStack->addWidget(Single, 0);
        RGBA = new QWidget(m_VariableSelectionStack);
        RGBA->setObjectName(QString::fromUtf8("RGBA"));
        hboxLayout1 = new QHBoxLayout(RGBA);
        hboxLayout1->setSpacing(6);
        hboxLayout1->setContentsMargins(11, 11, 11, 11);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        hboxLayout1->setContentsMargins(0, 0, 0, 0);
        m_RedLabel = new QLabel(RGBA);
        m_RedLabel->setObjectName(QString::fromUtf8("m_RedLabel"));
        m_RedLabel->setWordWrap(false);

        hboxLayout1->addWidget(m_RedLabel);

        m_RedBox = new QComboBox(RGBA);
        m_RedBox->setObjectName(QString::fromUtf8("m_RedBox"));
        QSizePolicy sizePolicy1(static_cast<QSizePolicy::Policy>(5), static_cast<QSizePolicy::Policy>(0));
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(m_RedBox->sizePolicy().hasHeightForWidth());
        m_RedBox->setSizePolicy(sizePolicy1);

        hboxLayout1->addWidget(m_RedBox);

        m_GreenLabel = new QLabel(RGBA);
        m_GreenLabel->setObjectName(QString::fromUtf8("m_GreenLabel"));
        m_GreenLabel->setWordWrap(false);

        hboxLayout1->addWidget(m_GreenLabel);

        m_GreenBox = new QComboBox(RGBA);
        m_GreenBox->setObjectName(QString::fromUtf8("m_GreenBox"));
        sizePolicy1.setHeightForWidth(m_GreenBox->sizePolicy().hasHeightForWidth());
        m_GreenBox->setSizePolicy(sizePolicy1);

        hboxLayout1->addWidget(m_GreenBox);

        m_BlueLabel = new QLabel(RGBA);
        m_BlueLabel->setObjectName(QString::fromUtf8("m_BlueLabel"));
        m_BlueLabel->setWordWrap(false);

        hboxLayout1->addWidget(m_BlueLabel);

        m_BlueBox = new QComboBox(RGBA);
        m_BlueBox->setObjectName(QString::fromUtf8("m_BlueBox"));
        sizePolicy1.setHeightForWidth(m_BlueBox->sizePolicy().hasHeightForWidth());
        m_BlueBox->setSizePolicy(sizePolicy1);

        hboxLayout1->addWidget(m_BlueBox);

        m_AlphaLabel = new QLabel(RGBA);
        m_AlphaLabel->setObjectName(QString::fromUtf8("m_AlphaLabel"));
        m_AlphaLabel->setWordWrap(false);

        hboxLayout1->addWidget(m_AlphaLabel);

        m_AlphaBox = new QComboBox(RGBA);
        m_AlphaBox->setObjectName(QString::fromUtf8("m_AlphaBox"));
        sizePolicy1.setHeightForWidth(m_AlphaBox->sizePolicy().hasHeightForWidth());
        m_AlphaBox->setSizePolicy(sizePolicy1);

        hboxLayout1->addWidget(m_AlphaBox);

        m_RGBATimeLabel = new QLabel(RGBA);
        m_RGBATimeLabel->setObjectName(QString::fromUtf8("m_RGBATimeLabel"));
        m_RGBATimeLabel->setWordWrap(false);

        hboxLayout1->addWidget(m_RGBATimeLabel);

        m_RGBATimeStep = new QSpinBox(RGBA);
        m_RGBATimeStep->setObjectName(QString::fromUtf8("m_RGBATimeStep"));

        hboxLayout1->addWidget(m_RGBATimeStep);

        m_VariableSelectionStack->addWidget(RGBA, 1);

        gridLayout1->addWidget(m_VariableSelectionStack, 0, 0, 1, 2);

        m_ViewTabs->addTab(ThreeDRover, QString());

        gridLayout->addWidget(m_ViewTabs, 0, 0, 1, 1);

        NewVolumeMainWindowBase->setCentralWidget(widget);
        menubar = new QMenuBar(NewVolumeMainWindowBase);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        m_FileMenu = new QMenu(menubar);
        m_FileMenu->setObjectName(QString::fromUtf8("m_FileMenu"));
        m_ViewMenu = new QMenu(menubar);
        m_ViewMenu->setObjectName(QString::fromUtf8("m_ViewMenu"));
        m_GeometryMenu = new QMenu(menubar);
        m_GeometryMenu->setObjectName(QString::fromUtf8("m_GeometryMenu"));
        popupMenu_39 = new QMenu(m_GeometryMenu);
        popupMenu_39->setObjectName(QString::fromUtf8("popupMenu_39"));
        m_ServersMenu = new QMenu(menubar);
        m_ServersMenu->setObjectName(QString::fromUtf8("m_ServersMenu"));
        m_AnimationMenu = new QMenu(menubar);
        m_AnimationMenu->setObjectName(QString::fromUtf8("m_AnimationMenu"));
        Tools = new QMenu(menubar);
        Tools->setObjectName(QString::fromUtf8("Tools"));
        m_HelpMenu = new QMenu(menubar);
        m_HelpMenu->setObjectName(QString::fromUtf8("m_HelpMenu"));
        QWidget::setTabOrder(Slider2, Slider1);

        menubar->addAction(m_FileMenu->menuAction());
        menubar->addAction(m_ViewMenu->menuAction());
        menubar->addAction(m_GeometryMenu->menuAction());
        menubar->addAction(m_ServersMenu->menuAction());
        menubar->addAction(m_AnimationMenu->menuAction());
        menubar->addAction(Tools->menuAction());
        menubar->addAction(m_HelpMenu->menuAction());
        m_FileMenu->addAction(ConnectToFile);
        m_FileMenu->addAction(OpenImageFileAction);
        m_FileMenu->addAction(ConnectToDC);
        m_FileMenu->addAction(SaveSubVolumeAction);
        m_FileMenu->addAction(SaveImageAction);
        m_FileMenu->addSeparator();
        m_FileMenu->addAction(OptionsAction);
        m_FileMenu->addSeparator();
        m_FileMenu->addAction(ExitAction);
        m_ViewMenu->addAction(CenterAction);
        m_ViewMenu->addAction(explorerChangedAction);
        m_ViewMenu->addSeparator();
        m_ViewMenu->addAction(WireCubeToggleAction);
        m_ViewMenu->addAction(DepthCueToggleAction);
        m_ViewMenu->addSeparator();
        m_ViewMenu->addAction(ShowTerminalAction);
        m_ViewMenu->addSeparator();
        m_ViewMenu->addAction(SliceRenderingAction);
        m_GeometryMenu->addAction(LoadGeometryAction);
        m_GeometryMenu->addAction(ClearGeometryAction);
        m_GeometryMenu->addAction(SaveGeometryAction);
        m_GeometryMenu->addSeparator();
        m_GeometryMenu->addAction(ExportZoomedOutIsosurfaceAction);
        m_GeometryMenu->addAction(ExportZoomedInIsosurfaceAction);
        m_GeometryMenu->addSeparator();
        m_GeometryMenu->addAction(SaveSkeletonAction);
        m_GeometryMenu->addAction(ClearSkeletonAction);
        m_GeometryMenu->addSeparator();
        m_GeometryMenu->addAction(WireframeRenderToggleAction);
        m_GeometryMenu->addAction(RenderSurfaceWithWireframeAction);
        m_GeometryMenu->addAction(ClipGeometryToVolumeBoxAction);
        m_GeometryMenu->addAction(EditTransformationAction);
        m_GeometryMenu->addAction(ClearTransformationAction);
        m_GeometryMenu->addSeparator();
        m_GeometryMenu->addAction(SmoothGeometryAction);
        m_GeometryMenu->addAction(SignedDistanceFunctionAction);
        m_GeometryMenu->addAction(MergeGeometryObjectsAction);
        m_GeometryMenu->addAction(ConvertCurrentIsosurfaceToGeometryAction);
        m_GeometryMenu->addAction(LBIEAction);
        m_GeometryMenu->addAction(popupMenu_39->menuAction());
        m_GeometryMenu->addAction(ProjectToSurfaceAction);
        m_GeometryMenu->addAction(ColorGeometryByVolumeAction);
        m_GeometryMenu->addAction(CullGeometryWithSubvolumeBoxAction);
        popupMenu_39->addAction(LBIEMeshingAction);
        popupMenu_39->addAction(LBIEQualityImprovementAction);
        m_ServersMenu->addAction(ConnectServerAction);
        m_ServersMenu->addAction(DisconnectServerAction);
        m_ServersMenu->addAction(ServerSettingAction);
        m_ServersMenu->addSeparator();
        m_ServersMenu->addAction(RenderFrameAction);
        m_AnimationMenu->addAction(StartRecordingAnimationAction);
        m_AnimationMenu->addAction(StopRecordingAnimationAction);
        m_AnimationMenu->addSeparator();
        m_AnimationMenu->addAction(PlayAnimationAction);
        m_AnimationMenu->addAction(StopAnimationAction);
        m_AnimationMenu->addSeparator();
        m_AnimationMenu->addAction(SaveAnimationAction);
        m_AnimationMenu->addAction(LoadAnimationAction);
        m_AnimationMenu->addSeparator();
        m_AnimationMenu->addAction(SaveFrameSequenceAction);
        Tools->addAction(BilateralFilterAction);
        Tools->addAction(SegmentVirusMapAction);
        Tools->addAction(ContrastEnhancementAction);
        Tools->addAction(PEDetectionAction);
        Tools->addAction(PocketTunnelAction);
        Tools->addAction(AnisotropicDiffusionAction);
        Tools->addAction(BoundaryPointCloudAction);
        Tools->addAction(TightCoconeAction);
        Tools->addAction(CurationAction);
        Tools->addAction(SkeletonizationAction);
        Tools->addAction(SecondaryStructureAction);
        Tools->addAction(HighLevelSetAction);
        Tools->addAction(HighLevelSetReconAction);
        Tools->addAction(GDTVFilterAction);
        Tools->addAction(MumfordShahLevelSetAction);
        m_HelpMenu->addAction(WhatsThisAction);

        retranslateUi(NewVolumeMainWindowBase);
        QObject::connect(ConnectToFile, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(actionSlot()));
        QObject::connect(BilateralFilterAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(bilateralFilterSlot()));
        QObject::connect(CenterAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(centerSlot()));
        QObject::connect(ClearGeometryAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(clearGeometrySlot()));
        QObject::connect(ExitAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(close()));
        QObject::connect(ConnectServerAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(connectServerSlot()));
        QObject::connect(ConnectToDC, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(connectToDCSlot()));
        QObject::connect(DisconnectServerAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(disconnectServerSlot()));
        QObject::connect(explorerChangedAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(explorerChangedSlot()));
        QObject::connect(ExportZoomedInIsosurfaceAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(exportZoomedInIsosurfaceSlot()));
        QObject::connect(ExportZoomedOutIsosurfaceAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(exportZoomedOutIsosurfaceSlot()));
        QObject::connect(LoadAnimationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(loadAnimationSlot()));
        QObject::connect(LoadGeometryAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(loadGeometrySlot()));
        QObject::connect(OptionsAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(optionsSlot()));
        QObject::connect(PlayAnimationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(playAnimationSlot()));
        QObject::connect(RenderAnimationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(renderAnimationSlot()));
        QObject::connect(RenderFrameAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(renderFrameSlot()));
        QObject::connect(SaveFrameSequenceAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(renderSequenceSlot()));
        QObject::connect(ClearTransformationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(resetGeometryTransformationSlot()));
        QObject::connect(SaveAnimationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(saveAnimationSlot()));
        QObject::connect(SaveImageAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(saveImageSlot()));
        QObject::connect(SaveSubVolumeAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(saveSubvolumeSlot()));
        QObject::connect(ServerSettingAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(serverSettingsSlot()));
        QObject::connect(Slider1, SIGNAL(valueChanged(int)), NewVolumeMainWindowBase, SLOT(setExplorerQualitySlot(int)));
        QObject::connect(Slider2, SIGNAL(valueChanged(int)), NewVolumeMainWindowBase, SLOT(setMainQualitySlot(int)));
        QObject::connect(StartRecordingAnimationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(startRecordingAnimationSlot()));
        QObject::connect(StopAnimationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(stopAnimationSlot()));
        QObject::connect(StopRecordingAnimationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(stopRecordingAnimationSlot()));
        QObject::connect(DepthCueToggleAction, SIGNAL(toggled(bool)), NewVolumeMainWindowBase, SLOT(toggleDepthCueSlot(bool)));
        QObject::connect(EditTransformationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(toggleGeometryTransformationSlot()));
        QObject::connect(ShowTerminalAction, SIGNAL(toggled(bool)), NewVolumeMainWindowBase, SLOT(toggleTerminalSlot(bool)));
        QObject::connect(WireCubeToggleAction, SIGNAL(toggled(bool)), NewVolumeMainWindowBase, SLOT(toggleWireCubeSlot(bool)));
        QObject::connect(WireframeRenderToggleAction, SIGNAL(toggled(bool)), NewVolumeMainWindowBase, SLOT(toggleWireframeRenderingSlot(bool)));
        QObject::connect(RenderSurfaceWithWireframeAction, SIGNAL(toggled(bool)), NewVolumeMainWindowBase, SLOT(toggleRenderSurfaceWithWireframeSlot(bool)));
        QObject::connect(m_TimeStep, SIGNAL(valueChanged(int)), NewVolumeMainWindowBase, SLOT(variableOrTimeChangeSlot()));
        QObject::connect(m_VariableBox, SIGNAL(activated(int)), NewVolumeMainWindowBase, SLOT(variableOrTimeChangeSlot()));
        QObject::connect(m_RedBox, SIGNAL(activated(int)), NewVolumeMainWindowBase, SLOT(variableOrTimeChangeSlot()));
        QObject::connect(m_GreenBox, SIGNAL(activated(int)), NewVolumeMainWindowBase, SLOT(variableOrTimeChangeSlot()));
        QObject::connect(m_BlueBox, SIGNAL(activated(int)), NewVolumeMainWindowBase, SLOT(variableOrTimeChangeSlot()));
        QObject::connect(m_AlphaBox, SIGNAL(activated(int)), NewVolumeMainWindowBase, SLOT(variableOrTimeChangeSlot()));
        QObject::connect(m_RGBATimeStep, SIGNAL(valueChanged(int)), NewVolumeMainWindowBase, SLOT(variableOrTimeChangeSlot()));
        QObject::connect(SegmentVirusMapAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(virusSegmentationSlot()));
        QObject::connect(WhatsThisAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(whatsThis()));
        QObject::connect(m_MainNearPlane, SIGNAL(valueChanged(int)), NewVolumeMainWindowBase, SLOT(zoomedInClipSlot(int)));
        QObject::connect(m_ExplorerNearPlane, SIGNAL(valueChanged(int)), NewVolumeMainWindowBase, SLOT(zoomedOutClipSlot(int)));
        QObject::connect(ContrastEnhancementAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(contrastEnhancementSlot()));
        QObject::connect(PEDetectionAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(PEDetectionSlot()));
        QObject::connect(PocketTunnelAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(pocketTunnelSlot()));
        QObject::connect(SaveGeometryAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(saveGeometrySlot()));
        QObject::connect(SmoothGeometryAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(smoothGeometrySlot()));
        QObject::connect(AnisotropicDiffusionAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(anisotropicDiffusionSlot()));
        QObject::connect(SliceRenderingAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(sliceRenderingSlot()));
        QObject::connect(TightCoconeAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(tightCoconeSlot()));
        QObject::connect(CurationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(curationSlot()));
        QObject::connect(SkeletonizationAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(skeletonizationSlot()));
        QObject::connect(ClipGeometryToVolumeBoxAction, SIGNAL(toggled(bool)), NewVolumeMainWindowBase, SLOT(clipGeometryToVolumeBoxSlot(bool)));
        QObject::connect(BoundaryPointCloudAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(boundaryPointCloudSlot()));
        QObject::connect(SaveSkeletonAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(saveSkeletonSlot()));
        QObject::connect(ClearSkeletonAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(clearSkeletonSlot()));
        QObject::connect(SignedDistanceFunctionAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(signedDistanceFunctionSlot()));
        QObject::connect(MergeGeometryObjectsAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(mergeGeometrySlot()));
        QObject::connect(ConvertCurrentIsosurfaceToGeometryAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(convertIsosurfaceToGeometrySlot()));
        QObject::connect(HighLevelSetAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(highLevelSetSlot()));
        QObject::connect(HighLevelSetReconAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(highLevelSetReconSlot()));
        QObject::connect(LBIEMeshingAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(LBIEMeshingSlot()));
        QObject::connect(LBIEQualityImprovementAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(LBIEQualityImprovementSlot()));
        QObject::connect(OpenImageFileAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(openImageFileSlot()));
        QObject::connect(ProjectToSurfaceAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(projectGeometrySlot()));
        QObject::connect(GDTVFilterAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(gdtvFilterSlot()));
        QObject::connect(MumfordShahLevelSetAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(MSLevelSetSlot()));
        QObject::connect(ColorGeometryByVolumeAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(colorGeometryByVolumeSlot()));
        QObject::connect(CullGeometryWithSubvolumeBoxAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(cullGeometryWithSubvolumeBoxSlot()));
        QObject::connect(SecondaryStructureAction, SIGNAL(activated()), NewVolumeMainWindowBase, SLOT(secondaryStructureElucidationSlot()));

        QMetaObject::connectSlotsByName(NewVolumeMainWindowBase);
    } // setupUi

    void retranslateUi(Q3MainWindow *NewVolumeMainWindowBase)
    {
        NewVolumeMainWindowBase->setWindowTitle(QApplication::translate("NewVolumeMainWindowBase", "NewVolume", 0, QApplication::UnicodeUTF8));
        ConnectToFile->setIconText(QApplication::translate("NewVolumeMainWindowBase", "&Open Volume File...", 0, QApplication::UnicodeUTF8));
        ConnectToFile->setText(QApplication::translate("NewVolumeMainWindowBase", "&Open Volume File...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ConnectToFile->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Opens a local file", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ConnectToFile->setShortcut(QApplication::translate("NewVolumeMainWindowBase", "Ctrl+O", 0, QApplication::UnicodeUTF8));
        explorerChangedAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Update", 0, QApplication::UnicodeUTF8));
        explorerChangedAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Update", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        explorerChangedAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Update the subvolume", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ConnectToDC->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Data Server...", 0, QApplication::UnicodeUTF8));
        ConnectToDC->setText(QApplication::translate("NewVolumeMainWindowBase", "&Data Server...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ConnectToDC->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Connects to a datacutter server", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ExitAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Exit", 0, QApplication::UnicodeUTF8));
        ExitAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Exit", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ExitAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Quits the program", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ExitAction->setShortcut(QApplication::translate("NewVolumeMainWindowBase", "Ctrl+Q", 0, QApplication::UnicodeUTF8));
        WhatsThisAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "WhatsThis", 0, QApplication::UnicodeUTF8));
        WhatsThisAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&WhatsThis", 0, QApplication::UnicodeUTF8));
        CenterAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Center On Preview", 0, QApplication::UnicodeUTF8));
        CenterAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Center On Preview", 0, QApplication::UnicodeUTF8));
        OptionsAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Options...", 0, QApplication::UnicodeUTF8));
        OptionsAction->setText(QApplication::translate("NewVolumeMainWindowBase", "O&ptions...", 0, QApplication::UnicodeUTF8));
        ConnectServerAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Connect...", 0, QApplication::UnicodeUTF8));
        ConnectServerAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Connect...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ConnectServerAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Connects to a remote render server.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        DisconnectServerAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Disconnect", 0, QApplication::UnicodeUTF8));
        DisconnectServerAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Disconnect", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        DisconnectServerAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Disconnects from the render server.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ServerSettingAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Server Settings...", 0, QApplication::UnicodeUTF8));
        ServerSettingAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Server Settings...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ServerSettingAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Settings for the remote server.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        RenderFrameAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Render Frame", 0, QApplication::UnicodeUTF8));
        RenderFrameAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Render &Frame", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        RenderFrameAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Renders a frame on the server.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        RenderFrameAction->setShortcut(QString());
        RenderAnimationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Render Animation", 0, QApplication::UnicodeUTF8));
        RenderAnimationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Render &Animation", 0, QApplication::UnicodeUTF8));
        BilateralFilterAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Bilateral Filter", 0, QApplication::UnicodeUTF8));
        BilateralFilterAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Bilateral Filter", 0, QApplication::UnicodeUTF8));
        SaveSubVolumeAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Save Subvolume", 0, QApplication::UnicodeUTF8));
        SaveSubVolumeAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Save Subvolume", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        SaveSubVolumeAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Creates a new volume file from the current subvolume.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        LoadGeometryAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Load Geometry", 0, QApplication::UnicodeUTF8));
        LoadGeometryAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Load Geometry", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        LoadGeometryAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Loads geometry from a file on disk.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ClearGeometryAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Clear Geometry", 0, QApplication::UnicodeUTF8));
        ClearGeometryAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Clear Geometry", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ClearGeometryAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Clears any geometry loaded from a file.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ExportZoomedInIsosurfaceAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Export Subvolume Isosurface", 0, QApplication::UnicodeUTF8));
        ExportZoomedInIsosurfaceAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Export Subvolume Isosurface", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ExportZoomedInIsosurfaceAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Saves all the isosurfaces from the left window to a file.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        ExportZoomedOutIsosurfaceAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Export Thumbnail Isosurface", 0, QApplication::UnicodeUTF8));
        ExportZoomedOutIsosurfaceAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Export &Thumbnail Isosurface", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ExportZoomedOutIsosurfaceAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Saves all isosurfaces from the right window to a file.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        WireCubeToggleAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Show Wire Cube", 0, QApplication::UnicodeUTF8));
        WireCubeToggleAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Show Wire Cube", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        WireCubeToggleAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Enables/Disables drawing of the wire cubes surrounding the volumes.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        WireframeRenderToggleAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Wireframe Rendering", 0, QApplication::UnicodeUTF8));
        WireframeRenderToggleAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Wireframe Rendering", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        WireframeRenderToggleAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Enables/Disables wireframe rendering.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        viewUnnamedAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Unnamed", 0, QApplication::UnicodeUTF8));
        viewUnnamedAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Unnamed", 0, QApplication::UnicodeUTF8));
        viewUnnamedAction_2->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Unnamed", 0, QApplication::UnicodeUTF8));
        viewUnnamedAction_2->setText(QApplication::translate("NewVolumeMainWindowBase", "Unnamed", 0, QApplication::UnicodeUTF8));
        DepthCueToggleAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Depth Cueing", 0, QApplication::UnicodeUTF8));
        DepthCueToggleAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Depth Cueing", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        DepthCueToggleAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Enables/Disables depth cueing.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        SaveImageAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Save Image...", 0, QApplication::UnicodeUTF8));
        SaveImageAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Save Image...", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        SaveImageAction->setWhatsThis(QApplication::translate("NewVolumeMainWindowBase", "Saves the contents of one or both render buffers to a file.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_WHATSTHIS
        StartRecordingAnimationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Start Recording", 0, QApplication::UnicodeUTF8));
        StartRecordingAnimationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Start Recording", 0, QApplication::UnicodeUTF8));
        StopRecordingAnimationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Stop Recording", 0, QApplication::UnicodeUTF8));
        StopRecordingAnimationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Stop Recording", 0, QApplication::UnicodeUTF8));
        PlayAnimationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Play Animation", 0, QApplication::UnicodeUTF8));
        PlayAnimationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Play Animation", 0, QApplication::UnicodeUTF8));
        SaveAnimationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Save Animation...", 0, QApplication::UnicodeUTF8));
        SaveAnimationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Save Animation...", 0, QApplication::UnicodeUTF8));
        LoadAnimationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Load Animation...", 0, QApplication::UnicodeUTF8));
        LoadAnimationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Load Animation...", 0, QApplication::UnicodeUTF8));
        StopAnimationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Stop Animation", 0, QApplication::UnicodeUTF8));
        StopAnimationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Stop Animation", 0, QApplication::UnicodeUTF8));
        SaveFrameSequenceAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Save Frame Sequence...", 0, QApplication::UnicodeUTF8));
        SaveFrameSequenceAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Save Frame Sequence...", 0, QApplication::UnicodeUTF8));
        SegmentVirusMapAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Segment Virus Map", 0, QApplication::UnicodeUTF8));
        SegmentVirusMapAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Segment Virus Map", 0, QApplication::UnicodeUTF8));
        EditTransformationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Edit Transformation", 0, QApplication::UnicodeUTF8));
        EditTransformationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Edit Transformation", 0, QApplication::UnicodeUTF8));
        EditTransformationAction->setShortcut(QApplication::translate("NewVolumeMainWindowBase", "Ctrl+T", 0, QApplication::UnicodeUTF8));
        ClearTransformationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Clear Transformation", 0, QApplication::UnicodeUTF8));
        ClearTransformationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Clear Transformation", 0, QApplication::UnicodeUTF8));
        ClearTransformationAction->setShortcut(QApplication::translate("NewVolumeMainWindowBase", "Ctrl+R", 0, QApplication::UnicodeUTF8));
        ShowTerminalAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Show Terminal", 0, QApplication::UnicodeUTF8));
        ShowTerminalAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Show Terminal", 0, QApplication::UnicodeUTF8));
        ShowCorrelationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Show Correlation", 0, QApplication::UnicodeUTF8));
        ShowCorrelationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Show Correlation", 0, QApplication::UnicodeUTF8));
        ContrastEnhancementAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Contrast Enhancement", 0, QApplication::UnicodeUTF8));
        ContrastEnhancementAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Contrast Enhancement", 0, QApplication::UnicodeUTF8));
        PEDetectionAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "&PEDetection", 0, QApplication::UnicodeUTF8));
        PEDetectionAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&PEDetection", 0, QApplication::UnicodeUTF8));
        PocketTunnelAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Pocket &Tunnel", 0, QApplication::UnicodeUTF8));
        PocketTunnelAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Pocket &Tunnel", 0, QApplication::UnicodeUTF8));
        SaveGeometryAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "&Save Geometry", 0, QApplication::UnicodeUTF8));
        SaveGeometryAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Save Geometry", 0, QApplication::UnicodeUTF8));
        geometrySmoothingAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Smoothing", 0, QApplication::UnicodeUTF8));
        geometrySmoothingAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Smoothing", 0, QApplication::UnicodeUTF8));
        geometrynew_itemAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "new item", 0, QApplication::UnicodeUTF8));
        geometrynew_itemAction->setText(QApplication::translate("NewVolumeMainWindowBase", "new item", 0, QApplication::UnicodeUTF8));
        SmoothGeometryAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Smoothing", 0, QApplication::UnicodeUTF8));
        SmoothGeometryAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Smoothing", 0, QApplication::UnicodeUTF8));
        AnisotropicDiffusionAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "&Anisotropic Diffusion", 0, QApplication::UnicodeUTF8));
        AnisotropicDiffusionAction->setText(QApplication::translate("NewVolumeMainWindowBase", "&Anisotropic Diffusion", 0, QApplication::UnicodeUTF8));
        SliceRenderingAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Slice Rendering", 0, QApplication::UnicodeUTF8));
        SliceRenderingAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Slice Rendering", 0, QApplication::UnicodeUTF8));
        TightCoconeAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "T&ight Cocone", 0, QApplication::UnicodeUTF8));
        TightCoconeAction->setText(QApplication::translate("NewVolumeMainWindowBase", "T&ight Cocone", 0, QApplication::UnicodeUTF8));
        CurationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Curation", 0, QApplication::UnicodeUTF8));
        CurationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Curation", 0, QApplication::UnicodeUTF8));
        SkeletonizationAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Skeletonization", 0, QApplication::UnicodeUTF8));
        SkeletonizationAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Skeletonization", 0, QApplication::UnicodeUTF8));
        ClipGeometryToVolumeBoxAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Clip Geometry to Volume Box", 0, QApplication::UnicodeUTF8));
        ClipGeometryToVolumeBoxAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Clip Geometry to Volume Box", 0, QApplication::UnicodeUTF8));
        BoundaryPointCloudAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Boundary Point Cloud", 0, QApplication::UnicodeUTF8));
        BoundaryPointCloudAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Boundary Point Cloud", 0, QApplication::UnicodeUTF8));
        SaveSkeletonAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Save Skeleton", 0, QApplication::UnicodeUTF8));
        ClearSkeletonAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Clear Skeleton", 0, QApplication::UnicodeUTF8));
        ClearSkeletonAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Clear Skeleton", 0, QApplication::UnicodeUTF8));
        SignedDistanceFunctionAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Signed Distance Function", 0, QApplication::UnicodeUTF8));
        SignedDistanceFunctionAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Signed Distance Function", 0, QApplication::UnicodeUTF8));
        MergeGeometryObjectsAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Merge Geometry Objects", 0, QApplication::UnicodeUTF8));
        MergeGeometryObjectsAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Merge Geometry Objects", 0, QApplication::UnicodeUTF8));
        ConvertCurrentIsosurfaceToGeometryAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Convert Current Isosurface to Geometry", 0, QApplication::UnicodeUTF8));
        ConvertCurrentIsosurfaceToGeometryAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Convert Current Isosurface to Geometry", 0, QApplication::UnicodeUTF8));
        HighLevelSetAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "High Level Set", 0, QApplication::UnicodeUTF8));
        HighLevelSetAction->setText(QApplication::translate("NewVolumeMainWindowBase", "High Level Set", 0, QApplication::UnicodeUTF8));
        HighLevelSetReconAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "High Level Set Recon", 0, QApplication::UnicodeUTF8));
        HighLevelSetReconAction->setText(QApplication::translate("NewVolumeMainWindowBase", "High Level Set Recon", 0, QApplication::UnicodeUTF8));
        LBIEAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "LBIE", 0, QApplication::UnicodeUTF8));
        LBIEAction->setText(QApplication::translate("NewVolumeMainWindowBase", "LBIE", 0, QApplication::UnicodeUTF8));
        LBIEMeshingAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Meshing", 0, QApplication::UnicodeUTF8));
        LBIEMeshingAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Meshing", 0, QApplication::UnicodeUTF8));
        LBIEQualityImprovementAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Quality Improvement", 0, QApplication::UnicodeUTF8));
        LBIEQualityImprovementAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Quality Improvement", 0, QApplication::UnicodeUTF8));
        RenderSurfaceWithWireframeAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Render Surface With Wireframe", 0, QApplication::UnicodeUTF8));
        RenderSurfaceWithWireframeAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Render Surface With Wireframe", 0, QApplication::UnicodeUTF8));
        OpenImageFileAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Open Image File...", 0, QApplication::UnicodeUTF8));
        OpenImageFileAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Open Image File...", 0, QApplication::UnicodeUTF8));
        ProjectToSurfaceAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Project to Surface", 0, QApplication::UnicodeUTF8));
        ProjectToSurfaceAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Project to Surface", 0, QApplication::UnicodeUTF8));
        GDTVFilterAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "GDTVFilter", 0, QApplication::UnicodeUTF8));
        GDTVFilterAction->setText(QApplication::translate("NewVolumeMainWindowBase", "GDTVFilter", 0, QApplication::UnicodeUTF8));
        MumfordShahLevelSetAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Mumford-Shah Level Set", 0, QApplication::UnicodeUTF8));
        MumfordShahLevelSetAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Mumford-Shah Level Set", 0, QApplication::UnicodeUTF8));
        ColorGeometryByVolumeAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Color Geometry By Volume", 0, QApplication::UnicodeUTF8));
        ColorGeometryByVolumeAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Color Geometry By Volume", 0, QApplication::UnicodeUTF8));
        CullGeometryWithSubvolumeBoxAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Cull Geometry with Subvolume Box", 0, QApplication::UnicodeUTF8));
        CullGeometryWithSubvolumeBoxAction->setText(QApplication::translate("NewVolumeMainWindowBase", "Cull Geometry with Subvolume Box", 0, QApplication::UnicodeUTF8));
        toolsnew_itemAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "new item", 0, QApplication::UnicodeUTF8));
        toolsnew_itemAction->setText(QApplication::translate("NewVolumeMainWindowBase", "new item", 0, QApplication::UnicodeUTF8));
        SecondaryStructureAction->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Secondary Structure ", 0, QApplication::UnicodeUTF8));
        RecentFile1->setIconText(QApplication::translate("NewVolumeMainWindowBase", "Action", 0, QApplication::UnicodeUTF8));
        RecentFile1->setText(QApplication::translate("NewVolumeMainWindowBase", "Action", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        m_ZoomedIn->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "<p><b>Subvolume Viewer Window</b></p>\n"
"\n"
"<p>Use this window to view the subvolume cube.  The right mouse button rotates the view.  The left mouse button translates.  The middle button, or scroll wheel zooms.\n"
"</p>", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        TextLabel1->setText(QApplication::translate("NewVolumeMainWindowBase", "Main Render Quality", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        TextLabel1->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Determines the quality of the main render.  Dragging the slider to the left has the following 3 effects:\n"
"\n"
"1. Render speed increases.\n"
"2. Render quality decreases.\n"
"3. Render becomes more transparent.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        TextLabel1_3->setText(QApplication::translate("NewVolumeMainWindowBase", "Main Near Clip Plane", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        TextLabel1_3->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Allows you to clip away the near part of the volume.  Moving the slider all the way to the left will display the whole volume.  Moving the slider to the right will clip away the near portions of the volume.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_WHATSTHIS
        Slider2->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Determines the quality of the main render.  Dragging the slider to the left has the following 3 effects:\n"
"\n"
"1. Render speed increases.\n"
"2. Render quality decreases.\n"
"3. Render becomes more transparent.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_WHATSTHIS
        m_MainNearPlane->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Allows you to clip away the near part of the volume.  Moving the slider all the way to the left will display the whole volume.  Moving the slider to the right will clip away the near portions of the volume.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_WHATSTHIS
        m_ZoomedOut->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "<p><b>Explorer Window</b></p>\n"
"\n"
"<p>Use this window to position the subvolume cube.  The right mouse button rotates the view.  To move the subvolume:\n"
"<ol>\n"
"<li>Click on one of the axes of the subvolume.\n"
"<li>Drag along that axis.\n"
"</ol>\n"
"To resize the subvolume:\n"
"<ol>\n"
"<li>Click on the end of one of the axes of the subvolume\n"
"<li>Drag along that axis.\n"
"</ol>\n"
"</p>", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_WHATSTHIS
        m_ExplorerNearPlane->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Allows you to clip away the near part of the volume.  Moving the slider all the way to the left will display the whole volume.  Moving the slider to the right will clip away the near portions of the volume.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        TextLabel2->setText(QApplication::translate("NewVolumeMainWindowBase", "Explorer Near Clip Plane", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        TextLabel2->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Allows you to clip away the near part of the volume.  Moving the slider all the way to the left will display the whole volume.  Moving the slider to the right will clip away the near portions of the volume.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_WHATSTHIS
        Slider1->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Determines the quality of the main render.  Dragging the slider to the left has the following 3 effects:\n"
"\n"
"1. Render speed increases.\n"
"2. Render quality decreases.\n"
"3. Render becomes more transparent.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        TextLabel1_2->setText(QApplication::translate("NewVolumeMainWindowBase", "Explorer Render Quality", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        TextLabel1_2->setProperty("whatsThis", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Determines the quality of the main render.  Dragging the slider to the left has the following 3 effects:\n"
"\n"
"1. Render speed increases.\n"
"2. Render quality decreases.\n"
"3. Render becomes more transparent.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        m_VariableLabel->setText(QApplication::translate("NewVolumeMainWindowBase", "Variable", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_VariableBox->setProperty("toolTip", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Variable", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_TimeStepLevel->setText(QApplication::translate("NewVolumeMainWindowBase", "Time Step", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_TimeStep->setProperty("toolTip", QVariant(QApplication::translate("NewVolumeMainWindowBase", "Time Step", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_RedLabel->setText(QApplication::translate("NewVolumeMainWindowBase", "Red", 0, QApplication::UnicodeUTF8));
        m_GreenLabel->setText(QApplication::translate("NewVolumeMainWindowBase", "Green", 0, QApplication::UnicodeUTF8));
        m_BlueLabel->setText(QApplication::translate("NewVolumeMainWindowBase", "Blue", 0, QApplication::UnicodeUTF8));
        m_AlphaLabel->setText(QApplication::translate("NewVolumeMainWindowBase", "Alpha", 0, QApplication::UnicodeUTF8));
        m_RGBATimeLabel->setText(QApplication::translate("NewVolumeMainWindowBase", "Time Step", 0, QApplication::UnicodeUTF8));
        m_ViewTabs->setTabText(m_ViewTabs->indexOf(ThreeDRover), QApplication::translate("NewVolumeMainWindowBase", "3D Rover", 0, QApplication::UnicodeUTF8));
        m_FileMenu->setTitle(QApplication::translate("NewVolumeMainWindowBase", "&File", 0, QApplication::UnicodeUTF8));
        m_ViewMenu->setTitle(QApplication::translate("NewVolumeMainWindowBase", "&View", 0, QApplication::UnicodeUTF8));
        m_GeometryMenu->setTitle(QApplication::translate("NewVolumeMainWindowBase", "&Geometry", 0, QApplication::UnicodeUTF8));
        popupMenu_39->setTitle(QApplication::translate("NewVolumeMainWindowBase", "LBIE", 0, QApplication::UnicodeUTF8));
        m_ServersMenu->setTitle(QApplication::translate("NewVolumeMainWindowBase", "&Servers", 0, QApplication::UnicodeUTF8));
        m_AnimationMenu->setTitle(QApplication::translate("NewVolumeMainWindowBase", "&Animation", 0, QApplication::UnicodeUTF8));
        Tools->setTitle(QApplication::translate("NewVolumeMainWindowBase", "Tools", 0, QApplication::UnicodeUTF8));
        m_HelpMenu->setTitle(QApplication::translate("NewVolumeMainWindowBase", "&Help", 0, QApplication::UnicodeUTF8));
    } // retranslateUi


protected:
    enum IconID
    {
        image0_ID,
        unknown_ID
    };
    static QPixmap qt_get_icon(IconID id)
    {
    static const unsigned char image0_data[] = { 
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x16,
    0x08, 0x06, 0x00, 0x00, 0x00, 0xc4, 0xb4, 0x6c, 0x3b, 0x00, 0x00, 0x04,
    0xab, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0xb5, 0x95, 0xdb, 0x6b, 0x5c,
    0x55, 0x14, 0x87, 0xbf, 0x33, 0xe7, 0xcc, 0x3d, 0x93, 0xcb, 0x4c, 0x6e,
    0x93, 0xa4, 0x49, 0xda, 0x1a, 0xdb, 0xc6, 0x58, 0x2f, 0xd5, 0x52, 0x51,
    0x6b, 0x4b, 0xd1, 0xb6, 0x0f, 0x16, 0x5a, 0x10, 0xda, 0x07, 0xb1, 0x28,
    0x28, 0x22, 0xfe, 0x03, 0x82, 0x20, 0x08, 0xfe, 0x0b, 0xa2, 0xa0, 0x2f,
    0x22, 0xfa, 0x26, 0x22, 0x85, 0xd6, 0x07, 0x91, 0x50, 0x63, 0x08, 0x78,
    0x69, 0xad, 0x84, 0xb4, 0x4e, 0xcc, 0x4c, 0x9a, 0x4c, 0xe6, 0x3e, 0xc9,
    0x99, 0x73, 0xe6, 0x9c, 0x33, 0x73, 0xf6, 0xf2, 0x61, 0x4c, 0x6a, 0xa8,
    0x4f, 0x82, 0xeb, 0x61, 0xb3, 0x2f, 0x6b, 0x7f, 0xfc, 0x36, 0x6b, 0xad,
    0xbd, 0xb4, 0xb9, 0xb9, 0x39, 0xb6, 0xed, 0x9d, 0xf7, 0x2f, 0x4a, 0x61,
    0xdd, 0xe2, 0xbf, 0xd8, 0xd0, 0x48, 0x9c, 0x0f, 0xde, 0xfd, 0x52, 0xdb,
    0x5e, 0x6b, 0x73, 0x73, 0x73, 0x7c, 0xf4, 0xc5, 0xeb, 0xb2, 0x30, 0x9b,
    0xe7, 0xf2, 0xcb, 0x6f, 0x11, 0x8b, 0xc6, 0x09, 0x87, 0x63, 0x38, 0x4e,
    0x13, 0xdf, 0xf7, 0x11, 0x91, 0x7f, 0x05, 0x89, 0x08, 0xbe, 0xf2, 0x71,
    0x9a, 0x4d, 0x72, 0xab, 0xcb, 0xac, 0xde, 0xcd, 0x92, 0xbb, 0x7b, 0x87,
    0xa3, 0xc7, 0xd3, 0xbc, 0x71, 0xe9, 0x63, 0xcd, 0x00, 0x58, 0x98, 0xcd,
    0xf3, 0xea, 0x2b, 0x6f, 0x33, 0x39, 0x31, 0x45, 0xb9, 0x5c, 0xa4, 0x5a,
    0xab, 0x60, 0xe8, 0x90, 0xe8, 0x89, 0xf2, 0xe0, 0x74, 0x1f, 0x18, 0x9b,
    0x38, 0xad, 0x32, 0x0d, 0xab, 0x4a, 0xf6, 0xb6, 0x46, 0xd3, 0x0c, 0xe2,
    0xda, 0x11, 0x44, 0x14, 0x9e, 0xe7, 0x11, 0x08, 0x18, 0x74, 0x27, 0xfa,
    0x18, 0x1a, 0x98, 0x60, 0x61, 0x36, 0xcb, 0x1b, 0x97, 0x40, 0x3b, 0x71,
    0x76, 0x5c, 0xce, 0x9e, 0xba, 0xcc, 0xbe, 0xbd, 0x07, 0x28, 0x14, 0x36,
    0x28, 0x95, 0x0b, 0xa4, 0xd3, 0x29, 0x1e, 0x7e, 0x7c, 0x92, 0xe4, 0x20,
    0xf8, 0x5a, 0x01, 0xdb, 0x5d, 0xc3, 0x6d, 0x57, 0xf1, 0xda, 0x9b, 0x98,
    0x56, 0x8d, 0x42, 0xde, 0x24, 0xb7, 0x98, 0xa0, 0xba, 0x36, 0x84, 0xdf,
    0x16, 0x2c, 0xab, 0x81, 0xe3, 0xb8, 0xd8, 0x76, 0x93, 0x5a, 0xad, 0x44,
    0xa4, 0xbb, 0x81, 0x51, 0x58, 0xb7, 0x88, 0x45, 0xe3, 0x94, 0xcb, 0x45,
    0x4a, 0xe5, 0x02, 0xe3, 0xe3, 0x83, 0x9c, 0x39, 0x77, 0x84, 0x2d, 0xe7,
    0x0e, 0x95, 0xc6, 0xaf, 0xd4, 0xed, 0x45, 0xdc, 0x56, 0x85, 0x40, 0x20,
    0x88, 0xa1, 0x77, 0xa1, 0x87, 0xa0, 0x7f, 0xb4, 0x49, 0x24, 0xb5, 0x44,
    0x75, 0xad, 0x97, 0xdf, 0xbe, 0x3f, 0x48, 0xbb, 0x1d, 0x40, 0x44, 0x10,
    0x25, 0x68, 0x9a, 0x41, 0x61, 0xdd, 0xc2, 0x00, 0x08, 0x87, 0x63, 0x54,
    0x6b, 0x15, 0x26, 0x26, 0x87, 0x39, 0x75, 0x76, 0x9a, 0xc2, 0xd6, 0x75,
    0xb2, 0x95, 0xaf, 0xd9, 0xb4, 0xfe, 0xa0, 0xbc, 0xd6, 0x8b, 0xdf, 0xd2,
    0xd1, 0x34, 0xd0, 0x03, 0x36, 0xfd, 0xc3, 0x11, 0x62, 0x89, 0x16, 0x00,
    0x7d, 0xe9, 0x1a, 0x33, 0x27, 0x6e, 0xf1, 0xcb, 0xb7, 0x87, 0xf0, 0x3c,
    0x10, 0x04, 0xad, 0x83, 0xec, 0x8c, 0x8e, 0xd3, 0xa4, 0x2f, 0x19, 0xe7,
    0xd4, 0x99, 0x69, 0xca, 0xd6, 0x3c, 0xb7, 0xf3, 0x9f, 0x52, 0x29, 0x7b,
    0x2c, 0xfd, 0x38, 0x4d, 0x29, 0x9b, 0xdc, 0x15, 0xb4, 0x50, 0xc4, 0x67,
    0xef, 0xe1, 0x0a, 0xfb, 0x8e, 0x74, 0x82, 0xda, 0x3b, 0x64, 0xb2, 0xe7,
    0xd0, 0x06, 0x8b, 0xf3, 0xc3, 0xf0, 0x8f, 0x38, 0x07, 0x00, 0x7c, 0xdf,
    0x27, 0x35, 0x10, 0xa5, 0xd1, 0xfa, 0x83, 0xd5, 0xca, 0x15, 0xca, 0xc5,
    0x16, 0xbf, 0x5e, 0xbb, 0x1f, 0xfa, 0xd8, 0x33, 0x1a, 0x7b, 0xa6, 0x14,
    0x4b, 0x0b, 0x83, 0x2c, 0xcd, 0x4f, 0x22, 0xaa, 0x93, 0x5d, 0x23, 0x07,
    0x36, 0x88, 0xc4, 0x3d, 0xe4, 0xef, 0x6c, 0xd9, 0x51, 0x2c, 0x22, 0x8c,
    0xef, 0x0f, 0x51, 0xb7, 0x6f, 0x52, 0x6d, 0x64, 0xc8, 0xfc, 0x34, 0x45,
    0xa3, 0x16, 0xdb, 0x05, 0x3d, 0x71, 0x7a, 0x2f, 0x4f, 0x3c, 0xdb, 0xcb,
    0x66, 0xf3, 0x0e, 0x4a, 0xf2, 0x2c, 0xff, 0x32, 0x46, 0x32, 0xbd, 0x49,
    0x72, 0xac, 0x46, 0x24, 0xee, 0x32, 0x38, 0x59, 0xa3, 0x52, 0xec, 0xde,
    0xad, 0x18, 0x40, 0x0f, 0x9b, 0x58, 0xee, 0x5d, 0xec, 0x5a, 0x8c, 0xc2,
    0x9f, 0xfd, 0xbb, 0xa0, 0x27, 0x5f, 0x38, 0xcc, 0xf3, 0xa7, 0x9f, 0x26,
    0xa8, 0x77, 0x01, 0xc2, 0xa3, 0x27, 0x8b, 0x04, 0xc3, 0x6d, 0x56, 0x7e,
    0x1b, 0xdd, 0xf1, 0xe9, 0x1f, 0xdb, 0xdc, 0x75, 0x67, 0x07, 0xac, 0x19,
    0x0d, 0x94, 0xf2, 0x10, 0xd1, 0x76, 0x9e, 0x08, 0x30, 0xf3, 0xc8, 0x38,
    0xc7, 0x9f, 0x7b, 0x12, 0x41, 0xe1, 0xb6, 0xeb, 0xd8, 0xcd, 0x2a, 0x3f,
    0x7f, 0x97, 0xa4, 0xed, 0xe9, 0xf8, 0xed, 0x00, 0x22, 0x1d, 0xdf, 0xe4,
    0xc8, 0xd6, 0xbf, 0x83, 0x7d, 0xe5, 0x11, 0x34, 0x12, 0x04, 0x8d, 0xf8,
    0xce, 0xe1, 0xd4, 0x74, 0x92, 0xf3, 0x17, 0x8f, 0x11, 0x8c, 0x36, 0xb1,
    0xbd, 0x3c, 0xb6, 0x97, 0xc7, 0xf1, 0x4c, 0xea, 0x85, 0x18, 0x22, 0x1a,
    0x9e, 0x1d, 0xc6, 0x31, 0x23, 0x00, 0xbb, 0xc4, 0xec, 0x02, 0x37, 0x1a,
    0x2e, 0x41, 0x3d, 0x41, 0x3c, 0xb4, 0x07, 0x80, 0x81, 0x11, 0x9d, 0x73,
    0x97, 0x26, 0xf0, 0xb5, 0x22, 0x75, 0x7b, 0x89, 0x9a, 0xf5, 0x3b, 0xb6,
    0x7b, 0x97, 0x60, 0x48, 0x71, 0xfc, 0xbc, 0x49, 0x77, 0xaa, 0x4d, 0x28,
    0xa2, 0x88, 0xf5, 0x34, 0x11, 0x01, 0x7b, 0x2b, 0xbc, 0x0b, 0x6c, 0x6c,
    0x4f, 0xbc, 0xd6, 0x26, 0x91, 0x90, 0x4f, 0x32, 0x95, 0x20, 0x35, 0xec,
    0xe2, 0xb9, 0x3e, 0x1b, 0xa5, 0x65, 0x12, 0xdd, 0x06, 0x4d, 0xaf, 0x4a,
    0xbd, 0x56, 0x23, 0x92, 0x70, 0x31, 0xf4, 0x18, 0x7d, 0x7d, 0x61, 0x4e,
    0x5c, 0xb0, 0x28, 0x16, 0x36, 0x10, 0x01, 0x04, 0x36, 0x32, 0xa9, 0xfb,
    0x15, 0x8b, 0x08, 0xeb, 0x39, 0x0b, 0xdb, 0xdb, 0x40, 0x0b, 0x36, 0x98,
    0x79, 0xca, 0xc6, 0xb1, 0x15, 0x57, 0xbe, 0xa8, 0xf0, 0x67, 0x76, 0x89,
    0xeb, 0x57, 0xb7, 0xb8, 0xfa, 0x79, 0x82, 0xb5, 0x8c, 0x41, 0xdb, 0xd3,
    0x01, 0x21, 0x14, 0xaf, 0xd3, 0x9b, 0x2e, 0x21, 0x0a, 0x7c, 0x5f, 0xa3,
    0xe5, 0xe9, 0xf7, 0x83, 0x7d, 0xe5, 0xb3, 0xba, 0xd4, 0x45, 0xa5, 0xbe,
    0xc2, 0x96, 0x93, 0xa1, 0x77, 0x6c, 0x99, 0x67, 0x5e, 0x5a, 0xc4, 0xb1,
    0x15, 0xd7, 0x3e, 0xeb, 0x27, 0x73, 0x33, 0x81, 0x6b, 0xeb, 0x5c, 0xff,
    0x3a, 0xc5, 0xe2, 0x4f, 0x8a, 0x86, 0x9b, 0xa3, 0xe1, 0xac, 0xa2, 0x14,
    0x28, 0x5f, 0x68, 0x9a, 0x61, 0x8a, 0x2b, 0x7d, 0x20, 0xc2, 0xf6, 0x67,
    0xd8, 0xa9, 0xbc, 0x66, 0x13, 0xd3, 0x0c, 0x92, 0x5d, 0x8c, 0x31, 0x7a,
    0x28, 0x0f, 0x40, 0x30, 0x0a, 0xc7, 0x2e, 0xdc, 0x20, 0x77, 0x2b, 0x4d,
    0xdb, 0x33, 0x08, 0x04, 0x74, 0x94, 0x12, 0x42, 0x3d, 0x25, 0x2c, 0xa7,
    0x86, 0xf2, 0x41, 0x29, 0x41, 0x29, 0xd0, 0x0c, 0x8f, 0xe9, 0x67, 0x6f,
    0xf3, 0xc3, 0x57, 0xe3, 0x88, 0x52, 0xf7, 0xc0, 0xb9, 0xd5, 0x65, 0x02,
    0x01, 0x83, 0xfc, 0xdc, 0x28, 0xad, 0xb6, 0xcd, 0xf8, 0x43, 0x1d, 0x78,
    0x28, 0xe2, 0xf2, 0xc0, 0x93, 0x2b, 0x68, 0x1a, 0x3b, 0x4a, 0x44, 0x41,
    0xbb, 0xd5, 0x01, 0x8a, 0xea, 0xec, 0x1b, 0xba, 0xa2, 0x7b, 0xb0, 0xc1,
    0xe4, 0x23, 0x45, 0x7e, 0xfe, 0xbe, 0x13, 0x36, 0x7d, 0xe6, 0xf1, 0x81,
    0xf7, 0x2c, 0x53, 0x21, 0x0a, 0x5a, 0x2d, 0x9f, 0x62, 0xb6, 0x9b, 0xcd,
    0x52, 0x8c, 0x50, 0xb4, 0x45, 0xb4, 0xcb, 0x45, 0xa4, 0x03, 0x50, 0x0a,
    0xc4, 0x07, 0x5f, 0x81, 0xf2, 0xa1, 0xba, 0x9e, 0x60, 0x6d, 0x69, 0x80,
    0x9e, 0x01, 0x0b, 0xc7, 0xd1, 0xd9, 0xc8, 0x74, 0xb1, 0x96, 0x09, 0x53,
    0x2d, 0x79, 0xa4, 0x86, 0x8c, 0x4e, 0x07, 0x79, 0xed, 0xcd, 0x17, 0x65,
    0x68, 0x60, 0x82, 0x68, 0x24, 0x81, 0x28, 0x41, 0x10, 0x10, 0x18, 0xde,
    0x6b, 0xd2, 0x97, 0xb6, 0x19, 0xd9, 0xdf, 0xa9, 0x2a, 0x11, 0x28, 0xe6,
    0xba, 0x28, 0xaf, 0x75, 0x51, 0x58, 0x49, 0xa0, 0xda, 0xb0, 0xe7, 0x60,
    0x95, 0xd4, 0xd8, 0x16, 0x99, 0x9b, 0x31, 0xd6, 0x73, 0x1e, 0xf5, 0x9a,
    0xc9, 0x27, 0x1f, 0x7e, 0xd3, 0xe9, 0x20, 0x47, 0x8f, 0xa7, 0x59, 0x98,
    0xcd, 0xd2, 0xd3, 0x35, 0x84, 0xa6, 0x19, 0x3b, 0x5f, 0xdf, 0xd6, 0x0d,
    0x0d, 0x6e, 0xc4, 0x81, 0x7b, 0x45, 0x73, 0xcf, 0x2c, 0x44, 0xe0, 0xd6,
    0x7c, 0x90, 0x60, 0x34, 0x8a, 0xd7, 0xf2, 0x30, 0x4d, 0x93, 0xa1, 0x91,
    0x8e, 0xaf, 0xf6, 0x7f, 0x35, 0xd3, 0xbf, 0x00, 0xa1, 0x32, 0x6e, 0xea,
    0xa5, 0x5a, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
    0xae, 0x42, 0x60, 0x82
};

    switch (id) {
        case image0_ID:  { QImage img; img.loadFromData(image0_data, sizeof(image0_data), "PNG"); return QPixmap::fromImage(img); }
        default: return QPixmap();
    } // switch
    } // icon

};

namespace Ui {
    class NewVolumeMainWindowBase: public Ui_NewVolumeMainWindowBase {};
} // namespace Ui

QT_END_NAMESPACE

class NewVolumeMainWindowBase : public Q3MainWindow, public Ui::NewVolumeMainWindowBase
{
    Q_OBJECT

public:
    NewVolumeMainWindowBase(QWidget* parent = 0, const char* name = 0, Qt::WindowFlags fl = Qt::WType_TopLevel);
    ~NewVolumeMainWindowBase();

public slots:
    virtual void actionSlot();
    virtual void functionChangedSlot();
    virtual void setExplorerQualitySlot( int );
    virtual void mouseReleasedMain();
    virtual void mouseReleasedPreview();
    virtual void explorerChangedSlot();
    virtual void connectToDCSlot();
    virtual void setMainQualitySlot( int );
    virtual void optionsSlot();
    virtual void explorerMoveSlot();
    virtual void explorerReleaseSlot();
    virtual void connectServerSlot();
    virtual void serverSettingsSlot();
    virtual void disconnectServerSlot();
    virtual void renderAnimationSlot();
    virtual void renderFrameSlot();
    virtual void isocontourNodeColorChangedSlot( int, double, double, double );
    virtual void isocontourNodeChangedSlot( int, double );
    virtual void isocontourNodeAddedSlot( int, double, double, double, double );
    virtual void isocontourNodeDeletedSlot( int );
    virtual void isocontourNodesAllChangedSlot();
    virtual void bilateralFilterSlot();
    virtual void saveSubvolumeSlot();
    virtual void zoomedOutClipSlot( int );
    virtual void zoomedInClipSlot( int );
    virtual void centerSlot();
    virtual void loadGeometrySlot();
    virtual void clearGeometrySlot();
    virtual void saveGeometrySlot();
    virtual void exportZoomedInIsosurfaceSlot();
    virtual void exportZoomedOutIsosurfaceSlot();
    virtual void variableOrTimeChangeSlot();
    virtual void contrastEnhancementSlot();
    virtual void PEDetectionSlot();
    virtual void pocketTunnelSlot();
    virtual void smoothGeometrySlot();
    virtual void anisotropicDiffusionSlot();
    virtual void sliceRenderingSlot();
    virtual void acquireConTreeSlot();
    virtual void acquireConSpecSlot();
    virtual void toggleWireframeRenderingSlot( bool );
    virtual void toggleRenderSurfaceWithWireframeSlot( bool );
    virtual void toggleWireCubeSlot( bool );
    virtual void toggleDepthCueSlot( bool );
    virtual void saveImageSlot();
    virtual void contourTreeNodeAddedSlot( int, int, double );
    virtual void contourTreeNodeDeletedSlot( int );
    virtual void contourTreeNodeChangedSlot( int, double );
    virtual void contourTreeNodeExploringSlot( int, double );
    virtual void startRecordingAnimationSlot();
    virtual void stopRecordingAnimationSlot();
    virtual void playAnimationSlot();
    virtual void saveAnimationSlot();
    virtual void loadAnimationSlot();
    virtual void stopAnimationSlot();
    virtual void renderSequenceSlot();
    virtual void virusSegmentationSlot();
    virtual void toggleGeometryTransformationSlot();
    virtual void resetGeometryTransformationSlot();
    virtual void toggleTerminalSlot( bool );
    virtual void boundaryPointCloudSlot();
    virtual void tightCoconeSlot();
    virtual void curationSlot();
    virtual void skeletonizationSlot();
    virtual void clipGeometryToVolumeBoxSlot( bool );
    virtual void saveSkeletonSlot();
    virtual void clearSkeletonSlot();
    virtual void signedDistanceFunctionSlot();
    virtual void mergeGeometrySlot();
    virtual void convertIsosurfaceToGeometrySlot();
    virtual void highLevelSetReconSlot();
    virtual void highLevelSetSlot();
    virtual void LBIEMeshingSlot();
    virtual void LBIEQualityImprovementSlot();
    virtual void openImageFileSlot();
    virtual void projectGeometrySlot();
    virtual void gdtvFilterSlot();
    virtual void MSLevelSetSlot();
    virtual void isocontourAskIsovalueSlot( int );
    virtual void colorGeometryByVolumeSlot();
    virtual void cullGeometryWithSubvolumeBoxSlot();
    virtual void secondaryStructureElucidationSlot();

protected slots:
    virtual void languageChange();

};

#endif // NEWVOLUMEMAINWINDOWBASE_H
