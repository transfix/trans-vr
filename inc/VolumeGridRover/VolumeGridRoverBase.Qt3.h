#ifndef VOLUMEGRIDROVERBASE_H
#define VOLUMEGRIDROVERBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
#include <Qt3Support/Q3Frame>
#include <Qt3Support/Q3GroupBox>
#include <Qt3Support/Q3Header>
#include <Qt3Support/Q3ListView>
#include <Qt3Support/Q3MimeSourceFactory>
#include <Qt3Support/Q3WidgetStack>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QTabWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VolumeGridRoverBase
{
public:
    QGridLayout *gridLayout;
    Q3GroupBox *m_SliceCanvasGroup;
    QGridLayout *gridLayout1;
    Q3ButtonGroup *m_SliceAxisGroup;
    QGridLayout *gridLayout2;
    QRadioButton *m_ZYSliceCanvasButton;
    QRadioButton *m_XZSliceCanvasButton;
    QRadioButton *m_XYSliceCanvasButton;
    Q3WidgetStack *m_SliceCanvasStack;
    QWidget *m_XYSlicePage;
    QGridLayout *gridLayout3;
    Q3Frame *m_XYSliceFrame;
    QSlider *m_XYDepthSlide;
    QPushButton *m_XYResetViewButton;
    QWidget *m_XZSlicePage;
    QGridLayout *gridLayout4;
    Q3Frame *m_XZSliceFrame;
    QSlider *m_XZDepthSlide;
    QPushButton *m_XZResetViewButton;
    QWidget *m_ZYSlicePage;
    QGridLayout *gridLayout5;
    Q3Frame *m_ZYSliceFrame;
    QSlider *m_ZYDepthSlide;
    QPushButton *m_ZYResetViewButton;
    QTabWidget *m_GridCellTabs;
    QWidget *VoxelCoordinates;
    QGridLayout *gridLayout6;
    Q3GroupBox *m_IndicesGroup;
    QGridLayout *gridLayout7;
    QLabel *m_XText;
    QLineEdit *m_X;
    QLabel *m_YText;
    QLineEdit *m_Y;
    QLabel *m_ZText;
    QLineEdit *m_Z;
    Q3GroupBox *m_ObjectCoordinatesGroup;
    QGridLayout *gridLayout8;
    QLineEdit *m_ObjX;
    QLabel *m_ObjXText;
    QLabel *m_ObjYText;
    QLineEdit *m_ObjY;
    QLabel *m_ObjZText;
    QLineEdit *m_ObjZ;
    QWidget *VoxelInfo;
    QGridLayout *gridLayout9;
    QHBoxLayout *hboxLayout;
    QLabel *m_RText;
    QLineEdit *m_R;
    QLabel *m_GText;
    QLineEdit *m_G;
    QLabel *m_BText;
    QLineEdit *m_B;
    QLabel *m_AText;
    QLineEdit *m_A;
    QHBoxLayout *hboxLayout1;
    QLabel *m_ValueText;
    QLineEdit *m_Value;
    QLabel *m_ColorNameText;
    QLineEdit *m_ColorName;
    QLabel *m_MappedValueText;
    QLineEdit *m_MappedValue;
    QSpacerItem *spacer1;
    QTabWidget *m_VariableInformationTabs;
    QWidget *VariableSelection;
    QHBoxLayout *hboxLayout2;
    QLabel *m_VariableText;
    QComboBox *m_Variable;
    QLabel *m_TimestepText;
    QSpinBox *m_Timestep;
    QWidget *VariableInformation;
    QGridLayout *gridLayout10;
    QLabel *m_MinimumValueText;
    QLineEdit *m_MinimumValue;
    QLabel *m_MaximumValueText;
    QLineEdit *m_MaximumValue;
    QTabWidget *m_OptionsAndSegmentation;
    QWidget *tab;
    QGridLayout *gridLayout11;
    Q3GroupBox *m_DisplayOptionsGroup;
    QGridLayout *gridLayout12;
    QHBoxLayout *hboxLayout3;
    QLabel *m_BackgroundColorText;
    QPushButton *m_BackgroundColor;
    QCheckBox *m_GreyScale;
    QSlider *m_PointSize;
    QLabel *m_PointSizeText;
    QCheckBox *m_RenderControlPoints;
    QCheckBox *m_RenderSDF;
    QCheckBox *m_Isocontouring;
    QTabWidget *m_GridCellMarkingToolSelection;
    QWidget *PointClasses;
    QGridLayout *gridLayout13;
    QPushButton *m_PointClassColor;
    QComboBox *m_PointClass;
    QLabel *m_PointClassText;
    QPushButton *m_AddPointClass;
    QPushButton *m_DeletePointClass;
    QPushButton *m_PointClassesLoadButton;
    QPushButton *m_PointClassesSaveButton;
    QWidget *Contours;
    QGridLayout *gridLayout14;
    Q3ListView *m_Objects;
    QLabel *m_InterpolationTypeText;
    QLabel *m_InterpolationSamplingText;
    QPushButton *m_DeleteContour;
    QPushButton *m_SaveContoursButton;
    QSpinBox *m_InterpolationSampling;
    QPushButton *m_AddContour;
    QPushButton *m_ContourColor;
    QPushButton *m_LoadContourButton;
    QComboBox *m_InterpolationType;
    QWidget *tab1;
    QVBoxLayout *vboxLayout;
    Q3GroupBox *m_SegmentationThresholdGroup;
    QHBoxLayout *hboxLayout4;
    QLabel *m_TresholdLowText;
    QLineEdit *m_ThresholdLow;
    QLabel *m_ThresholdHighText;
    QLineEdit *m_ThresholdHigh;
    Q3GroupBox *m_LocalSegmentationGroup;
    QGridLayout *gridLayout15;
    QLabel *m_OutputFileText;
    QLineEdit *m_LocalOutputFile;
    QPushButton *m_LocalOutputFileSelection;
    QPushButton *m_LocalRun;
    Q3GroupBox *m_RemoteSegmentationGroup;
    QGridLayout *gridLayout16;
    QLabel *m_RemoteOutputFileText;
    QLineEdit *m_RemoteFile;
    QPushButton *m_RemoteOutputFileSelection;
    QGridLayout *gridLayout17;
    QPushButton *m_RemoteRun;
    QLabel *m_PortText;
    QLineEdit *m_Hostname;
    QLineEdit *m_Port;
    QLabel *m_HostnameText;
    QWidget *tab2;
    QPushButton *m_EMClusteringRun;
    QWidget *Tiling;
    QGridLayout *gridLayout18;
    Q3GroupBox *m_TilingOutputFilenameGroup;
    QGridLayout *gridLayout19;
    QLabel *m_TilingOutputDirectoryText;
    QLineEdit *m_TilingOutputDirectory;
    QPushButton *m_TilingOutputDirectorySelect;
    QSpacerItem *spacer11;
    Q3ButtonGroup *m_TilingOutputOptions;
    QGridLayout *gridLayout20;
    QRadioButton *m_TilingOutputInMemory;
    QRadioButton *m_TilingOptionSaveToFile;
    QSpacerItem *spacer12;
    QPushButton *m_Run2DSDF;
    QPushButton *m_RunTiling;
    QWidget *TabPage;
    QVBoxLayout *vboxLayout1;
    Q3GroupBox *m_CurrentSliceProcessGroup;
    QGridLayout *gridLayout21;
    QPushButton *m_CalcMedialAxis;
    QPushButton *m_SDFOptions;
    QPushButton *m_CalcDelaunayVoronoi;
    QPushButton *m_ContourCuration;

    void setupUi(QWidget *VolumeGridRoverBase)
    {
        if (VolumeGridRoverBase->objectName().isEmpty())
            VolumeGridRoverBase->setObjectName(QString::fromUtf8("VolumeGridRoverBase"));
        VolumeGridRoverBase->resize(839, 767);
        gridLayout = new QGridLayout(VolumeGridRoverBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_SliceCanvasGroup = new Q3GroupBox(VolumeGridRoverBase);
        m_SliceCanvasGroup->setObjectName(QString::fromUtf8("m_SliceCanvasGroup"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(3), static_cast<QSizePolicy::Policy>(3));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_SliceCanvasGroup->sizePolicy().hasHeightForWidth());
        m_SliceCanvasGroup->setSizePolicy(sizePolicy);
        QFont font;
        m_SliceCanvasGroup->setFont(font);
        m_SliceCanvasGroup->setColumnLayout(0, Qt::Vertical);
        m_SliceCanvasGroup->layout()->setSpacing(6);
        m_SliceCanvasGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_SliceCanvasGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_SliceAxisGroup = new Q3ButtonGroup(m_SliceCanvasGroup);
        m_SliceAxisGroup->setObjectName(QString::fromUtf8("m_SliceAxisGroup"));
        m_SliceAxisGroup->setFrameShape(Q3GroupBox::GroupBoxPanel);
        m_SliceAxisGroup->setFrameShadow(Q3GroupBox::Sunken);
        m_SliceAxisGroup->setChecked(false);
        m_SliceAxisGroup->setRadioButtonExclusive(true);
        m_SliceAxisGroup->setColumnLayout(0, Qt::Vertical);
        m_SliceAxisGroup->layout()->setSpacing(6);
        m_SliceAxisGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_SliceAxisGroup->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        m_ZYSliceCanvasButton = new QRadioButton(m_SliceAxisGroup);
        m_ZYSliceCanvasButton->setObjectName(QString::fromUtf8("m_ZYSliceCanvasButton"));
        m_SliceAxisGroup->insert(m_ZYSliceCanvasButton, 2);

        gridLayout2->addWidget(m_ZYSliceCanvasButton, 0, 2, 1, 1);

        m_XZSliceCanvasButton = new QRadioButton(m_SliceAxisGroup);
        m_XZSliceCanvasButton->setObjectName(QString::fromUtf8("m_XZSliceCanvasButton"));

        gridLayout2->addWidget(m_XZSliceCanvasButton, 0, 1, 1, 1);

        m_XYSliceCanvasButton = new QRadioButton(m_SliceAxisGroup);
        m_XYSliceCanvasButton->setObjectName(QString::fromUtf8("m_XYSliceCanvasButton"));
        m_XYSliceCanvasButton->setChecked(true);
        m_SliceAxisGroup->insert(m_XYSliceCanvasButton, 0);

        gridLayout2->addWidget(m_XYSliceCanvasButton, 0, 0, 1, 1);


        gridLayout1->addWidget(m_SliceAxisGroup, 0, 0, 1, 1);

        m_SliceCanvasStack = new Q3WidgetStack(m_SliceCanvasGroup);
        m_SliceCanvasStack->setObjectName(QString::fromUtf8("m_SliceCanvasStack"));
        sizePolicy.setHeightForWidth(m_SliceCanvasStack->sizePolicy().hasHeightForWidth());
        m_SliceCanvasStack->setSizePolicy(sizePolicy);
        m_XYSlicePage = new QWidget(m_SliceCanvasStack);
        m_XYSlicePage->setObjectName(QString::fromUtf8("m_XYSlicePage"));
        gridLayout3 = new QGridLayout(m_XYSlicePage);
        gridLayout3->setSpacing(6);
        gridLayout3->setContentsMargins(11, 11, 11, 11);
        gridLayout3->setObjectName(QString::fromUtf8("gridLayout3"));
        gridLayout3->setContentsMargins(0, 0, 0, 0);
        m_XYSliceFrame = new Q3Frame(m_XYSlicePage);
        m_XYSliceFrame->setObjectName(QString::fromUtf8("m_XYSliceFrame"));
        sizePolicy.setHeightForWidth(m_XYSliceFrame->sizePolicy().hasHeightForWidth());
        m_XYSliceFrame->setSizePolicy(sizePolicy);
        m_XYSliceFrame->setMinimumSize(QSize(250, 250));
        QFont font1;
        font1.setFamily(QString::fromUtf8("Bitstream Charter"));
        m_XYSliceFrame->setFont(font1);
        m_XYSliceFrame->setFrameShape(QFrame::StyledPanel);
        m_XYSliceFrame->setFrameShadow(QFrame::Raised);

        gridLayout3->addWidget(m_XYSliceFrame, 0, 0, 1, 2);

        m_XYDepthSlide = new QSlider(m_XYSlicePage);
        m_XYDepthSlide->setObjectName(QString::fromUtf8("m_XYDepthSlide"));
        m_XYDepthSlide->setMaximumSize(QSize(32767, 20));
        m_XYDepthSlide->setOrientation(Qt::Horizontal);

        gridLayout3->addWidget(m_XYDepthSlide, 1, 0, 1, 1);

        m_XYResetViewButton = new QPushButton(m_XYSlicePage);
        m_XYResetViewButton->setObjectName(QString::fromUtf8("m_XYResetViewButton"));
        m_XYResetViewButton->setFont(font);

        gridLayout3->addWidget(m_XYResetViewButton, 1, 1, 1, 1);

        m_SliceCanvasStack->addWidget(m_XYSlicePage, 0);
        m_XZSlicePage = new QWidget(m_SliceCanvasStack);
        m_XZSlicePage->setObjectName(QString::fromUtf8("m_XZSlicePage"));
        gridLayout4 = new QGridLayout(m_XZSlicePage);
        gridLayout4->setSpacing(6);
        gridLayout4->setContentsMargins(11, 11, 11, 11);
        gridLayout4->setObjectName(QString::fromUtf8("gridLayout4"));
        gridLayout4->setContentsMargins(0, 0, 0, 0);
        m_XZSliceFrame = new Q3Frame(m_XZSlicePage);
        m_XZSliceFrame->setObjectName(QString::fromUtf8("m_XZSliceFrame"));
        sizePolicy.setHeightForWidth(m_XZSliceFrame->sizePolicy().hasHeightForWidth());
        m_XZSliceFrame->setSizePolicy(sizePolicy);
        m_XZSliceFrame->setMinimumSize(QSize(250, 250));
        m_XZSliceFrame->setFont(font1);
        m_XZSliceFrame->setFrameShape(QFrame::StyledPanel);
        m_XZSliceFrame->setFrameShadow(QFrame::Raised);

        gridLayout4->addWidget(m_XZSliceFrame, 0, 0, 1, 2);

        m_XZDepthSlide = new QSlider(m_XZSlicePage);
        m_XZDepthSlide->setObjectName(QString::fromUtf8("m_XZDepthSlide"));
        m_XZDepthSlide->setMaximumSize(QSize(32767, 20));
        m_XZDepthSlide->setOrientation(Qt::Horizontal);

        gridLayout4->addWidget(m_XZDepthSlide, 1, 0, 1, 1);

        m_XZResetViewButton = new QPushButton(m_XZSlicePage);
        m_XZResetViewButton->setObjectName(QString::fromUtf8("m_XZResetViewButton"));
        m_XZResetViewButton->setFont(font);

        gridLayout4->addWidget(m_XZResetViewButton, 1, 1, 1, 1);

        m_SliceCanvasStack->addWidget(m_XZSlicePage, 1);
        m_ZYSlicePage = new QWidget(m_SliceCanvasStack);
        m_ZYSlicePage->setObjectName(QString::fromUtf8("m_ZYSlicePage"));
        gridLayout5 = new QGridLayout(m_ZYSlicePage);
        gridLayout5->setSpacing(6);
        gridLayout5->setContentsMargins(11, 11, 11, 11);
        gridLayout5->setObjectName(QString::fromUtf8("gridLayout5"));
        gridLayout5->setContentsMargins(0, 0, 0, 0);
        m_ZYSliceFrame = new Q3Frame(m_ZYSlicePage);
        m_ZYSliceFrame->setObjectName(QString::fromUtf8("m_ZYSliceFrame"));
        sizePolicy.setHeightForWidth(m_ZYSliceFrame->sizePolicy().hasHeightForWidth());
        m_ZYSliceFrame->setSizePolicy(sizePolicy);
        m_ZYSliceFrame->setMinimumSize(QSize(250, 250));
        m_ZYSliceFrame->setFont(font1);
        m_ZYSliceFrame->setFrameShape(QFrame::StyledPanel);
        m_ZYSliceFrame->setFrameShadow(QFrame::Raised);

        gridLayout5->addWidget(m_ZYSliceFrame, 0, 0, 1, 2);

        m_ZYDepthSlide = new QSlider(m_ZYSlicePage);
        m_ZYDepthSlide->setObjectName(QString::fromUtf8("m_ZYDepthSlide"));
        m_ZYDepthSlide->setMaximumSize(QSize(32767, 20));
        m_ZYDepthSlide->setOrientation(Qt::Horizontal);

        gridLayout5->addWidget(m_ZYDepthSlide, 1, 0, 1, 1);

        m_ZYResetViewButton = new QPushButton(m_ZYSlicePage);
        m_ZYResetViewButton->setObjectName(QString::fromUtf8("m_ZYResetViewButton"));
        m_ZYResetViewButton->setFont(font);

        gridLayout5->addWidget(m_ZYResetViewButton, 1, 1, 1, 1);

        m_SliceCanvasStack->addWidget(m_ZYSlicePage, 2);

        gridLayout1->addWidget(m_SliceCanvasStack, 1, 0, 1, 1);


        gridLayout->addWidget(m_SliceCanvasGroup, 0, 0, 3, 1);

        m_GridCellTabs = new QTabWidget(VolumeGridRoverBase);
        m_GridCellTabs->setObjectName(QString::fromUtf8("m_GridCellTabs"));
        QSizePolicy sizePolicy1(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(0));
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(m_GridCellTabs->sizePolicy().hasHeightForWidth());
        m_GridCellTabs->setSizePolicy(sizePolicy1);
        VoxelCoordinates = new QWidget();
        VoxelCoordinates->setObjectName(QString::fromUtf8("VoxelCoordinates"));
        gridLayout6 = new QGridLayout(VoxelCoordinates);
        gridLayout6->setSpacing(6);
        gridLayout6->setContentsMargins(11, 11, 11, 11);
        gridLayout6->setObjectName(QString::fromUtf8("gridLayout6"));
        m_IndicesGroup = new Q3GroupBox(VoxelCoordinates);
        m_IndicesGroup->setObjectName(QString::fromUtf8("m_IndicesGroup"));
        m_IndicesGroup->setColumnLayout(0, Qt::Vertical);
        m_IndicesGroup->layout()->setSpacing(6);
        m_IndicesGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout7 = new QGridLayout();
        QBoxLayout *boxlayout2 = qobject_cast<QBoxLayout *>(m_IndicesGroup->layout());
        if (boxlayout2)
            boxlayout2->addLayout(gridLayout7);
        gridLayout7->setAlignment(Qt::AlignTop);
        gridLayout7->setObjectName(QString::fromUtf8("gridLayout7"));
        m_XText = new QLabel(m_IndicesGroup);
        m_XText->setObjectName(QString::fromUtf8("m_XText"));
        m_XText->setFont(font);
        m_XText->setWordWrap(false);

        gridLayout7->addWidget(m_XText, 0, 0, 1, 1);

        m_X = new QLineEdit(m_IndicesGroup);
        m_X->setObjectName(QString::fromUtf8("m_X"));
        QSizePolicy sizePolicy2(static_cast<QSizePolicy::Policy>(0), static_cast<QSizePolicy::Policy>(0));
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(m_X->sizePolicy().hasHeightForWidth());
        m_X->setSizePolicy(sizePolicy2);
        m_X->setMaximumSize(QSize(100, 32767));
        m_X->setFont(font);
        m_X->setReadOnly(true);

        gridLayout7->addWidget(m_X, 0, 1, 1, 1);

        m_YText = new QLabel(m_IndicesGroup);
        m_YText->setObjectName(QString::fromUtf8("m_YText"));
        m_YText->setFont(font);
        m_YText->setWordWrap(false);

        gridLayout7->addWidget(m_YText, 0, 2, 1, 1);

        m_Y = new QLineEdit(m_IndicesGroup);
        m_Y->setObjectName(QString::fromUtf8("m_Y"));
        sizePolicy2.setHeightForWidth(m_Y->sizePolicy().hasHeightForWidth());
        m_Y->setSizePolicy(sizePolicy2);
        m_Y->setMaximumSize(QSize(100, 32767));
        m_Y->setFont(font);
        m_Y->setReadOnly(true);

        gridLayout7->addWidget(m_Y, 0, 3, 1, 1);

        m_ZText = new QLabel(m_IndicesGroup);
        m_ZText->setObjectName(QString::fromUtf8("m_ZText"));
        m_ZText->setFont(font);
        m_ZText->setWordWrap(false);

        gridLayout7->addWidget(m_ZText, 0, 4, 1, 1);

        m_Z = new QLineEdit(m_IndicesGroup);
        m_Z->setObjectName(QString::fromUtf8("m_Z"));
        sizePolicy2.setHeightForWidth(m_Z->sizePolicy().hasHeightForWidth());
        m_Z->setSizePolicy(sizePolicy2);
        m_Z->setMaximumSize(QSize(100, 32767));
        m_Z->setFont(font);
        m_Z->setReadOnly(true);

        gridLayout7->addWidget(m_Z, 0, 5, 1, 1);


        gridLayout6->addWidget(m_IndicesGroup, 0, 0, 1, 1);

        m_ObjectCoordinatesGroup = new Q3GroupBox(VoxelCoordinates);
        m_ObjectCoordinatesGroup->setObjectName(QString::fromUtf8("m_ObjectCoordinatesGroup"));
        m_ObjectCoordinatesGroup->setColumnLayout(0, Qt::Vertical);
        m_ObjectCoordinatesGroup->layout()->setSpacing(6);
        m_ObjectCoordinatesGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout8 = new QGridLayout();
        QBoxLayout *boxlayout3 = qobject_cast<QBoxLayout *>(m_ObjectCoordinatesGroup->layout());
        if (boxlayout3)
            boxlayout3->addLayout(gridLayout8);
        gridLayout8->setAlignment(Qt::AlignTop);
        gridLayout8->setObjectName(QString::fromUtf8("gridLayout8"));
        m_ObjX = new QLineEdit(m_ObjectCoordinatesGroup);
        m_ObjX->setObjectName(QString::fromUtf8("m_ObjX"));
        sizePolicy2.setHeightForWidth(m_ObjX->sizePolicy().hasHeightForWidth());
        m_ObjX->setSizePolicy(sizePolicy2);
        m_ObjX->setMaximumSize(QSize(100, 32767));

        gridLayout8->addWidget(m_ObjX, 0, 1, 1, 1);

        m_ObjXText = new QLabel(m_ObjectCoordinatesGroup);
        m_ObjXText->setObjectName(QString::fromUtf8("m_ObjXText"));
        m_ObjXText->setWordWrap(false);

        gridLayout8->addWidget(m_ObjXText, 0, 0, 1, 1);

        m_ObjYText = new QLabel(m_ObjectCoordinatesGroup);
        m_ObjYText->setObjectName(QString::fromUtf8("m_ObjYText"));
        m_ObjYText->setWordWrap(false);

        gridLayout8->addWidget(m_ObjYText, 0, 2, 1, 1);

        m_ObjY = new QLineEdit(m_ObjectCoordinatesGroup);
        m_ObjY->setObjectName(QString::fromUtf8("m_ObjY"));
        sizePolicy2.setHeightForWidth(m_ObjY->sizePolicy().hasHeightForWidth());
        m_ObjY->setSizePolicy(sizePolicy2);
        m_ObjY->setMaximumSize(QSize(100, 32767));

        gridLayout8->addWidget(m_ObjY, 0, 3, 1, 1);

        m_ObjZText = new QLabel(m_ObjectCoordinatesGroup);
        m_ObjZText->setObjectName(QString::fromUtf8("m_ObjZText"));
        m_ObjZText->setWordWrap(false);

        gridLayout8->addWidget(m_ObjZText, 0, 4, 1, 1);

        m_ObjZ = new QLineEdit(m_ObjectCoordinatesGroup);
        m_ObjZ->setObjectName(QString::fromUtf8("m_ObjZ"));
        sizePolicy2.setHeightForWidth(m_ObjZ->sizePolicy().hasHeightForWidth());
        m_ObjZ->setSizePolicy(sizePolicy2);
        m_ObjZ->setMaximumSize(QSize(100, 32767));

        gridLayout8->addWidget(m_ObjZ, 0, 5, 1, 1);


        gridLayout6->addWidget(m_ObjectCoordinatesGroup, 1, 0, 1, 1);

        m_GridCellTabs->addTab(VoxelCoordinates, QString());
        VoxelInfo = new QWidget();
        VoxelInfo->setObjectName(QString::fromUtf8("VoxelInfo"));
        gridLayout9 = new QGridLayout(VoxelInfo);
        gridLayout9->setSpacing(6);
        gridLayout9->setContentsMargins(11, 11, 11, 11);
        gridLayout9->setObjectName(QString::fromUtf8("gridLayout9"));
        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        m_RText = new QLabel(VoxelInfo);
        m_RText->setObjectName(QString::fromUtf8("m_RText"));
        m_RText->setFont(font);
        m_RText->setWordWrap(false);

        hboxLayout->addWidget(m_RText);

        m_R = new QLineEdit(VoxelInfo);
        m_R->setObjectName(QString::fromUtf8("m_R"));
        QSizePolicy sizePolicy3(static_cast<QSizePolicy::Policy>(4), static_cast<QSizePolicy::Policy>(0));
        sizePolicy3.setHorizontalStretch(255);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(m_R->sizePolicy().hasHeightForWidth());
        m_R->setSizePolicy(sizePolicy3);
        m_R->setMaximumSize(QSize(32767, 32767));
        m_R->setFont(font);
        m_R->setReadOnly(true);

        hboxLayout->addWidget(m_R);

        m_GText = new QLabel(VoxelInfo);
        m_GText->setObjectName(QString::fromUtf8("m_GText"));
        m_GText->setFont(font);
        m_GText->setWordWrap(false);

        hboxLayout->addWidget(m_GText);

        m_G = new QLineEdit(VoxelInfo);
        m_G->setObjectName(QString::fromUtf8("m_G"));
        sizePolicy3.setHeightForWidth(m_G->sizePolicy().hasHeightForWidth());
        m_G->setSizePolicy(sizePolicy3);
        m_G->setMaximumSize(QSize(32767, 32767));
        m_G->setFont(font);
        m_G->setReadOnly(true);

        hboxLayout->addWidget(m_G);

        m_BText = new QLabel(VoxelInfo);
        m_BText->setObjectName(QString::fromUtf8("m_BText"));
        m_BText->setFont(font);
        m_BText->setWordWrap(false);

        hboxLayout->addWidget(m_BText);

        m_B = new QLineEdit(VoxelInfo);
        m_B->setObjectName(QString::fromUtf8("m_B"));
        sizePolicy3.setHeightForWidth(m_B->sizePolicy().hasHeightForWidth());
        m_B->setSizePolicy(sizePolicy3);
        m_B->setMaximumSize(QSize(32767, 32767));
        m_B->setFont(font);
        m_B->setReadOnly(true);

        hboxLayout->addWidget(m_B);

        m_AText = new QLabel(VoxelInfo);
        m_AText->setObjectName(QString::fromUtf8("m_AText"));
        m_AText->setFont(font);
        m_AText->setWordWrap(false);

        hboxLayout->addWidget(m_AText);

        m_A = new QLineEdit(VoxelInfo);
        m_A->setObjectName(QString::fromUtf8("m_A"));
        sizePolicy3.setHeightForWidth(m_A->sizePolicy().hasHeightForWidth());
        m_A->setSizePolicy(sizePolicy3);
        m_A->setMaximumSize(QSize(32767, 32767));
        m_A->setFont(font);
        m_A->setReadOnly(true);

        hboxLayout->addWidget(m_A);


        gridLayout9->addLayout(hboxLayout, 0, 0, 1, 3);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        m_ValueText = new QLabel(VoxelInfo);
        m_ValueText->setObjectName(QString::fromUtf8("m_ValueText"));
        m_ValueText->setFont(font);
        m_ValueText->setWordWrap(false);

        hboxLayout1->addWidget(m_ValueText);

        m_Value = new QLineEdit(VoxelInfo);
        m_Value->setObjectName(QString::fromUtf8("m_Value"));
        m_Value->setFont(font);
        m_Value->setReadOnly(true);

        hboxLayout1->addWidget(m_Value);

        m_ColorNameText = new QLabel(VoxelInfo);
        m_ColorNameText->setObjectName(QString::fromUtf8("m_ColorNameText"));
        m_ColorNameText->setFont(font);
        m_ColorNameText->setWordWrap(false);

        hboxLayout1->addWidget(m_ColorNameText);

        m_ColorName = new QLineEdit(VoxelInfo);
        m_ColorName->setObjectName(QString::fromUtf8("m_ColorName"));
        m_ColorName->setReadOnly(true);

        hboxLayout1->addWidget(m_ColorName);


        gridLayout9->addLayout(hboxLayout1, 1, 0, 1, 3);

        m_MappedValueText = new QLabel(VoxelInfo);
        m_MappedValueText->setObjectName(QString::fromUtf8("m_MappedValueText"));
        QSizePolicy sizePolicy4(static_cast<QSizePolicy::Policy>(0), static_cast<QSizePolicy::Policy>(5));
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(m_MappedValueText->sizePolicy().hasHeightForWidth());
        m_MappedValueText->setSizePolicy(sizePolicy4);
        m_MappedValueText->setFont(font);
        m_MappedValueText->setWordWrap(false);

        gridLayout9->addWidget(m_MappedValueText, 2, 0, 1, 1);

        m_MappedValue = new QLineEdit(VoxelInfo);
        m_MappedValue->setObjectName(QString::fromUtf8("m_MappedValue"));
        sizePolicy2.setHeightForWidth(m_MappedValue->sizePolicy().hasHeightForWidth());
        m_MappedValue->setSizePolicy(sizePolicy2);
        m_MappedValue->setReadOnly(true);

        gridLayout9->addWidget(m_MappedValue, 2, 1, 1, 1);

        spacer1 = new QSpacerItem(131, 21, QSizePolicy::Fixed, QSizePolicy::Minimum);

        gridLayout9->addItem(spacer1, 2, 2, 1, 1);

        m_GridCellTabs->addTab(VoxelInfo, QString());

        gridLayout->addWidget(m_GridCellTabs, 1, 1, 1, 1);

        m_VariableInformationTabs = new QTabWidget(VolumeGridRoverBase);
        m_VariableInformationTabs->setObjectName(QString::fromUtf8("m_VariableInformationTabs"));
        sizePolicy2.setHeightForWidth(m_VariableInformationTabs->sizePolicy().hasHeightForWidth());
        m_VariableInformationTabs->setSizePolicy(sizePolicy2);
        VariableSelection = new QWidget();
        VariableSelection->setObjectName(QString::fromUtf8("VariableSelection"));
        hboxLayout2 = new QHBoxLayout(VariableSelection);
        hboxLayout2->setSpacing(6);
        hboxLayout2->setContentsMargins(11, 11, 11, 11);
        hboxLayout2->setObjectName(QString::fromUtf8("hboxLayout2"));
        m_VariableText = new QLabel(VariableSelection);
        m_VariableText->setObjectName(QString::fromUtf8("m_VariableText"));
        m_VariableText->setFont(font);
        m_VariableText->setWordWrap(false);

        hboxLayout2->addWidget(m_VariableText);

        m_Variable = new QComboBox(VariableSelection);
        m_Variable->setObjectName(QString::fromUtf8("m_Variable"));

        hboxLayout2->addWidget(m_Variable);

        m_TimestepText = new QLabel(VariableSelection);
        m_TimestepText->setObjectName(QString::fromUtf8("m_TimestepText"));
        m_TimestepText->setFont(font);
        m_TimestepText->setWordWrap(false);

        hboxLayout2->addWidget(m_TimestepText);

        m_Timestep = new QSpinBox(VariableSelection);
        m_Timestep->setObjectName(QString::fromUtf8("m_Timestep"));
        m_Timestep->setWrapping(true);
        m_Timestep->setMaximum(0);

        hboxLayout2->addWidget(m_Timestep);

        m_VariableInformationTabs->addTab(VariableSelection, QString());
        VariableInformation = new QWidget();
        VariableInformation->setObjectName(QString::fromUtf8("VariableInformation"));
        gridLayout10 = new QGridLayout(VariableInformation);
        gridLayout10->setSpacing(6);
        gridLayout10->setContentsMargins(11, 11, 11, 11);
        gridLayout10->setObjectName(QString::fromUtf8("gridLayout10"));
        m_MinimumValueText = new QLabel(VariableInformation);
        m_MinimumValueText->setObjectName(QString::fromUtf8("m_MinimumValueText"));
        m_MinimumValueText->setFont(font);
        m_MinimumValueText->setWordWrap(false);

        gridLayout10->addWidget(m_MinimumValueText, 0, 0, 1, 1);

        m_MinimumValue = new QLineEdit(VariableInformation);
        m_MinimumValue->setObjectName(QString::fromUtf8("m_MinimumValue"));
        m_MinimumValue->setFont(font);
        m_MinimumValue->setReadOnly(true);

        gridLayout10->addWidget(m_MinimumValue, 0, 1, 1, 1);

        m_MaximumValueText = new QLabel(VariableInformation);
        m_MaximumValueText->setObjectName(QString::fromUtf8("m_MaximumValueText"));
        m_MaximumValueText->setFont(font);
        m_MaximumValueText->setWordWrap(false);

        gridLayout10->addWidget(m_MaximumValueText, 0, 2, 1, 1);

        m_MaximumValue = new QLineEdit(VariableInformation);
        m_MaximumValue->setObjectName(QString::fromUtf8("m_MaximumValue"));
        m_MaximumValue->setFont(font);
        m_MaximumValue->setReadOnly(true);

        gridLayout10->addWidget(m_MaximumValue, 0, 3, 1, 1);

        m_VariableInformationTabs->addTab(VariableInformation, QString());

        gridLayout->addWidget(m_VariableInformationTabs, 0, 1, 1, 1);

        m_OptionsAndSegmentation = new QTabWidget(VolumeGridRoverBase);
        m_OptionsAndSegmentation->setObjectName(QString::fromUtf8("m_OptionsAndSegmentation"));
        QSizePolicy sizePolicy5(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(4));
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(m_OptionsAndSegmentation->sizePolicy().hasHeightForWidth());
        m_OptionsAndSegmentation->setSizePolicy(sizePolicy5);
        m_OptionsAndSegmentation->setFont(font);
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        gridLayout11 = new QGridLayout(tab);
        gridLayout11->setSpacing(6);
        gridLayout11->setContentsMargins(11, 11, 11, 11);
        gridLayout11->setObjectName(QString::fromUtf8("gridLayout11"));
        m_DisplayOptionsGroup = new Q3GroupBox(tab);
        m_DisplayOptionsGroup->setObjectName(QString::fromUtf8("m_DisplayOptionsGroup"));
        m_DisplayOptionsGroup->setFont(font);
        m_DisplayOptionsGroup->setColumnLayout(0, Qt::Vertical);
        m_DisplayOptionsGroup->layout()->setSpacing(6);
        m_DisplayOptionsGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout12 = new QGridLayout();
        QBoxLayout *boxlayout4 = qobject_cast<QBoxLayout *>(m_DisplayOptionsGroup->layout());
        if (boxlayout4)
            boxlayout4->addLayout(gridLayout12);
        gridLayout12->setAlignment(Qt::AlignTop);
        gridLayout12->setObjectName(QString::fromUtf8("gridLayout12"));
        hboxLayout3 = new QHBoxLayout();
        hboxLayout3->setSpacing(6);
        hboxLayout3->setObjectName(QString::fromUtf8("hboxLayout3"));
        m_BackgroundColorText = new QLabel(m_DisplayOptionsGroup);
        m_BackgroundColorText->setObjectName(QString::fromUtf8("m_BackgroundColorText"));
        m_BackgroundColorText->setFont(font);
        m_BackgroundColorText->setWordWrap(false);

        hboxLayout3->addWidget(m_BackgroundColorText);

        m_BackgroundColor = new QPushButton(m_DisplayOptionsGroup);
        m_BackgroundColor->setObjectName(QString::fromUtf8("m_BackgroundColor"));

        hboxLayout3->addWidget(m_BackgroundColor);


        gridLayout12->addLayout(hboxLayout3, 2, 0, 1, 2);

        m_GreyScale = new QCheckBox(m_DisplayOptionsGroup);
        m_GreyScale->setObjectName(QString::fromUtf8("m_GreyScale"));

        gridLayout12->addWidget(m_GreyScale, 1, 0, 1, 2);

        m_PointSize = new QSlider(m_DisplayOptionsGroup);
        m_PointSize->setObjectName(QString::fromUtf8("m_PointSize"));
        m_PointSize->setMinimum(1);
        m_PointSize->setMaximum(50);
        m_PointSize->setOrientation(Qt::Horizontal);

        gridLayout12->addWidget(m_PointSize, 0, 1, 1, 2);

        m_PointSizeText = new QLabel(m_DisplayOptionsGroup);
        m_PointSizeText->setObjectName(QString::fromUtf8("m_PointSizeText"));
        m_PointSizeText->setFont(font);
        m_PointSizeText->setWordWrap(false);

        gridLayout12->addWidget(m_PointSizeText, 0, 0, 1, 1);

        m_RenderControlPoints = new QCheckBox(m_DisplayOptionsGroup);
        m_RenderControlPoints->setObjectName(QString::fromUtf8("m_RenderControlPoints"));
        m_RenderControlPoints->setChecked(true);

        gridLayout12->addWidget(m_RenderControlPoints, 0, 3, 1, 1);

        m_RenderSDF = new QCheckBox(m_DisplayOptionsGroup);
        m_RenderSDF->setObjectName(QString::fromUtf8("m_RenderSDF"));
        m_RenderSDF->setEnabled(false);

        gridLayout12->addWidget(m_RenderSDF, 1, 2, 1, 2);

        m_Isocontouring = new QCheckBox(m_DisplayOptionsGroup);
        m_Isocontouring->setObjectName(QString::fromUtf8("m_Isocontouring"));

        gridLayout12->addWidget(m_Isocontouring, 2, 2, 1, 2);


        gridLayout11->addWidget(m_DisplayOptionsGroup, 1, 0, 1, 1);

        m_GridCellMarkingToolSelection = new QTabWidget(tab);
        m_GridCellMarkingToolSelection->setObjectName(QString::fromUtf8("m_GridCellMarkingToolSelection"));
        PointClasses = new QWidget();
        PointClasses->setObjectName(QString::fromUtf8("PointClasses"));
        gridLayout13 = new QGridLayout(PointClasses);
        gridLayout13->setSpacing(6);
        gridLayout13->setContentsMargins(11, 11, 11, 11);
        gridLayout13->setObjectName(QString::fromUtf8("gridLayout13"));
        m_PointClassColor = new QPushButton(PointClasses);
        m_PointClassColor->setObjectName(QString::fromUtf8("m_PointClassColor"));
        sizePolicy1.setHeightForWidth(m_PointClassColor->sizePolicy().hasHeightForWidth());
        m_PointClassColor->setSizePolicy(sizePolicy1);
        m_PointClassColor->setMaximumSize(QSize(32767, 32767));
        m_PointClassColor->setFont(font);

        gridLayout13->addWidget(m_PointClassColor, 0, 3, 1, 1);

        m_PointClass = new QComboBox(PointClasses);
        m_PointClass->setObjectName(QString::fromUtf8("m_PointClass"));
        QSizePolicy sizePolicy6(static_cast<QSizePolicy::Policy>(5), static_cast<QSizePolicy::Policy>(5));
        sizePolicy6.setHorizontalStretch(0);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(m_PointClass->sizePolicy().hasHeightForWidth());
        m_PointClass->setSizePolicy(sizePolicy6);
        m_PointClass->setFont(font);
        m_PointClass->setEditable(false);

        gridLayout13->addWidget(m_PointClass, 0, 1, 1, 2);

        m_PointClassText = new QLabel(PointClasses);
        m_PointClassText->setObjectName(QString::fromUtf8("m_PointClassText"));
        m_PointClassText->setFont(font);
        m_PointClassText->setWordWrap(false);

        gridLayout13->addWidget(m_PointClassText, 0, 0, 1, 1);

        m_AddPointClass = new QPushButton(PointClasses);
        m_AddPointClass->setObjectName(QString::fromUtf8("m_AddPointClass"));
        m_AddPointClass->setFont(font);

        gridLayout13->addWidget(m_AddPointClass, 1, 0, 1, 2);

        m_DeletePointClass = new QPushButton(PointClasses);
        m_DeletePointClass->setObjectName(QString::fromUtf8("m_DeletePointClass"));
        m_DeletePointClass->setFont(font);

        gridLayout13->addWidget(m_DeletePointClass, 1, 2, 1, 2);

        m_PointClassesLoadButton = new QPushButton(PointClasses);
        m_PointClassesLoadButton->setObjectName(QString::fromUtf8("m_PointClassesLoadButton"));
        m_PointClassesLoadButton->setFont(font);

        gridLayout13->addWidget(m_PointClassesLoadButton, 2, 2, 1, 2);

        m_PointClassesSaveButton = new QPushButton(PointClasses);
        m_PointClassesSaveButton->setObjectName(QString::fromUtf8("m_PointClassesSaveButton"));
        m_PointClassesSaveButton->setFont(font);

        gridLayout13->addWidget(m_PointClassesSaveButton, 2, 0, 1, 2);

        m_GridCellMarkingToolSelection->addTab(PointClasses, QString());
        Contours = new QWidget();
        Contours->setObjectName(QString::fromUtf8("Contours"));
        gridLayout14 = new QGridLayout(Contours);
        gridLayout14->setSpacing(6);
        gridLayout14->setContentsMargins(11, 11, 11, 11);
        gridLayout14->setObjectName(QString::fromUtf8("gridLayout14"));
        m_Objects = new Q3ListView(Contours);
        m_Objects->addColumn(QApplication::translate("VolumeGridRoverBase", "Object Name", 0, QApplication::UnicodeUTF8));
        m_Objects->header()->setClickEnabled(true, m_Objects->header()->count() - 1);
        m_Objects->header()->setResizeEnabled(true, m_Objects->header()->count() - 1);
        m_Objects->setObjectName(QString::fromUtf8("m_Objects"));
        m_Objects->setSelectionMode(Q3ListView::Extended);

        gridLayout14->addWidget(m_Objects, 0, 0, 5, 1);

        m_InterpolationTypeText = new QLabel(Contours);
        m_InterpolationTypeText->setObjectName(QString::fromUtf8("m_InterpolationTypeText"));
        m_InterpolationTypeText->setWordWrap(false);

        gridLayout14->addWidget(m_InterpolationTypeText, 2, 1, 1, 2);

        m_InterpolationSamplingText = new QLabel(Contours);
        m_InterpolationSamplingText->setObjectName(QString::fromUtf8("m_InterpolationSamplingText"));
        m_InterpolationSamplingText->setWordWrap(false);

        gridLayout14->addWidget(m_InterpolationSamplingText, 3, 1, 1, 2);

        m_DeleteContour = new QPushButton(Contours);
        m_DeleteContour->setObjectName(QString::fromUtf8("m_DeleteContour"));
        QSizePolicy sizePolicy7(static_cast<QSizePolicy::Policy>(4), static_cast<QSizePolicy::Policy>(0));
        sizePolicy7.setHorizontalStretch(0);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(m_DeleteContour->sizePolicy().hasHeightForWidth());
        m_DeleteContour->setSizePolicy(sizePolicy7);

        gridLayout14->addWidget(m_DeleteContour, 1, 1, 1, 3);

        m_SaveContoursButton = new QPushButton(Contours);
        m_SaveContoursButton->setObjectName(QString::fromUtf8("m_SaveContoursButton"));
        sizePolicy7.setHeightForWidth(m_SaveContoursButton->sizePolicy().hasHeightForWidth());
        m_SaveContoursButton->setSizePolicy(sizePolicy7);

        gridLayout14->addWidget(m_SaveContoursButton, 4, 1, 1, 1);

        m_InterpolationSampling = new QSpinBox(Contours);
        m_InterpolationSampling->setObjectName(QString::fromUtf8("m_InterpolationSampling"));
        sizePolicy7.setHeightForWidth(m_InterpolationSampling->sizePolicy().hasHeightForWidth());
        m_InterpolationSampling->setSizePolicy(sizePolicy7);
        m_InterpolationSampling->setMaximum(999);
        m_InterpolationSampling->setMinimum(0);
        m_InterpolationSampling->setValue(5);

        gridLayout14->addWidget(m_InterpolationSampling, 3, 3, 1, 2);

        m_AddContour = new QPushButton(Contours);
        m_AddContour->setObjectName(QString::fromUtf8("m_AddContour"));
        sizePolicy7.setHeightForWidth(m_AddContour->sizePolicy().hasHeightForWidth());
        m_AddContour->setSizePolicy(sizePolicy7);

        gridLayout14->addWidget(m_AddContour, 0, 1, 1, 3);

        m_ContourColor = new QPushButton(Contours);
        m_ContourColor->setObjectName(QString::fromUtf8("m_ContourColor"));
        sizePolicy2.setHeightForWidth(m_ContourColor->sizePolicy().hasHeightForWidth());
        m_ContourColor->setSizePolicy(sizePolicy2);

        gridLayout14->addWidget(m_ContourColor, 0, 4, 1, 1);

        m_LoadContourButton = new QPushButton(Contours);
        m_LoadContourButton->setObjectName(QString::fromUtf8("m_LoadContourButton"));
        sizePolicy7.setHeightForWidth(m_LoadContourButton->sizePolicy().hasHeightForWidth());
        m_LoadContourButton->setSizePolicy(sizePolicy7);

        gridLayout14->addWidget(m_LoadContourButton, 4, 2, 1, 2);

        m_InterpolationType = new QComboBox(Contours);
        m_InterpolationType->setObjectName(QString::fromUtf8("m_InterpolationType"));
        sizePolicy2.setHeightForWidth(m_InterpolationType->sizePolicy().hasHeightForWidth());
        m_InterpolationType->setSizePolicy(sizePolicy2);
        m_InterpolationType->setMaximumSize(QSize(100, 32767));

        gridLayout14->addWidget(m_InterpolationType, 2, 3, 1, 2);

        m_GridCellMarkingToolSelection->addTab(Contours, QString());

        gridLayout11->addWidget(m_GridCellMarkingToolSelection, 0, 0, 1, 1);

        m_OptionsAndSegmentation->addTab(tab, QString());
        tab1 = new QWidget();
        tab1->setObjectName(QString::fromUtf8("tab1"));
        vboxLayout = new QVBoxLayout(tab1);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        m_SegmentationThresholdGroup = new Q3GroupBox(tab1);
        m_SegmentationThresholdGroup->setObjectName(QString::fromUtf8("m_SegmentationThresholdGroup"));
        m_SegmentationThresholdGroup->setColumnLayout(0, Qt::Vertical);
        m_SegmentationThresholdGroup->layout()->setSpacing(6);
        m_SegmentationThresholdGroup->layout()->setContentsMargins(11, 11, 11, 11);
        hboxLayout4 = new QHBoxLayout();
        QBoxLayout *boxlayout5 = qobject_cast<QBoxLayout *>(m_SegmentationThresholdGroup->layout());
        if (boxlayout5)
            boxlayout5->addLayout(hboxLayout4);
        hboxLayout4->setAlignment(Qt::AlignTop);
        hboxLayout4->setObjectName(QString::fromUtf8("hboxLayout4"));
        m_TresholdLowText = new QLabel(m_SegmentationThresholdGroup);
        m_TresholdLowText->setObjectName(QString::fromUtf8("m_TresholdLowText"));
        m_TresholdLowText->setWordWrap(false);

        hboxLayout4->addWidget(m_TresholdLowText);

        m_ThresholdLow = new QLineEdit(m_SegmentationThresholdGroup);
        m_ThresholdLow->setObjectName(QString::fromUtf8("m_ThresholdLow"));

        hboxLayout4->addWidget(m_ThresholdLow);

        m_ThresholdHighText = new QLabel(m_SegmentationThresholdGroup);
        m_ThresholdHighText->setObjectName(QString::fromUtf8("m_ThresholdHighText"));
        m_ThresholdHighText->setWordWrap(false);

        hboxLayout4->addWidget(m_ThresholdHighText);

        m_ThresholdHigh = new QLineEdit(m_SegmentationThresholdGroup);
        m_ThresholdHigh->setObjectName(QString::fromUtf8("m_ThresholdHigh"));

        hboxLayout4->addWidget(m_ThresholdHigh);


        vboxLayout->addWidget(m_SegmentationThresholdGroup);

        m_LocalSegmentationGroup = new Q3GroupBox(tab1);
        m_LocalSegmentationGroup->setObjectName(QString::fromUtf8("m_LocalSegmentationGroup"));
        m_LocalSegmentationGroup->setColumnLayout(0, Qt::Vertical);
        m_LocalSegmentationGroup->layout()->setSpacing(6);
        m_LocalSegmentationGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout15 = new QGridLayout();
        QBoxLayout *boxlayout6 = qobject_cast<QBoxLayout *>(m_LocalSegmentationGroup->layout());
        if (boxlayout6)
            boxlayout6->addLayout(gridLayout15);
        gridLayout15->setAlignment(Qt::AlignTop);
        gridLayout15->setObjectName(QString::fromUtf8("gridLayout15"));
        m_OutputFileText = new QLabel(m_LocalSegmentationGroup);
        m_OutputFileText->setObjectName(QString::fromUtf8("m_OutputFileText"));
        m_OutputFileText->setWordWrap(false);

        gridLayout15->addWidget(m_OutputFileText, 0, 0, 1, 1);

        m_LocalOutputFile = new QLineEdit(m_LocalSegmentationGroup);
        m_LocalOutputFile->setObjectName(QString::fromUtf8("m_LocalOutputFile"));

        gridLayout15->addWidget(m_LocalOutputFile, 0, 1, 1, 1);

        m_LocalOutputFileSelection = new QPushButton(m_LocalSegmentationGroup);
        m_LocalOutputFileSelection->setObjectName(QString::fromUtf8("m_LocalOutputFileSelection"));

        gridLayout15->addWidget(m_LocalOutputFileSelection, 0, 2, 1, 1);

        m_LocalRun = new QPushButton(m_LocalSegmentationGroup);
        m_LocalRun->setObjectName(QString::fromUtf8("m_LocalRun"));

        gridLayout15->addWidget(m_LocalRun, 0, 3, 1, 1);


        vboxLayout->addWidget(m_LocalSegmentationGroup);

        m_RemoteSegmentationGroup = new Q3GroupBox(tab1);
        m_RemoteSegmentationGroup->setObjectName(QString::fromUtf8("m_RemoteSegmentationGroup"));
        m_RemoteSegmentationGroup->setColumnLayout(0, Qt::Vertical);
        m_RemoteSegmentationGroup->layout()->setSpacing(6);
        m_RemoteSegmentationGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout16 = new QGridLayout();
        QBoxLayout *boxlayout7 = qobject_cast<QBoxLayout *>(m_RemoteSegmentationGroup->layout());
        if (boxlayout7)
            boxlayout7->addLayout(gridLayout16);
        gridLayout16->setAlignment(Qt::AlignTop);
        gridLayout16->setObjectName(QString::fromUtf8("gridLayout16"));
        m_RemoteOutputFileText = new QLabel(m_RemoteSegmentationGroup);
        m_RemoteOutputFileText->setObjectName(QString::fromUtf8("m_RemoteOutputFileText"));
        m_RemoteOutputFileText->setWordWrap(false);

        gridLayout16->addWidget(m_RemoteOutputFileText, 0, 0, 1, 1);

        m_RemoteFile = new QLineEdit(m_RemoteSegmentationGroup);
        m_RemoteFile->setObjectName(QString::fromUtf8("m_RemoteFile"));

        gridLayout16->addWidget(m_RemoteFile, 0, 1, 1, 1);

        m_RemoteOutputFileSelection = new QPushButton(m_RemoteSegmentationGroup);
        m_RemoteOutputFileSelection->setObjectName(QString::fromUtf8("m_RemoteOutputFileSelection"));

        gridLayout16->addWidget(m_RemoteOutputFileSelection, 0, 2, 1, 1);

        gridLayout17 = new QGridLayout();
        gridLayout17->setSpacing(6);
        gridLayout17->setObjectName(QString::fromUtf8("gridLayout17"));
        m_RemoteRun = new QPushButton(m_RemoteSegmentationGroup);
        m_RemoteRun->setObjectName(QString::fromUtf8("m_RemoteRun"));

        gridLayout17->addWidget(m_RemoteRun, 0, 2, 2, 1);

        m_PortText = new QLabel(m_RemoteSegmentationGroup);
        m_PortText->setObjectName(QString::fromUtf8("m_PortText"));
        sizePolicy4.setHeightForWidth(m_PortText->sizePolicy().hasHeightForWidth());
        m_PortText->setSizePolicy(sizePolicy4);
        m_PortText->setWordWrap(false);

        gridLayout17->addWidget(m_PortText, 1, 0, 1, 1);

        m_Hostname = new QLineEdit(m_RemoteSegmentationGroup);
        m_Hostname->setObjectName(QString::fromUtf8("m_Hostname"));
        sizePolicy2.setHeightForWidth(m_Hostname->sizePolicy().hasHeightForWidth());
        m_Hostname->setSizePolicy(sizePolicy2);

        gridLayout17->addWidget(m_Hostname, 0, 1, 1, 1);

        m_Port = new QLineEdit(m_RemoteSegmentationGroup);
        m_Port->setObjectName(QString::fromUtf8("m_Port"));
        sizePolicy2.setHeightForWidth(m_Port->sizePolicy().hasHeightForWidth());
        m_Port->setSizePolicy(sizePolicy2);

        gridLayout17->addWidget(m_Port, 1, 1, 1, 1);

        m_HostnameText = new QLabel(m_RemoteSegmentationGroup);
        m_HostnameText->setObjectName(QString::fromUtf8("m_HostnameText"));
        sizePolicy4.setHeightForWidth(m_HostnameText->sizePolicy().hasHeightForWidth());
        m_HostnameText->setSizePolicy(sizePolicy4);
        m_HostnameText->setWordWrap(false);

        gridLayout17->addWidget(m_HostnameText, 0, 0, 1, 1);


        gridLayout16->addLayout(gridLayout17, 1, 0, 1, 3);


        vboxLayout->addWidget(m_RemoteSegmentationGroup);

        m_OptionsAndSegmentation->addTab(tab1, QString());
        tab2 = new QWidget();
        tab2->setObjectName(QString::fromUtf8("tab2"));
        m_EMClusteringRun = new QPushButton(tab2);
        m_EMClusteringRun->setObjectName(QString::fromUtf8("m_EMClusteringRun"));
        m_EMClusteringRun->setGeometry(QRect(15, 25, 81, 31));
        m_OptionsAndSegmentation->addTab(tab2, QString());
        Tiling = new QWidget();
        Tiling->setObjectName(QString::fromUtf8("Tiling"));
        gridLayout18 = new QGridLayout(Tiling);
        gridLayout18->setSpacing(6);
        gridLayout18->setContentsMargins(11, 11, 11, 11);
        gridLayout18->setObjectName(QString::fromUtf8("gridLayout18"));
        m_TilingOutputFilenameGroup = new Q3GroupBox(Tiling);
        m_TilingOutputFilenameGroup->setObjectName(QString::fromUtf8("m_TilingOutputFilenameGroup"));
        m_TilingOutputFilenameGroup->setEnabled(false);
        m_TilingOutputFilenameGroup->setColumnLayout(0, Qt::Vertical);
        m_TilingOutputFilenameGroup->layout()->setSpacing(6);
        m_TilingOutputFilenameGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout19 = new QGridLayout();
        QBoxLayout *boxlayout8 = qobject_cast<QBoxLayout *>(m_TilingOutputFilenameGroup->layout());
        if (boxlayout8)
            boxlayout8->addLayout(gridLayout19);
        gridLayout19->setAlignment(Qt::AlignTop);
        gridLayout19->setObjectName(QString::fromUtf8("gridLayout19"));
        m_TilingOutputDirectoryText = new QLabel(m_TilingOutputFilenameGroup);
        m_TilingOutputDirectoryText->setObjectName(QString::fromUtf8("m_TilingOutputDirectoryText"));
        m_TilingOutputDirectoryText->setWordWrap(false);

        gridLayout19->addWidget(m_TilingOutputDirectoryText, 0, 0, 1, 1);

        m_TilingOutputDirectory = new QLineEdit(m_TilingOutputFilenameGroup);
        m_TilingOutputDirectory->setObjectName(QString::fromUtf8("m_TilingOutputDirectory"));

        gridLayout19->addWidget(m_TilingOutputDirectory, 0, 1, 1, 1);

        m_TilingOutputDirectorySelect = new QPushButton(m_TilingOutputFilenameGroup);
        m_TilingOutputDirectorySelect->setObjectName(QString::fromUtf8("m_TilingOutputDirectorySelect"));

        gridLayout19->addWidget(m_TilingOutputDirectorySelect, 0, 2, 1, 1);


        gridLayout18->addWidget(m_TilingOutputFilenameGroup, 1, 0, 1, 2);

        spacer11 = new QSpacerItem(191, 81, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout18->addItem(spacer11, 2, 1, 2, 1);

        m_TilingOutputOptions = new Q3ButtonGroup(Tiling);
        m_TilingOutputOptions->setObjectName(QString::fromUtf8("m_TilingOutputOptions"));
        m_TilingOutputOptions->setChecked(false);
        m_TilingOutputOptions->setExclusive(false);
        m_TilingOutputOptions->setColumnLayout(0, Qt::Vertical);
        m_TilingOutputOptions->layout()->setSpacing(6);
        m_TilingOutputOptions->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout20 = new QGridLayout();
        QBoxLayout *boxlayout9 = qobject_cast<QBoxLayout *>(m_TilingOutputOptions->layout());
        if (boxlayout9)
            boxlayout9->addLayout(gridLayout20);
        gridLayout20->setAlignment(Qt::AlignTop);
        gridLayout20->setObjectName(QString::fromUtf8("gridLayout20"));
        m_TilingOutputInMemory = new QRadioButton(m_TilingOutputOptions);
        m_TilingOutputInMemory->setObjectName(QString::fromUtf8("m_TilingOutputInMemory"));
        m_TilingOutputInMemory->setChecked(true);

        gridLayout20->addWidget(m_TilingOutputInMemory, 0, 0, 1, 1);

        m_TilingOptionSaveToFile = new QRadioButton(m_TilingOutputOptions);
        m_TilingOptionSaveToFile->setObjectName(QString::fromUtf8("m_TilingOptionSaveToFile"));

        gridLayout20->addWidget(m_TilingOptionSaveToFile, 1, 0, 1, 1);


        gridLayout18->addWidget(m_TilingOutputOptions, 0, 0, 1, 2);

        spacer12 = new QSpacerItem(431, 161, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout18->addItem(spacer12, 4, 0, 1, 2);

        m_Run2DSDF = new QPushButton(Tiling);
        m_Run2DSDF->setObjectName(QString::fromUtf8("m_Run2DSDF"));
        QSizePolicy sizePolicy8(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(1));
        sizePolicy8.setHorizontalStretch(0);
        sizePolicy8.setVerticalStretch(0);
        sizePolicy8.setHeightForWidth(m_Run2DSDF->sizePolicy().hasHeightForWidth());
        m_Run2DSDF->setSizePolicy(sizePolicy8);

        gridLayout18->addWidget(m_Run2DSDF, 3, 0, 1, 1);

        m_RunTiling = new QPushButton(Tiling);
        m_RunTiling->setObjectName(QString::fromUtf8("m_RunTiling"));
        sizePolicy8.setHeightForWidth(m_RunTiling->sizePolicy().hasHeightForWidth());
        m_RunTiling->setSizePolicy(sizePolicy8);

        gridLayout18->addWidget(m_RunTiling, 2, 0, 1, 1);

        m_OptionsAndSegmentation->addTab(Tiling, QString());
        TabPage = new QWidget();
        TabPage->setObjectName(QString::fromUtf8("TabPage"));
        vboxLayout1 = new QVBoxLayout(TabPage);
        vboxLayout1->setSpacing(6);
        vboxLayout1->setContentsMargins(11, 11, 11, 11);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        m_CurrentSliceProcessGroup = new Q3GroupBox(TabPage);
        m_CurrentSliceProcessGroup->setObjectName(QString::fromUtf8("m_CurrentSliceProcessGroup"));
        m_CurrentSliceProcessGroup->setColumnLayout(0, Qt::Vertical);
        m_CurrentSliceProcessGroup->layout()->setSpacing(6);
        m_CurrentSliceProcessGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout21 = new QGridLayout();
        QBoxLayout *boxlayout10 = qobject_cast<QBoxLayout *>(m_CurrentSliceProcessGroup->layout());
        if (boxlayout10)
            boxlayout10->addLayout(gridLayout21);
        gridLayout21->setAlignment(Qt::AlignTop);
        gridLayout21->setObjectName(QString::fromUtf8("gridLayout21"));
        m_CalcMedialAxis = new QPushButton(m_CurrentSliceProcessGroup);
        m_CalcMedialAxis->setObjectName(QString::fromUtf8("m_CalcMedialAxis"));

        gridLayout21->addWidget(m_CalcMedialAxis, 2, 0, 1, 1);

        m_SDFOptions = new QPushButton(m_CurrentSliceProcessGroup);
        m_SDFOptions->setObjectName(QString::fromUtf8("m_SDFOptions"));

        gridLayout21->addWidget(m_SDFOptions, 0, 0, 1, 1);

        m_CalcDelaunayVoronoi = new QPushButton(m_CurrentSliceProcessGroup);
        m_CalcDelaunayVoronoi->setObjectName(QString::fromUtf8("m_CalcDelaunayVoronoi"));

        gridLayout21->addWidget(m_CalcDelaunayVoronoi, 1, 0, 1, 1);

        m_ContourCuration = new QPushButton(m_CurrentSliceProcessGroup);
        m_ContourCuration->setObjectName(QString::fromUtf8("m_ContourCuration"));

        gridLayout21->addWidget(m_ContourCuration, 3, 0, 1, 1);


        vboxLayout1->addWidget(m_CurrentSliceProcessGroup);

        m_OptionsAndSegmentation->addTab(TabPage, QString());

        gridLayout->addWidget(m_OptionsAndSegmentation, 2, 1, 1, 1);

#ifndef QT_NO_SHORTCUT
        m_RText->setBuddy(m_R);
        m_ValueText->setBuddy(m_Value);
#endif // QT_NO_SHORTCUT
        QWidget::setTabOrder(m_XYSliceCanvasButton, m_XYDepthSlide);
        QWidget::setTabOrder(m_XYDepthSlide, m_XZDepthSlide);
        QWidget::setTabOrder(m_XZDepthSlide, m_ZYDepthSlide);
        QWidget::setTabOrder(m_ZYDepthSlide, m_Variable);
        QWidget::setTabOrder(m_Variable, m_Timestep);
        QWidget::setTabOrder(m_Timestep, m_X);
        QWidget::setTabOrder(m_X, m_Y);
        QWidget::setTabOrder(m_Y, m_Z);
        QWidget::setTabOrder(m_Z, m_R);
        QWidget::setTabOrder(m_R, m_G);
        QWidget::setTabOrder(m_G, m_B);
        QWidget::setTabOrder(m_B, m_A);
        QWidget::setTabOrder(m_A, m_Value);
        QWidget::setTabOrder(m_Value, m_ColorName);
        QWidget::setTabOrder(m_ColorName, m_OptionsAndSegmentation);
        QWidget::setTabOrder(m_OptionsAndSegmentation, m_PointClass);
        QWidget::setTabOrder(m_PointClass, m_PointClassColor);
        QWidget::setTabOrder(m_PointClassColor, m_AddPointClass);
        QWidget::setTabOrder(m_AddPointClass, m_DeletePointClass);
        QWidget::setTabOrder(m_DeletePointClass, m_PointSize);
        QWidget::setTabOrder(m_PointSize, m_GreyScale);
        QWidget::setTabOrder(m_GreyScale, m_LocalOutputFile);
        QWidget::setTabOrder(m_LocalOutputFile, m_LocalOutputFileSelection);
        QWidget::setTabOrder(m_LocalOutputFileSelection, m_LocalRun);
        QWidget::setTabOrder(m_LocalRun, m_RemoteFile);
        QWidget::setTabOrder(m_RemoteFile, m_RemoteOutputFileSelection);
        QWidget::setTabOrder(m_RemoteOutputFileSelection, m_Hostname);
        QWidget::setTabOrder(m_Hostname, m_Port);
        QWidget::setTabOrder(m_Port, m_RemoteRun);

        retranslateUi(VolumeGridRoverBase);
        QObject::connect(m_SliceAxisGroup, SIGNAL(clicked(int)), m_SliceCanvasStack, SLOT(raiseWidget(int)));
        QObject::connect(m_PointClassColor, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(colorSlot()));
        QObject::connect(m_AddPointClass, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(addPointClassSlot()));
        QObject::connect(m_DeletePointClass, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(deletePointClassSlot()));
        QObject::connect(m_LocalOutputFileSelection, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(getLocalOutputFileSlot()));
        QObject::connect(m_LocalRun, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(localSegmentationRunSlot()));
        QObject::connect(m_RemoteRun, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(remoteSegmentationRunSlot()));
        QObject::connect(m_RemoteOutputFileSelection, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(getRemoteFileSlot()));
        QObject::connect(m_SliceAxisGroup, SIGNAL(clicked(int)), VolumeGridRoverBase, SLOT(sliceAxisChangedSlot()));
        QObject::connect(m_Z, SIGNAL(returnPressed()), VolumeGridRoverBase, SLOT(zChangedSlot()));
        QObject::connect(m_Y, SIGNAL(returnPressed()), VolumeGridRoverBase, SLOT(yChangedSlot()));
        QObject::connect(m_X, SIGNAL(returnPressed()), VolumeGridRoverBase, SLOT(xChangedSlot()));
        QObject::connect(m_EMClusteringRun, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(EMClusteringRunSlot()));
        QObject::connect(m_BackgroundColor, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(backgroundColorSlot()));
        QObject::connect(m_PointClassesLoadButton, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(loadPointClassesSlot()));
        QObject::connect(m_PointClassesSaveButton, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(savePointClassesSlot()));
        QObject::connect(m_AddContour, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(addContourSlot()));
        QObject::connect(m_DeleteContour, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(deleteContourSlot()));
        QObject::connect(m_ContourColor, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(contourColorSlot()));
        QObject::connect(m_RunTiling, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(tilingRunSlot()));
        QObject::connect(m_GridCellMarkingToolSelection, SIGNAL(currentChanged(QWidget*)), VolumeGridRoverBase, SLOT(cellMarkingModeTabChangedSlot(QWidget*)));
        QObject::connect(m_LoadContourButton, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(loadContoursSlot()));
        QObject::connect(m_InterpolationType, SIGNAL(activated(int)), VolumeGridRoverBase, SLOT(setInterpolationTypeSlot(int)));
        QObject::connect(m_InterpolationSampling, SIGNAL(valueChanged(int)), VolumeGridRoverBase, SLOT(setInterpolationSamplingSlot(int)));
        QObject::connect(m_TilingOutputDirectorySelect, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(getTilingOutputDirectorySlot()));
        QObject::connect(m_TilingOutputOptions, SIGNAL(clicked(int)), VolumeGridRoverBase, SLOT(handleTilingOutputDestinationSelectionSlot(int)));
        QObject::connect(m_SaveContoursButton, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(saveContoursSlot()));
        QObject::connect(m_Run2DSDF, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(sdfCurationSlot()));
        QObject::connect(m_SDFOptions, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(sdfOptionsSlot()));
        QObject::connect(m_CalcMedialAxis, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(medialAxisSlot()));
        QObject::connect(m_ContourCuration, SIGNAL(clicked()), VolumeGridRoverBase, SLOT(curateContoursSlot()));

        m_InterpolationType->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(VolumeGridRoverBase);
    } // setupUi

    void retranslateUi(QWidget *VolumeGridRoverBase)
    {
        VolumeGridRoverBase->setWindowTitle(QApplication::translate("VolumeGridRoverBase", "Volume Grid Rover", 0, QApplication::UnicodeUTF8));
        m_SliceCanvasGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Slice Canvas", 0, QApplication::UnicodeUTF8));
        m_SliceAxisGroup->setTitle(QString());
        m_ZYSliceCanvasButton->setText(QApplication::translate("VolumeGridRoverBase", "ZY", 0, QApplication::UnicodeUTF8));
        m_XZSliceCanvasButton->setText(QApplication::translate("VolumeGridRoverBase", "XZ", 0, QApplication::UnicodeUTF8));
        m_XYSliceCanvasButton->setText(QApplication::translate("VolumeGridRoverBase", "XY", 0, QApplication::UnicodeUTF8));
        m_XYResetViewButton->setText(QApplication::translate("VolumeGridRoverBase", "Reset View", 0, QApplication::UnicodeUTF8));
        m_XZResetViewButton->setText(QApplication::translate("VolumeGridRoverBase", "Reset View", 0, QApplication::UnicodeUTF8));
        m_ZYResetViewButton->setText(QApplication::translate("VolumeGridRoverBase", "Reset View", 0, QApplication::UnicodeUTF8));
        m_IndicesGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Image Index", 0, QApplication::UnicodeUTF8));
        m_XText->setText(QApplication::translate("VolumeGridRoverBase", "X:", 0, QApplication::UnicodeUTF8));
        m_X->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_YText->setText(QApplication::translate("VolumeGridRoverBase", "Y:", 0, QApplication::UnicodeUTF8));
        m_Y->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_ZText->setText(QApplication::translate("VolumeGridRoverBase", "Z:", 0, QApplication::UnicodeUTF8));
        m_Z->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_ObjectCoordinatesGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Object Coordinates", 0, QApplication::UnicodeUTF8));
        m_ObjX->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_ObjXText->setText(QApplication::translate("VolumeGridRoverBase", "X:", 0, QApplication::UnicodeUTF8));
        m_ObjYText->setText(QApplication::translate("VolumeGridRoverBase", "Y:", 0, QApplication::UnicodeUTF8));
        m_ObjY->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_ObjZText->setText(QApplication::translate("VolumeGridRoverBase", "Z:", 0, QApplication::UnicodeUTF8));
        m_ObjZ->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_GridCellTabs->setTabText(m_GridCellTabs->indexOf(VoxelCoordinates), QApplication::translate("VolumeGridRoverBase", "Voxel Coordinates", 0, QApplication::UnicodeUTF8));
        m_RText->setText(QApplication::translate("VolumeGridRoverBase", "R:", 0, QApplication::UnicodeUTF8));
        m_R->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_GText->setText(QApplication::translate("VolumeGridRoverBase", "G:", 0, QApplication::UnicodeUTF8));
        m_G->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_BText->setText(QApplication::translate("VolumeGridRoverBase", "B:", 0, QApplication::UnicodeUTF8));
        m_B->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_AText->setText(QApplication::translate("VolumeGridRoverBase", "A:", 0, QApplication::UnicodeUTF8));
        m_A->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_ValueText->setText(QApplication::translate("VolumeGridRoverBase", "Value:", 0, QApplication::UnicodeUTF8));
        m_Value->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_ColorNameText->setText(QApplication::translate("VolumeGridRoverBase", "Color Name: ", 0, QApplication::UnicodeUTF8));
        m_ColorName->setText(QString());
        m_MappedValueText->setText(QApplication::translate("VolumeGridRoverBase", "Mapped Value:", 0, QApplication::UnicodeUTF8));
        m_MappedValue->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_MappedValue->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "This is the value mapped to an unsigned character value.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_GridCellTabs->setTabText(m_GridCellTabs->indexOf(VoxelInfo), QApplication::translate("VolumeGridRoverBase", "Voxel Info", 0, QApplication::UnicodeUTF8));
        m_VariableText->setText(QApplication::translate("VolumeGridRoverBase", "Variable:", 0, QApplication::UnicodeUTF8));
        m_TimestepText->setText(QApplication::translate("VolumeGridRoverBase", "Timestep:", 0, QApplication::UnicodeUTF8));
        m_VariableInformationTabs->setTabText(m_VariableInformationTabs->indexOf(VariableSelection), QApplication::translate("VolumeGridRoverBase", "Variable Selection", 0, QApplication::UnicodeUTF8));
        m_MinimumValueText->setText(QApplication::translate("VolumeGridRoverBase", "Min Value:", 0, QApplication::UnicodeUTF8));
        m_MinimumValue->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_MaximumValueText->setText(QApplication::translate("VolumeGridRoverBase", "Max Value:", 0, QApplication::UnicodeUTF8));
        m_MaximumValue->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
        m_VariableInformationTabs->setTabText(m_VariableInformationTabs->indexOf(VariableInformation), QApplication::translate("VolumeGridRoverBase", "Variable Information", 0, QApplication::UnicodeUTF8));
        m_DisplayOptionsGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Display Options", 0, QApplication::UnicodeUTF8));
        m_BackgroundColorText->setText(QApplication::translate("VolumeGridRoverBase", "Background Color:", 0, QApplication::UnicodeUTF8));
        m_BackgroundColor->setText(QString());
        m_GreyScale->setText(QApplication::translate("VolumeGridRoverBase", "Grey Scale Density Map", 0, QApplication::UnicodeUTF8));
        m_PointSizeText->setText(QApplication::translate("VolumeGridRoverBase", "Point Size: ", 0, QApplication::UnicodeUTF8));
        m_RenderControlPoints->setText(QApplication::translate("VolumeGridRoverBase", "Render Points", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_RenderControlPoints->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "Render curve control points", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_RenderSDF->setText(QApplication::translate("VolumeGridRoverBase", "Show Signed Distance Function", 0, QApplication::UnicodeUTF8));
        m_Isocontouring->setText(QApplication::translate("VolumeGridRoverBase", "Isocontouring", 0, QApplication::UnicodeUTF8));
        m_PointClassColor->setText(QString());
        m_PointClassText->setText(QApplication::translate("VolumeGridRoverBase", "Point Class:", 0, QApplication::UnicodeUTF8));
        m_AddPointClass->setText(QApplication::translate("VolumeGridRoverBase", "Add Class", 0, QApplication::UnicodeUTF8));
        m_DeletePointClass->setText(QApplication::translate("VolumeGridRoverBase", "Delete Class", 0, QApplication::UnicodeUTF8));
        m_PointClassesLoadButton->setText(QApplication::translate("VolumeGridRoverBase", "Load Point Classes", 0, QApplication::UnicodeUTF8));
        m_PointClassesSaveButton->setText(QApplication::translate("VolumeGridRoverBase", "Save Point Classes", 0, QApplication::UnicodeUTF8));
        m_GridCellMarkingToolSelection->setTabText(m_GridCellMarkingToolSelection->indexOf(PointClasses), QApplication::translate("VolumeGridRoverBase", "Point Classes", 0, QApplication::UnicodeUTF8));
        m_Objects->header()->setLabel(0, QApplication::translate("VolumeGridRoverBase", "Object Name", 0, QApplication::UnicodeUTF8));
        m_InterpolationTypeText->setText(QApplication::translate("VolumeGridRoverBase", "<p align=\"left\">Interpolation\n"
"Type:</p>", 0, QApplication::UnicodeUTF8));
        m_InterpolationSamplingText->setText(QApplication::translate("VolumeGridRoverBase", "<p align=\"left\">Interp Sampling:</p>", 0, QApplication::UnicodeUTF8));
        m_DeleteContour->setText(QApplication::translate("VolumeGridRoverBase", "Delete Contour", 0, QApplication::UnicodeUTF8));
        m_SaveContoursButton->setText(QApplication::translate("VolumeGridRoverBase", "Save", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_InterpolationSampling->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "Number of samples between control points.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_WHATSTHIS
        m_InterpolationSampling->setProperty("whatsThis", QVariant(QApplication::translate("VolumeGridRoverBase", "<p align=\"left\">The interpolation sampling widget determines the number of samples between control points of the contours for the selected objects. Applies only to non-linear interpolation.</p>", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        m_AddContour->setText(QApplication::translate("VolumeGridRoverBase", "Add Contour", 0, QApplication::UnicodeUTF8));
        m_ContourColor->setText(QString());
        m_LoadContourButton->setText(QApplication::translate("VolumeGridRoverBase", "Load", 0, QApplication::UnicodeUTF8));
        m_InterpolationType->clear();
        m_InterpolationType->insertItems(0, QStringList()
         << QApplication::translate("VolumeGridRoverBase", "Linear", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("VolumeGridRoverBase", "Polynomial", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("VolumeGridRoverBase", "Cubic Spline", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("VolumeGridRoverBase", "Cubic Spline (Periodic)", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("VolumeGridRoverBase", "Akima spline", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("VolumeGridRoverBase", "Akima spline (periodic)", 0, QApplication::UnicodeUTF8)
        );
        m_GridCellMarkingToolSelection->setTabText(m_GridCellMarkingToolSelection->indexOf(Contours), QApplication::translate("VolumeGridRoverBase", "Contours", 0, QApplication::UnicodeUTF8));
        m_OptionsAndSegmentation->setTabText(m_OptionsAndSegmentation->indexOf(tab), QApplication::translate("VolumeGridRoverBase", "Grid Cell Marking", 0, QApplication::UnicodeUTF8));
        m_SegmentationThresholdGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Segmentation Threshold", 0, QApplication::UnicodeUTF8));
        m_TresholdLowText->setText(QApplication::translate("VolumeGridRoverBase", "Low:", 0, QApplication::UnicodeUTF8));
        m_ThresholdLow->setText(QApplication::translate("VolumeGridRoverBase", "0", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_ThresholdLow->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "These values are unsigned character mapped density values.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_ThresholdHighText->setText(QApplication::translate("VolumeGridRoverBase", "High:", 0, QApplication::UnicodeUTF8));
        m_ThresholdHigh->setText(QApplication::translate("VolumeGridRoverBase", "255", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_ThresholdHigh->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "These values are unsigned character mapped density values.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_LocalSegmentationGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Local Segmentation", 0, QApplication::UnicodeUTF8));
        m_OutputFileText->setText(QApplication::translate("VolumeGridRoverBase", "File:", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_LocalOutputFile->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "Enter output file name here. Appended to this filename will be the suffixes of the output segmentation files.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_LocalOutputFileSelection->setText(QApplication::translate("VolumeGridRoverBase", "...", 0, QApplication::UnicodeUTF8));
        m_LocalRun->setText(QApplication::translate("VolumeGridRoverBase", "Run", 0, QApplication::UnicodeUTF8));
        m_RemoteSegmentationGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Remote Segmentation", 0, QApplication::UnicodeUTF8));
        m_RemoteOutputFileText->setText(QApplication::translate("VolumeGridRoverBase", "Remote File:", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_RemoteFile->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "Enter output file name here.", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_RemoteOutputFileSelection->setText(QApplication::translate("VolumeGridRoverBase", "...", 0, QApplication::UnicodeUTF8));
        m_RemoteRun->setText(QApplication::translate("VolumeGridRoverBase", "Run", 0, QApplication::UnicodeUTF8));
        m_PortText->setText(QApplication::translate("VolumeGridRoverBase", "Port:", 0, QApplication::UnicodeUTF8));
        m_HostnameText->setText(QApplication::translate("VolumeGridRoverBase", "Hostname:", 0, QApplication::UnicodeUTF8));
        m_OptionsAndSegmentation->setTabText(m_OptionsAndSegmentation->indexOf(tab1), QApplication::translate("VolumeGridRoverBase", "Segmentation", 0, QApplication::UnicodeUTF8));
        m_EMClusteringRun->setText(QApplication::translate("VolumeGridRoverBase", "Run", 0, QApplication::UnicodeUTF8));
        m_OptionsAndSegmentation->setTabText(m_OptionsAndSegmentation->indexOf(tab2), QApplication::translate("VolumeGridRoverBase", "EMClustering", 0, QApplication::UnicodeUTF8));
        m_TilingOutputFilenameGroup->setTitle(QString());
        m_TilingOutputDirectoryText->setText(QApplication::translate("VolumeGridRoverBase", "Output Directory:", 0, QApplication::UnicodeUTF8));
        m_TilingOutputDirectorySelect->setText(QApplication::translate("VolumeGridRoverBase", "...", 0, QApplication::UnicodeUTF8));
        m_TilingOutputOptions->setTitle(QApplication::translate("VolumeGridRoverBase", "Output Destination", 0, QApplication::UnicodeUTF8));
        m_TilingOutputInMemory->setText(QApplication::translate("VolumeGridRoverBase", "In Core (Render Immediately)", 0, QApplication::UnicodeUTF8));
        m_TilingOptionSaveToFile->setText(QApplication::translate("VolumeGridRoverBase", "To Files", 0, QApplication::UnicodeUTF8));
        m_Run2DSDF->setText(QApplication::translate("VolumeGridRoverBase", "Run 2D SDF Contour Curation", 0, QApplication::UnicodeUTF8));
        m_RunTiling->setText(QApplication::translate("VolumeGridRoverBase", "Run Tiling On Selected Contours", 0, QApplication::UnicodeUTF8));
        m_OptionsAndSegmentation->setTabText(m_OptionsAndSegmentation->indexOf(Tiling), QApplication::translate("VolumeGridRoverBase", "Tiling", 0, QApplication::UnicodeUTF8));
        m_CurrentSliceProcessGroup->setTitle(QApplication::translate("VolumeGridRoverBase", "Processing for current selected slice...", 0, QApplication::UnicodeUTF8));
        m_CalcMedialAxis->setText(QApplication::translate("VolumeGridRoverBase", "Calculate Medial Axis for Selected Contours", 0, QApplication::UnicodeUTF8));
        m_SDFOptions->setText(QApplication::translate("VolumeGridRoverBase", "Calculate Signed Distance Function", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_SDFOptions->setProperty("toolTip", QVariant(QApplication::translate("VolumeGridRoverBase", "Signed Distance Function Options", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_CalcDelaunayVoronoi->setText(QApplication::translate("VolumeGridRoverBase", "Calculate Delaunay Triangulation && Voronoi Diagram", 0, QApplication::UnicodeUTF8));
        m_ContourCuration->setText(QApplication::translate("VolumeGridRoverBase", "Curate Selected Contours", 0, QApplication::UnicodeUTF8));
        m_OptionsAndSegmentation->setTabText(m_OptionsAndSegmentation->indexOf(TabPage), QApplication::translate("VolumeGridRoverBase", "Processing", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class VolumeGridRoverBase: public Ui_VolumeGridRoverBase {};
} // namespace Ui

QT_END_NAMESPACE

class VolumeGridRoverBase : public QWidget, public Ui::VolumeGridRoverBase
{
    Q_OBJECT

public:
    VolumeGridRoverBase(QWidget* parent = 0, const char* name = 0, Qt::WindowFlags fl = 0);
    ~VolumeGridRoverBase();

public slots:
    virtual void getLocalOutputFileSlot();
    virtual void localSegmentationRunSlot();
    virtual void remoteSegmentationRunSlot();
    virtual void getRemoteFileSlot();
    virtual void sliceAxisChangedSlot();
    virtual void EMClusteringRunSlot();
    virtual void sdfOptionsSlot();
    virtual void medialAxisSlot();
    virtual void curateContoursSlot();

protected slots:
    virtual void languageChange();

    virtual void colorSlot();
    virtual void addPointClassSlot();
    virtual void deletePointClassSlot();
    virtual void zChangedSlot();
    virtual void yChangedSlot();
    virtual void xChangedSlot();
    virtual void backgroundColorSlot();
    virtual void savePointClassesSlot();
    virtual void loadPointClassesSlot();
    virtual void addContourSlot();
    virtual void deleteContourSlot();
    virtual void contourColorSlot();
    virtual void tilingRunSlot();
    virtual void cellMarkingModeTabChangedSlot(QWidget *w);
    virtual void loadContoursSlot();
    virtual void setInterpolationTypeSlot(int);
    virtual void setInterpolationSamplingSlot(int);
    virtual void getTilingOutputDirectorySlot();
    virtual void handleTilingOutputDestinationSelectionSlot(int);
    virtual void saveContoursSlot();
    virtual void sdfCurationSlot();


};

#endif // VOLUMEGRIDROVERBASE_H
