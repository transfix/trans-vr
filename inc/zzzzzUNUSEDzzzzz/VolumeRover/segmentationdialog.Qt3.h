#ifndef SEGMENTATIONDIALOGBASE_H
#define SEGMENTATIONDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
#include <Qt3Support/Q3GroupBox>
#include <Qt3Support/Q3MimeSourceFactory>
#include <Qt3Support/Q3WidgetStack>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SegmentationDialogBase
{
public:
    QGridLayout *gridLayout;
    Q3WidgetStack *m_OptionsStack;
    QWidget *CapsidOptions;
    QGridLayout *gridLayout1;
    Q3GroupBox *m_CapsidOptions;
    QGridLayout *gridLayout2;
    Q3WidgetStack *m_CapsidOptionsStack;
    QWidget *Type0;
    QGridLayout *gridLayout3;
    QLineEdit *m_TLowEditType0;
    QSpacerItem *spacer4;
    QLabel *m_Z0TextType0;
    QLabel *m_Y0TextType0;
    QLineEdit *m_Z0EditType0;
    QLineEdit *m_Y0EditType0;
    QLabel *m_X0TextType0;
    QLabel *m_TLowTextType0;
    QLabel *m_SeedPoint0TextType0;
    QLineEdit *m_X0EditType0;
    QCheckBox *m_RunDiffusionType0;
    QWidget *Type2;
    QGridLayout *gridLayout4;
    QLineEdit *m_TLowEditType1;
    QLabel *m_TLowTextType1;
    QLabel *m_SeedPoint0TextType1;
    QLabel *m_X0TextType1;
    QLineEdit *m_X0EditType1;
    QLabel *m_Y0TextType1;
    QLineEdit *m_Y0EditType1;
    QLabel *m_Z0TextType1;
    QLineEdit *m_X1EditType1;
    QLabel *m_Y1TextType1;
    QLabel *m_SeedPoint1TextType1;
    QLabel *m_X1TextType1;
    QLineEdit *m_Y1EditType1;
    QLabel *m_Z1TextType1;
    QLineEdit *m_Z1EditType1;
    QLineEdit *m_Z0EditType1;
    QCheckBox *m_RunDiffusionType1;
    QWidget *Type21;
    QGridLayout *gridLayout5;
    QLabel *m_RadiiTextType2;
    QLabel *m_SmallRadiusTextType2;
    QLabel *m_LargeRadiusTextType2;
    QLineEdit *m_LargeRadiusEditType2;
    QLineEdit *m_SmallRadiusEditType2;
    QLineEdit *m_TLowEditType2;
    QLabel *m_TLowTextType2;
    QSpacerItem *spacer2;
    QWidget *Type3;
    QGridLayout *gridLayout6;
    QLabel *m_OuterLayerTextType3;
    QLabel *m_RadiiTextType3;
    QLabel *m_SmallRadiusTextType3;
    QLineEdit *m_SmallRadiusEditType3;
    QLabel *m_LargeRadiusTextType3;
    QLineEdit *m_LargeRadiusEditType3;
    QLabel *m_3FoldTextType3;
    QLineEdit *m_3FoldEditType3;
    QLabel *m_5FoldTextType3;
    QLineEdit *m_5FoldEditType3;
    QLabel *m_6FoldTextType3;
    QLineEdit *m_6FoldEditType3;
    QLineEdit *m_TLowEditType3;
    QLabel *m_TLowTextType3;
    QLabel *m_CapsidLayerTypeText;
    QComboBox *m_CapsidLayerType;
    QWidget *MonomerOptions;
    QGridLayout *gridLayout7;
    Q3GroupBox *m_MonomerOptions;
    QGridLayout *gridLayout8;
    QLabel *m_FoldNumText;
    QLineEdit *m_FoldNumEdit;
    QWidget *SubunitOptions;
    QGridLayout *gridLayout9;
    Q3GroupBox *m_SubunitOptions;
    QGridLayout *gridLayout10;
    QLabel *m_HNumText;
    QLineEdit *m_HNumEdit;
    QLabel *m_KNumText;
    QLineEdit *m_KNumEdit;
    QLabel *m_3FoldText;
    QLineEdit *m_3FoldEdit;
    QLabel *m_5FoldText;
    QLineEdit *m_5FoldEdit;
    QLabel *m_6FoldText;
    QLineEdit *m_6FoldEdit;
    QLabel *m_InitRadiusText;
    QLineEdit *m_InitRadiusEdit;
    QSpacerItem *spacer3;
    QWidget *WStackPage;
    QGridLayout *gridLayout11;
    Q3GroupBox *m_SecondaryStructureOptions;
    QLabel *m_HelixWidthText;
    QLabel *m_SheetWidthText;
    QLineEdit *m_SheetWidth;
    QLabel *m_MinHelixWidthRatioText;
    QLineEdit *m_MinHelixWidthRatio;
    QLabel *m_MinSheetWidthRatioText;
    QLineEdit *m_MinSheetWidthRatio;
    QLineEdit *m_MaxSheetWidthRatio;
    QLabel *m_MaxSheetWidthRatioText;
    QLineEdit *m_MaxHelixWidthRatio;
    QLabel *m_MaxHelixWidthRatioText;
    QLabel *m_MinHelixLengthText;
    QLabel *m_SheetExtendText;
    QLineEdit *m_SheetExtend;
    QLineEdit *m_HelixWidth;
    QLineEdit *m_MinHelixLength;
    QLineEdit *m_Threshold;
    QRadioButton *m_ThresholdCheck;
    QPushButton *m_RunButton;
    QPushButton *m_CancelButton;
    Q3ButtonGroup *m_ExecutionLocation;
    QGridLayout *gridLayout12;
    QRadioButton *m_LocalSegmentationButton;
    QRadioButton *m_RemoteSegmentationButton;
    Q3GroupBox *m_RemoteSegmentationGroup;
    QGridLayout *gridLayout13;
    QLabel *m_RemoteSegmentationHostnameText;
    QLineEdit *m_RemoteSegmentationHostname;
    QLabel *m_RemoteSegmentationPortText;
    QLineEdit *m_RemoteSegmentationPort;
    QLabel *m_RemoteSegmentationFilenameText;
    QLineEdit *m_RemoteSegmentationFilename;
    QLabel *m_SegTypeText;
    QComboBox *m_SegTypeSelection;

    void setupUi(QDialog *SegmentationDialogBase)
    {
        if (SegmentationDialogBase->objectName().isEmpty())
            SegmentationDialogBase->setObjectName(QString::fromUtf8("SegmentationDialogBase"));
        SegmentationDialogBase->resize(473, 557);
        gridLayout = new QGridLayout(SegmentationDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setResizeMode(QGridLayout::Fixed);
        m_OptionsStack = new Q3WidgetStack(SegmentationDialogBase);
        m_OptionsStack->setObjectName(QString::fromUtf8("m_OptionsStack"));
        CapsidOptions = new QWidget(m_OptionsStack);
        CapsidOptions->setObjectName(QString::fromUtf8("CapsidOptions"));
        gridLayout1 = new QGridLayout(CapsidOptions);
        gridLayout1->setSpacing(6);
        gridLayout1->setContentsMargins(11, 11, 11, 11);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        gridLayout1->setContentsMargins(0, 0, 0, 0);
        m_CapsidOptions = new Q3GroupBox(CapsidOptions);
        m_CapsidOptions->setObjectName(QString::fromUtf8("m_CapsidOptions"));
        m_CapsidOptions->setColumnLayout(0, Qt::Vertical);
        m_CapsidOptions->layout()->setSpacing(6);
        m_CapsidOptions->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_CapsidOptions->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        m_CapsidOptionsStack = new Q3WidgetStack(m_CapsidOptions);
        m_CapsidOptionsStack->setObjectName(QString::fromUtf8("m_CapsidOptionsStack"));
        Type0 = new QWidget(m_CapsidOptionsStack);
        Type0->setObjectName(QString::fromUtf8("Type0"));
        gridLayout3 = new QGridLayout(Type0);
        gridLayout3->setSpacing(6);
        gridLayout3->setContentsMargins(11, 11, 11, 11);
        gridLayout3->setObjectName(QString::fromUtf8("gridLayout3"));
        gridLayout3->setContentsMargins(0, 0, 0, 0);
        m_TLowEditType0 = new QLineEdit(Type0);
        m_TLowEditType0->setObjectName(QString::fromUtf8("m_TLowEditType0"));

        gridLayout3->addWidget(m_TLowEditType0, 0, 4, 1, 2);

        spacer4 = new QSpacerItem(41, 30, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout3->addItem(spacer4, 3, 1, 1, 1);

        m_Z0TextType0 = new QLabel(Type0);
        m_Z0TextType0->setObjectName(QString::fromUtf8("m_Z0TextType0"));
        m_Z0TextType0->setWordWrap(false);

        gridLayout3->addWidget(m_Z0TextType0, 2, 4, 1, 1);

        m_Y0TextType0 = new QLabel(Type0);
        m_Y0TextType0->setObjectName(QString::fromUtf8("m_Y0TextType0"));
        m_Y0TextType0->setWordWrap(false);

        gridLayout3->addWidget(m_Y0TextType0, 2, 2, 1, 1);

        m_Z0EditType0 = new QLineEdit(Type0);
        m_Z0EditType0->setObjectName(QString::fromUtf8("m_Z0EditType0"));

        gridLayout3->addWidget(m_Z0EditType0, 2, 5, 1, 1);

        m_Y0EditType0 = new QLineEdit(Type0);
        m_Y0EditType0->setObjectName(QString::fromUtf8("m_Y0EditType0"));

        gridLayout3->addWidget(m_Y0EditType0, 2, 3, 1, 1);

        m_X0TextType0 = new QLabel(Type0);
        m_X0TextType0->setObjectName(QString::fromUtf8("m_X0TextType0"));
        m_X0TextType0->setWordWrap(false);

        gridLayout3->addWidget(m_X0TextType0, 2, 0, 1, 1);

        m_TLowTextType0 = new QLabel(Type0);
        m_TLowTextType0->setObjectName(QString::fromUtf8("m_TLowTextType0"));
        m_TLowTextType0->setWordWrap(false);

        gridLayout3->addWidget(m_TLowTextType0, 0, 0, 1, 4);

        m_SeedPoint0TextType0 = new QLabel(Type0);
        m_SeedPoint0TextType0->setObjectName(QString::fromUtf8("m_SeedPoint0TextType0"));
        m_SeedPoint0TextType0->setWordWrap(false);

        gridLayout3->addWidget(m_SeedPoint0TextType0, 1, 0, 1, 2);

        m_X0EditType0 = new QLineEdit(Type0);
        m_X0EditType0->setObjectName(QString::fromUtf8("m_X0EditType0"));

        gridLayout3->addWidget(m_X0EditType0, 2, 1, 1, 1);

        m_RunDiffusionType0 = new QCheckBox(Type0);
        m_RunDiffusionType0->setObjectName(QString::fromUtf8("m_RunDiffusionType0"));
        m_RunDiffusionType0->setChecked(true);

        gridLayout3->addWidget(m_RunDiffusionType0, 1, 4, 1, 2);

        m_CapsidOptionsStack->addWidget(Type0, 0);
        Type2 = new QWidget(m_CapsidOptionsStack);
        Type2->setObjectName(QString::fromUtf8("Type2"));
        gridLayout4 = new QGridLayout(Type2);
        gridLayout4->setSpacing(6);
        gridLayout4->setContentsMargins(11, 11, 11, 11);
        gridLayout4->setObjectName(QString::fromUtf8("gridLayout4"));
        gridLayout4->setContentsMargins(0, 0, 0, 0);
        m_TLowEditType1 = new QLineEdit(Type2);
        m_TLowEditType1->setObjectName(QString::fromUtf8("m_TLowEditType1"));

        gridLayout4->addWidget(m_TLowEditType1, 0, 7, 1, 1);

        m_TLowTextType1 = new QLabel(Type2);
        m_TLowTextType1->setObjectName(QString::fromUtf8("m_TLowTextType1"));
        m_TLowTextType1->setWordWrap(false);

        gridLayout4->addWidget(m_TLowTextType1, 0, 0, 1, 7);

        m_SeedPoint0TextType1 = new QLabel(Type2);
        m_SeedPoint0TextType1->setObjectName(QString::fromUtf8("m_SeedPoint0TextType1"));
        m_SeedPoint0TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_SeedPoint0TextType1, 1, 0, 1, 6);

        m_X0TextType1 = new QLabel(Type2);
        m_X0TextType1->setObjectName(QString::fromUtf8("m_X0TextType1"));
        m_X0TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_X0TextType1, 2, 0, 1, 1);

        m_X0EditType1 = new QLineEdit(Type2);
        m_X0EditType1->setObjectName(QString::fromUtf8("m_X0EditType1"));

        gridLayout4->addWidget(m_X0EditType1, 2, 1, 1, 2);

        m_Y0TextType1 = new QLabel(Type2);
        m_Y0TextType1->setObjectName(QString::fromUtf8("m_Y0TextType1"));
        m_Y0TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_Y0TextType1, 2, 3, 1, 1);

        m_Y0EditType1 = new QLineEdit(Type2);
        m_Y0EditType1->setObjectName(QString::fromUtf8("m_Y0EditType1"));

        gridLayout4->addWidget(m_Y0EditType1, 2, 4, 1, 1);

        m_Z0TextType1 = new QLabel(Type2);
        m_Z0TextType1->setObjectName(QString::fromUtf8("m_Z0TextType1"));
        m_Z0TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_Z0TextType1, 2, 5, 1, 1);

        m_X1EditType1 = new QLineEdit(Type2);
        m_X1EditType1->setObjectName(QString::fromUtf8("m_X1EditType1"));

        gridLayout4->addWidget(m_X1EditType1, 4, 2, 1, 1);

        m_Y1TextType1 = new QLabel(Type2);
        m_Y1TextType1->setObjectName(QString::fromUtf8("m_Y1TextType1"));
        m_Y1TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_Y1TextType1, 4, 3, 1, 1);

        m_SeedPoint1TextType1 = new QLabel(Type2);
        m_SeedPoint1TextType1->setObjectName(QString::fromUtf8("m_SeedPoint1TextType1"));
        m_SeedPoint1TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_SeedPoint1TextType1, 3, 0, 1, 5);

        m_X1TextType1 = new QLabel(Type2);
        m_X1TextType1->setObjectName(QString::fromUtf8("m_X1TextType1"));
        m_X1TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_X1TextType1, 4, 0, 1, 2);

        m_Y1EditType1 = new QLineEdit(Type2);
        m_Y1EditType1->setObjectName(QString::fromUtf8("m_Y1EditType1"));

        gridLayout4->addWidget(m_Y1EditType1, 4, 4, 1, 1);

        m_Z1TextType1 = new QLabel(Type2);
        m_Z1TextType1->setObjectName(QString::fromUtf8("m_Z1TextType1"));
        m_Z1TextType1->setWordWrap(false);

        gridLayout4->addWidget(m_Z1TextType1, 4, 5, 1, 1);

        m_Z1EditType1 = new QLineEdit(Type2);
        m_Z1EditType1->setObjectName(QString::fromUtf8("m_Z1EditType1"));

        gridLayout4->addWidget(m_Z1EditType1, 4, 6, 1, 1);

        m_Z0EditType1 = new QLineEdit(Type2);
        m_Z0EditType1->setObjectName(QString::fromUtf8("m_Z0EditType1"));

        gridLayout4->addWidget(m_Z0EditType1, 2, 6, 1, 1);

        m_RunDiffusionType1 = new QCheckBox(Type2);
        m_RunDiffusionType1->setObjectName(QString::fromUtf8("m_RunDiffusionType1"));
        m_RunDiffusionType1->setChecked(true);

        gridLayout4->addWidget(m_RunDiffusionType1, 1, 6, 1, 2);

        m_CapsidOptionsStack->addWidget(Type2, 1);
        Type21 = new QWidget(m_CapsidOptionsStack);
        Type21->setObjectName(QString::fromUtf8("Type21"));
        gridLayout5 = new QGridLayout(Type21);
        gridLayout5->setSpacing(6);
        gridLayout5->setContentsMargins(11, 11, 11, 11);
        gridLayout5->setObjectName(QString::fromUtf8("gridLayout5"));
        gridLayout5->setContentsMargins(0, 0, 0, 0);
        m_RadiiTextType2 = new QLabel(Type21);
        m_RadiiTextType2->setObjectName(QString::fromUtf8("m_RadiiTextType2"));
        m_RadiiTextType2->setWordWrap(false);

        gridLayout5->addWidget(m_RadiiTextType2, 1, 0, 1, 4);

        m_SmallRadiusTextType2 = new QLabel(Type21);
        m_SmallRadiusTextType2->setObjectName(QString::fromUtf8("m_SmallRadiusTextType2"));
        m_SmallRadiusTextType2->setWordWrap(false);

        gridLayout5->addWidget(m_SmallRadiusTextType2, 2, 0, 1, 1);

        m_LargeRadiusTextType2 = new QLabel(Type21);
        m_LargeRadiusTextType2->setObjectName(QString::fromUtf8("m_LargeRadiusTextType2"));
        m_LargeRadiusTextType2->setWordWrap(false);

        gridLayout5->addWidget(m_LargeRadiusTextType2, 2, 2, 1, 1);

        m_LargeRadiusEditType2 = new QLineEdit(Type21);
        m_LargeRadiusEditType2->setObjectName(QString::fromUtf8("m_LargeRadiusEditType2"));
        m_LargeRadiusEditType2->setAcceptDrops(false);

        gridLayout5->addWidget(m_LargeRadiusEditType2, 2, 3, 1, 1);

        m_SmallRadiusEditType2 = new QLineEdit(Type21);
        m_SmallRadiusEditType2->setObjectName(QString::fromUtf8("m_SmallRadiusEditType2"));
        m_SmallRadiusEditType2->setAcceptDrops(false);

        gridLayout5->addWidget(m_SmallRadiusEditType2, 2, 1, 1, 1);

        m_TLowEditType2 = new QLineEdit(Type21);
        m_TLowEditType2->setObjectName(QString::fromUtf8("m_TLowEditType2"));

        gridLayout5->addWidget(m_TLowEditType2, 0, 4, 1, 1);

        m_TLowTextType2 = new QLabel(Type21);
        m_TLowTextType2->setObjectName(QString::fromUtf8("m_TLowTextType2"));
        m_TLowTextType2->setWordWrap(false);

        gridLayout5->addWidget(m_TLowTextType2, 0, 0, 1, 4);

        spacer2 = new QSpacerItem(51, 61, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout5->addItem(spacer2, 3, 1, 1, 2);

        m_CapsidOptionsStack->addWidget(Type21, 2);
        Type3 = new QWidget(m_CapsidOptionsStack);
        Type3->setObjectName(QString::fromUtf8("Type3"));
        gridLayout6 = new QGridLayout(Type3);
        gridLayout6->setSpacing(6);
        gridLayout6->setContentsMargins(11, 11, 11, 11);
        gridLayout6->setObjectName(QString::fromUtf8("gridLayout6"));
        gridLayout6->setContentsMargins(0, 0, 0, 0);
        m_OuterLayerTextType3 = new QLabel(Type3);
        m_OuterLayerTextType3->setObjectName(QString::fromUtf8("m_OuterLayerTextType3"));
        m_OuterLayerTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_OuterLayerTextType3, 1, 0, 1, 3);

        m_RadiiTextType3 = new QLabel(Type3);
        m_RadiiTextType3->setObjectName(QString::fromUtf8("m_RadiiTextType3"));
        m_RadiiTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_RadiiTextType3, 3, 0, 1, 6);

        m_SmallRadiusTextType3 = new QLabel(Type3);
        m_SmallRadiusTextType3->setObjectName(QString::fromUtf8("m_SmallRadiusTextType3"));
        m_SmallRadiusTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_SmallRadiusTextType3, 4, 0, 1, 2);

        m_SmallRadiusEditType3 = new QLineEdit(Type3);
        m_SmallRadiusEditType3->setObjectName(QString::fromUtf8("m_SmallRadiusEditType3"));
        m_SmallRadiusEditType3->setAcceptDrops(false);

        gridLayout6->addWidget(m_SmallRadiusEditType3, 4, 2, 1, 2);

        m_LargeRadiusTextType3 = new QLabel(Type3);
        m_LargeRadiusTextType3->setObjectName(QString::fromUtf8("m_LargeRadiusTextType3"));
        m_LargeRadiusTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_LargeRadiusTextType3, 4, 4, 1, 2);

        m_LargeRadiusEditType3 = new QLineEdit(Type3);
        m_LargeRadiusEditType3->setObjectName(QString::fromUtf8("m_LargeRadiusEditType3"));
        m_LargeRadiusEditType3->setAcceptDrops(false);

        gridLayout6->addWidget(m_LargeRadiusEditType3, 4, 6, 1, 2);

        m_3FoldTextType3 = new QLabel(Type3);
        m_3FoldTextType3->setObjectName(QString::fromUtf8("m_3FoldTextType3"));
        m_3FoldTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_3FoldTextType3, 2, 0, 1, 1);

        m_3FoldEditType3 = new QLineEdit(Type3);
        m_3FoldEditType3->setObjectName(QString::fromUtf8("m_3FoldEditType3"));

        gridLayout6->addWidget(m_3FoldEditType3, 2, 1, 1, 2);

        m_5FoldTextType3 = new QLabel(Type3);
        m_5FoldTextType3->setObjectName(QString::fromUtf8("m_5FoldTextType3"));
        m_5FoldTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_5FoldTextType3, 2, 3, 1, 2);

        m_5FoldEditType3 = new QLineEdit(Type3);
        m_5FoldEditType3->setObjectName(QString::fromUtf8("m_5FoldEditType3"));

        gridLayout6->addWidget(m_5FoldEditType3, 2, 5, 1, 2);

        m_6FoldTextType3 = new QLabel(Type3);
        m_6FoldTextType3->setObjectName(QString::fromUtf8("m_6FoldTextType3"));
        m_6FoldTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_6FoldTextType3, 2, 7, 1, 1);

        m_6FoldEditType3 = new QLineEdit(Type3);
        m_6FoldEditType3->setObjectName(QString::fromUtf8("m_6FoldEditType3"));

        gridLayout6->addWidget(m_6FoldEditType3, 2, 8, 1, 1);

        m_TLowEditType3 = new QLineEdit(Type3);
        m_TLowEditType3->setObjectName(QString::fromUtf8("m_TLowEditType3"));

        gridLayout6->addWidget(m_TLowEditType3, 0, 8, 1, 1);

        m_TLowTextType3 = new QLabel(Type3);
        m_TLowTextType3->setObjectName(QString::fromUtf8("m_TLowTextType3"));
        m_TLowTextType3->setWordWrap(false);

        gridLayout6->addWidget(m_TLowTextType3, 0, 0, 1, 8);

        m_CapsidOptionsStack->addWidget(Type3, 3);

        gridLayout2->addWidget(m_CapsidOptionsStack, 1, 0, 1, 2);

        m_CapsidLayerTypeText = new QLabel(m_CapsidOptions);
        m_CapsidLayerTypeText->setObjectName(QString::fromUtf8("m_CapsidLayerTypeText"));
        m_CapsidLayerTypeText->setWordWrap(false);

        gridLayout2->addWidget(m_CapsidLayerTypeText, 0, 0, 1, 1);

        m_CapsidLayerType = new QComboBox(m_CapsidOptions);
        m_CapsidLayerType->setObjectName(QString::fromUtf8("m_CapsidLayerType"));

        gridLayout2->addWidget(m_CapsidLayerType, 0, 1, 1, 1);


        gridLayout1->addWidget(m_CapsidOptions, 0, 0, 1, 1);

        m_OptionsStack->addWidget(CapsidOptions, 0);
        MonomerOptions = new QWidget(m_OptionsStack);
        MonomerOptions->setObjectName(QString::fromUtf8("MonomerOptions"));
        gridLayout7 = new QGridLayout(MonomerOptions);
        gridLayout7->setSpacing(6);
        gridLayout7->setContentsMargins(11, 11, 11, 11);
        gridLayout7->setObjectName(QString::fromUtf8("gridLayout7"));
        gridLayout7->setContentsMargins(0, 0, 0, 0);
        m_MonomerOptions = new Q3GroupBox(MonomerOptions);
        m_MonomerOptions->setObjectName(QString::fromUtf8("m_MonomerOptions"));
        m_MonomerOptions->setColumnLayout(0, Qt::Vertical);
        m_MonomerOptions->layout()->setSpacing(6);
        m_MonomerOptions->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout8 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_MonomerOptions->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout8);
        gridLayout8->setAlignment(Qt::AlignTop);
        gridLayout8->setObjectName(QString::fromUtf8("gridLayout8"));
        m_FoldNumText = new QLabel(m_MonomerOptions);
        m_FoldNumText->setObjectName(QString::fromUtf8("m_FoldNumText"));
        m_FoldNumText->setWordWrap(false);

        gridLayout8->addWidget(m_FoldNumText, 0, 0, 1, 1);

        m_FoldNumEdit = new QLineEdit(m_MonomerOptions);
        m_FoldNumEdit->setObjectName(QString::fromUtf8("m_FoldNumEdit"));

        gridLayout8->addWidget(m_FoldNumEdit, 0, 1, 1, 1);


        gridLayout7->addWidget(m_MonomerOptions, 0, 0, 1, 1);

        m_OptionsStack->addWidget(MonomerOptions, 1);
        SubunitOptions = new QWidget(m_OptionsStack);
        SubunitOptions->setObjectName(QString::fromUtf8("SubunitOptions"));
        gridLayout9 = new QGridLayout(SubunitOptions);
        gridLayout9->setSpacing(6);
        gridLayout9->setContentsMargins(11, 11, 11, 11);
        gridLayout9->setObjectName(QString::fromUtf8("gridLayout9"));
        gridLayout9->setContentsMargins(0, 0, 0, 0);
        m_SubunitOptions = new Q3GroupBox(SubunitOptions);
        m_SubunitOptions->setObjectName(QString::fromUtf8("m_SubunitOptions"));
        m_SubunitOptions->setColumnLayout(0, Qt::Vertical);
        m_SubunitOptions->layout()->setSpacing(6);
        m_SubunitOptions->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout10 = new QGridLayout();
        QBoxLayout *boxlayout2 = qobject_cast<QBoxLayout *>(m_SubunitOptions->layout());
        if (boxlayout2)
            boxlayout2->addLayout(gridLayout10);
        gridLayout10->setAlignment(Qt::AlignTop);
        gridLayout10->setObjectName(QString::fromUtf8("gridLayout10"));
        m_HNumText = new QLabel(m_SubunitOptions);
        m_HNumText->setObjectName(QString::fromUtf8("m_HNumText"));
        m_HNumText->setWordWrap(false);

        gridLayout10->addWidget(m_HNumText, 0, 0, 1, 2);

        m_HNumEdit = new QLineEdit(m_SubunitOptions);
        m_HNumEdit->setObjectName(QString::fromUtf8("m_HNumEdit"));

        gridLayout10->addWidget(m_HNumEdit, 0, 2, 1, 3);

        m_KNumText = new QLabel(m_SubunitOptions);
        m_KNumText->setObjectName(QString::fromUtf8("m_KNumText"));
        m_KNumText->setWordWrap(false);

        gridLayout10->addWidget(m_KNumText, 0, 5, 1, 2);

        m_KNumEdit = new QLineEdit(m_SubunitOptions);
        m_KNumEdit->setObjectName(QString::fromUtf8("m_KNumEdit"));

        gridLayout10->addWidget(m_KNumEdit, 0, 7, 1, 2);

        m_3FoldText = new QLabel(m_SubunitOptions);
        m_3FoldText->setObjectName(QString::fromUtf8("m_3FoldText"));
        m_3FoldText->setWordWrap(false);

        gridLayout10->addWidget(m_3FoldText, 1, 0, 1, 1);

        m_3FoldEdit = new QLineEdit(m_SubunitOptions);
        m_3FoldEdit->setObjectName(QString::fromUtf8("m_3FoldEdit"));

        gridLayout10->addWidget(m_3FoldEdit, 1, 1, 1, 3);

        m_5FoldText = new QLabel(m_SubunitOptions);
        m_5FoldText->setObjectName(QString::fromUtf8("m_5FoldText"));
        m_5FoldText->setWordWrap(false);

        gridLayout10->addWidget(m_5FoldText, 1, 4, 1, 2);

        m_5FoldEdit = new QLineEdit(m_SubunitOptions);
        m_5FoldEdit->setObjectName(QString::fromUtf8("m_5FoldEdit"));

        gridLayout10->addWidget(m_5FoldEdit, 1, 6, 1, 2);

        m_6FoldText = new QLabel(m_SubunitOptions);
        m_6FoldText->setObjectName(QString::fromUtf8("m_6FoldText"));
        m_6FoldText->setWordWrap(false);

        gridLayout10->addWidget(m_6FoldText, 1, 8, 1, 1);

        m_6FoldEdit = new QLineEdit(m_SubunitOptions);
        m_6FoldEdit->setObjectName(QString::fromUtf8("m_6FoldEdit"));

        gridLayout10->addWidget(m_6FoldEdit, 1, 9, 1, 1);

        m_InitRadiusText = new QLabel(m_SubunitOptions);
        m_InitRadiusText->setObjectName(QString::fromUtf8("m_InitRadiusText"));
        m_InitRadiusText->setWordWrap(false);

        gridLayout10->addWidget(m_InitRadiusText, 2, 0, 1, 3);

        m_InitRadiusEdit = new QLineEdit(m_SubunitOptions);
        m_InitRadiusEdit->setObjectName(QString::fromUtf8("m_InitRadiusEdit"));

        gridLayout10->addWidget(m_InitRadiusEdit, 2, 3, 1, 3);

        spacer3 = new QSpacerItem(281, 31, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout10->addItem(spacer3, 2, 6, 1, 4);


        gridLayout9->addWidget(m_SubunitOptions, 0, 0, 1, 1);

        m_OptionsStack->addWidget(SubunitOptions, 2);
        WStackPage = new QWidget(m_OptionsStack);
        WStackPage->setObjectName(QString::fromUtf8("WStackPage"));
        gridLayout11 = new QGridLayout(WStackPage);
        gridLayout11->setSpacing(6);
        gridLayout11->setContentsMargins(11, 11, 11, 11);
        gridLayout11->setObjectName(QString::fromUtf8("gridLayout11"));
        gridLayout11->setContentsMargins(0, 0, 0, 0);
        m_SecondaryStructureOptions = new Q3GroupBox(WStackPage);
        m_SecondaryStructureOptions->setObjectName(QString::fromUtf8("m_SecondaryStructureOptions"));
        m_HelixWidthText = new QLabel(m_SecondaryStructureOptions);
        m_HelixWidthText->setObjectName(QString::fromUtf8("m_HelixWidthText"));
        m_HelixWidthText->setGeometry(QRect(11, 21, 139, 22));
        m_HelixWidthText->setWordWrap(false);
        m_SheetWidthText = new QLabel(m_SecondaryStructureOptions);
        m_SheetWidthText->setObjectName(QString::fromUtf8("m_SheetWidthText"));
        m_SheetWidthText->setGeometry(QRect(211, 21, 145, 22));
        m_SheetWidthText->setWordWrap(false);
        m_SheetWidth = new QLineEdit(m_SecondaryStructureOptions);
        m_SheetWidth->setObjectName(QString::fromUtf8("m_SheetWidth"));
        m_SheetWidth->setGeometry(QRect(362, 21, 50, 22));
        m_MinHelixWidthRatioText = new QLabel(m_SecondaryStructureOptions);
        m_MinHelixWidthRatioText->setObjectName(QString::fromUtf8("m_MinHelixWidthRatioText"));
        m_MinHelixWidthRatioText->setGeometry(QRect(11, 49, 139, 22));
        m_MinHelixWidthRatioText->setWordWrap(false);
        m_MinHelixWidthRatio = new QLineEdit(m_SecondaryStructureOptions);
        m_MinHelixWidthRatio->setObjectName(QString::fromUtf8("m_MinHelixWidthRatio"));
        m_MinHelixWidthRatio->setGeometry(QRect(156, 49, 49, 22));
        m_MinSheetWidthRatioText = new QLabel(m_SecondaryStructureOptions);
        m_MinSheetWidthRatioText->setObjectName(QString::fromUtf8("m_MinSheetWidthRatioText"));
        m_MinSheetWidthRatioText->setGeometry(QRect(211, 49, 145, 22));
        m_MinSheetWidthRatioText->setWordWrap(false);
        m_MinSheetWidthRatio = new QLineEdit(m_SecondaryStructureOptions);
        m_MinSheetWidthRatio->setObjectName(QString::fromUtf8("m_MinSheetWidthRatio"));
        m_MinSheetWidthRatio->setGeometry(QRect(362, 49, 50, 22));
        m_MaxSheetWidthRatio = new QLineEdit(m_SecondaryStructureOptions);
        m_MaxSheetWidthRatio->setObjectName(QString::fromUtf8("m_MaxSheetWidthRatio"));
        m_MaxSheetWidthRatio->setGeometry(QRect(362, 77, 50, 22));
        m_MaxSheetWidthRatioText = new QLabel(m_SecondaryStructureOptions);
        m_MaxSheetWidthRatioText->setObjectName(QString::fromUtf8("m_MaxSheetWidthRatioText"));
        m_MaxSheetWidthRatioText->setGeometry(QRect(211, 77, 145, 22));
        m_MaxSheetWidthRatioText->setWordWrap(false);
        m_MaxHelixWidthRatio = new QLineEdit(m_SecondaryStructureOptions);
        m_MaxHelixWidthRatio->setObjectName(QString::fromUtf8("m_MaxHelixWidthRatio"));
        m_MaxHelixWidthRatio->setGeometry(QRect(156, 77, 49, 22));
        m_MaxHelixWidthRatioText = new QLabel(m_SecondaryStructureOptions);
        m_MaxHelixWidthRatioText->setObjectName(QString::fromUtf8("m_MaxHelixWidthRatioText"));
        m_MaxHelixWidthRatioText->setGeometry(QRect(11, 77, 139, 22));
        m_MaxHelixWidthRatioText->setWordWrap(false);
        m_MinHelixLengthText = new QLabel(m_SecondaryStructureOptions);
        m_MinHelixLengthText->setObjectName(QString::fromUtf8("m_MinHelixLengthText"));
        m_MinHelixLengthText->setGeometry(QRect(11, 105, 139, 22));
        m_MinHelixLengthText->setWordWrap(false);
        m_SheetExtendText = new QLabel(m_SecondaryStructureOptions);
        m_SheetExtendText->setObjectName(QString::fromUtf8("m_SheetExtendText"));
        m_SheetExtendText->setGeometry(QRect(211, 105, 145, 22));
        m_SheetExtendText->setWordWrap(false);
        m_SheetExtend = new QLineEdit(m_SecondaryStructureOptions);
        m_SheetExtend->setObjectName(QString::fromUtf8("m_SheetExtend"));
        m_SheetExtend->setGeometry(QRect(362, 105, 50, 22));
        m_HelixWidth = new QLineEdit(m_SecondaryStructureOptions);
        m_HelixWidth->setObjectName(QString::fromUtf8("m_HelixWidth"));
        m_HelixWidth->setGeometry(QRect(156, 21, 49, 22));
        m_MinHelixLength = new QLineEdit(m_SecondaryStructureOptions);
        m_MinHelixLength->setObjectName(QString::fromUtf8("m_MinHelixLength"));
        m_MinHelixLength->setGeometry(QRect(156, 105, 49, 22));
        m_Threshold = new QLineEdit(m_SecondaryStructureOptions);
        m_Threshold->setObjectName(QString::fromUtf8("m_Threshold"));
        m_Threshold->setEnabled(false);
        m_Threshold->setGeometry(QRect(155, 135, 48, 21));
        m_ThresholdCheck = new QRadioButton(m_SecondaryStructureOptions);
        m_ThresholdCheck->setObjectName(QString::fromUtf8("m_ThresholdCheck"));
        m_ThresholdCheck->setGeometry(QRect(10, 130, 140, 31));

        gridLayout11->addWidget(m_SecondaryStructureOptions, 0, 0, 1, 1);

        m_OptionsStack->addWidget(WStackPage, 3);

        gridLayout->addWidget(m_OptionsStack, 2, 0, 1, 3);

        m_RunButton = new QPushButton(SegmentationDialogBase);
        m_RunButton->setObjectName(QString::fromUtf8("m_RunButton"));

        gridLayout->addWidget(m_RunButton, 3, 0, 1, 2);

        m_CancelButton = new QPushButton(SegmentationDialogBase);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));

        gridLayout->addWidget(m_CancelButton, 3, 2, 1, 1);

        m_ExecutionLocation = new Q3ButtonGroup(SegmentationDialogBase);
        m_ExecutionLocation->setObjectName(QString::fromUtf8("m_ExecutionLocation"));
        m_ExecutionLocation->setColumnLayout(0, Qt::Vertical);
        m_ExecutionLocation->layout()->setSpacing(6);
        m_ExecutionLocation->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout12 = new QGridLayout();
        QBoxLayout *boxlayout3 = qobject_cast<QBoxLayout *>(m_ExecutionLocation->layout());
        if (boxlayout3)
            boxlayout3->addLayout(gridLayout12);
        gridLayout12->setAlignment(Qt::AlignTop);
        gridLayout12->setObjectName(QString::fromUtf8("gridLayout12"));
        m_LocalSegmentationButton = new QRadioButton(m_ExecutionLocation);
        m_LocalSegmentationButton->setObjectName(QString::fromUtf8("m_LocalSegmentationButton"));
        m_LocalSegmentationButton->setChecked(true);

        gridLayout12->addWidget(m_LocalSegmentationButton, 0, 0, 1, 1);

        m_RemoteSegmentationButton = new QRadioButton(m_ExecutionLocation);
        m_RemoteSegmentationButton->setObjectName(QString::fromUtf8("m_RemoteSegmentationButton"));

        gridLayout12->addWidget(m_RemoteSegmentationButton, 1, 0, 1, 1);


        gridLayout->addWidget(m_ExecutionLocation, 0, 0, 1, 1);

        m_RemoteSegmentationGroup = new Q3GroupBox(SegmentationDialogBase);
        m_RemoteSegmentationGroup->setObjectName(QString::fromUtf8("m_RemoteSegmentationGroup"));
        m_RemoteSegmentationGroup->setEnabled(false);
        m_RemoteSegmentationGroup->setColumnLayout(0, Qt::Vertical);
        m_RemoteSegmentationGroup->layout()->setSpacing(6);
        m_RemoteSegmentationGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout13 = new QGridLayout();
        QBoxLayout *boxlayout4 = qobject_cast<QBoxLayout *>(m_RemoteSegmentationGroup->layout());
        if (boxlayout4)
            boxlayout4->addLayout(gridLayout13);
        gridLayout13->setAlignment(Qt::AlignTop);
        gridLayout13->setObjectName(QString::fromUtf8("gridLayout13"));
        m_RemoteSegmentationHostnameText = new QLabel(m_RemoteSegmentationGroup);
        m_RemoteSegmentationHostnameText->setObjectName(QString::fromUtf8("m_RemoteSegmentationHostnameText"));
        m_RemoteSegmentationHostnameText->setWordWrap(false);

        gridLayout13->addWidget(m_RemoteSegmentationHostnameText, 0, 0, 1, 1);

        m_RemoteSegmentationHostname = new QLineEdit(m_RemoteSegmentationGroup);
        m_RemoteSegmentationHostname->setObjectName(QString::fromUtf8("m_RemoteSegmentationHostname"));

        gridLayout13->addWidget(m_RemoteSegmentationHostname, 0, 1, 1, 2);

        m_RemoteSegmentationPortText = new QLabel(m_RemoteSegmentationGroup);
        m_RemoteSegmentationPortText->setObjectName(QString::fromUtf8("m_RemoteSegmentationPortText"));
        m_RemoteSegmentationPortText->setWordWrap(false);

        gridLayout13->addWidget(m_RemoteSegmentationPortText, 1, 0, 1, 1);

        m_RemoteSegmentationPort = new QLineEdit(m_RemoteSegmentationGroup);
        m_RemoteSegmentationPort->setObjectName(QString::fromUtf8("m_RemoteSegmentationPort"));

        gridLayout13->addWidget(m_RemoteSegmentationPort, 1, 1, 1, 2);

        m_RemoteSegmentationFilenameText = new QLabel(m_RemoteSegmentationGroup);
        m_RemoteSegmentationFilenameText->setObjectName(QString::fromUtf8("m_RemoteSegmentationFilenameText"));
        m_RemoteSegmentationFilenameText->setWordWrap(false);

        gridLayout13->addWidget(m_RemoteSegmentationFilenameText, 2, 0, 1, 2);

        m_RemoteSegmentationFilename = new QLineEdit(m_RemoteSegmentationGroup);
        m_RemoteSegmentationFilename->setObjectName(QString::fromUtf8("m_RemoteSegmentationFilename"));

        gridLayout13->addWidget(m_RemoteSegmentationFilename, 2, 2, 1, 1);


        gridLayout->addWidget(m_RemoteSegmentationGroup, 0, 1, 1, 2);

        m_SegTypeText = new QLabel(SegmentationDialogBase);
        m_SegTypeText->setObjectName(QString::fromUtf8("m_SegTypeText"));
        m_SegTypeText->setWordWrap(false);

        gridLayout->addWidget(m_SegTypeText, 1, 0, 1, 1);

        m_SegTypeSelection = new QComboBox(SegmentationDialogBase);
        m_SegTypeSelection->setObjectName(QString::fromUtf8("m_SegTypeSelection"));

        gridLayout->addWidget(m_SegTypeSelection, 1, 1, 1, 2);

        QWidget::setTabOrder(m_LocalSegmentationButton, m_RemoteSegmentationHostname);
        QWidget::setTabOrder(m_RemoteSegmentationHostname, m_RemoteSegmentationPort);
        QWidget::setTabOrder(m_RemoteSegmentationPort, m_RemoteSegmentationFilename);
        QWidget::setTabOrder(m_RemoteSegmentationFilename, m_SegTypeSelection);
        QWidget::setTabOrder(m_SegTypeSelection, m_CapsidLayerType);
        QWidget::setTabOrder(m_CapsidLayerType, m_TLowEditType3);
        QWidget::setTabOrder(m_TLowEditType3, m_3FoldEditType3);
        QWidget::setTabOrder(m_3FoldEditType3, m_5FoldEditType3);
        QWidget::setTabOrder(m_5FoldEditType3, m_6FoldEditType3);
        QWidget::setTabOrder(m_6FoldEditType3, m_SmallRadiusEditType3);
        QWidget::setTabOrder(m_SmallRadiusEditType3, m_LargeRadiusEditType3);
        QWidget::setTabOrder(m_LargeRadiusEditType3, m_FoldNumEdit);
        QWidget::setTabOrder(m_FoldNumEdit, m_HNumEdit);
        QWidget::setTabOrder(m_HNumEdit, m_KNumEdit);
        QWidget::setTabOrder(m_KNumEdit, m_3FoldEdit);
        QWidget::setTabOrder(m_3FoldEdit, m_5FoldEdit);
        QWidget::setTabOrder(m_5FoldEdit, m_6FoldEdit);
        QWidget::setTabOrder(m_6FoldEdit, m_InitRadiusEdit);
        QWidget::setTabOrder(m_InitRadiusEdit, m_HelixWidth);
        QWidget::setTabOrder(m_HelixWidth, m_MinHelixWidthRatio);
        QWidget::setTabOrder(m_MinHelixWidthRatio, m_MaxHelixWidthRatio);
        QWidget::setTabOrder(m_MaxHelixWidthRatio, m_MinHelixLength);
        QWidget::setTabOrder(m_MinHelixLength, m_SheetWidth);
        QWidget::setTabOrder(m_SheetWidth, m_MinSheetWidthRatio);
        QWidget::setTabOrder(m_MinSheetWidthRatio, m_MaxSheetWidthRatio);
        QWidget::setTabOrder(m_MaxSheetWidthRatio, m_SheetExtend);
        QWidget::setTabOrder(m_SheetExtend, m_RunButton);
        QWidget::setTabOrder(m_RunButton, m_CancelButton);
        QWidget::setTabOrder(m_CancelButton, m_TLowEditType0);
        QWidget::setTabOrder(m_TLowEditType0, m_TLowEditType1);
        QWidget::setTabOrder(m_TLowEditType1, m_X0EditType1);
        QWidget::setTabOrder(m_X0EditType1, m_Y0EditType1);
        QWidget::setTabOrder(m_Y0EditType1, m_X1EditType1);
        QWidget::setTabOrder(m_X1EditType1, m_Y1EditType1);
        QWidget::setTabOrder(m_Y1EditType1, m_Z1EditType1);
        QWidget::setTabOrder(m_Z1EditType1, m_Z0EditType1);
        QWidget::setTabOrder(m_Z0EditType1, m_LargeRadiusEditType2);
        QWidget::setTabOrder(m_LargeRadiusEditType2, m_SmallRadiusEditType2);
        QWidget::setTabOrder(m_SmallRadiusEditType2, m_TLowEditType2);

        retranslateUi(SegmentationDialogBase);
        QObject::connect(m_RunButton, SIGNAL(clicked()), SegmentationDialogBase, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), SegmentationDialogBase, SLOT(reject()));
        QObject::connect(m_SegTypeSelection, SIGNAL(activated(int)), m_OptionsStack, SLOT(raiseWidget(int)));
        QObject::connect(m_CapsidLayerType, SIGNAL(activated(int)), m_CapsidOptionsStack, SLOT(raiseWidget(int)));
        QObject::connect(m_ExecutionLocation, SIGNAL(clicked(int)), SegmentationDialogBase, SLOT(changeExecutionLocationSlot(int)));
        QObject::connect(m_ThresholdCheck, SIGNAL(toggled(bool)), m_Threshold, SLOT(setEnabled(bool)));

        QMetaObject::connectSlotsByName(SegmentationDialogBase);
    } // setupUi

    void retranslateUi(QDialog *SegmentationDialogBase)
    {
        SegmentationDialogBase->setWindowTitle(QApplication::translate("SegmentationDialogBase", "Segment Virus Map", 0, QApplication::UnicodeUTF8));
        m_CapsidOptions->setTitle(QApplication::translate("SegmentationDialogBase", "Capsid Segmentation Options", 0, QApplication::UnicodeUTF8));
        m_Z0TextType0->setText(QApplication::translate("SegmentationDialogBase", "z", 0, QApplication::UnicodeUTF8));
        m_Y0TextType0->setText(QApplication::translate("SegmentationDialogBase", "y", 0, QApplication::UnicodeUTF8));
        m_X0TextType0->setText(QApplication::translate("SegmentationDialogBase", "x", 0, QApplication::UnicodeUTF8));
        m_TLowTextType0->setText(QApplication::translate("SegmentationDialogBase", "Lowest threshold for segmentation (0-255)", 0, QApplication::UnicodeUTF8));
        m_SeedPoint0TextType0->setText(QApplication::translate("SegmentationDialogBase", "Seed point of capsid:", 0, QApplication::UnicodeUTF8));
        m_RunDiffusionType0->setText(QApplication::translate("SegmentationDialogBase", "Run Diffusion", 0, QApplication::UnicodeUTF8));
        m_TLowTextType1->setText(QApplication::translate("SegmentationDialogBase", "Lowest threshold for segmentation (0-255)", 0, QApplication::UnicodeUTF8));
        m_SeedPoint0TextType1->setText(QApplication::translate("SegmentationDialogBase", "Seed point of genomic structures:", 0, QApplication::UnicodeUTF8));
        m_X0TextType1->setText(QApplication::translate("SegmentationDialogBase", "x", 0, QApplication::UnicodeUTF8));
        m_Y0TextType1->setText(QApplication::translate("SegmentationDialogBase", "y", 0, QApplication::UnicodeUTF8));
        m_Z0TextType1->setText(QApplication::translate("SegmentationDialogBase", "z", 0, QApplication::UnicodeUTF8));
        m_Y1TextType1->setText(QApplication::translate("SegmentationDialogBase", "y", 0, QApplication::UnicodeUTF8));
        m_SeedPoint1TextType1->setText(QApplication::translate("SegmentationDialogBase", "Seed point of capsid:", 0, QApplication::UnicodeUTF8));
        m_X1TextType1->setText(QApplication::translate("SegmentationDialogBase", "x", 0, QApplication::UnicodeUTF8));
        m_Z1TextType1->setText(QApplication::translate("SegmentationDialogBase", "z", 0, QApplication::UnicodeUTF8));
        m_RunDiffusionType1->setText(QApplication::translate("SegmentationDialogBase", "Run Diffusion", 0, QApplication::UnicodeUTF8));
        m_RadiiTextType2->setText(QApplication::translate("SegmentationDialogBase", "Estimated Radii:", 0, QApplication::UnicodeUTF8));
        m_SmallRadiusTextType2->setText(QApplication::translate("SegmentationDialogBase", "Small", 0, QApplication::UnicodeUTF8));
        m_LargeRadiusTextType2->setText(QApplication::translate("SegmentationDialogBase", "Large", 0, QApplication::UnicodeUTF8));
        m_TLowTextType2->setText(QApplication::translate("SegmentationDialogBase", "Lowest threshold for segmentation (0-255)", 0, QApplication::UnicodeUTF8));
        m_OuterLayerTextType3->setText(QApplication::translate("SegmentationDialogBase", "Outer Layer:", 0, QApplication::UnicodeUTF8));
        m_RadiiTextType3->setText(QApplication::translate("SegmentationDialogBase", "Estimated Radii:", 0, QApplication::UnicodeUTF8));
        m_SmallRadiusTextType3->setText(QApplication::translate("SegmentationDialogBase", "Small", 0, QApplication::UnicodeUTF8));
        m_LargeRadiusTextType3->setText(QApplication::translate("SegmentationDialogBase", "Large", 0, QApplication::UnicodeUTF8));
        m_3FoldTextType3->setText(QApplication::translate("SegmentationDialogBase", "3-fold", 0, QApplication::UnicodeUTF8));
        m_5FoldTextType3->setText(QApplication::translate("SegmentationDialogBase", "5-fold", 0, QApplication::UnicodeUTF8));
        m_6FoldTextType3->setText(QApplication::translate("SegmentationDialogBase", "6-fold", 0, QApplication::UnicodeUTF8));
        m_TLowTextType3->setText(QApplication::translate("SegmentationDialogBase", "Lowest threshold for segmentation (0-255)", 0, QApplication::UnicodeUTF8));
        m_CapsidLayerTypeText->setText(QApplication::translate("SegmentationDialogBase", "Capsid Layer Type", 0, QApplication::UnicodeUTF8));
        m_CapsidLayerType->clear();
        m_CapsidLayerType->insertItems(0, QStringList()
         << QApplication::translate("SegmentationDialogBase", "Single Capsid, distinct", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("SegmentationDialogBase", "Single Capsid", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("SegmentationDialogBase", "Double Capsid, initial segmentation", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("SegmentationDialogBase", "Double Capsid, refined segmentation", 0, QApplication::UnicodeUTF8)
        );
        m_MonomerOptions->setTitle(QApplication::translate("SegmentationDialogBase", "Monomer Segmentation Options", 0, QApplication::UnicodeUTF8));
        m_FoldNumText->setText(QApplication::translate("SegmentationDialogBase", "Number of symmetry folding", 0, QApplication::UnicodeUTF8));
        m_SubunitOptions->setTitle(QApplication::translate("SegmentationDialogBase", "Subunit Segmentation Options", 0, QApplication::UnicodeUTF8));
        m_HNumText->setText(QApplication::translate("SegmentationDialogBase", "h-num", 0, QApplication::UnicodeUTF8));
        m_KNumText->setText(QApplication::translate("SegmentationDialogBase", "k-num", 0, QApplication::UnicodeUTF8));
        m_3FoldText->setText(QApplication::translate("SegmentationDialogBase", "3-fold", 0, QApplication::UnicodeUTF8));
        m_5FoldText->setText(QApplication::translate("SegmentationDialogBase", "5-fold", 0, QApplication::UnicodeUTF8));
        m_6FoldText->setText(QApplication::translate("SegmentationDialogBase", "6-fold", 0, QApplication::UnicodeUTF8));
        m_InitRadiusText->setText(QApplication::translate("SegmentationDialogBase", "Initial Radius:", 0, QApplication::UnicodeUTF8));
        m_SecondaryStructureOptions->setTitle(QApplication::translate("SegmentationDialogBase", "Secondary Structure Detection Options", 0, QApplication::UnicodeUTF8));
        m_HelixWidthText->setText(QApplication::translate("SegmentationDialogBase", "Helix Width:", 0, QApplication::UnicodeUTF8));
        m_SheetWidthText->setText(QApplication::translate("SegmentationDialogBase", "Sheet Width:", 0, QApplication::UnicodeUTF8));
        m_SheetWidth->setText(QApplication::translate("SegmentationDialogBase", "2.6", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_SheetWidth->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the thickness of typical sheets (in pixels)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_MinHelixWidthRatioText->setText(QApplication::translate("SegmentationDialogBase", "Min Helix Width Ratio:", 0, QApplication::UnicodeUTF8));
        m_MinHelixWidthRatio->setText(QApplication::translate("SegmentationDialogBase", "0.001", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_MinHelixWidthRatio->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the low ratio of thickness (0~1)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_MinSheetWidthRatioText->setText(QApplication::translate("SegmentationDialogBase", "Min Sheet Width Ratio:", 0, QApplication::UnicodeUTF8));
        m_MinSheetWidthRatio->setText(QApplication::translate("SegmentationDialogBase", "0.01", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_MinSheetWidthRatio->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the low ratio of thickness (0~1)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_MaxSheetWidthRatio->setText(QApplication::translate("SegmentationDialogBase", "8.0", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_MaxSheetWidthRatio->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the high ratio of thickness (> 1)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_MaxSheetWidthRatioText->setText(QApplication::translate("SegmentationDialogBase", "Max Sheet Width Ratio:", 0, QApplication::UnicodeUTF8));
        m_MaxHelixWidthRatio->setText(QApplication::translate("SegmentationDialogBase", "8.0", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_MaxHelixWidthRatio->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the high fofthickness (> 1)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_MaxHelixWidthRatioText->setText(QApplication::translate("SegmentationDialogBase", "Max Helix Width Ratio:", 0, QApplication::UnicodeUTF8));
        m_MinHelixLengthText->setText(QApplication::translate("SegmentationDialogBase", "Min Helix Length:", 0, QApplication::UnicodeUTF8));
        m_SheetExtendText->setText(QApplication::translate("SegmentationDialogBase", "Sheet Extend:", 0, QApplication::UnicodeUTF8));
        m_SheetExtend->setText(QApplication::translate("SegmentationDialogBase", "1.5", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_SheetExtend->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the extension ratio of sheets (1~2)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_HelixWidth->setText(QApplication::translate("SegmentationDialogBase", "2.5", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_HelixWidth->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the thickness of typical helices (in pixels)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_MinHelixLength->setText(QApplication::translate("SegmentationDialogBase", "1.8", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        m_MinHelixLength->setProperty("toolTip", QVariant(QApplication::translate("SegmentationDialogBase", "the shortest helices (in pixels)", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_TOOLTIP
        m_Threshold->setText(QApplication::translate("SegmentationDialogBase", "125", 0, QApplication::UnicodeUTF8));
        m_ThresholdCheck->setText(QApplication::translate("SegmentationDialogBase", "Threshold (0~255)", 0, QApplication::UnicodeUTF8));
        m_RunButton->setText(QApplication::translate("SegmentationDialogBase", "Run", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("SegmentationDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_ExecutionLocation->setTitle(QString());
        m_LocalSegmentationButton->setText(QApplication::translate("SegmentationDialogBase", "Local Segmentation", 0, QApplication::UnicodeUTF8));
        m_RemoteSegmentationButton->setText(QApplication::translate("SegmentationDialogBase", "Remote Segmentation", 0, QApplication::UnicodeUTF8));
        m_RemoteSegmentationGroup->setTitle(QApplication::translate("SegmentationDialogBase", "Remote Segmentation Host", 0, QApplication::UnicodeUTF8));
        m_RemoteSegmentationHostnameText->setText(QApplication::translate("SegmentationDialogBase", "Hostname:", 0, QApplication::UnicodeUTF8));
        m_RemoteSegmentationPortText->setText(QApplication::translate("SegmentationDialogBase", "Port:", 0, QApplication::UnicodeUTF8));
        m_RemoteSegmentationFilenameText->setText(QApplication::translate("SegmentationDialogBase", "Remote File:", 0, QApplication::UnicodeUTF8));
        m_SegTypeText->setText(QApplication::translate("SegmentationDialogBase", "Segmentation Type", 0, QApplication::UnicodeUTF8));
        m_SegTypeSelection->clear();
        m_SegTypeSelection->insertItems(0, QStringList()
         << QApplication::translate("SegmentationDialogBase", "Capsid", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("SegmentationDialogBase", "Monomer", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("SegmentationDialogBase", "Subunit", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("SegmentationDialogBase", "Secondary Structure Detection", 0, QApplication::UnicodeUTF8)
        );
    } // retranslateUi

};

namespace Ui {
    class SegmentationDialogBase: public Ui_SegmentationDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class SegmentationDialogBase : public QDialog, public Ui::SegmentationDialogBase
{
    Q_OBJECT

public:
    SegmentationDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~SegmentationDialogBase();

protected slots:
    virtual void languageChange();

    virtual void changeExecutionLocationSlot( int );


};

#endif // SEGMENTATIONDIALOGBASE_H
