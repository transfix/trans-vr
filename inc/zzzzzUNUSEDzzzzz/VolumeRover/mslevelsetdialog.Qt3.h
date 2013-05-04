#ifndef MSLEVELSETDIALOG_H
#define MSLEVELSETDIALOG_H

#include <qvariant.h>


#include <Qt3Support/Q3GroupBox>
#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include "MSLevelSet/levelset3D.h"

QT_BEGIN_NAMESPACE

class Ui_MSLevelSetDialog
{
public:
    QGridLayout *gridLayout;
    QSpacerItem *spacer5;
    QHBoxLayout *hboxLayout;
    QPushButton *m_CancelButton;
    QPushButton *m_RunButton;
    Q3GroupBox *m_CUDAParamsGroupBox;
    QGridLayout *gridLayout1;
    QLabel *m_BlockDimText;
    QComboBox *m_BlockDimComboBox;
    QLabel *m_SubvolDimText;
    QLineEdit *m_SubvolDimEdit;
    QLineEdit *m_BlockDimEdit;
    Q3GroupBox *m_LevelSetParamsGroupBox;
    QGridLayout *gridLayout2;
    QGridLayout *gridLayout3;
    QLabel *m_Lambda2Text;
    QLineEdit *m_Lambda1Edit;
    QLabel *m_DeltaTText;
    QLineEdit *m_MuEdit;
    QLabel *m_MuText;
    QLabel *m_NuText;
    QLabel *m_EpsilonText;
    QLineEdit *m_Lambda2Edit;
    QLineEdit *m_NuEdit;
    QLineEdit *m_DeltaTEdit;
    QLineEdit *m_EpsilonEdit;
    QLabel *m_Lambda1Text;
    QHBoxLayout *hboxLayout1;
    QLabel *m_MaxSolverIterText;
    QSpacerItem *spacer1;
    QLineEdit *m_MaxSolverIterEdit;
    QHBoxLayout *hboxLayout2;
    QLabel *m_MaxMedianIterText;
    QSpacerItem *spacer3;
    QLineEdit *m_MaxMedianIterEdit;
    QHBoxLayout *hboxLayout3;
    QLabel *m_MedianTolText;
    QSpacerItem *spacer4;
    QLineEdit *m_MedianTolEdit;
    QHBoxLayout *hboxLayout4;
    QLabel *m_DTInitText;
    QComboBox *m_DTInitComboBox;
    QHBoxLayout *hboxLayout5;
    QLabel *m_DTWidthText;
    QSpacerItem *spacer2;
    QLineEdit *m_DTWidthEdit;
    QHBoxLayout *hboxLayout6;
    QSpacerItem *spacer6;
    QLabel *m_EllipsoidPowerText;
    QLineEdit *m_EllipsoidPowerEdit;
    QLabel *m_BBoxOffsetText;
    QLineEdit *m_BBoxOffsetEdit;
    QCheckBox *m_Preview;

    void setupUi(QDialog *MSLevelSetDialog)
    {
        if (MSLevelSetDialog->objectName().isEmpty())
            MSLevelSetDialog->setObjectName(QString::fromUtf8("MSLevelSetDialog"));
        MSLevelSetDialog->resize(449, 427);
        gridLayout = new QGridLayout(MSLevelSetDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setResizeMode(QGridLayout::Fixed);
        spacer5 = new QSpacerItem(60, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(spacer5, 2, 1, 1, 1);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        m_CancelButton = new QPushButton(MSLevelSetDialog);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));

        hboxLayout->addWidget(m_CancelButton);

        m_RunButton = new QPushButton(MSLevelSetDialog);
        m_RunButton->setObjectName(QString::fromUtf8("m_RunButton"));

        hboxLayout->addWidget(m_RunButton);


        gridLayout->addLayout(hboxLayout, 2, 2, 1, 1);

        m_CUDAParamsGroupBox = new Q3GroupBox(MSLevelSetDialog);
        m_CUDAParamsGroupBox->setObjectName(QString::fromUtf8("m_CUDAParamsGroupBox"));
        m_CUDAParamsGroupBox->setColumnLayout(0, Qt::Vertical);
        m_CUDAParamsGroupBox->layout()->setSpacing(6);
        m_CUDAParamsGroupBox->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_CUDAParamsGroupBox->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_BlockDimText = new QLabel(m_CUDAParamsGroupBox);
        m_BlockDimText->setObjectName(QString::fromUtf8("m_BlockDimText"));
        m_BlockDimText->setWordWrap(false);

        gridLayout1->addWidget(m_BlockDimText, 1, 0, 1, 1);

        m_BlockDimComboBox = new QComboBox(m_CUDAParamsGroupBox);
        m_BlockDimComboBox->setObjectName(QString::fromUtf8("m_BlockDimComboBox"));

        gridLayout1->addWidget(m_BlockDimComboBox, 1, 1, 1, 1);

        m_SubvolDimText = new QLabel(m_CUDAParamsGroupBox);
        m_SubvolDimText->setObjectName(QString::fromUtf8("m_SubvolDimText"));
        m_SubvolDimText->setWordWrap(false);

        gridLayout1->addWidget(m_SubvolDimText, 0, 0, 1, 2);

        m_SubvolDimEdit = new QLineEdit(m_CUDAParamsGroupBox);
        m_SubvolDimEdit->setObjectName(QString::fromUtf8("m_SubvolDimEdit"));

        gridLayout1->addWidget(m_SubvolDimEdit, 0, 2, 1, 1);

        m_BlockDimEdit = new QLineEdit(m_CUDAParamsGroupBox);
        m_BlockDimEdit->setObjectName(QString::fromUtf8("m_BlockDimEdit"));

        gridLayout1->addWidget(m_BlockDimEdit, 1, 2, 1, 1);


        gridLayout->addWidget(m_CUDAParamsGroupBox, 1, 0, 1, 3);

        m_LevelSetParamsGroupBox = new Q3GroupBox(MSLevelSetDialog);
        m_LevelSetParamsGroupBox->setObjectName(QString::fromUtf8("m_LevelSetParamsGroupBox"));
        m_LevelSetParamsGroupBox->setColumnLayout(0, Qt::Vertical);
        m_LevelSetParamsGroupBox->layout()->setSpacing(6);
        m_LevelSetParamsGroupBox->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_LevelSetParamsGroupBox->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        gridLayout3 = new QGridLayout();
        gridLayout3->setSpacing(6);
        gridLayout3->setObjectName(QString::fromUtf8("gridLayout3"));
        m_Lambda2Text = new QLabel(m_LevelSetParamsGroupBox);
        m_Lambda2Text->setObjectName(QString::fromUtf8("m_Lambda2Text"));
        m_Lambda2Text->setWordWrap(false);

        gridLayout3->addWidget(m_Lambda2Text, 1, 0, 1, 1);

        m_Lambda1Edit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_Lambda1Edit->setObjectName(QString::fromUtf8("m_Lambda1Edit"));

        gridLayout3->addWidget(m_Lambda1Edit, 0, 1, 1, 1);

        m_DeltaTText = new QLabel(m_LevelSetParamsGroupBox);
        m_DeltaTText->setObjectName(QString::fromUtf8("m_DeltaTText"));
        m_DeltaTText->setAlignment(Qt::AlignVCenter);
        m_DeltaTText->setWordWrap(false);

        gridLayout3->addWidget(m_DeltaTText, 1, 4, 1, 1);

        m_MuEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_MuEdit->setObjectName(QString::fromUtf8("m_MuEdit"));

        gridLayout3->addWidget(m_MuEdit, 0, 3, 1, 1);

        m_MuText = new QLabel(m_LevelSetParamsGroupBox);
        m_MuText->setObjectName(QString::fromUtf8("m_MuText"));
        m_MuText->setWordWrap(false);

        gridLayout3->addWidget(m_MuText, 0, 2, 1, 1);

        m_NuText = new QLabel(m_LevelSetParamsGroupBox);
        m_NuText->setObjectName(QString::fromUtf8("m_NuText"));
        m_NuText->setWordWrap(false);

        gridLayout3->addWidget(m_NuText, 1, 2, 1, 1);

        m_EpsilonText = new QLabel(m_LevelSetParamsGroupBox);
        m_EpsilonText->setObjectName(QString::fromUtf8("m_EpsilonText"));
        m_EpsilonText->setWordWrap(false);

        gridLayout3->addWidget(m_EpsilonText, 0, 4, 1, 1);

        m_Lambda2Edit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_Lambda2Edit->setObjectName(QString::fromUtf8("m_Lambda2Edit"));

        gridLayout3->addWidget(m_Lambda2Edit, 1, 1, 1, 1);

        m_NuEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_NuEdit->setObjectName(QString::fromUtf8("m_NuEdit"));

        gridLayout3->addWidget(m_NuEdit, 1, 3, 1, 1);

        m_DeltaTEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_DeltaTEdit->setObjectName(QString::fromUtf8("m_DeltaTEdit"));

        gridLayout3->addWidget(m_DeltaTEdit, 1, 5, 1, 1);

        m_EpsilonEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_EpsilonEdit->setObjectName(QString::fromUtf8("m_EpsilonEdit"));

        gridLayout3->addWidget(m_EpsilonEdit, 0, 5, 1, 1);

        m_Lambda1Text = new QLabel(m_LevelSetParamsGroupBox);
        m_Lambda1Text->setObjectName(QString::fromUtf8("m_Lambda1Text"));
        m_Lambda1Text->setWordWrap(false);

        gridLayout3->addWidget(m_Lambda1Text, 0, 0, 1, 1);


        gridLayout2->addLayout(gridLayout3, 0, 0, 1, 1);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        m_MaxSolverIterText = new QLabel(m_LevelSetParamsGroupBox);
        m_MaxSolverIterText->setObjectName(QString::fromUtf8("m_MaxSolverIterText"));
        m_MaxSolverIterText->setWordWrap(false);

        hboxLayout1->addWidget(m_MaxSolverIterText);

        spacer1 = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout1->addItem(spacer1);

        m_MaxSolverIterEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_MaxSolverIterEdit->setObjectName(QString::fromUtf8("m_MaxSolverIterEdit"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(0), static_cast<QSizePolicy::Policy>(0));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_MaxSolverIterEdit->sizePolicy().hasHeightForWidth());
        m_MaxSolverIterEdit->setSizePolicy(sizePolicy);

        hboxLayout1->addWidget(m_MaxSolverIterEdit);


        gridLayout2->addLayout(hboxLayout1, 1, 0, 1, 1);

        hboxLayout2 = new QHBoxLayout();
        hboxLayout2->setSpacing(6);
        hboxLayout2->setObjectName(QString::fromUtf8("hboxLayout2"));
        m_MaxMedianIterText = new QLabel(m_LevelSetParamsGroupBox);
        m_MaxMedianIterText->setObjectName(QString::fromUtf8("m_MaxMedianIterText"));
        m_MaxMedianIterText->setAlignment(Qt::AlignVCenter);
        m_MaxMedianIterText->setWordWrap(false);

        hboxLayout2->addWidget(m_MaxMedianIterText);

        spacer3 = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout2->addItem(spacer3);

        m_MaxMedianIterEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_MaxMedianIterEdit->setObjectName(QString::fromUtf8("m_MaxMedianIterEdit"));
        sizePolicy.setHeightForWidth(m_MaxMedianIterEdit->sizePolicy().hasHeightForWidth());
        m_MaxMedianIterEdit->setSizePolicy(sizePolicy);

        hboxLayout2->addWidget(m_MaxMedianIterEdit);


        gridLayout2->addLayout(hboxLayout2, 3, 0, 1, 1);

        hboxLayout3 = new QHBoxLayout();
        hboxLayout3->setSpacing(6);
        hboxLayout3->setObjectName(QString::fromUtf8("hboxLayout3"));
        m_MedianTolText = new QLabel(m_LevelSetParamsGroupBox);
        m_MedianTolText->setObjectName(QString::fromUtf8("m_MedianTolText"));
        m_MedianTolText->setAlignment(Qt::AlignVCenter);
        m_MedianTolText->setWordWrap(false);

        hboxLayout3->addWidget(m_MedianTolText);

        spacer4 = new QSpacerItem(16, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout3->addItem(spacer4);

        m_MedianTolEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_MedianTolEdit->setObjectName(QString::fromUtf8("m_MedianTolEdit"));
        sizePolicy.setHeightForWidth(m_MedianTolEdit->sizePolicy().hasHeightForWidth());
        m_MedianTolEdit->setSizePolicy(sizePolicy);

        hboxLayout3->addWidget(m_MedianTolEdit);


        gridLayout2->addLayout(hboxLayout3, 4, 0, 1, 1);

        hboxLayout4 = new QHBoxLayout();
        hboxLayout4->setSpacing(6);
        hboxLayout4->setObjectName(QString::fromUtf8("hboxLayout4"));
        m_DTInitText = new QLabel(m_LevelSetParamsGroupBox);
        m_DTInitText->setObjectName(QString::fromUtf8("m_DTInitText"));
        m_DTInitText->setWordWrap(false);

        hboxLayout4->addWidget(m_DTInitText);

        m_DTInitComboBox = new QComboBox(m_LevelSetParamsGroupBox);
        m_DTInitComboBox->setObjectName(QString::fromUtf8("m_DTInitComboBox"));

        hboxLayout4->addWidget(m_DTInitComboBox);


        gridLayout2->addLayout(hboxLayout4, 5, 0, 1, 1);

        hboxLayout5 = new QHBoxLayout();
        hboxLayout5->setSpacing(6);
        hboxLayout5->setObjectName(QString::fromUtf8("hboxLayout5"));
        m_DTWidthText = new QLabel(m_LevelSetParamsGroupBox);
        m_DTWidthText->setObjectName(QString::fromUtf8("m_DTWidthText"));
        m_DTWidthText->setWordWrap(false);

        hboxLayout5->addWidget(m_DTWidthText);

        spacer2 = new QSpacerItem(180, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout5->addItem(spacer2);

        m_DTWidthEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_DTWidthEdit->setObjectName(QString::fromUtf8("m_DTWidthEdit"));
        sizePolicy.setHeightForWidth(m_DTWidthEdit->sizePolicy().hasHeightForWidth());
        m_DTWidthEdit->setSizePolicy(sizePolicy);

        hboxLayout5->addWidget(m_DTWidthEdit);


        gridLayout2->addLayout(hboxLayout5, 2, 0, 1, 1);

        hboxLayout6 = new QHBoxLayout();
        hboxLayout6->setSpacing(6);
        hboxLayout6->setObjectName(QString::fromUtf8("hboxLayout6"));
        spacer6 = new QSpacerItem(77, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout6->addItem(spacer6);

        m_EllipsoidPowerText = new QLabel(m_LevelSetParamsGroupBox);
        m_EllipsoidPowerText->setObjectName(QString::fromUtf8("m_EllipsoidPowerText"));
        m_EllipsoidPowerText->setWordWrap(false);

        hboxLayout6->addWidget(m_EllipsoidPowerText);

        m_EllipsoidPowerEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_EllipsoidPowerEdit->setObjectName(QString::fromUtf8("m_EllipsoidPowerEdit"));
        sizePolicy.setHeightForWidth(m_EllipsoidPowerEdit->sizePolicy().hasHeightForWidth());
        m_EllipsoidPowerEdit->setSizePolicy(sizePolicy);

        hboxLayout6->addWidget(m_EllipsoidPowerEdit);

        m_BBoxOffsetText = new QLabel(m_LevelSetParamsGroupBox);
        m_BBoxOffsetText->setObjectName(QString::fromUtf8("m_BBoxOffsetText"));
        m_BBoxOffsetText->setWordWrap(false);

        hboxLayout6->addWidget(m_BBoxOffsetText);

        m_BBoxOffsetEdit = new QLineEdit(m_LevelSetParamsGroupBox);
        m_BBoxOffsetEdit->setObjectName(QString::fromUtf8("m_BBoxOffsetEdit"));
        sizePolicy.setHeightForWidth(m_BBoxOffsetEdit->sizePolicy().hasHeightForWidth());
        m_BBoxOffsetEdit->setSizePolicy(sizePolicy);

        hboxLayout6->addWidget(m_BBoxOffsetEdit);


        gridLayout2->addLayout(hboxLayout6, 6, 0, 1, 1);


        gridLayout->addWidget(m_LevelSetParamsGroupBox, 0, 0, 1, 3);

        m_Preview = new QCheckBox(MSLevelSetDialog);
        m_Preview->setObjectName(QString::fromUtf8("m_Preview"));
        m_Preview->setChecked(true);

        gridLayout->addWidget(m_Preview, 2, 0, 1, 1);


        retranslateUi(MSLevelSetDialog);
        QObject::connect(m_DTInitComboBox, SIGNAL(activated(int)), MSLevelSetDialog, SLOT(on_DTInitComboBox_activated(int)));
        QObject::connect(m_BlockDimEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_BlockDimEdit_textChangedSlot()));
        QObject::connect(m_DTWidthEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_DTWidth_textChangedSlot()));
        QObject::connect(m_DeltaTEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_DeltaTEdit_textChangedSlot()));
        QObject::connect(m_EllipsoidPowerEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_EllipsoidPowerEdit_textChangedSlot()));
        QObject::connect(m_EpsilonEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_EpsilonEdit_textChangedSlot()));
        QObject::connect(m_Lambda1Edit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_Lambda1Edit_textChangedSlot()));
        QObject::connect(m_Lambda2Edit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_Lambda2Edit_textChangedSlot()));
        QObject::connect(m_MaxMedianIterEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_MaxMedianIterEdit_textChangedSlot()));
        QObject::connect(m_MaxSolverIterEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_MaxSolverIterEdit_textChangedSlot()));
        QObject::connect(m_MedianTolEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_MedianTolEdit_textChangedSlot()));
        QObject::connect(m_MuEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_MuEdit_textChangedSLot()));
        QObject::connect(m_NuEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_NuEdit_textChangedSLot()));
        QObject::connect(m_SubvolDimEdit, SIGNAL(textChanged(QString)), MSLevelSetDialog, SLOT(on_SubvolDimEdit_textChangedSlot()));
        QObject::connect(m_BlockDimComboBox, SIGNAL(activated(int)), MSLevelSetDialog, SLOT(on_BlockDimComboBox_activatedSlot(int)));
        QObject::connect(m_RunButton, SIGNAL(clicked()), MSLevelSetDialog, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), MSLevelSetDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(MSLevelSetDialog);
    } // setupUi

    void retranslateUi(QDialog *MSLevelSetDialog)
    {
        MSLevelSetDialog->setWindowTitle(QApplication::translate("MSLevelSetDialog", "Mumford-Shah Level Set Segmentation", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("MSLevelSetDialog", "Cancel", 0, QApplication::UnicodeUTF8));
        m_RunButton->setText(QApplication::translate("MSLevelSetDialog", "Run", 0, QApplication::UnicodeUTF8));
        m_CUDAParamsGroupBox->setTitle(QApplication::translate("MSLevelSetDialog", "CUDA Parameters", 0, QApplication::UnicodeUTF8));
        m_BlockDimText->setText(QApplication::translate("MSLevelSetDialog", "Block dimension", 0, QApplication::UnicodeUTF8));
        m_BlockDimComboBox->clear();
        m_BlockDimComboBox->insertItems(0, QStringList()
         << QApplication::translate("MSLevelSetDialog", "SDT kernel", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MSLevelSetDialog", "Averaging kernel", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MSLevelSetDialog", "Median kernel", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MSLevelSetDialog", "PDE kernel", 0, QApplication::UnicodeUTF8)
        );
        m_SubvolDimText->setText(QApplication::translate("MSLevelSetDialog", "Subvolume dimension", 0, QApplication::UnicodeUTF8));
        m_SubvolDimEdit->setText(QApplication::translate("MSLevelSetDialog", "128", 0, QApplication::UnicodeUTF8));
        m_BlockDimEdit->setText(QApplication::translate("MSLevelSetDialog", "4", 0, QApplication::UnicodeUTF8));
        m_LevelSetParamsGroupBox->setTitle(QApplication::translate("MSLevelSetDialog", "Level Set Parameters", 0, QApplication::UnicodeUTF8));
        m_Lambda2Text->setText(QApplication::translate("MSLevelSetDialog", "\316\273<font size=\"-1\">2</font>", 0, QApplication::UnicodeUTF8));
        m_Lambda1Edit->setText(QApplication::translate("MSLevelSetDialog", "1.0", 0, QApplication::UnicodeUTF8));
        m_DeltaTText->setText(QApplication::translate("MSLevelSetDialog", "\316\224t", 0, QApplication::UnicodeUTF8));
        m_MuEdit->setText(QApplication::translate("MSLevelSetDialog", "32.5125", 0, QApplication::UnicodeUTF8));
        m_MuText->setText(QApplication::translate("MSLevelSetDialog", "\316\274", 0, QApplication::UnicodeUTF8));
        m_NuText->setText(QApplication::translate("MSLevelSetDialog", "\316\275", 0, QApplication::UnicodeUTF8));
        m_EpsilonText->setText(QApplication::translate("MSLevelSetDialog", "\317\265", 0, QApplication::UnicodeUTF8));
        m_Lambda2Edit->setText(QApplication::translate("MSLevelSetDialog", "1.0", 0, QApplication::UnicodeUTF8));
        m_NuEdit->setText(QApplication::translate("MSLevelSetDialog", "0.0", 0, QApplication::UnicodeUTF8));
        m_DeltaTEdit->setText(QApplication::translate("MSLevelSetDialog", "0.01", 0, QApplication::UnicodeUTF8));
        m_EpsilonEdit->setText(QApplication::translate("MSLevelSetDialog", "1.0", 0, QApplication::UnicodeUTF8));
        m_Lambda1Text->setText(QApplication::translate("MSLevelSetDialog", "\316\273<font size=\"-1\">1</font>", 0, QApplication::UnicodeUTF8));
        m_MaxSolverIterText->setText(QApplication::translate("MSLevelSetDialog", "Maximum solver iterations", 0, QApplication::UnicodeUTF8));
        m_MaxSolverIterEdit->setText(QApplication::translate("MSLevelSetDialog", "15", 0, QApplication::UnicodeUTF8));
        m_MaxMedianIterText->setText(QApplication::translate("MSLevelSetDialog", "Maximum weighted median iterations", 0, QApplication::UnicodeUTF8));
        m_MaxMedianIterEdit->setText(QApplication::translate("MSLevelSetDialog", "10", 0, QApplication::UnicodeUTF8));
        m_MedianTolText->setText(QApplication::translate("MSLevelSetDialog", "Weighted median tolerance", 0, QApplication::UnicodeUTF8));
        m_MedianTolEdit->setText(QApplication::translate("MSLevelSetDialog", "0.0005", 0, QApplication::UnicodeUTF8));
        m_DTInitText->setText(QApplication::translate("MSLevelSetDialog", "Distance field interface (\317\225) initialization", 0, QApplication::UnicodeUTF8));
        m_DTInitComboBox->clear();
        m_DTInitComboBox->insertItems(0, QStringList()
         << QApplication::translate("MSLevelSetDialog", "Bounding box", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("MSLevelSetDialog", "Super-ellipsoid", 0, QApplication::UnicodeUTF8)
        );
        m_DTWidthText->setText(QApplication::translate("MSLevelSetDialog", "DT band width", 0, QApplication::UnicodeUTF8));
        m_DTWidthEdit->setText(QApplication::translate("MSLevelSetDialog", "10", 0, QApplication::UnicodeUTF8));
        m_EllipsoidPowerText->setText(QApplication::translate("MSLevelSetDialog", "n", 0, QApplication::UnicodeUTF8));
        m_EllipsoidPowerEdit->setText(QApplication::translate("MSLevelSetDialog", "8", 0, QApplication::UnicodeUTF8));
        m_BBoxOffsetText->setText(QApplication::translate("MSLevelSetDialog", "BBox offset", 0, QApplication::UnicodeUTF8));
        m_BBoxOffsetEdit->setText(QApplication::translate("MSLevelSetDialog", "5", 0, QApplication::UnicodeUTF8));
        m_Preview->setText(QApplication::translate("MSLevelSetDialog", "Preview", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MSLevelSetDialog: public Ui_MSLevelSetDialog {};
} // namespace Ui

QT_END_NAMESPACE

class MSLevelSetDialog : public QDialog, public Ui::MSLevelSetDialog
{
    Q_OBJECT

public:
    MSLevelSetDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~MSLevelSetDialog();

    virtual void paramReference( MSLevelSetParams * mslsp );
    virtual void init();

public slots:
    virtual void on_DTInitComboBox_activated( int idx );
    virtual void on_BlockDimEdit_textChangedSlot();
    virtual void on_DTWidth_textChangedSlot();
    virtual void on_DeltaTEdit_textChangedSlot();
    virtual void on_EllipsoidPowerEdit_textChangedSlot();
    virtual void on_EpsilonEdit_textChangedSlot();
    virtual void on_Lambda1Edit_textChangedSlot();
    virtual void on_Lambda2Edit_textChangedSlot();
    virtual void on_MaxMedianIterEdit_textChangedSlot();
    virtual void on_MaxSolverIterEdit_textChangedSlot();
    virtual void on_MedianTolEdit_textChangedSlot();
    virtual void on_MuEdit_textChangedSLot();
    virtual void on_NuEdit_textChangedSLot();
    virtual void on_SubvolDimEdit_textChangedSlot();
    virtual void on_BlockDimComboBox_activatedSlot( int idx );
    virtual void on_BBoxOffsetEdit_textChangedSlot();

protected slots:
    virtual void languageChange();

private:
    MSLevelSetParams *MSLSParams;

};

#endif // MSLEVELSETDIALOG_H
