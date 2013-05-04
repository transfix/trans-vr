#ifndef LBIEMESHINGDIALOGBASE_H
#define LBIEMESHINGDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
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
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>

QT_BEGIN_NAMESPACE

class Ui_LBIEMeshingDialogBase
{
public:
    QGridLayout *gridLayout;
    Q3ButtonGroup *m_WhichIsovaluesGroup;
    QGridLayout *gridLayout1;
    QRadioButton *m_IsovaluesManual;
    QRadioButton *m_IsovaluesFromColorTable;
    Q3GroupBox *m_IsovalueGroup;
    QGridLayout *gridLayout2;
    QLabel *m_OuterIsoValueText;
    QLineEdit *m_OuterIsoValue;
    QLabel *m_InnerIsoValueText;
    QLineEdit *m_InnerIsoValue;
    QLabel *m_ErrorToleranceText;
    QCheckBox *m_DualContouring;
    QPushButton *m_Cancel;
    QLineEdit *m_Iterations;
    QLabel *m_ImproveMethodText;
    QLineEdit *m_ErrorTolerance;
    QComboBox *m_NormalType;
    QComboBox *m_ImproveMethod;
    QLabel *m_MeshTypeText;
    QCheckBox *m_Preview;
    QComboBox *m_MeshType;
    QLabel *m_InnerErrorToleranceText;
    QPushButton *m_Ok;
    QLineEdit *m_InnerErrorTolerance;
    QLabel *m_NormalTypeText;
    QLabel *m_IterationsText;
    QLabel *m_MeshExtractionMethodText;
    QComboBox *m_MeshExtractionMethod;

    void setupUi(QDialog *LBIEMeshingDialogBase)
    {
        if (LBIEMeshingDialogBase->objectName().isEmpty())
            LBIEMeshingDialogBase->setObjectName(QString::fromUtf8("LBIEMeshingDialogBase"));
        LBIEMeshingDialogBase->resize(386, 461);
        gridLayout = new QGridLayout(LBIEMeshingDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_WhichIsovaluesGroup = new Q3ButtonGroup(LBIEMeshingDialogBase);
        m_WhichIsovaluesGroup->setObjectName(QString::fromUtf8("m_WhichIsovaluesGroup"));
        m_WhichIsovaluesGroup->setColumnLayout(0, Qt::Vertical);
        m_WhichIsovaluesGroup->layout()->setSpacing(6);
        m_WhichIsovaluesGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_WhichIsovaluesGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_IsovaluesManual = new QRadioButton(m_WhichIsovaluesGroup);
        m_IsovaluesManual->setObjectName(QString::fromUtf8("m_IsovaluesManual"));
        m_IsovaluesManual->setChecked(false);
        m_WhichIsovaluesGroup->insert(m_IsovaluesManual, 1);

        gridLayout1->addWidget(m_IsovaluesManual, 1, 0, 1, 1);

        m_IsovaluesFromColorTable = new QRadioButton(m_WhichIsovaluesGroup);
        m_IsovaluesFromColorTable->setObjectName(QString::fromUtf8("m_IsovaluesFromColorTable"));
        m_IsovaluesFromColorTable->setChecked(true);
        m_WhichIsovaluesGroup->insert(m_IsovaluesFromColorTable, 0);

        gridLayout1->addWidget(m_IsovaluesFromColorTable, 0, 0, 1, 1);


        gridLayout->addWidget(m_WhichIsovaluesGroup, 0, 0, 1, 4);

        m_IsovalueGroup = new Q3GroupBox(LBIEMeshingDialogBase);
        m_IsovalueGroup->setObjectName(QString::fromUtf8("m_IsovalueGroup"));
        m_IsovalueGroup->setEnabled(false);
        m_IsovalueGroup->setColumnLayout(0, Qt::Vertical);
        m_IsovalueGroup->layout()->setSpacing(6);
        m_IsovalueGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_IsovalueGroup->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        m_OuterIsoValueText = new QLabel(m_IsovalueGroup);
        m_OuterIsoValueText->setObjectName(QString::fromUtf8("m_OuterIsoValueText"));
        m_OuterIsoValueText->setWordWrap(false);

        gridLayout2->addWidget(m_OuterIsoValueText, 0, 0, 1, 1);

        m_OuterIsoValue = new QLineEdit(m_IsovalueGroup);
        m_OuterIsoValue->setObjectName(QString::fromUtf8("m_OuterIsoValue"));

        gridLayout2->addWidget(m_OuterIsoValue, 0, 1, 1, 1);

        m_InnerIsoValueText = new QLabel(m_IsovalueGroup);
        m_InnerIsoValueText->setObjectName(QString::fromUtf8("m_InnerIsoValueText"));
        m_InnerIsoValueText->setWordWrap(false);

        gridLayout2->addWidget(m_InnerIsoValueText, 1, 0, 1, 1);

        m_InnerIsoValue = new QLineEdit(m_IsovalueGroup);
        m_InnerIsoValue->setObjectName(QString::fromUtf8("m_InnerIsoValue"));

        gridLayout2->addWidget(m_InnerIsoValue, 1, 1, 1, 1);


        gridLayout->addWidget(m_IsovalueGroup, 1, 0, 1, 4);

        m_ErrorToleranceText = new QLabel(LBIEMeshingDialogBase);
        m_ErrorToleranceText->setObjectName(QString::fromUtf8("m_ErrorToleranceText"));
        m_ErrorToleranceText->setWordWrap(false);

        gridLayout->addWidget(m_ErrorToleranceText, 3, 0, 1, 1);

        m_DualContouring = new QCheckBox(LBIEMeshingDialogBase);
        m_DualContouring->setObjectName(QString::fromUtf8("m_DualContouring"));

        gridLayout->addWidget(m_DualContouring, 9, 0, 1, 4);

        m_Cancel = new QPushButton(LBIEMeshingDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));

        gridLayout->addWidget(m_Cancel, 10, 1, 1, 2);

        m_Iterations = new QLineEdit(LBIEMeshingDialogBase);
        m_Iterations->setObjectName(QString::fromUtf8("m_Iterations"));

        gridLayout->addWidget(m_Iterations, 7, 2, 1, 2);

        m_ImproveMethodText = new QLabel(LBIEMeshingDialogBase);
        m_ImproveMethodText->setObjectName(QString::fromUtf8("m_ImproveMethodText"));
        m_ImproveMethodText->setWordWrap(false);

        gridLayout->addWidget(m_ImproveMethodText, 6, 0, 1, 2);

        m_ErrorTolerance = new QLineEdit(LBIEMeshingDialogBase);
        m_ErrorTolerance->setObjectName(QString::fromUtf8("m_ErrorTolerance"));

        gridLayout->addWidget(m_ErrorTolerance, 3, 2, 1, 2);

        m_NormalType = new QComboBox(LBIEMeshingDialogBase);
        m_NormalType->setObjectName(QString::fromUtf8("m_NormalType"));

        gridLayout->addWidget(m_NormalType, 8, 2, 1, 2);

        m_ImproveMethod = new QComboBox(LBIEMeshingDialogBase);
        m_ImproveMethod->setObjectName(QString::fromUtf8("m_ImproveMethod"));

        gridLayout->addWidget(m_ImproveMethod, 6, 2, 1, 2);

        m_MeshTypeText = new QLabel(LBIEMeshingDialogBase);
        m_MeshTypeText->setObjectName(QString::fromUtf8("m_MeshTypeText"));
        m_MeshTypeText->setWordWrap(false);

        gridLayout->addWidget(m_MeshTypeText, 5, 0, 1, 1);

        m_Preview = new QCheckBox(LBIEMeshingDialogBase);
        m_Preview->setObjectName(QString::fromUtf8("m_Preview"));
        m_Preview->setChecked(true);

        gridLayout->addWidget(m_Preview, 10, 0, 1, 1);

        m_MeshType = new QComboBox(LBIEMeshingDialogBase);
        m_MeshType->setObjectName(QString::fromUtf8("m_MeshType"));

        gridLayout->addWidget(m_MeshType, 5, 2, 1, 2);

        m_InnerErrorToleranceText = new QLabel(LBIEMeshingDialogBase);
        m_InnerErrorToleranceText->setObjectName(QString::fromUtf8("m_InnerErrorToleranceText"));
        m_InnerErrorToleranceText->setWordWrap(false);

        gridLayout->addWidget(m_InnerErrorToleranceText, 4, 0, 1, 1);

        m_Ok = new QPushButton(LBIEMeshingDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));

        gridLayout->addWidget(m_Ok, 10, 3, 1, 1);

        m_InnerErrorTolerance = new QLineEdit(LBIEMeshingDialogBase);
        m_InnerErrorTolerance->setObjectName(QString::fromUtf8("m_InnerErrorTolerance"));

        gridLayout->addWidget(m_InnerErrorTolerance, 4, 2, 1, 2);

        m_NormalTypeText = new QLabel(LBIEMeshingDialogBase);
        m_NormalTypeText->setObjectName(QString::fromUtf8("m_NormalTypeText"));
        m_NormalTypeText->setWordWrap(false);

        gridLayout->addWidget(m_NormalTypeText, 8, 0, 1, 1);

        m_IterationsText = new QLabel(LBIEMeshingDialogBase);
        m_IterationsText->setObjectName(QString::fromUtf8("m_IterationsText"));
        m_IterationsText->setWordWrap(false);

        gridLayout->addWidget(m_IterationsText, 7, 0, 1, 2);

        m_MeshExtractionMethodText = new QLabel(LBIEMeshingDialogBase);
        m_MeshExtractionMethodText->setObjectName(QString::fromUtf8("m_MeshExtractionMethodText"));
        m_MeshExtractionMethodText->setWordWrap(false);

        gridLayout->addWidget(m_MeshExtractionMethodText, 2, 0, 1, 2);

        m_MeshExtractionMethod = new QComboBox(LBIEMeshingDialogBase);
        m_MeshExtractionMethod->setObjectName(QString::fromUtf8("m_MeshExtractionMethod"));

        gridLayout->addWidget(m_MeshExtractionMethod, 2, 2, 1, 2);


        retranslateUi(LBIEMeshingDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), LBIEMeshingDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), LBIEMeshingDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(LBIEMeshingDialogBase);
    } // setupUi

    void retranslateUi(QDialog *LBIEMeshingDialogBase)
    {
        LBIEMeshingDialogBase->setWindowTitle(QApplication::translate("LBIEMeshingDialogBase", "LBIE Meshing", 0, QApplication::UnicodeUTF8));
        m_WhichIsovaluesGroup->setTitle(QString());
        m_IsovaluesManual->setText(QApplication::translate("LBIEMeshingDialogBase", "Manually define iso-values", 0, QApplication::UnicodeUTF8));
        m_IsovaluesFromColorTable->setText(QApplication::translate("LBIEMeshingDialogBase", "Use min/max iso-values from color table", 0, QApplication::UnicodeUTF8));
        m_IsovalueGroup->setTitle(QApplication::translate("LBIEMeshingDialogBase", "Iso-values", 0, QApplication::UnicodeUTF8));
        m_OuterIsoValueText->setText(QApplication::translate("LBIEMeshingDialogBase", "Outer Iso Value:", 0, QApplication::UnicodeUTF8));
        m_InnerIsoValueText->setText(QApplication::translate("LBIEMeshingDialogBase", "Inner Iso Value:", 0, QApplication::UnicodeUTF8));
        m_ErrorToleranceText->setText(QApplication::translate("LBIEMeshingDialogBase", "Error Tolerance:", 0, QApplication::UnicodeUTF8));
        m_DualContouring->setText(QApplication::translate("LBIEMeshingDialogBase", "Dual Contouring", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("LBIEMeshingDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Iterations->setText(QApplication::translate("LBIEMeshingDialogBase", "1", 0, QApplication::UnicodeUTF8));
        m_ImproveMethodText->setText(QApplication::translate("LBIEMeshingDialogBase", "Improvement Method:", 0, QApplication::UnicodeUTF8));
        m_NormalType->clear();
        m_NormalType->insertItems(0, QStringList()
         << QApplication::translate("LBIEMeshingDialogBase", "B-Spline Convolution", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Central Difference", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "B-Spline Interpolation", 0, QApplication::UnicodeUTF8)
        );
        m_ImproveMethod->clear();
        m_ImproveMethod->insertItems(0, QStringList()
         << QApplication::translate("LBIEMeshingDialogBase", "No Improvement", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Geometric Flow", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Edge Contraction", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Joe Liu", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Minimal Volume", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Optimization", 0, QApplication::UnicodeUTF8)
        );
        m_MeshTypeText->setText(QApplication::translate("LBIEMeshingDialogBase", "Mesh Type:", 0, QApplication::UnicodeUTF8));
        m_Preview->setText(QApplication::translate("LBIEMeshingDialogBase", "Preview", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        m_Preview->setProperty("whatsThis", QVariant(QString()));
#endif // QT_NO_WHATSTHIS
        m_MeshType->clear();
        m_MeshType->insertItems(0, QStringList()
         << QApplication::translate("LBIEMeshingDialogBase", "Single", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Tetra", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Quad", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Hexa", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Double", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "Tetra2", 0, QApplication::UnicodeUTF8)
        );
        m_InnerErrorToleranceText->setText(QApplication::translate("LBIEMeshingDialogBase", "Inner Error Tolerance:", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("LBIEMeshingDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
        m_NormalTypeText->setText(QApplication::translate("LBIEMeshingDialogBase", "Normal Type:", 0, QApplication::UnicodeUTF8));
        m_IterationsText->setText(QApplication::translate("LBIEMeshingDialogBase", "Improvement Iterations:", 0, QApplication::UnicodeUTF8));
        m_MeshExtractionMethodText->setText(QApplication::translate("LBIEMeshingDialogBase", "Mesh Extraction Method:", 0, QApplication::UnicodeUTF8));
        m_MeshExtractionMethod->clear();
        m_MeshExtractionMethod->insertItems(0, QStringList()
         << QApplication::translate("LBIEMeshingDialogBase", "libLBIE (duallib)", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "FastContouring", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEMeshingDialogBase", "libcontour", 0, QApplication::UnicodeUTF8)
        );
    } // retranslateUi

};

namespace Ui {
    class LBIEMeshingDialogBase: public Ui_LBIEMeshingDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class LBIEMeshingDialogBase : public QDialog, public Ui::LBIEMeshingDialogBase
{
    Q_OBJECT

public:
    LBIEMeshingDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~LBIEMeshingDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // LBIEMESHINGDIALOGBASE_H
