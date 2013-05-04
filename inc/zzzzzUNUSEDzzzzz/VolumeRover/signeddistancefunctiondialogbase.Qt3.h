#ifndef SIGNEDDISTANCEFUNCTIONDIALOGBASE_H
#define SIGNEDDISTANCEFUNCTIONDIALOGBASE_H

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
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_SignedDistanceFunctionDialogBase
{
public:
    QGridLayout *gridLayout;
    Q3GroupBox *m_MethodGroup;
    QGridLayout *gridLayout1;
    QComboBox *m_Method;
    Q3GroupBox *m_DimensionGroup;
    QGridLayout *gridLayout2;
    QLabel *m_DimXText;
    QLineEdit *m_DimX;
    QLabel *m_DimYText;
    QLineEdit *m_DimY;
    QLabel *m_DimZText;
    QLineEdit *m_DimZ;
    Q3GroupBox *m_BoundingBoxGroup;
    QGridLayout *gridLayout3;
    Q3GroupBox *m_MinGroup;
    QGridLayout *gridLayout4;
    QLabel *m_MinXText;
    QLabel *m_MinYText;
    QLineEdit *m_MinX;
    QLineEdit *m_MinY;
    QLabel *m_MinZText;
    QLineEdit *m_MinZ;
    Q3GroupBox *m_MaxGroup;
    QGridLayout *gridLayout5;
    QLabel *m_MaxXText;
    QLabel *m_MaxYText;
    QLineEdit *m_MaxX;
    QLabel *m_MaxZText;
    QLineEdit *m_MaxZ;
    QLineEdit *m_MaxY;
    QPushButton *m_Cancel;
    QPushButton *m_Ok;
    QCheckBox *m_UseBoundingBox;
    QPushButton *m_SubVolBoxButton;

    void setupUi(QDialog *SignedDistanceFunctionDialogBase)
    {
        if (SignedDistanceFunctionDialogBase->objectName().isEmpty())
            SignedDistanceFunctionDialogBase->setObjectName(QString::fromUtf8("SignedDistanceFunctionDialogBase"));
        SignedDistanceFunctionDialogBase->resize(409, 304);
        gridLayout = new QGridLayout(SignedDistanceFunctionDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_MethodGroup = new Q3GroupBox(SignedDistanceFunctionDialogBase);
        m_MethodGroup->setObjectName(QString::fromUtf8("m_MethodGroup"));
        m_MethodGroup->setColumnLayout(0, Qt::Vertical);
        m_MethodGroup->layout()->setSpacing(6);
        m_MethodGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_MethodGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_Method = new QComboBox(m_MethodGroup);
        m_Method->setObjectName(QString::fromUtf8("m_Method"));

        gridLayout1->addWidget(m_Method, 0, 0, 1, 1);


        gridLayout->addWidget(m_MethodGroup, 0, 0, 1, 2);

        m_DimensionGroup = new Q3GroupBox(SignedDistanceFunctionDialogBase);
        m_DimensionGroup->setObjectName(QString::fromUtf8("m_DimensionGroup"));
        m_DimensionGroup->setColumnLayout(0, Qt::Vertical);
        m_DimensionGroup->layout()->setSpacing(6);
        m_DimensionGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_DimensionGroup->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        m_DimXText = new QLabel(m_DimensionGroup);
        m_DimXText->setObjectName(QString::fromUtf8("m_DimXText"));
        m_DimXText->setWordWrap(false);

        gridLayout2->addWidget(m_DimXText, 0, 0, 1, 1);

        m_DimX = new QLineEdit(m_DimensionGroup);
        m_DimX->setObjectName(QString::fromUtf8("m_DimX"));

        gridLayout2->addWidget(m_DimX, 0, 1, 1, 1);

        m_DimYText = new QLabel(m_DimensionGroup);
        m_DimYText->setObjectName(QString::fromUtf8("m_DimYText"));
        m_DimYText->setWordWrap(false);

        gridLayout2->addWidget(m_DimYText, 1, 0, 1, 1);

        m_DimY = new QLineEdit(m_DimensionGroup);
        m_DimY->setObjectName(QString::fromUtf8("m_DimY"));

        gridLayout2->addWidget(m_DimY, 1, 1, 1, 1);

        m_DimZText = new QLabel(m_DimensionGroup);
        m_DimZText->setObjectName(QString::fromUtf8("m_DimZText"));
        m_DimZText->setWordWrap(false);

        gridLayout2->addWidget(m_DimZText, 2, 0, 1, 1);

        m_DimZ = new QLineEdit(m_DimensionGroup);
        m_DimZ->setObjectName(QString::fromUtf8("m_DimZ"));

        gridLayout2->addWidget(m_DimZ, 2, 1, 1, 1);


        gridLayout->addWidget(m_DimensionGroup, 1, 0, 1, 2);

        m_BoundingBoxGroup = new Q3GroupBox(SignedDistanceFunctionDialogBase);
        m_BoundingBoxGroup->setObjectName(QString::fromUtf8("m_BoundingBoxGroup"));
        m_BoundingBoxGroup->setEnabled(false);
        m_BoundingBoxGroup->setColumnLayout(0, Qt::Vertical);
        m_BoundingBoxGroup->layout()->setSpacing(6);
        m_BoundingBoxGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout3 = new QGridLayout();
        QBoxLayout *boxlayout2 = qobject_cast<QBoxLayout *>(m_BoundingBoxGroup->layout());
        if (boxlayout2)
            boxlayout2->addLayout(gridLayout3);
        gridLayout3->setAlignment(Qt::AlignTop);
        gridLayout3->setObjectName(QString::fromUtf8("gridLayout3"));
        m_MinGroup = new Q3GroupBox(m_BoundingBoxGroup);
        m_MinGroup->setObjectName(QString::fromUtf8("m_MinGroup"));
        m_MinGroup->setColumnLayout(0, Qt::Vertical);
        m_MinGroup->layout()->setSpacing(6);
        m_MinGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout4 = new QGridLayout();
        QBoxLayout *boxlayout3 = qobject_cast<QBoxLayout *>(m_MinGroup->layout());
        if (boxlayout3)
            boxlayout3->addLayout(gridLayout4);
        gridLayout4->setAlignment(Qt::AlignTop);
        gridLayout4->setObjectName(QString::fromUtf8("gridLayout4"));
        m_MinXText = new QLabel(m_MinGroup);
        m_MinXText->setObjectName(QString::fromUtf8("m_MinXText"));
        m_MinXText->setWordWrap(false);

        gridLayout4->addWidget(m_MinXText, 0, 0, 1, 1);

        m_MinYText = new QLabel(m_MinGroup);
        m_MinYText->setObjectName(QString::fromUtf8("m_MinYText"));
        m_MinYText->setWordWrap(false);

        gridLayout4->addWidget(m_MinYText, 1, 0, 1, 1);

        m_MinX = new QLineEdit(m_MinGroup);
        m_MinX->setObjectName(QString::fromUtf8("m_MinX"));

        gridLayout4->addWidget(m_MinX, 0, 1, 1, 1);

        m_MinY = new QLineEdit(m_MinGroup);
        m_MinY->setObjectName(QString::fromUtf8("m_MinY"));

        gridLayout4->addWidget(m_MinY, 1, 1, 1, 1);

        m_MinZText = new QLabel(m_MinGroup);
        m_MinZText->setObjectName(QString::fromUtf8("m_MinZText"));
        m_MinZText->setWordWrap(false);

        gridLayout4->addWidget(m_MinZText, 2, 0, 1, 1);

        m_MinZ = new QLineEdit(m_MinGroup);
        m_MinZ->setObjectName(QString::fromUtf8("m_MinZ"));

        gridLayout4->addWidget(m_MinZ, 2, 1, 1, 1);


        gridLayout3->addWidget(m_MinGroup, 0, 0, 1, 1);

        m_MaxGroup = new Q3GroupBox(m_BoundingBoxGroup);
        m_MaxGroup->setObjectName(QString::fromUtf8("m_MaxGroup"));
        m_MaxGroup->setColumnLayout(0, Qt::Vertical);
        m_MaxGroup->layout()->setSpacing(6);
        m_MaxGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout5 = new QGridLayout();
        QBoxLayout *boxlayout4 = qobject_cast<QBoxLayout *>(m_MaxGroup->layout());
        if (boxlayout4)
            boxlayout4->addLayout(gridLayout5);
        gridLayout5->setAlignment(Qt::AlignTop);
        gridLayout5->setObjectName(QString::fromUtf8("gridLayout5"));
        m_MaxXText = new QLabel(m_MaxGroup);
        m_MaxXText->setObjectName(QString::fromUtf8("m_MaxXText"));
        m_MaxXText->setWordWrap(false);

        gridLayout5->addWidget(m_MaxXText, 0, 0, 1, 1);

        m_MaxYText = new QLabel(m_MaxGroup);
        m_MaxYText->setObjectName(QString::fromUtf8("m_MaxYText"));
        m_MaxYText->setWordWrap(false);

        gridLayout5->addWidget(m_MaxYText, 1, 0, 1, 1);

        m_MaxX = new QLineEdit(m_MaxGroup);
        m_MaxX->setObjectName(QString::fromUtf8("m_MaxX"));

        gridLayout5->addWidget(m_MaxX, 0, 1, 1, 1);

        m_MaxZText = new QLabel(m_MaxGroup);
        m_MaxZText->setObjectName(QString::fromUtf8("m_MaxZText"));
        m_MaxZText->setWordWrap(false);

        gridLayout5->addWidget(m_MaxZText, 2, 0, 1, 1);

        m_MaxZ = new QLineEdit(m_MaxGroup);
        m_MaxZ->setObjectName(QString::fromUtf8("m_MaxZ"));

        gridLayout5->addWidget(m_MaxZ, 2, 1, 1, 1);

        m_MaxY = new QLineEdit(m_MaxGroup);
        m_MaxY->setObjectName(QString::fromUtf8("m_MaxY"));

        gridLayout5->addWidget(m_MaxY, 1, 1, 1, 1);


        gridLayout3->addWidget(m_MaxGroup, 1, 0, 1, 1);


        gridLayout->addWidget(m_BoundingBoxGroup, 0, 2, 5, 1);

        m_Cancel = new QPushButton(SignedDistanceFunctionDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));

        gridLayout->addWidget(m_Cancel, 4, 0, 1, 1);

        m_Ok = new QPushButton(SignedDistanceFunctionDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));

        gridLayout->addWidget(m_Ok, 4, 1, 1, 1);

        m_UseBoundingBox = new QCheckBox(SignedDistanceFunctionDialogBase);
        m_UseBoundingBox->setObjectName(QString::fromUtf8("m_UseBoundingBox"));

        gridLayout->addWidget(m_UseBoundingBox, 2, 0, 1, 2);

        m_SubVolBoxButton = new QPushButton(SignedDistanceFunctionDialogBase);
        m_SubVolBoxButton->setObjectName(QString::fromUtf8("m_SubVolBoxButton"));
        m_SubVolBoxButton->setEnabled(false);

        gridLayout->addWidget(m_SubVolBoxButton, 3, 0, 1, 2);

        QWidget::setTabOrder(m_DimX, m_DimY);
        QWidget::setTabOrder(m_DimY, m_DimZ);
        QWidget::setTabOrder(m_DimZ, m_UseBoundingBox);
        QWidget::setTabOrder(m_UseBoundingBox, m_MinX);
        QWidget::setTabOrder(m_MinX, m_MinY);
        QWidget::setTabOrder(m_MinY, m_MinZ);
        QWidget::setTabOrder(m_MinZ, m_MaxX);
        QWidget::setTabOrder(m_MaxX, m_MaxY);
        QWidget::setTabOrder(m_MaxY, m_MaxZ);
        QWidget::setTabOrder(m_MaxZ, m_Cancel);
        QWidget::setTabOrder(m_Cancel, m_Ok);
        QWidget::setTabOrder(m_Ok, m_Method);

        retranslateUi(SignedDistanceFunctionDialogBase);
        QObject::connect(m_UseBoundingBox, SIGNAL(toggled(bool)), m_BoundingBoxGroup, SLOT(setEnabled(bool)));
        QObject::connect(m_Cancel, SIGNAL(clicked()), SignedDistanceFunctionDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), SignedDistanceFunctionDialogBase, SLOT(accept()));
        QObject::connect(m_UseBoundingBox, SIGNAL(toggled(bool)), m_SubVolBoxButton, SLOT(setEnabled(bool)));
        QObject::connect(m_SubVolBoxButton, SIGNAL(clicked()), SignedDistanceFunctionDialogBase, SLOT(grabSubVolBox()));

        QMetaObject::connectSlotsByName(SignedDistanceFunctionDialogBase);
    } // setupUi

    void retranslateUi(QDialog *SignedDistanceFunctionDialogBase)
    {
        SignedDistanceFunctionDialogBase->setWindowTitle(QApplication::translate("SignedDistanceFunctionDialogBase", "Signed Distance Function", 0, QApplication::UnicodeUTF8));
        m_MethodGroup->setTitle(QApplication::translate("SignedDistanceFunctionDialogBase", "Method", 0, QApplication::UnicodeUTF8));
        m_Method->clear();
        m_Method->insertItems(0, QStringList()
         << QApplication::translate("SignedDistanceFunctionDialogBase", "multi_sdf", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("SignedDistanceFunctionDialogBase", "SignDistanceFunction", 0, QApplication::UnicodeUTF8)
        );
        m_DimensionGroup->setTitle(QApplication::translate("SignedDistanceFunctionDialogBase", "Output Volume Dimension", 0, QApplication::UnicodeUTF8));
        m_DimXText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "X:", 0, QApplication::UnicodeUTF8));
        m_DimX->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "64", 0, QApplication::UnicodeUTF8));
        m_DimYText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Y:", 0, QApplication::UnicodeUTF8));
        m_DimY->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "64", 0, QApplication::UnicodeUTF8));
        m_DimZText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Z:", 0, QApplication::UnicodeUTF8));
        m_DimZ->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "64", 0, QApplication::UnicodeUTF8));
        m_BoundingBoxGroup->setTitle(QApplication::translate("SignedDistanceFunctionDialogBase", "Bounding Box", 0, QApplication::UnicodeUTF8));
        m_MinGroup->setTitle(QApplication::translate("SignedDistanceFunctionDialogBase", "Minimum", 0, QApplication::UnicodeUTF8));
        m_MinXText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "X:", 0, QApplication::UnicodeUTF8));
        m_MinYText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Y:", 0, QApplication::UnicodeUTF8));
        m_MinX->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "0.0", 0, QApplication::UnicodeUTF8));
        m_MinY->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "0.0", 0, QApplication::UnicodeUTF8));
        m_MinZText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Z:", 0, QApplication::UnicodeUTF8));
        m_MinZ->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "0.0", 0, QApplication::UnicodeUTF8));
        m_MaxGroup->setTitle(QApplication::translate("SignedDistanceFunctionDialogBase", "Maximum", 0, QApplication::UnicodeUTF8));
        m_MaxXText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "X:", 0, QApplication::UnicodeUTF8));
        m_MaxYText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Y:", 0, QApplication::UnicodeUTF8));
        m_MaxX->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "0.0", 0, QApplication::UnicodeUTF8));
        m_MaxZText->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Z:", 0, QApplication::UnicodeUTF8));
        m_MaxZ->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "0.0", 0, QApplication::UnicodeUTF8));
        m_MaxY->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "0.0", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
        m_UseBoundingBox->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Manually Define Bounding Box", 0, QApplication::UnicodeUTF8));
        m_SubVolBoxButton->setText(QApplication::translate("SignedDistanceFunctionDialogBase", "Bounding Box from SubVolume", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SignedDistanceFunctionDialogBase: public Ui_SignedDistanceFunctionDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class SignedDistanceFunctionDialogBase : public QDialog, public Ui::SignedDistanceFunctionDialogBase
{
    Q_OBJECT

public:
    SignedDistanceFunctionDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~SignedDistanceFunctionDialogBase();

public slots:
    virtual void grabSubVolBox();

protected slots:
    virtual void languageChange();

};

#endif // SIGNEDDISTANCEFUNCTIONDIALOGBASE_H
