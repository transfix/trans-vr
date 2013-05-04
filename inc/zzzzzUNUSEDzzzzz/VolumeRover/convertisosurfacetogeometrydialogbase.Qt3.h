#ifndef CONVERTISOSURFACETOGEOMETRYDIALOGBASE_H
#define CONVERTISOSURFACETOGEOMETRYDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>

QT_BEGIN_NAMESPACE

class Ui_ConvertIsosurfaceToGeometryDialogBase
{
public:
    QGridLayout *gridLayout;
    Q3ButtonGroup *m_IsosurfaceConversionOptionsGroup;
    QGridLayout *gridLayout1;
    QRadioButton *m_subVolume;
    QRadioButton *m_Volume;
    QPushButton *m_Cancel;
    QPushButton *m_Ok;

    void setupUi(QDialog *ConvertIsosurfaceToGeometryDialogBase)
    {
        if (ConvertIsosurfaceToGeometryDialogBase->objectName().isEmpty())
            ConvertIsosurfaceToGeometryDialogBase->setObjectName(QString::fromUtf8("ConvertIsosurfaceToGeometryDialogBase"));
        ConvertIsosurfaceToGeometryDialogBase->resize(227, 146);
        gridLayout = new QGridLayout(ConvertIsosurfaceToGeometryDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_IsosurfaceConversionOptionsGroup = new Q3ButtonGroup(ConvertIsosurfaceToGeometryDialogBase);
        m_IsosurfaceConversionOptionsGroup->setObjectName(QString::fromUtf8("m_IsosurfaceConversionOptionsGroup"));
        m_IsosurfaceConversionOptionsGroup->setColumnLayout(0, Qt::Vertical);
        m_IsosurfaceConversionOptionsGroup->layout()->setSpacing(6);
        m_IsosurfaceConversionOptionsGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_IsosurfaceConversionOptionsGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_subVolume = new QRadioButton(m_IsosurfaceConversionOptionsGroup);
        m_subVolume->setObjectName(QString::fromUtf8("m_subVolume"));
        m_subVolume->setChecked(true);

        gridLayout1->addWidget(m_subVolume, 0, 0, 1, 1);

        m_Volume = new QRadioButton(m_IsosurfaceConversionOptionsGroup);
        m_Volume->setObjectName(QString::fromUtf8("m_Volume"));

        gridLayout1->addWidget(m_Volume, 1, 0, 1, 1);


        gridLayout->addWidget(m_IsosurfaceConversionOptionsGroup, 0, 0, 1, 2);

        m_Cancel = new QPushButton(ConvertIsosurfaceToGeometryDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));

        gridLayout->addWidget(m_Cancel, 1, 0, 1, 1);

        m_Ok = new QPushButton(ConvertIsosurfaceToGeometryDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));

        gridLayout->addWidget(m_Ok, 1, 1, 1, 1);


        retranslateUi(ConvertIsosurfaceToGeometryDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), ConvertIsosurfaceToGeometryDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), ConvertIsosurfaceToGeometryDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(ConvertIsosurfaceToGeometryDialogBase);
    } // setupUi

    void retranslateUi(QDialog *ConvertIsosurfaceToGeometryDialogBase)
    {
        ConvertIsosurfaceToGeometryDialogBase->setWindowTitle(QApplication::translate("ConvertIsosurfaceToGeometryDialogBase", "Convert Isosurface", 0, QApplication::UnicodeUTF8));
        m_IsosurfaceConversionOptionsGroup->setTitle(QApplication::translate("ConvertIsosurfaceToGeometryDialogBase", "Conversion Options", 0, QApplication::UnicodeUTF8));
        m_subVolume->setText(QApplication::translate("ConvertIsosurfaceToGeometryDialogBase", "Use Zoomed-in Volume", 0, QApplication::UnicodeUTF8));
        m_Volume->setText(QApplication::translate("ConvertIsosurfaceToGeometryDialogBase", "Use Thumbnail Volume", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("ConvertIsosurfaceToGeometryDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("ConvertIsosurfaceToGeometryDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ConvertIsosurfaceToGeometryDialogBase: public Ui_ConvertIsosurfaceToGeometryDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class ConvertIsosurfaceToGeometryDialogBase : public QDialog, public Ui::ConvertIsosurfaceToGeometryDialogBase
{
    Q_OBJECT

public:
    ConvertIsosurfaceToGeometryDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~ConvertIsosurfaceToGeometryDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // CONVERTISOSURFACETOGEOMETRYDIALOGBASE_H
