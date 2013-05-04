#ifndef SMOOTHINGDIALOGBASE_H
#define SMOOTHINGDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_SmoothingDialogBase
{
public:
    QGridLayout *gridLayout;
    QLabel *m_DeltaText;
    QLineEdit *m_Delta;
    QCheckBox *m_FixBoundary;
    QPushButton *m_Cancel;
    QPushButton *m_Ok;

    void setupUi(QDialog *SmoothingDialogBase)
    {
        if (SmoothingDialogBase->objectName().isEmpty())
            SmoothingDialogBase->setObjectName(QString::fromUtf8("SmoothingDialogBase"));
        SmoothingDialogBase->resize(234, 110);
        gridLayout = new QGridLayout(SmoothingDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_DeltaText = new QLabel(SmoothingDialogBase);
        m_DeltaText->setObjectName(QString::fromUtf8("m_DeltaText"));
        m_DeltaText->setWordWrap(false);

        gridLayout->addWidget(m_DeltaText, 0, 0, 1, 1);

        m_Delta = new QLineEdit(SmoothingDialogBase);
        m_Delta->setObjectName(QString::fromUtf8("m_Delta"));

        gridLayout->addWidget(m_Delta, 0, 1, 1, 2);

        m_FixBoundary = new QCheckBox(SmoothingDialogBase);
        m_FixBoundary->setObjectName(QString::fromUtf8("m_FixBoundary"));

        gridLayout->addWidget(m_FixBoundary, 1, 0, 1, 3);

        m_Cancel = new QPushButton(SmoothingDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));

        gridLayout->addWidget(m_Cancel, 2, 0, 1, 2);

        m_Ok = new QPushButton(SmoothingDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));

        gridLayout->addWidget(m_Ok, 2, 2, 1, 1);


        retranslateUi(SmoothingDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), SmoothingDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), SmoothingDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(SmoothingDialogBase);
    } // setupUi

    void retranslateUi(QDialog *SmoothingDialogBase)
    {
        SmoothingDialogBase->setWindowTitle(QApplication::translate("SmoothingDialogBase", "Smoothing", 0, QApplication::UnicodeUTF8));
        m_DeltaText->setText(QApplication::translate("SmoothingDialogBase", "Delta:", 0, QApplication::UnicodeUTF8));
        m_Delta->setText(QApplication::translate("SmoothingDialogBase", "0.1", 0, QApplication::UnicodeUTF8));
        m_FixBoundary->setText(QApplication::translate("SmoothingDialogBase", "Fix Boundary", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("SmoothingDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("SmoothingDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SmoothingDialogBase: public Ui_SmoothingDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class SmoothingDialogBase : public QDialog, public Ui::SmoothingDialogBase
{
    Q_OBJECT

public:
    SmoothingDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~SmoothingDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // SMOOTHINGDIALOGBASE_H
