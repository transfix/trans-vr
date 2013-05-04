#ifndef CONTRASTENHANCEMENTDIALOG_H
#define CONTRASTENHANCEMENTDIALOG_H

#include <qvariant.h>


#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_ContrastEnhancementDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *m_ResistorText;
    QLineEdit *m_Resistor;
    QHBoxLayout *hboxLayout;
    QCheckBox *m_Preview;
    QPushButton *m_CancelButton;
    QPushButton *m_RunButton;

    void setupUi(QDialog *ContrastEnhancementDialog)
    {
        if (ContrastEnhancementDialog->objectName().isEmpty())
            ContrastEnhancementDialog->setObjectName(QString::fromUtf8("ContrastEnhancementDialog"));
        ContrastEnhancementDialog->resize(254, 84);
        gridLayout = new QGridLayout(ContrastEnhancementDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setResizeMode(QGridLayout::Fixed);
        m_ResistorText = new QLabel(ContrastEnhancementDialog);
        m_ResistorText->setObjectName(QString::fromUtf8("m_ResistorText"));
        m_ResistorText->setWordWrap(false);

        gridLayout->addWidget(m_ResistorText, 0, 0, 1, 1);

        m_Resistor = new QLineEdit(ContrastEnhancementDialog);
        m_Resistor->setObjectName(QString::fromUtf8("m_Resistor"));

        gridLayout->addWidget(m_Resistor, 0, 1, 1, 1);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        m_Preview = new QCheckBox(ContrastEnhancementDialog);
        m_Preview->setObjectName(QString::fromUtf8("m_Preview"));
        m_Preview->setChecked(true);

        hboxLayout->addWidget(m_Preview);

        m_CancelButton = new QPushButton(ContrastEnhancementDialog);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));

        hboxLayout->addWidget(m_CancelButton);

        m_RunButton = new QPushButton(ContrastEnhancementDialog);
        m_RunButton->setObjectName(QString::fromUtf8("m_RunButton"));

        hboxLayout->addWidget(m_RunButton);


        gridLayout->addLayout(hboxLayout, 1, 0, 1, 2);


        retranslateUi(ContrastEnhancementDialog);
        QObject::connect(m_RunButton, SIGNAL(clicked()), ContrastEnhancementDialog, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), ContrastEnhancementDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(ContrastEnhancementDialog);
    } // setupUi

    void retranslateUi(QDialog *ContrastEnhancementDialog)
    {
        ContrastEnhancementDialog->setWindowTitle(QApplication::translate("ContrastEnhancementDialog", "Contrast Enhancement", 0, QApplication::UnicodeUTF8));
        m_ResistorText->setText(QApplication::translate("ContrastEnhancementDialog", "Resistor", 0, QApplication::UnicodeUTF8));
        m_Resistor->setText(QApplication::translate("ContrastEnhancementDialog", "0.95", 0, QApplication::UnicodeUTF8));
        m_Preview->setText(QApplication::translate("ContrastEnhancementDialog", "Preview", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("ContrastEnhancementDialog", "Cancel", 0, QApplication::UnicodeUTF8));
        m_RunButton->setText(QApplication::translate("ContrastEnhancementDialog", "Run", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ContrastEnhancementDialog: public Ui_ContrastEnhancementDialog {};
} // namespace Ui

QT_END_NAMESPACE

class ContrastEnhancementDialog : public QDialog, public Ui::ContrastEnhancementDialog
{
    Q_OBJECT

public:
    ContrastEnhancementDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~ContrastEnhancementDialog();

protected slots:
    virtual void languageChange();

};

#endif // CONTRASTENHANCEMENTDIALOG_H
