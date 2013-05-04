#ifndef ANISOTROPICDIFFUSIONDIALOG_H
#define ANISOTROPICDIFFUSIONDIALOG_H

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

class Ui_AnisotropicDiffusionDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *m_IterationsText;
    QLineEdit *m_Iterations;
    QHBoxLayout *hboxLayout;
    QCheckBox *m_Preview;
    QPushButton *m_CancelButton;
    QPushButton *m_RunButton;

    void setupUi(QDialog *AnisotropicDiffusionDialog)
    {
        if (AnisotropicDiffusionDialog->objectName().isEmpty())
            AnisotropicDiffusionDialog->setObjectName(QString::fromUtf8("AnisotropicDiffusionDialog"));
        AnisotropicDiffusionDialog->resize(254, 84);
        gridLayout = new QGridLayout(AnisotropicDiffusionDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_IterationsText = new QLabel(AnisotropicDiffusionDialog);
        m_IterationsText->setObjectName(QString::fromUtf8("m_IterationsText"));
        m_IterationsText->setWordWrap(false);

        gridLayout->addWidget(m_IterationsText, 0, 0, 1, 1);

        m_Iterations = new QLineEdit(AnisotropicDiffusionDialog);
        m_Iterations->setObjectName(QString::fromUtf8("m_Iterations"));

        gridLayout->addWidget(m_Iterations, 0, 1, 1, 1);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        m_Preview = new QCheckBox(AnisotropicDiffusionDialog);
        m_Preview->setObjectName(QString::fromUtf8("m_Preview"));
        m_Preview->setChecked(true);

        hboxLayout->addWidget(m_Preview);

        m_CancelButton = new QPushButton(AnisotropicDiffusionDialog);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));

        hboxLayout->addWidget(m_CancelButton);

        m_RunButton = new QPushButton(AnisotropicDiffusionDialog);
        m_RunButton->setObjectName(QString::fromUtf8("m_RunButton"));

        hboxLayout->addWidget(m_RunButton);


        gridLayout->addLayout(hboxLayout, 1, 0, 1, 2);


        retranslateUi(AnisotropicDiffusionDialog);
        QObject::connect(m_RunButton, SIGNAL(clicked()), AnisotropicDiffusionDialog, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), AnisotropicDiffusionDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(AnisotropicDiffusionDialog);
    } // setupUi

    void retranslateUi(QDialog *AnisotropicDiffusionDialog)
    {
        AnisotropicDiffusionDialog->setWindowTitle(QApplication::translate("AnisotropicDiffusionDialog", "Anisotropic Diffusion", 0, QApplication::UnicodeUTF8));
        m_IterationsText->setText(QApplication::translate("AnisotropicDiffusionDialog", "Iterations:", 0, QApplication::UnicodeUTF8));
        m_Iterations->setText(QApplication::translate("AnisotropicDiffusionDialog", "20", 0, QApplication::UnicodeUTF8));
        m_Preview->setText(QApplication::translate("AnisotropicDiffusionDialog", "Preview", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("AnisotropicDiffusionDialog", "Cancel", 0, QApplication::UnicodeUTF8));
        m_RunButton->setText(QApplication::translate("AnisotropicDiffusionDialog", "Run", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class AnisotropicDiffusionDialog: public Ui_AnisotropicDiffusionDialog {};
} // namespace Ui

QT_END_NAMESPACE

class AnisotropicDiffusionDialog : public QDialog, public Ui::AnisotropicDiffusionDialog
{
    Q_OBJECT

public:
    AnisotropicDiffusionDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~AnisotropicDiffusionDialog();

protected slots:
    virtual void languageChange();

};

#endif // ANISOTROPICDIFFUSIONDIALOG_H
