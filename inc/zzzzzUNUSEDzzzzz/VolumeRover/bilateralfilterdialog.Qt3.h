#ifndef BILATERALFILTERDIALOG_H
#define BILATERALFILTERDIALOG_H

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

class Ui_BilateralFilterDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *m_SpatSigText;
    QLabel *m_FilRadText;
    QLabel *m_RadSigText;
    QLineEdit *m_RadSigEdit;
    QLineEdit *m_SpatSigEdit;
    QLineEdit *m_FilRadEdit;
    QHBoxLayout *hboxLayout;
    QCheckBox *m_Preview;
    QPushButton *m_CancelButton;
    QPushButton *m_RunButton;

    void setupUi(QDialog *BilateralFilterDialog)
    {
        if (BilateralFilterDialog->objectName().isEmpty())
            BilateralFilterDialog->setObjectName(QString::fromUtf8("BilateralFilterDialog"));
        BilateralFilterDialog->resize(282, 140);
        BilateralFilterDialog->setModal(false);
        gridLayout = new QGridLayout(BilateralFilterDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setResizeMode(QGridLayout::Fixed);
        m_SpatSigText = new QLabel(BilateralFilterDialog);
        m_SpatSigText->setObjectName(QString::fromUtf8("m_SpatSigText"));
        m_SpatSigText->setAlignment(Qt::AlignVCenter);
        m_SpatSigText->setWordWrap(false);

        gridLayout->addWidget(m_SpatSigText, 1, 0, 1, 1);

        m_FilRadText = new QLabel(BilateralFilterDialog);
        m_FilRadText->setObjectName(QString::fromUtf8("m_FilRadText"));
        m_FilRadText->setAlignment(Qt::AlignVCenter);
        m_FilRadText->setWordWrap(false);

        gridLayout->addWidget(m_FilRadText, 2, 0, 1, 1);

        m_RadSigText = new QLabel(BilateralFilterDialog);
        m_RadSigText->setObjectName(QString::fromUtf8("m_RadSigText"));
        m_RadSigText->setAlignment(Qt::AlignVCenter);
        m_RadSigText->setWordWrap(false);

        gridLayout->addWidget(m_RadSigText, 0, 0, 1, 1);

        m_RadSigEdit = new QLineEdit(BilateralFilterDialog);
        m_RadSigEdit->setObjectName(QString::fromUtf8("m_RadSigEdit"));

        gridLayout->addWidget(m_RadSigEdit, 0, 1, 1, 1);

        m_SpatSigEdit = new QLineEdit(BilateralFilterDialog);
        m_SpatSigEdit->setObjectName(QString::fromUtf8("m_SpatSigEdit"));

        gridLayout->addWidget(m_SpatSigEdit, 1, 1, 1, 1);

        m_FilRadEdit = new QLineEdit(BilateralFilterDialog);
        m_FilRadEdit->setObjectName(QString::fromUtf8("m_FilRadEdit"));

        gridLayout->addWidget(m_FilRadEdit, 2, 1, 1, 1);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        m_Preview = new QCheckBox(BilateralFilterDialog);
        m_Preview->setObjectName(QString::fromUtf8("m_Preview"));
        m_Preview->setChecked(true);

        hboxLayout->addWidget(m_Preview);

        m_CancelButton = new QPushButton(BilateralFilterDialog);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));

        hboxLayout->addWidget(m_CancelButton);

        m_RunButton = new QPushButton(BilateralFilterDialog);
        m_RunButton->setObjectName(QString::fromUtf8("m_RunButton"));

        hboxLayout->addWidget(m_RunButton);


        gridLayout->addLayout(hboxLayout, 3, 0, 1, 2);


        retranslateUi(BilateralFilterDialog);
        QObject::connect(m_RunButton, SIGNAL(clicked()), BilateralFilterDialog, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), BilateralFilterDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(BilateralFilterDialog);
    } // setupUi

    void retranslateUi(QDialog *BilateralFilterDialog)
    {
        BilateralFilterDialog->setWindowTitle(QApplication::translate("BilateralFilterDialog", "Bilateral Filter", 0, QApplication::UnicodeUTF8));
        m_SpatSigText->setText(QApplication::translate("BilateralFilterDialog", "Spatial Sigma", 0, QApplication::UnicodeUTF8));
        m_FilRadText->setText(QApplication::translate("BilateralFilterDialog", "Filter Radius", 0, QApplication::UnicodeUTF8));
        m_RadSigText->setText(QApplication::translate("BilateralFilterDialog", "Radiometric Sigma", 0, QApplication::UnicodeUTF8));
        m_RadSigEdit->setText(QApplication::translate("BilateralFilterDialog", "200", 0, QApplication::UnicodeUTF8));
        m_RadSigEdit->setInputMask(QString());
        m_SpatSigEdit->setText(QApplication::translate("BilateralFilterDialog", "1.5", 0, QApplication::UnicodeUTF8));
        m_FilRadEdit->setText(QApplication::translate("BilateralFilterDialog", "2", 0, QApplication::UnicodeUTF8));
        m_Preview->setText(QApplication::translate("BilateralFilterDialog", "Preview", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("BilateralFilterDialog", "Cancel", 0, QApplication::UnicodeUTF8));
        m_RunButton->setText(QApplication::translate("BilateralFilterDialog", "Run", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class BilateralFilterDialog: public Ui_BilateralFilterDialog {};
} // namespace Ui

QT_END_NAMESPACE

class BilateralFilterDialog : public QDialog, public Ui::BilateralFilterDialog
{
    Q_OBJECT

public:
    BilateralFilterDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~BilateralFilterDialog();

protected slots:
    virtual void languageChange();

private:
    void init();

};

#endif // BILATERALFILTERDIALOG_H
