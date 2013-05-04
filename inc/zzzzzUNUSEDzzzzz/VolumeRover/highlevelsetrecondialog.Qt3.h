#ifndef HIGHLEVELSETRECONDIALOG_H
#define HIGHLEVELSETRECONDIALOG_H

#include <qvariant.h>


#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_HighlevelsetReconDialog
{
public:
    QLabel *m_MaxdimText;
    QLabel *m_edgelengthText;
    QLabel *m_dimText;
    QLineEdit *m_dimEdit;
    QLineEdit *m_endEdit;
    QLineEdit *m_edgelengthEdit;
    QLineEdit *m_MaxdimEdit;
    QLabel *m_endText;
    QPushButton *m_RunButton;
    QPushButton *m_CancelButton;

    void setupUi(QDialog *HighlevelsetReconDialog)
    {
        if (HighlevelsetReconDialog->objectName().isEmpty())
            HighlevelsetReconDialog->setObjectName(QString::fromUtf8("HighlevelsetReconDialog"));
        HighlevelsetReconDialog->resize(291, 261);
        m_MaxdimText = new QLabel(HighlevelsetReconDialog);
        m_MaxdimText->setObjectName(QString::fromUtf8("m_MaxdimText"));
        m_MaxdimText->setGeometry(QRect(30, 150, 91, 31));
        m_MaxdimText->setWordWrap(false);
        m_edgelengthText = new QLabel(HighlevelsetReconDialog);
        m_edgelengthText->setObjectName(QString::fromUtf8("m_edgelengthText"));
        m_edgelengthText->setGeometry(QRect(30, 110, 90, 31));
        m_edgelengthText->setWordWrap(false);
        m_dimText = new QLabel(HighlevelsetReconDialog);
        m_dimText->setObjectName(QString::fromUtf8("m_dimText"));
        m_dimText->setGeometry(QRect(30, 30, 70, 22));
        m_dimText->setWordWrap(false);
        m_dimEdit = new QLineEdit(HighlevelsetReconDialog);
        m_dimEdit->setObjectName(QString::fromUtf8("m_dimEdit"));
        m_dimEdit->setGeometry(QRect(140, 30, 120, 30));
        m_endEdit = new QLineEdit(HighlevelsetReconDialog);
        m_endEdit->setObjectName(QString::fromUtf8("m_endEdit"));
        m_endEdit->setGeometry(QRect(140, 70, 120, 31));
        m_edgelengthEdit = new QLineEdit(HighlevelsetReconDialog);
        m_edgelengthEdit->setObjectName(QString::fromUtf8("m_edgelengthEdit"));
        m_edgelengthEdit->setGeometry(QRect(140, 110, 120, 30));
        m_MaxdimEdit = new QLineEdit(HighlevelsetReconDialog);
        m_MaxdimEdit->setObjectName(QString::fromUtf8("m_MaxdimEdit"));
        m_MaxdimEdit->setGeometry(QRect(140, 150, 120, 30));
        m_endText = new QLabel(HighlevelsetReconDialog);
        m_endText->setObjectName(QString::fromUtf8("m_endText"));
        m_endText->setGeometry(QRect(30, 70, 80, 22));
        m_endText->setWordWrap(false);
        m_RunButton = new QPushButton(HighlevelsetReconDialog);
        m_RunButton->setObjectName(QString::fromUtf8("m_RunButton"));
        m_RunButton->setGeometry(QRect(10, 200, 110, 40));
        m_CancelButton = new QPushButton(HighlevelsetReconDialog);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));
        m_CancelButton->setGeometry(QRect(160, 200, 110, 40));

        retranslateUi(HighlevelsetReconDialog);
        QObject::connect(m_RunButton, SIGNAL(clicked()), HighlevelsetReconDialog, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), HighlevelsetReconDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(HighlevelsetReconDialog);
    } // setupUi

    void retranslateUi(QDialog *HighlevelsetReconDialog)
    {
        HighlevelsetReconDialog->setWindowTitle(QApplication::translate("HighlevelsetReconDialog", "HighlevelsetReconDialog", 0, QApplication::UnicodeUTF8));
        m_MaxdimText->setText(QApplication::translate("HighlevelsetReconDialog", "Max_dim", 0, QApplication::UnicodeUTF8));
        m_edgelengthText->setText(QApplication::translate("HighlevelsetReconDialog", "edgelength", 0, QApplication::UnicodeUTF8));
        m_dimText->setText(QApplication::translate("HighlevelsetReconDialog", "dim", 0, QApplication::UnicodeUTF8));
        m_dimEdit->setText(QApplication::translate("HighlevelsetReconDialog", "128", 0, QApplication::UnicodeUTF8));
        m_endEdit->setText(QApplication::translate("HighlevelsetReconDialog", "1", 0, QApplication::UnicodeUTF8));
        m_edgelengthEdit->setText(QApplication::translate("HighlevelsetReconDialog", "1.0", 0, QApplication::UnicodeUTF8));
        m_MaxdimEdit->setText(QApplication::translate("HighlevelsetReconDialog", "100", 0, QApplication::UnicodeUTF8));
        m_endText->setText(QApplication::translate("HighlevelsetReconDialog", "end", 0, QApplication::UnicodeUTF8));
        m_RunButton->setText(QApplication::translate("HighlevelsetReconDialog", "Run", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("HighlevelsetReconDialog", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class HighlevelsetReconDialog: public Ui_HighlevelsetReconDialog {};
} // namespace Ui

QT_END_NAMESPACE

class HighlevelsetReconDialog : public QDialog, public Ui::HighlevelsetReconDialog
{
    Q_OBJECT

public:
    HighlevelsetReconDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~HighlevelsetReconDialog();

protected slots:
    virtual void languageChange();

private:
    void init();

};

#endif // HIGHLEVELSETRECONDIALOG_H
