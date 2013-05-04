#ifndef GDTVFILTERDIALOG_H
#define GDTVFILTERDIALOG_H

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

class Ui_GDTVFilterDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *m_ParaterqText;
    QLabel *m_LambdaText;
    QLabel *m_IterationText;
    QLabel *m_NeigbourText;
    QLineEdit *m_ParaterqEdit;
    QLineEdit *m_LambdaEdit;
    QLineEdit *m_IterationEdit;
    QLineEdit *m_NeigbourEdit;
    QCheckBox *m_Preview;
    QPushButton *m_RunButton;
    QPushButton *m_CancelButton;

    void setupUi(QDialog *GDTVFilterDialog)
    {
        if (GDTVFilterDialog->objectName().isEmpty())
            GDTVFilterDialog->setObjectName(QString::fromUtf8("GDTVFilterDialog"));
        GDTVFilterDialog->resize(240, 199);
        gridLayout = new QGridLayout(GDTVFilterDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_ParaterqText = new QLabel(GDTVFilterDialog);
        m_ParaterqText->setObjectName(QString::fromUtf8("m_ParaterqText"));
        m_ParaterqText->setWordWrap(false);

        gridLayout->addWidget(m_ParaterqText, 0, 0, 1, 1);

        m_LambdaText = new QLabel(GDTVFilterDialog);
        m_LambdaText->setObjectName(QString::fromUtf8("m_LambdaText"));
        m_LambdaText->setWordWrap(false);

        gridLayout->addWidget(m_LambdaText, 1, 0, 1, 1);

        m_IterationText = new QLabel(GDTVFilterDialog);
        m_IterationText->setObjectName(QString::fromUtf8("m_IterationText"));
        m_IterationText->setWordWrap(false);

        gridLayout->addWidget(m_IterationText, 2, 0, 1, 1);

        m_NeigbourText = new QLabel(GDTVFilterDialog);
        m_NeigbourText->setObjectName(QString::fromUtf8("m_NeigbourText"));
        m_NeigbourText->setWordWrap(false);

        gridLayout->addWidget(m_NeigbourText, 3, 0, 1, 1);

        m_ParaterqEdit = new QLineEdit(GDTVFilterDialog);
        m_ParaterqEdit->setObjectName(QString::fromUtf8("m_ParaterqEdit"));

        gridLayout->addWidget(m_ParaterqEdit, 0, 1, 1, 1);

        m_LambdaEdit = new QLineEdit(GDTVFilterDialog);
        m_LambdaEdit->setObjectName(QString::fromUtf8("m_LambdaEdit"));

        gridLayout->addWidget(m_LambdaEdit, 1, 1, 1, 1);

        m_IterationEdit = new QLineEdit(GDTVFilterDialog);
        m_IterationEdit->setObjectName(QString::fromUtf8("m_IterationEdit"));

        gridLayout->addWidget(m_IterationEdit, 2, 1, 1, 1);

        m_NeigbourEdit = new QLineEdit(GDTVFilterDialog);
        m_NeigbourEdit->setObjectName(QString::fromUtf8("m_NeigbourEdit"));

        gridLayout->addWidget(m_NeigbourEdit, 3, 1, 1, 1);

        m_Preview = new QCheckBox(GDTVFilterDialog);
        m_Preview->setObjectName(QString::fromUtf8("m_Preview"));
        m_Preview->setChecked(true);

        gridLayout->addWidget(m_Preview, 4, 0, 1, 2);

        m_RunButton = new QPushButton(GDTVFilterDialog);
        m_RunButton->setObjectName(QString::fromUtf8("m_RunButton"));
        m_RunButton->setDefault(true);

        gridLayout->addWidget(m_RunButton, 5, 1, 1, 1);

        m_CancelButton = new QPushButton(GDTVFilterDialog);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));
        m_CancelButton->setAutoDefault(false);

        gridLayout->addWidget(m_CancelButton, 5, 0, 1, 1);


        retranslateUi(GDTVFilterDialog);
        QObject::connect(m_RunButton, SIGNAL(clicked()), GDTVFilterDialog, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), GDTVFilterDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(GDTVFilterDialog);
    } // setupUi

    void retranslateUi(QDialog *GDTVFilterDialog)
    {
        GDTVFilterDialog->setWindowTitle(QApplication::translate("GDTVFilterDialog", "GDTV Filter", 0, QApplication::UnicodeUTF8));
        m_ParaterqText->setText(QApplication::translate("GDTVFilterDialog", "Parameter q", 0, QApplication::UnicodeUTF8));
        m_LambdaText->setText(QApplication::translate("GDTVFilterDialog", "Lambda", 0, QApplication::UnicodeUTF8));
        m_IterationText->setText(QApplication::translate("GDTVFilterDialog", "Iteration Num", 0, QApplication::UnicodeUTF8));
        m_NeigbourText->setText(QApplication::translate("GDTVFilterDialog", "Neighbourhood", 0, QApplication::UnicodeUTF8));
        m_ParaterqEdit->setText(QApplication::translate("GDTVFilterDialog", "1.0", 0, QApplication::UnicodeUTF8));
        m_LambdaEdit->setText(QApplication::translate("GDTVFilterDialog", "0.0001", 0, QApplication::UnicodeUTF8));
        m_IterationEdit->setText(QApplication::translate("GDTVFilterDialog", "1", 0, QApplication::UnicodeUTF8));
        m_NeigbourEdit->setText(QApplication::translate("GDTVFilterDialog", "0", 0, QApplication::UnicodeUTF8));
        m_Preview->setText(QApplication::translate("GDTVFilterDialog", "Preview", 0, QApplication::UnicodeUTF8));
        m_RunButton->setText(QApplication::translate("GDTVFilterDialog", "Run", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("GDTVFilterDialog", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GDTVFilterDialog: public Ui_GDTVFilterDialog {};
} // namespace Ui

QT_END_NAMESPACE

class GDTVFilterDialog : public QDialog, public Ui::GDTVFilterDialog
{
    Q_OBJECT

public:
    GDTVFilterDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~GDTVFilterDialog();

protected slots:
    virtual void languageChange();

};

#endif // GDTVFILTERDIALOG_H
