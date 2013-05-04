#ifndef PEDETECTIONDIALOG_H
#define PEDETECTIONDIALOG_H

#include <qvariant.h>


#include <Qt3Support/Q3GroupBox>
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

class Ui_PEDetectionDialog
{
public:
    QGridLayout *gridLayout;
    QCheckBox *m_RunRemotely;
    Q3GroupBox *m_RemoteComputationGroup;
    QGridLayout *gridLayout1;
    QLabel *m_HostnameText;
    QLineEdit *m_Hostname;
    QLabel *m_PortText;
    QLineEdit *m_Port;
    QLabel *m_RemoteFileText;
    QLineEdit *m_RemoteFile;
    QPushButton *m_Run;
    QPushButton *m_Cancel;

    void setupUi(QDialog *PEDetectionDialog)
    {
        if (PEDetectionDialog->objectName().isEmpty())
            PEDetectionDialog->setObjectName(QString::fromUtf8("PEDetectionDialog"));
        PEDetectionDialog->resize(273, 201);
        gridLayout = new QGridLayout(PEDetectionDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_RunRemotely = new QCheckBox(PEDetectionDialog);
        m_RunRemotely->setObjectName(QString::fromUtf8("m_RunRemotely"));

        gridLayout->addWidget(m_RunRemotely, 0, 0, 1, 2);

        m_RemoteComputationGroup = new Q3GroupBox(PEDetectionDialog);
        m_RemoteComputationGroup->setObjectName(QString::fromUtf8("m_RemoteComputationGroup"));
        m_RemoteComputationGroup->setEnabled(false);
        m_RemoteComputationGroup->setColumnLayout(0, Qt::Vertical);
        m_RemoteComputationGroup->layout()->setSpacing(6);
        m_RemoteComputationGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_RemoteComputationGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_HostnameText = new QLabel(m_RemoteComputationGroup);
        m_HostnameText->setObjectName(QString::fromUtf8("m_HostnameText"));
        m_HostnameText->setWordWrap(false);

        gridLayout1->addWidget(m_HostnameText, 0, 0, 1, 1);

        m_Hostname = new QLineEdit(m_RemoteComputationGroup);
        m_Hostname->setObjectName(QString::fromUtf8("m_Hostname"));

        gridLayout1->addWidget(m_Hostname, 0, 1, 1, 1);

        m_PortText = new QLabel(m_RemoteComputationGroup);
        m_PortText->setObjectName(QString::fromUtf8("m_PortText"));
        m_PortText->setWordWrap(false);

        gridLayout1->addWidget(m_PortText, 1, 0, 1, 1);

        m_Port = new QLineEdit(m_RemoteComputationGroup);
        m_Port->setObjectName(QString::fromUtf8("m_Port"));

        gridLayout1->addWidget(m_Port, 1, 1, 1, 1);

        m_RemoteFileText = new QLabel(m_RemoteComputationGroup);
        m_RemoteFileText->setObjectName(QString::fromUtf8("m_RemoteFileText"));
        m_RemoteFileText->setWordWrap(false);

        gridLayout1->addWidget(m_RemoteFileText, 2, 0, 1, 1);

        m_RemoteFile = new QLineEdit(m_RemoteComputationGroup);
        m_RemoteFile->setObjectName(QString::fromUtf8("m_RemoteFile"));

        gridLayout1->addWidget(m_RemoteFile, 2, 1, 1, 1);


        gridLayout->addWidget(m_RemoteComputationGroup, 1, 0, 1, 2);

        m_Run = new QPushButton(PEDetectionDialog);
        m_Run->setObjectName(QString::fromUtf8("m_Run"));

        gridLayout->addWidget(m_Run, 2, 0, 1, 1);

        m_Cancel = new QPushButton(PEDetectionDialog);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));

        gridLayout->addWidget(m_Cancel, 2, 1, 1, 1);


        retranslateUi(PEDetectionDialog);
        QObject::connect(m_RunRemotely, SIGNAL(toggled(bool)), m_RemoteComputationGroup, SLOT(setEnabled(bool)));
        QObject::connect(m_Run, SIGNAL(clicked()), PEDetectionDialog, SLOT(accept()));
        QObject::connect(m_Cancel, SIGNAL(clicked()), PEDetectionDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(PEDetectionDialog);
    } // setupUi

    void retranslateUi(QDialog *PEDetectionDialog)
    {
        PEDetectionDialog->setWindowTitle(QApplication::translate("PEDetectionDialog", "Pulmonary Embolus Detection", 0, QApplication::UnicodeUTF8));
        m_RunRemotely->setText(QApplication::translate("PEDetectionDialog", "Run remotely", 0, QApplication::UnicodeUTF8));
        m_RemoteComputationGroup->setTitle(QApplication::translate("PEDetectionDialog", "Remote Computation", 0, QApplication::UnicodeUTF8));
        m_HostnameText->setText(QApplication::translate("PEDetectionDialog", "Hostname:", 0, QApplication::UnicodeUTF8));
        m_PortText->setText(QApplication::translate("PEDetectionDialog", "Port:", 0, QApplication::UnicodeUTF8));
        m_RemoteFileText->setText(QApplication::translate("PEDetectionDialog", "Remote File:", 0, QApplication::UnicodeUTF8));
        m_Run->setText(QApplication::translate("PEDetectionDialog", "Run", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("PEDetectionDialog", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class PEDetectionDialog: public Ui_PEDetectionDialog {};
} // namespace Ui

QT_END_NAMESPACE

class PEDetectionDialog : public QDialog, public Ui::PEDetectionDialog
{
    Q_OBJECT

public:
    PEDetectionDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~PEDetectionDialog();

protected slots:
    virtual void languageChange();

};

#endif // PEDETECTIONDIALOG_H
