#ifndef VOLUMESAVEDIALOG_H
#define VOLUMESAVEDIALOG_H

#include <qvariant.h>


#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VolumeSaveDialog
{
public:
    QWidget *layout5;
    QVBoxLayout *vboxLayout;
    QHBoxLayout *hboxLayout;
    QLabel *textLabel1;
    QComboBox *volumeFormatMenu;
    QHBoxLayout *hboxLayout1;
    QSpacerItem *spacer1;
    QPushButton *okButton;

    void setupUi(QDialog *VolumeSaveDialog)
    {
        if (VolumeSaveDialog->objectName().isEmpty())
            VolumeSaveDialog->setObjectName(QString::fromUtf8("VolumeSaveDialog"));
        VolumeSaveDialog->resize(193, 94);
        layout5 = new QWidget(VolumeSaveDialog);
        layout5->setObjectName(QString::fromUtf8("layout5"));
        layout5->setGeometry(QRect(10, 10, 170, 70));
        vboxLayout = new QVBoxLayout(layout5);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        vboxLayout->setContentsMargins(0, 0, 0, 0);
        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        textLabel1 = new QLabel(layout5);
        textLabel1->setObjectName(QString::fromUtf8("textLabel1"));
        textLabel1->setWordWrap(false);

        hboxLayout->addWidget(textLabel1);

        volumeFormatMenu = new QComboBox(layout5);
        volumeFormatMenu->setObjectName(QString::fromUtf8("volumeFormatMenu"));

        hboxLayout->addWidget(volumeFormatMenu);


        vboxLayout->addLayout(hboxLayout);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        spacer1 = new QSpacerItem(80, 21, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout1->addItem(spacer1);

        okButton = new QPushButton(layout5);
        okButton->setObjectName(QString::fromUtf8("okButton"));

        hboxLayout1->addWidget(okButton);


        vboxLayout->addLayout(hboxLayout1);


        retranslateUi(VolumeSaveDialog);
        QObject::connect(okButton, SIGNAL(clicked()), VolumeSaveDialog, SLOT(accept()));

        QMetaObject::connectSlotsByName(VolumeSaveDialog);
    } // setupUi

    void retranslateUi(QDialog *VolumeSaveDialog)
    {
        VolumeSaveDialog->setWindowTitle(QApplication::translate("VolumeSaveDialog", "Select a File Format", 0, QApplication::UnicodeUTF8));
        textLabel1->setText(QApplication::translate("VolumeSaveDialog", "Volume Format:", 0, QApplication::UnicodeUTF8));
        volumeFormatMenu->clear();
        volumeFormatMenu->insertItems(0, QStringList()
         << QApplication::translate("VolumeSaveDialog", "rawiv", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("VolumeSaveDialog", "MRC", 0, QApplication::UnicodeUTF8)
        );
        okButton->setText(QApplication::translate("VolumeSaveDialog", "OK", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class VolumeSaveDialog: public Ui_VolumeSaveDialog {};
} // namespace Ui

QT_END_NAMESPACE

class VolumeSaveDialog : public QDialog, public Ui::VolumeSaveDialog
{
    Q_OBJECT

public:
    VolumeSaveDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~VolumeSaveDialog();

protected slots:
    virtual void languageChange();

};

#endif // VOLUMESAVEDIALOG_H
