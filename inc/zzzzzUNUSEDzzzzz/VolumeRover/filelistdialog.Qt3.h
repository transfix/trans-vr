#ifndef FILELISTDIALOGBASE_H
#define FILELISTDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ListBox>
#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_FileListDialogBase
{
public:
    QVBoxLayout *vboxLayout;
    Q3ListBox *FileList;
    QHBoxLayout *hboxLayout;
    QSpacerItem *Horizontal_Spacing2;
    QPushButton *buttonOk;
    QPushButton *buttonCancel;

    void setupUi(QDialog *FileListDialogBase)
    {
        if (FileListDialogBase->objectName().isEmpty())
            FileListDialogBase->setObjectName(QString::fromUtf8("FileListDialogBase"));
        FileListDialogBase->resize(519, 285);
        FileListDialogBase->setSizeGripEnabled(true);
        vboxLayout = new QVBoxLayout(FileListDialogBase);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        FileList = new Q3ListBox(FileListDialogBase);
        FileList->setObjectName(QString::fromUtf8("FileList"));

        vboxLayout->addWidget(FileList);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setContentsMargins(0, 0, 0, 0);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        Horizontal_Spacing2 = new QSpacerItem(20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout->addItem(Horizontal_Spacing2);

        buttonOk = new QPushButton(FileListDialogBase);
        buttonOk->setObjectName(QString::fromUtf8("buttonOk"));
        buttonOk->setAutoDefault(true);
        buttonOk->setDefault(true);

        hboxLayout->addWidget(buttonOk);

        buttonCancel = new QPushButton(FileListDialogBase);
        buttonCancel->setObjectName(QString::fromUtf8("buttonCancel"));
        buttonCancel->setAutoDefault(true);

        hboxLayout->addWidget(buttonCancel);


        vboxLayout->addLayout(hboxLayout);


        retranslateUi(FileListDialogBase);
        QObject::connect(buttonOk, SIGNAL(clicked()), FileListDialogBase, SLOT(accept()));
        QObject::connect(buttonCancel, SIGNAL(clicked()), FileListDialogBase, SLOT(reject()));

        QMetaObject::connectSlotsByName(FileListDialogBase);
    } // setupUi

    void retranslateUi(QDialog *FileListDialogBase)
    {
        FileListDialogBase->setWindowTitle(QApplication::translate("FileListDialogBase", "DataCutter File List", 0, QApplication::UnicodeUTF8));
        buttonOk->setWindowTitle(QString());
        buttonOk->setText(QApplication::translate("FileListDialogBase", "&OK", 0, QApplication::UnicodeUTF8));
        buttonCancel->setText(QApplication::translate("FileListDialogBase", "&Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class FileListDialogBase: public Ui_FileListDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class FileListDialogBase : public QDialog, public Ui::FileListDialogBase
{
    Q_OBJECT

public:
    FileListDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~FileListDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // FILELISTDIALOGBASE_H
