#ifndef SERVERSELECTORDIALOGBASE_H
#define SERVERSELECTORDIALOGBASE_H

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
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_ServerSelectorDialogBase
{
public:
    QVBoxLayout *vboxLayout;
    QLabel *TextLabel3;
    Q3ListBox *m_ServerListBox;
    QHBoxLayout *hboxLayout;
    QSpacerItem *Horizontal_Spacing2;
    QPushButton *buttonOk;
    QPushButton *buttonCancel;

    void setupUi(QDialog *ServerSelectorDialogBase)
    {
        if (ServerSelectorDialogBase->objectName().isEmpty())
            ServerSelectorDialogBase->setObjectName(QString::fromUtf8("ServerSelectorDialogBase"));
        ServerSelectorDialogBase->resize(283, 161);
        ServerSelectorDialogBase->setSizeGripEnabled(true);
        vboxLayout = new QVBoxLayout(ServerSelectorDialogBase);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        TextLabel3 = new QLabel(ServerSelectorDialogBase);
        TextLabel3->setObjectName(QString::fromUtf8("TextLabel3"));
        TextLabel3->setWordWrap(false);

        vboxLayout->addWidget(TextLabel3);

        m_ServerListBox = new Q3ListBox(ServerSelectorDialogBase);
        m_ServerListBox->setObjectName(QString::fromUtf8("m_ServerListBox"));
        m_ServerListBox->setCurrentItem(-1);

        vboxLayout->addWidget(m_ServerListBox);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setContentsMargins(0, 0, 0, 0);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        Horizontal_Spacing2 = new QSpacerItem(20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout->addItem(Horizontal_Spacing2);

        buttonOk = new QPushButton(ServerSelectorDialogBase);
        buttonOk->setObjectName(QString::fromUtf8("buttonOk"));
        buttonOk->setAutoDefault(true);
        buttonOk->setDefault(true);

        hboxLayout->addWidget(buttonOk);

        buttonCancel = new QPushButton(ServerSelectorDialogBase);
        buttonCancel->setObjectName(QString::fromUtf8("buttonCancel"));
        buttonCancel->setAutoDefault(true);

        hboxLayout->addWidget(buttonCancel);


        vboxLayout->addLayout(hboxLayout);


        retranslateUi(ServerSelectorDialogBase);
        QObject::connect(buttonOk, SIGNAL(clicked()), ServerSelectorDialogBase, SLOT(accept()));
        QObject::connect(buttonCancel, SIGNAL(clicked()), ServerSelectorDialogBase, SLOT(reject()));

        QMetaObject::connectSlotsByName(ServerSelectorDialogBase);
    } // setupUi

    void retranslateUi(QDialog *ServerSelectorDialogBase)
    {
        ServerSelectorDialogBase->setWindowTitle(QApplication::translate("ServerSelectorDialogBase", "ServerSelectorDialog", 0, QApplication::UnicodeUTF8));
        TextLabel3->setText(QApplication::translate("ServerSelectorDialogBase", "Files:", 0, QApplication::UnicodeUTF8));
        m_ServerListBox->clear();
        m_ServerListBox->insertItem(QApplication::translate("ServerSelectorDialogBase", "Raycasting Server", 0, QApplication::UnicodeUTF8));
        m_ServerListBox->insertItem(QApplication::translate("ServerSelectorDialogBase", "Texturebased Server", 0, QApplication::UnicodeUTF8));
        m_ServerListBox->insertItem(QApplication::translate("ServerSelectorDialogBase", "Isocontour Server", 0, QApplication::UnicodeUTF8));
        buttonOk->setWindowTitle(QString());
        buttonOk->setText(QApplication::translate("ServerSelectorDialogBase", "&OK", 0, QApplication::UnicodeUTF8));
        buttonCancel->setText(QApplication::translate("ServerSelectorDialogBase", "&Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ServerSelectorDialogBase: public Ui_ServerSelectorDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class ServerSelectorDialogBase : public QDialog, public Ui::ServerSelectorDialogBase
{
    Q_OBJECT

public:
    ServerSelectorDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~ServerSelectorDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // SERVERSELECTORDIALOGBASE_H
