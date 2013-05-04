#ifndef TEXTURESERVERSETTINGSDIALOGBASE_H
#define TEXTURESERVERSETTINGSDIALOGBASE_H

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

class Ui_TextureServerSettingsDialogBase
{
public:
    QVBoxLayout *vboxLayout;
    QLabel *TextLabel3;
    Q3ListBox *m_FileListBox;
    QHBoxLayout *hboxLayout;
    QSpacerItem *Horizontal_Spacing2;
    QPushButton *buttonOk;
    QPushButton *buttonCancel;

    void setupUi(QDialog *TextureServerSettingsDialogBase)
    {
        if (TextureServerSettingsDialogBase->objectName().isEmpty())
            TextureServerSettingsDialogBase->setObjectName(QString::fromUtf8("TextureServerSettingsDialogBase"));
        TextureServerSettingsDialogBase->resize(282, 159);
        TextureServerSettingsDialogBase->setSizeGripEnabled(true);
        vboxLayout = new QVBoxLayout(TextureServerSettingsDialogBase);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        TextLabel3 = new QLabel(TextureServerSettingsDialogBase);
        TextLabel3->setObjectName(QString::fromUtf8("TextLabel3"));
        TextLabel3->setWordWrap(false);

        vboxLayout->addWidget(TextLabel3);

        m_FileListBox = new Q3ListBox(TextureServerSettingsDialogBase);
        m_FileListBox->setObjectName(QString::fromUtf8("m_FileListBox"));

        vboxLayout->addWidget(m_FileListBox);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setContentsMargins(0, 0, 0, 0);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        Horizontal_Spacing2 = new QSpacerItem(20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout->addItem(Horizontal_Spacing2);

        buttonOk = new QPushButton(TextureServerSettingsDialogBase);
        buttonOk->setObjectName(QString::fromUtf8("buttonOk"));
        buttonOk->setAutoDefault(true);
        buttonOk->setDefault(true);

        hboxLayout->addWidget(buttonOk);

        buttonCancel = new QPushButton(TextureServerSettingsDialogBase);
        buttonCancel->setObjectName(QString::fromUtf8("buttonCancel"));
        buttonCancel->setAutoDefault(true);

        hboxLayout->addWidget(buttonCancel);


        vboxLayout->addLayout(hboxLayout);


        retranslateUi(TextureServerSettingsDialogBase);
        QObject::connect(buttonOk, SIGNAL(clicked()), TextureServerSettingsDialogBase, SLOT(accept()));
        QObject::connect(buttonCancel, SIGNAL(clicked()), TextureServerSettingsDialogBase, SLOT(reject()));

        QMetaObject::connectSlotsByName(TextureServerSettingsDialogBase);
    } // setupUi

    void retranslateUi(QDialog *TextureServerSettingsDialogBase)
    {
        TextureServerSettingsDialogBase->setWindowTitle(QApplication::translate("TextureServerSettingsDialogBase", "TextureServerSettingsDialog", 0, QApplication::UnicodeUTF8));
        TextLabel3->setText(QApplication::translate("TextureServerSettingsDialogBase", "Files:", 0, QApplication::UnicodeUTF8));
        buttonOk->setWindowTitle(QString());
        buttonOk->setText(QApplication::translate("TextureServerSettingsDialogBase", "&OK", 0, QApplication::UnicodeUTF8));
        buttonCancel->setText(QApplication::translate("TextureServerSettingsDialogBase", "&Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class TextureServerSettingsDialogBase: public Ui_TextureServerSettingsDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class TextureServerSettingsDialogBase : public QDialog, public Ui::TextureServerSettingsDialogBase
{
    Q_OBJECT

public:
    TextureServerSettingsDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~TextureServerSettingsDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // TEXTURESERVERSETTINGSDIALOGBASE_H
