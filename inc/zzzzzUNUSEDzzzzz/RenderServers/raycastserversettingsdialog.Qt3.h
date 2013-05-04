#ifndef RAYCASTSERVERSETTINGSDIALOGBASE_H
#define RAYCASTSERVERSETTINGSDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
#include <Qt3Support/Q3GroupBox>
#include <Qt3Support/Q3ListBox>
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
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_RaycastServerSettingsDialogBase
{
public:
    QVBoxLayout *vboxLayout;
    QHBoxLayout *hboxLayout;
    Q3ButtonGroup *ButtonGroup1;
    QVBoxLayout *vboxLayout1;
    QRadioButton *m_ShadedButton;
    QRadioButton *m_UnshadedButton;
    Q3GroupBox *GroupBox1;
    QGridLayout *gridLayout;
    QLineEdit *m_HeightEditBox;
    QLabel *TextLabel2;
    QLineEdit *m_WidthEditBox;
    QLabel *TextLabel1;
    QVBoxLayout *vboxLayout2;
    QCheckBox *m_IsosurfacingBox;
    QLabel *TextLabel3;
    Q3ListBox *m_FileListBox;
    QHBoxLayout *hboxLayout1;
    QSpacerItem *Horizontal_Spacing2;
    QPushButton *buttonOk;
    QPushButton *buttonCancel;

    void setupUi(QDialog *RaycastServerSettingsDialogBase)
    {
        if (RaycastServerSettingsDialogBase->objectName().isEmpty())
            RaycastServerSettingsDialogBase->setObjectName(QString::fromUtf8("RaycastServerSettingsDialogBase"));
        RaycastServerSettingsDialogBase->resize(283, 248);
        RaycastServerSettingsDialogBase->setSizeGripEnabled(true);
        vboxLayout = new QVBoxLayout(RaycastServerSettingsDialogBase);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setContentsMargins(0, 0, 0, 0);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        ButtonGroup1 = new Q3ButtonGroup(RaycastServerSettingsDialogBase);
        ButtonGroup1->setObjectName(QString::fromUtf8("ButtonGroup1"));
        ButtonGroup1->setColumnLayout(0, Qt::Vertical);
        ButtonGroup1->layout()->setSpacing(6);
        ButtonGroup1->layout()->setContentsMargins(11, 11, 11, 11);
        vboxLayout1 = new QVBoxLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(ButtonGroup1->layout());
        if (boxlayout)
            boxlayout->addLayout(vboxLayout1);
        vboxLayout1->setAlignment(Qt::AlignTop);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        m_ShadedButton = new QRadioButton(ButtonGroup1);
        m_ShadedButton->setObjectName(QString::fromUtf8("m_ShadedButton"));

        vboxLayout1->addWidget(m_ShadedButton);

        m_UnshadedButton = new QRadioButton(ButtonGroup1);
        m_UnshadedButton->setObjectName(QString::fromUtf8("m_UnshadedButton"));

        vboxLayout1->addWidget(m_UnshadedButton);


        hboxLayout->addWidget(ButtonGroup1);

        GroupBox1 = new Q3GroupBox(RaycastServerSettingsDialogBase);
        GroupBox1->setObjectName(QString::fromUtf8("GroupBox1"));
        GroupBox1->setColumnLayout(0, Qt::Vertical);
        GroupBox1->layout()->setSpacing(6);
        GroupBox1->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(GroupBox1->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout);
        gridLayout->setAlignment(Qt::AlignTop);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_HeightEditBox = new QLineEdit(GroupBox1);
        m_HeightEditBox->setObjectName(QString::fromUtf8("m_HeightEditBox"));

        gridLayout->addWidget(m_HeightEditBox, 1, 1, 1, 1);

        TextLabel2 = new QLabel(GroupBox1);
        TextLabel2->setObjectName(QString::fromUtf8("TextLabel2"));
        TextLabel2->setWordWrap(false);

        gridLayout->addWidget(TextLabel2, 1, 0, 1, 1);

        m_WidthEditBox = new QLineEdit(GroupBox1);
        m_WidthEditBox->setObjectName(QString::fromUtf8("m_WidthEditBox"));

        gridLayout->addWidget(m_WidthEditBox, 0, 1, 1, 1);

        TextLabel1 = new QLabel(GroupBox1);
        TextLabel1->setObjectName(QString::fromUtf8("TextLabel1"));
        TextLabel1->setWordWrap(false);

        gridLayout->addWidget(TextLabel1, 0, 0, 1, 1);


        hboxLayout->addWidget(GroupBox1);


        vboxLayout->addLayout(hboxLayout);

        vboxLayout2 = new QVBoxLayout();
        vboxLayout2->setSpacing(6);
        vboxLayout2->setContentsMargins(0, 0, 0, 0);
        vboxLayout2->setObjectName(QString::fromUtf8("vboxLayout2"));
        m_IsosurfacingBox = new QCheckBox(RaycastServerSettingsDialogBase);
        m_IsosurfacingBox->setObjectName(QString::fromUtf8("m_IsosurfacingBox"));

        vboxLayout2->addWidget(m_IsosurfacingBox);

        TextLabel3 = new QLabel(RaycastServerSettingsDialogBase);
        TextLabel3->setObjectName(QString::fromUtf8("TextLabel3"));
        TextLabel3->setWordWrap(false);

        vboxLayout2->addWidget(TextLabel3);

        m_FileListBox = new Q3ListBox(RaycastServerSettingsDialogBase);
        m_FileListBox->setObjectName(QString::fromUtf8("m_FileListBox"));

        vboxLayout2->addWidget(m_FileListBox);


        vboxLayout->addLayout(vboxLayout2);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setContentsMargins(0, 0, 0, 0);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        Horizontal_Spacing2 = new QSpacerItem(20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout1->addItem(Horizontal_Spacing2);

        buttonOk = new QPushButton(RaycastServerSettingsDialogBase);
        buttonOk->setObjectName(QString::fromUtf8("buttonOk"));
        buttonOk->setAutoDefault(true);
        buttonOk->setDefault(true);

        hboxLayout1->addWidget(buttonOk);

        buttonCancel = new QPushButton(RaycastServerSettingsDialogBase);
        buttonCancel->setObjectName(QString::fromUtf8("buttonCancel"));
        buttonCancel->setAutoDefault(true);

        hboxLayout1->addWidget(buttonCancel);


        vboxLayout->addLayout(hboxLayout1);

        QWidget::setTabOrder(m_ShadedButton, m_UnshadedButton);
        QWidget::setTabOrder(m_UnshadedButton, m_WidthEditBox);
        QWidget::setTabOrder(m_WidthEditBox, m_HeightEditBox);
        QWidget::setTabOrder(m_HeightEditBox, m_IsosurfacingBox);
        QWidget::setTabOrder(m_IsosurfacingBox, m_FileListBox);
        QWidget::setTabOrder(m_FileListBox, buttonOk);
        QWidget::setTabOrder(buttonOk, buttonCancel);

        retranslateUi(RaycastServerSettingsDialogBase);
        QObject::connect(buttonOk, SIGNAL(clicked()), RaycastServerSettingsDialogBase, SLOT(accept()));
        QObject::connect(buttonCancel, SIGNAL(clicked()), RaycastServerSettingsDialogBase, SLOT(reject()));

        QMetaObject::connectSlotsByName(RaycastServerSettingsDialogBase);
    } // setupUi

    void retranslateUi(QDialog *RaycastServerSettingsDialogBase)
    {
        RaycastServerSettingsDialogBase->setWindowTitle(QApplication::translate("RaycastServerSettingsDialogBase", "RaycastServerSettingsDialog", 0, QApplication::UnicodeUTF8));
        ButtonGroup1->setTitle(QApplication::translate("RaycastServerSettingsDialogBase", "Render Style", 0, QApplication::UnicodeUTF8));
        m_ShadedButton->setText(QApplication::translate("RaycastServerSettingsDialogBase", "Shaded", 0, QApplication::UnicodeUTF8));
        m_UnshadedButton->setText(QApplication::translate("RaycastServerSettingsDialogBase", "Unshaded", 0, QApplication::UnicodeUTF8));
        GroupBox1->setTitle(QApplication::translate("RaycastServerSettingsDialogBase", "Render Resolution", 0, QApplication::UnicodeUTF8));
        TextLabel2->setText(QApplication::translate("RaycastServerSettingsDialogBase", "Height", 0, QApplication::UnicodeUTF8));
        TextLabel1->setText(QApplication::translate("RaycastServerSettingsDialogBase", "Width", 0, QApplication::UnicodeUTF8));
        m_IsosurfacingBox->setText(QApplication::translate("RaycastServerSettingsDialogBase", "Isosurfaces", 0, QApplication::UnicodeUTF8));
        TextLabel3->setText(QApplication::translate("RaycastServerSettingsDialogBase", "Files:", 0, QApplication::UnicodeUTF8));
        buttonOk->setWindowTitle(QString());
        buttonOk->setText(QApplication::translate("RaycastServerSettingsDialogBase", "&OK", 0, QApplication::UnicodeUTF8));
        buttonCancel->setText(QApplication::translate("RaycastServerSettingsDialogBase", "&Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class RaycastServerSettingsDialogBase: public Ui_RaycastServerSettingsDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class RaycastServerSettingsDialogBase : public QDialog, public Ui::RaycastServerSettingsDialogBase
{
    Q_OBJECT

public:
    RaycastServerSettingsDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~RaycastServerSettingsDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // RAYCASTSERVERSETTINGSDIALOGBASE_H
