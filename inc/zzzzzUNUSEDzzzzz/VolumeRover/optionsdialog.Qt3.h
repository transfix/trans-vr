#ifndef OPTIONSDIALOGBASE_H
#define OPTIONSDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
#include <Qt3Support/Q3GroupBox>
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

class Ui_OptionsDialogBase
{
public:
    QGridLayout *gridLayout;
    Q3GroupBox *IsosurfacingOptionsBox;
    QGridLayout *gridLayout1;
    QCheckBox *m_ZoomedOutSurface;
    QCheckBox *m_ZoomedInSurface;
    Q3ButtonGroup *ButtonGroup2_2;
    QVBoxLayout *vboxLayout;
    QRadioButton *m_Single;
    QRadioButton *m_RGBA;
    Q3ButtonGroup *ButtonGroup2;
    QVBoxLayout *vboxLayout1;
    QRadioButton *m_Interactive;
    QRadioButton *m_Delayed;
    QRadioButton *m_Manual;
    Q3ButtonGroup *ButtonGroup3;
    QGridLayout *gridLayout2;
    QRadioButton *m_Shaded;
    QRadioButton *m_Unshaded;
    Q3GroupBox *GroupBox2;
    QVBoxLayout *vboxLayout2;
    QHBoxLayout *hboxLayout;
    QLabel *TextLabel1;
    QLineEdit *m_CacheDir;
    QPushButton *m_BrowseButton;
    QHBoxLayout *hboxLayout1;
    QLabel *TextLabel2;
    QSpacerItem *Spacer2;
    QPushButton *m_ColorButton;
    QHBoxLayout *hboxLayout2;
    QSpacerItem *Horizontal_Spacing2;
    QPushButton *buttonOk;
    QPushButton *buttonCancel;
    Q3ButtonGroup *ButtonGroup7;
    QGridLayout *gridLayout3;
    QRadioButton *m_1DTransferFunc;
    QRadioButton *m_2DTransferFunc;
    QRadioButton *m_3DTransferFunc;

    void setupUi(QDialog *OptionsDialogBase)
    {
        if (OptionsDialogBase->objectName().isEmpty())
            OptionsDialogBase->setObjectName(QString::fromUtf8("OptionsDialogBase"));
        OptionsDialogBase->resize(501, 444);
        OptionsDialogBase->setSizeGripEnabled(true);
        gridLayout = new QGridLayout(OptionsDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setResizeMode(QGridLayout::Fixed);
        IsosurfacingOptionsBox = new Q3GroupBox(OptionsDialogBase);
        IsosurfacingOptionsBox->setObjectName(QString::fromUtf8("IsosurfacingOptionsBox"));
        IsosurfacingOptionsBox->setColumnLayout(0, Qt::Vertical);
        IsosurfacingOptionsBox->layout()->setSpacing(6);
        IsosurfacingOptionsBox->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(IsosurfacingOptionsBox->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_ZoomedOutSurface = new QCheckBox(IsosurfacingOptionsBox);
        m_ZoomedOutSurface->setObjectName(QString::fromUtf8("m_ZoomedOutSurface"));

        gridLayout1->addWidget(m_ZoomedOutSurface, 0, 0, 1, 1);

        m_ZoomedInSurface = new QCheckBox(IsosurfacingOptionsBox);
        m_ZoomedInSurface->setObjectName(QString::fromUtf8("m_ZoomedInSurface"));

        gridLayout1->addWidget(m_ZoomedInSurface, 1, 0, 1, 1);


        gridLayout->addWidget(IsosurfacingOptionsBox, 0, 1, 1, 2);

        ButtonGroup2_2 = new Q3ButtonGroup(OptionsDialogBase);
        ButtonGroup2_2->setObjectName(QString::fromUtf8("ButtonGroup2_2"));
        ButtonGroup2_2->setColumnLayout(0, Qt::Vertical);
        ButtonGroup2_2->layout()->setSpacing(6);
        ButtonGroup2_2->layout()->setContentsMargins(11, 11, 11, 11);
        vboxLayout = new QVBoxLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(ButtonGroup2_2->layout());
        if (boxlayout1)
            boxlayout1->addLayout(vboxLayout);
        vboxLayout->setAlignment(Qt::AlignTop);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        m_Single = new QRadioButton(ButtonGroup2_2);
        m_Single->setObjectName(QString::fromUtf8("m_Single"));

        vboxLayout->addWidget(m_Single);

        m_RGBA = new QRadioButton(ButtonGroup2_2);
        m_RGBA->setObjectName(QString::fromUtf8("m_RGBA"));

        vboxLayout->addWidget(m_RGBA);


        gridLayout->addWidget(ButtonGroup2_2, 1, 1, 1, 2);

        ButtonGroup2 = new Q3ButtonGroup(OptionsDialogBase);
        ButtonGroup2->setObjectName(QString::fromUtf8("ButtonGroup2"));
        ButtonGroup2->setColumnLayout(0, Qt::Vertical);
        ButtonGroup2->layout()->setSpacing(6);
        ButtonGroup2->layout()->setContentsMargins(11, 11, 11, 11);
        vboxLayout1 = new QVBoxLayout();
        QBoxLayout *boxlayout2 = qobject_cast<QBoxLayout *>(ButtonGroup2->layout());
        if (boxlayout2)
            boxlayout2->addLayout(vboxLayout1);
        vboxLayout1->setAlignment(Qt::AlignTop);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        m_Interactive = new QRadioButton(ButtonGroup2);
        m_Interactive->setObjectName(QString::fromUtf8("m_Interactive"));

        vboxLayout1->addWidget(m_Interactive);

        m_Delayed = new QRadioButton(ButtonGroup2);
        m_Delayed->setObjectName(QString::fromUtf8("m_Delayed"));

        vboxLayout1->addWidget(m_Delayed);

        m_Manual = new QRadioButton(ButtonGroup2);
        m_Manual->setObjectName(QString::fromUtf8("m_Manual"));

        vboxLayout1->addWidget(m_Manual);


        gridLayout->addWidget(ButtonGroup2, 0, 0, 1, 1);

        ButtonGroup3 = new Q3ButtonGroup(OptionsDialogBase);
        ButtonGroup3->setObjectName(QString::fromUtf8("ButtonGroup3"));
        ButtonGroup3->setColumnLayout(0, Qt::Vertical);
        ButtonGroup3->layout()->setSpacing(6);
        ButtonGroup3->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout3 = qobject_cast<QBoxLayout *>(ButtonGroup3->layout());
        if (boxlayout3)
            boxlayout3->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        m_Shaded = new QRadioButton(ButtonGroup3);
        m_Shaded->setObjectName(QString::fromUtf8("m_Shaded"));

        gridLayout2->addWidget(m_Shaded, 1, 0, 1, 1);

        m_Unshaded = new QRadioButton(ButtonGroup3);
        m_Unshaded->setObjectName(QString::fromUtf8("m_Unshaded"));

        gridLayout2->addWidget(m_Unshaded, 0, 0, 1, 1);


        gridLayout->addWidget(ButtonGroup3, 1, 0, 1, 1);

        GroupBox2 = new Q3GroupBox(OptionsDialogBase);
        GroupBox2->setObjectName(QString::fromUtf8("GroupBox2"));
        GroupBox2->setColumnLayout(0, Qt::Vertical);
        GroupBox2->layout()->setSpacing(6);
        GroupBox2->layout()->setContentsMargins(11, 11, 11, 11);
        vboxLayout2 = new QVBoxLayout();
        QBoxLayout *boxlayout4 = qobject_cast<QBoxLayout *>(GroupBox2->layout());
        if (boxlayout4)
            boxlayout4->addLayout(vboxLayout2);
        vboxLayout2->setAlignment(Qt::AlignTop);
        vboxLayout2->setObjectName(QString::fromUtf8("vboxLayout2"));
        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setContentsMargins(0, 0, 0, 0);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        TextLabel1 = new QLabel(GroupBox2);
        TextLabel1->setObjectName(QString::fromUtf8("TextLabel1"));
        TextLabel1->setWordWrap(false);

        hboxLayout->addWidget(TextLabel1);

        m_CacheDir = new QLineEdit(GroupBox2);
        m_CacheDir->setObjectName(QString::fromUtf8("m_CacheDir"));

        hboxLayout->addWidget(m_CacheDir);

        m_BrowseButton = new QPushButton(GroupBox2);
        m_BrowseButton->setObjectName(QString::fromUtf8("m_BrowseButton"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(0), static_cast<QSizePolicy::Policy>(0));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_BrowseButton->sizePolicy().hasHeightForWidth());
        m_BrowseButton->setSizePolicy(sizePolicy);
        m_BrowseButton->setMaximumSize(QSize(25, 32767));

        hboxLayout->addWidget(m_BrowseButton);


        vboxLayout2->addLayout(hboxLayout);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setContentsMargins(0, 0, 0, 0);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        TextLabel2 = new QLabel(GroupBox2);
        TextLabel2->setObjectName(QString::fromUtf8("TextLabel2"));
        TextLabel2->setWordWrap(false);

        hboxLayout1->addWidget(TextLabel2);

        Spacer2 = new QSpacerItem(20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout1->addItem(Spacer2);

        m_ColorButton = new QPushButton(GroupBox2);
        m_ColorButton->setObjectName(QString::fromUtf8("m_ColorButton"));
        QSizePolicy sizePolicy1(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(0));
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(m_ColorButton->sizePolicy().hasHeightForWidth());
        m_ColorButton->setSizePolicy(sizePolicy1);
        m_ColorButton->setMaximumSize(QSize(25, 32767));

        hboxLayout1->addWidget(m_ColorButton);


        vboxLayout2->addLayout(hboxLayout1);


        gridLayout->addWidget(GroupBox2, 3, 0, 1, 3);

        hboxLayout2 = new QHBoxLayout();
        hboxLayout2->setSpacing(6);
        hboxLayout2->setContentsMargins(0, 0, 0, 0);
        hboxLayout2->setObjectName(QString::fromUtf8("hboxLayout2"));
        Horizontal_Spacing2 = new QSpacerItem(20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout2->addItem(Horizontal_Spacing2);

        buttonOk = new QPushButton(OptionsDialogBase);
        buttonOk->setObjectName(QString::fromUtf8("buttonOk"));
        buttonOk->setAutoDefault(true);
        buttonOk->setDefault(true);

        hboxLayout2->addWidget(buttonOk);

        buttonCancel = new QPushButton(OptionsDialogBase);
        buttonCancel->setObjectName(QString::fromUtf8("buttonCancel"));
        buttonCancel->setAutoDefault(true);

        hboxLayout2->addWidget(buttonCancel);


        gridLayout->addLayout(hboxLayout2, 4, 0, 1, 3);

        ButtonGroup7 = new Q3ButtonGroup(OptionsDialogBase);
        ButtonGroup7->setObjectName(QString::fromUtf8("ButtonGroup7"));
        ButtonGroup7->setColumnLayout(0, Qt::Vertical);
        ButtonGroup7->layout()->setSpacing(6);
        ButtonGroup7->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout3 = new QGridLayout();
        QBoxLayout *boxlayout5 = qobject_cast<QBoxLayout *>(ButtonGroup7->layout());
        if (boxlayout5)
            boxlayout5->addLayout(gridLayout3);
        gridLayout3->setAlignment(Qt::AlignTop);
        gridLayout3->setObjectName(QString::fromUtf8("gridLayout3"));
        m_1DTransferFunc = new QRadioButton(ButtonGroup7);
        m_1DTransferFunc->setObjectName(QString::fromUtf8("m_1DTransferFunc"));

        gridLayout3->addWidget(m_1DTransferFunc, 0, 0, 1, 1);

        m_2DTransferFunc = new QRadioButton(ButtonGroup7);
        m_2DTransferFunc->setObjectName(QString::fromUtf8("m_2DTransferFunc"));

        gridLayout3->addWidget(m_2DTransferFunc, 0, 1, 1, 1);

        m_3DTransferFunc = new QRadioButton(ButtonGroup7);
        m_3DTransferFunc->setObjectName(QString::fromUtf8("m_3DTransferFunc"));

        gridLayout3->addWidget(m_3DTransferFunc, 0, 2, 1, 1);


        gridLayout->addWidget(ButtonGroup7, 2, 0, 1, 2);

        QWidget::setTabOrder(m_Interactive, m_Delayed);
        QWidget::setTabOrder(m_Delayed, m_Manual);
        QWidget::setTabOrder(m_Manual, m_ZoomedOutSurface);
        QWidget::setTabOrder(m_ZoomedOutSurface, m_ZoomedInSurface);
        QWidget::setTabOrder(m_ZoomedInSurface, m_Unshaded);
        QWidget::setTabOrder(m_Unshaded, m_Shaded);
        QWidget::setTabOrder(m_Shaded, m_Single);
        QWidget::setTabOrder(m_Single, m_RGBA);
        QWidget::setTabOrder(m_RGBA, m_1DTransferFunc);
        QWidget::setTabOrder(m_1DTransferFunc, m_2DTransferFunc);
        QWidget::setTabOrder(m_2DTransferFunc, m_3DTransferFunc);
        QWidget::setTabOrder(m_3DTransferFunc, m_CacheDir);
        QWidget::setTabOrder(m_CacheDir, m_BrowseButton);
        QWidget::setTabOrder(m_BrowseButton, m_ColorButton);
        QWidget::setTabOrder(m_ColorButton, buttonOk);
        QWidget::setTabOrder(buttonOk, buttonCancel);

        retranslateUi(OptionsDialogBase);
        QObject::connect(buttonOk, SIGNAL(clicked()), OptionsDialogBase, SLOT(accept()));
        QObject::connect(buttonCancel, SIGNAL(clicked()), OptionsDialogBase, SLOT(reject()));
        QObject::connect(m_BrowseButton, SIGNAL(clicked()), OptionsDialogBase, SLOT(browseSlot()));
        QObject::connect(m_ColorButton, SIGNAL(clicked()), OptionsDialogBase, SLOT(colorSlot()));

        QMetaObject::connectSlotsByName(OptionsDialogBase);
    } // setupUi

    void retranslateUi(QDialog *OptionsDialogBase)
    {
        OptionsDialogBase->setWindowTitle(QApplication::translate("OptionsDialogBase", "OptionsDialog", 0, QApplication::UnicodeUTF8));
        IsosurfacingOptionsBox->setTitle(QApplication::translate("OptionsDialogBase", "Isosurfacing", 0, QApplication::UnicodeUTF8));
        m_ZoomedOutSurface->setText(QApplication::translate("OptionsDialogBase", "Show Thumbnail Isosurface", 0, QApplication::UnicodeUTF8));
        m_ZoomedInSurface->setText(QApplication::translate("OptionsDialogBase", "Show Magnified Isosurface", 0, QApplication::UnicodeUTF8));
        ButtonGroup2_2->setTitle(QApplication::translate("OptionsDialogBase", "Render Style", 0, QApplication::UnicodeUTF8));
        m_Single->setText(QApplication::translate("OptionsDialogBase", "Single Variable", 0, QApplication::UnicodeUTF8));
        m_RGBA->setText(QApplication::translate("OptionsDialogBase", "RGBA Combined", 0, QApplication::UnicodeUTF8));
        ButtonGroup2->setTitle(QApplication::translate("OptionsDialogBase", "Update Method", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        ButtonGroup2->setProperty("whatsThis", QVariant(QApplication::translate("OptionsDialogBase", "<p><b> Update Method </p></b>\n"
"\n"
"<p>Determines when the data in the zoomed in view is updated as you explore the volume using the subvolume explorer. </p>", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        m_Interactive->setText(QApplication::translate("OptionsDialogBase", "Interactive", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        m_Interactive->setProperty("whatsThis", QVariant(QApplication::translate("OptionsDialogBase", "<p><b> Interactive </b></p>\n"
"\n"
"<p>The data will update immediately every time you move the subcube volume </p>", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        m_Delayed->setText(QApplication::translate("OptionsDialogBase", "Delayed", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        m_Delayed->setProperty("whatsThis", QVariant(QApplication::translate("OptionsDialogBase", "<p><b>Delayed</b></p>\n"
"\n"
"<p>The data will not update until you release the mouse button after dragging the subcube.</p>", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        m_Manual->setText(QApplication::translate("OptionsDialogBase", "Manual", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_WHATSTHIS
        m_Manual->setProperty("whatsThis", QVariant(QApplication::translate("OptionsDialogBase", "<p><b>Manual</p></b>\n"
"\n"
"<p>The data will not update untill you manually press the update button </p>", 0, QApplication::UnicodeUTF8)));
#endif // QT_NO_WHATSTHIS
        ButtonGroup3->setTitle(QApplication::translate("OptionsDialogBase", "Volume Rendering", 0, QApplication::UnicodeUTF8));
        m_Shaded->setText(QApplication::translate("OptionsDialogBase", "Shaded", 0, QApplication::UnicodeUTF8));
        m_Unshaded->setText(QApplication::translate("OptionsDialogBase", "Unshaded", 0, QApplication::UnicodeUTF8));
        GroupBox2->setTitle(QApplication::translate("OptionsDialogBase", "General", 0, QApplication::UnicodeUTF8));
        TextLabel1->setText(QApplication::translate("OptionsDialogBase", "Cache Directory:", 0, QApplication::UnicodeUTF8));
        m_BrowseButton->setText(QApplication::translate("OptionsDialogBase", "...", 0, QApplication::UnicodeUTF8));
        TextLabel2->setText(QApplication::translate("OptionsDialogBase", "Background Color:", 0, QApplication::UnicodeUTF8));
        m_ColorButton->setText(QString());
        buttonOk->setWindowTitle(QString());
        buttonOk->setText(QApplication::translate("OptionsDialogBase", "&OK", 0, QApplication::UnicodeUTF8));
        buttonCancel->setText(QApplication::translate("OptionsDialogBase", "&Cancel", 0, QApplication::UnicodeUTF8));
        ButtonGroup7->setTitle(QApplication::translate("OptionsDialogBase", "Transfer Function Dimension", 0, QApplication::UnicodeUTF8));
        m_1DTransferFunc->setText(QApplication::translate("OptionsDialogBase", "1D", 0, QApplication::UnicodeUTF8));
        m_2DTransferFunc->setText(QApplication::translate("OptionsDialogBase", "2D", 0, QApplication::UnicodeUTF8));
        m_3DTransferFunc->setText(QApplication::translate("OptionsDialogBase", "3D", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class OptionsDialogBase: public Ui_OptionsDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class OptionsDialogBase : public QDialog, public Ui::OptionsDialogBase
{
    Q_OBJECT

public:
    OptionsDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~OptionsDialogBase();

public slots:
    virtual void browseSlot();
    virtual void colorSlot();

protected slots:
    virtual void languageChange();

};

#endif // OPTIONSDIALOGBASE_H
