#ifndef IMAGESAVEDIALOGBASE_H
#define IMAGESAVEDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QFrame>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_ImageSaveDialogBase
{
public:
    QVBoxLayout *vboxLayout;
    Q3ButtonGroup *imageContentsButtons;
    QVBoxLayout *vboxLayout1;
    QRadioButton *subVolumeButton;
    QRadioButton *thumbVolumeButton;
    QRadioButton *bothButton;
    QHBoxLayout *hboxLayout;
    QLabel *textLabel1;
    QComboBox *imageFormatMenu;
    QFrame *line1;
    QHBoxLayout *hboxLayout1;
    QSpacerItem *spacer11;
    QPushButton *okButton;
    QPushButton *cancelButton;

    void setupUi(QDialog *ImageSaveDialogBase)
    {
        if (ImageSaveDialogBase->objectName().isEmpty())
            ImageSaveDialogBase->setObjectName(QString::fromUtf8("ImageSaveDialogBase"));
        ImageSaveDialogBase->resize(234, 223);
        vboxLayout = new QVBoxLayout(ImageSaveDialogBase);
        vboxLayout->setSpacing(6);
        vboxLayout->setContentsMargins(11, 11, 11, 11);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        imageContentsButtons = new Q3ButtonGroup(ImageSaveDialogBase);
        imageContentsButtons->setObjectName(QString::fromUtf8("imageContentsButtons"));
        imageContentsButtons->setColumnLayout(0, Qt::Vertical);
        imageContentsButtons->layout()->setSpacing(6);
        imageContentsButtons->layout()->setContentsMargins(11, 11, 11, 11);
        vboxLayout1 = new QVBoxLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(imageContentsButtons->layout());
        if (boxlayout)
            boxlayout->addLayout(vboxLayout1);
        vboxLayout1->setAlignment(Qt::AlignTop);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        subVolumeButton = new QRadioButton(imageContentsButtons);
        subVolumeButton->setObjectName(QString::fromUtf8("subVolumeButton"));
        subVolumeButton->setChecked(true);

        vboxLayout1->addWidget(subVolumeButton);

        thumbVolumeButton = new QRadioButton(imageContentsButtons);
        thumbVolumeButton->setObjectName(QString::fromUtf8("thumbVolumeButton"));

        vboxLayout1->addWidget(thumbVolumeButton);

        bothButton = new QRadioButton(imageContentsButtons);
        bothButton->setObjectName(QString::fromUtf8("bothButton"));
        bothButton->setEnabled(false);

        vboxLayout1->addWidget(bothButton);


        vboxLayout->addWidget(imageContentsButtons);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(4);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        textLabel1 = new QLabel(ImageSaveDialogBase);
        textLabel1->setObjectName(QString::fromUtf8("textLabel1"));
        textLabel1->setWordWrap(false);

        hboxLayout->addWidget(textLabel1);

        imageFormatMenu = new QComboBox(ImageSaveDialogBase);
        imageFormatMenu->setObjectName(QString::fromUtf8("imageFormatMenu"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(7), static_cast<QSizePolicy::Policy>(0));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(imageFormatMenu->sizePolicy().hasHeightForWidth());
        imageFormatMenu->setSizePolicy(sizePolicy);

        hboxLayout->addWidget(imageFormatMenu);


        vboxLayout->addLayout(hboxLayout);

        line1 = new QFrame(ImageSaveDialogBase);
        line1->setObjectName(QString::fromUtf8("line1"));
        line1->setFrameShape(QFrame::HLine);
        line1->setFrameShadow(QFrame::Sunken);

        vboxLayout->addWidget(line1);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        spacer11 = new QSpacerItem(32, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout1->addItem(spacer11);

        okButton = new QPushButton(ImageSaveDialogBase);
        okButton->setObjectName(QString::fromUtf8("okButton"));

        hboxLayout1->addWidget(okButton);

        cancelButton = new QPushButton(ImageSaveDialogBase);
        cancelButton->setObjectName(QString::fromUtf8("cancelButton"));

        hboxLayout1->addWidget(cancelButton);


        vboxLayout->addLayout(hboxLayout1);


        retranslateUi(ImageSaveDialogBase);
        QObject::connect(okButton, SIGNAL(clicked()), ImageSaveDialogBase, SLOT(accept()));
        QObject::connect(cancelButton, SIGNAL(clicked()), ImageSaveDialogBase, SLOT(reject()));

        QMetaObject::connectSlotsByName(ImageSaveDialogBase);
    } // setupUi

    void retranslateUi(QDialog *ImageSaveDialogBase)
    {
        ImageSaveDialogBase->setWindowTitle(QApplication::translate("ImageSaveDialogBase", "Save Image", 0, QApplication::UnicodeUTF8));
        imageContentsButtons->setTitle(QApplication::translate("ImageSaveDialogBase", "Image Contents", 0, QApplication::UnicodeUTF8));
        subVolumeButton->setText(QApplication::translate("ImageSaveDialogBase", "SubVolume (left side)", 0, QApplication::UnicodeUTF8));
        thumbVolumeButton->setText(QApplication::translate("ImageSaveDialogBase", "Thumbnail Volume (right side)", 0, QApplication::UnicodeUTF8));
        bothButton->setText(QApplication::translate("ImageSaveDialogBase", "Both", 0, QApplication::UnicodeUTF8));
        textLabel1->setText(QApplication::translate("ImageSaveDialogBase", "Image Format:", 0, QApplication::UnicodeUTF8));
        okButton->setText(QApplication::translate("ImageSaveDialogBase", "OK", 0, QApplication::UnicodeUTF8));
        cancelButton->setText(QApplication::translate("ImageSaveDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ImageSaveDialogBase: public Ui_ImageSaveDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class ImageSaveDialogBase : public QDialog, public Ui::ImageSaveDialogBase
{
    Q_OBJECT

public:
    ImageSaveDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~ImageSaveDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // IMAGESAVEDIALOGBASE_H
