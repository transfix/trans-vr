#ifndef BOUNDARYPOINTCLOUDDIALOGBASE_H
#define BOUNDARYPOINTCLOUDDIALOGBASE_H

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
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_BoundaryPointCloudDialogBase
{
public:
    QGridLayout *gridLayout;
    Q3GroupBox *m_CaptionGroup;
    QGridLayout *gridLayout1;
    QLabel *m_Caption;
    QLineEdit *m_THigh;
    QLabel *m_TLowText;
    QLabel *m_THighText;
    QLineEdit *m_TLow;
    QHBoxLayout *hboxLayout;
    QCheckBox *m_Preview;
    QPushButton *m_Cancel;
    QPushButton *m_Ok;

    void setupUi(QDialog *BoundaryPointCloudDialogBase)
    {
        if (BoundaryPointCloudDialogBase->objectName().isEmpty())
            BoundaryPointCloudDialogBase->setObjectName(QString::fromUtf8("BoundaryPointCloudDialogBase"));
        BoundaryPointCloudDialogBase->resize(248, 245);
        gridLayout = new QGridLayout(BoundaryPointCloudDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setResizeMode(QGridLayout::Fixed);
        m_CaptionGroup = new Q3GroupBox(BoundaryPointCloudDialogBase);
        m_CaptionGroup->setObjectName(QString::fromUtf8("m_CaptionGroup"));
        m_CaptionGroup->setMargin(0);
        m_CaptionGroup->setColumnLayout(0, Qt::Vertical);
        m_CaptionGroup->layout()->setSpacing(6);
        m_CaptionGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_CaptionGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_Caption = new QLabel(m_CaptionGroup);
        m_Caption->setObjectName(QString::fromUtf8("m_Caption"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(1));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_Caption->sizePolicy().hasHeightForWidth());
        m_Caption->setSizePolicy(sizePolicy);
        m_Caption->setMargin(0);
        m_Caption->setAlignment(Qt::AlignVCenter);
        m_Caption->setWordWrap(true);

        gridLayout1->addWidget(m_Caption, 0, 0, 1, 1);


        gridLayout->addWidget(m_CaptionGroup, 0, 0, 1, 2);

        m_THigh = new QLineEdit(BoundaryPointCloudDialogBase);
        m_THigh->setObjectName(QString::fromUtf8("m_THigh"));

        gridLayout->addWidget(m_THigh, 2, 1, 1, 1);

        m_TLowText = new QLabel(BoundaryPointCloudDialogBase);
        m_TLowText->setObjectName(QString::fromUtf8("m_TLowText"));
        m_TLowText->setWordWrap(false);

        gridLayout->addWidget(m_TLowText, 1, 0, 1, 1);

        m_THighText = new QLabel(BoundaryPointCloudDialogBase);
        m_THighText->setObjectName(QString::fromUtf8("m_THighText"));
        m_THighText->setWordWrap(false);

        gridLayout->addWidget(m_THighText, 2, 0, 1, 1);

        m_TLow = new QLineEdit(BoundaryPointCloudDialogBase);
        m_TLow->setObjectName(QString::fromUtf8("m_TLow"));

        gridLayout->addWidget(m_TLow, 1, 1, 1, 1);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        m_Preview = new QCheckBox(BoundaryPointCloudDialogBase);
        m_Preview->setObjectName(QString::fromUtf8("m_Preview"));
        m_Preview->setChecked(true);

        hboxLayout->addWidget(m_Preview);

        m_Cancel = new QPushButton(BoundaryPointCloudDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));

        hboxLayout->addWidget(m_Cancel);

        m_Ok = new QPushButton(BoundaryPointCloudDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));

        hboxLayout->addWidget(m_Ok);


        gridLayout->addLayout(hboxLayout, 3, 0, 1, 2);


        retranslateUi(BoundaryPointCloudDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), BoundaryPointCloudDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), BoundaryPointCloudDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(BoundaryPointCloudDialogBase);
    } // setupUi

    void retranslateUi(QDialog *BoundaryPointCloudDialogBase)
    {
        BoundaryPointCloudDialogBase->setWindowTitle(QApplication::translate("BoundaryPointCloudDialogBase", "Boundary Point Cloud", 0, QApplication::UnicodeUTF8));
        m_CaptionGroup->setTitle(QString());
        m_Caption->setText(QApplication::translate("BoundaryPointCloudDialogBase", "<p>Low/High Threshold values are silently clamped to [0.0,255.0]. 0.0 corresponds to minimum voxel value in the volume, while 255.0 corresponds to the maximum.</p>", 0, QApplication::UnicodeUTF8));
        m_THigh->setText(QApplication::translate("BoundaryPointCloudDialogBase", "1", 0, QApplication::UnicodeUTF8));
        m_TLowText->setText(QApplication::translate("BoundaryPointCloudDialogBase", "Low Threshold:", 0, QApplication::UnicodeUTF8));
        m_THighText->setText(QApplication::translate("BoundaryPointCloudDialogBase", "High Threshold:", 0, QApplication::UnicodeUTF8));
        m_TLow->setText(QApplication::translate("BoundaryPointCloudDialogBase", "0", 0, QApplication::UnicodeUTF8));
        m_Preview->setText(QApplication::translate("BoundaryPointCloudDialogBase", "Preview", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("BoundaryPointCloudDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("BoundaryPointCloudDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class BoundaryPointCloudDialogBase: public Ui_BoundaryPointCloudDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class BoundaryPointCloudDialogBase : public QDialog, public Ui::BoundaryPointCloudDialogBase
{
    Q_OBJECT

public:
    BoundaryPointCloudDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~BoundaryPointCloudDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // BOUNDARYPOINTCLOUDDIALOGBASE_H
