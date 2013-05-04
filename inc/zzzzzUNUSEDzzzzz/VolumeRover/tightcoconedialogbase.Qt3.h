#ifndef TIGHTCOCONEDIALOGBASE_H
#define TIGHTCOCONEDIALOGBASE_H

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
#include <QtGui/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_TightCoconeDialogBase
{
public:
    QGridLayout *gridLayout;
    Q3GroupBox *m_RobustCoconeGroup;
    QGridLayout *gridLayout1;
    QLabel *m_BigBallRatioText;
    QLabel *m_ThetaIFText;
    QLabel *m_ThetaFFText;
    QLineEdit *m_BigBallRatio;
    QLineEdit *m_ThetaIF;
    QLineEdit *m_ThetaFF;
    QCheckBox *m_EnableRobustCocone;
    Q3GroupBox *m_FlatnessMarkingGroup;
    QGridLayout *gridLayout2;
    QLabel *m_FlatnessRatioText;
    QLabel *m_CoconePhiText;
    QLabel *m_FlatPhiText;
    QLineEdit *m_FlatnessRatio;
    QLineEdit *m_CoconePhi;
    QLineEdit *m_FlatPhi;
    QPushButton *m_Ok;
    QPushButton *m_Cancel;
    QSpacerItem *spacer2;

    void setupUi(QDialog *TightCoconeDialogBase)
    {
        if (TightCoconeDialogBase->objectName().isEmpty())
            TightCoconeDialogBase->setObjectName(QString::fromUtf8("TightCoconeDialogBase"));
        TightCoconeDialogBase->resize(580, 220);
        gridLayout = new QGridLayout(TightCoconeDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_RobustCoconeGroup = new Q3GroupBox(TightCoconeDialogBase);
        m_RobustCoconeGroup->setObjectName(QString::fromUtf8("m_RobustCoconeGroup"));
        m_RobustCoconeGroup->setColumnLayout(0, Qt::Vertical);
        m_RobustCoconeGroup->layout()->setSpacing(6);
        m_RobustCoconeGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_RobustCoconeGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_BigBallRatioText = new QLabel(m_RobustCoconeGroup);
        m_BigBallRatioText->setObjectName(QString::fromUtf8("m_BigBallRatioText"));
        m_BigBallRatioText->setWordWrap(false);

        gridLayout1->addWidget(m_BigBallRatioText, 1, 0, 1, 1);

        m_ThetaIFText = new QLabel(m_RobustCoconeGroup);
        m_ThetaIFText->setObjectName(QString::fromUtf8("m_ThetaIFText"));
        m_ThetaIFText->setWordWrap(false);

        gridLayout1->addWidget(m_ThetaIFText, 2, 0, 1, 1);

        m_ThetaFFText = new QLabel(m_RobustCoconeGroup);
        m_ThetaFFText->setObjectName(QString::fromUtf8("m_ThetaFFText"));
        m_ThetaFFText->setWordWrap(false);

        gridLayout1->addWidget(m_ThetaFFText, 3, 0, 1, 1);

        m_BigBallRatio = new QLineEdit(m_RobustCoconeGroup);
        m_BigBallRatio->setObjectName(QString::fromUtf8("m_BigBallRatio"));

        gridLayout1->addWidget(m_BigBallRatio, 1, 1, 1, 1);

        m_ThetaIF = new QLineEdit(m_RobustCoconeGroup);
        m_ThetaIF->setObjectName(QString::fromUtf8("m_ThetaIF"));

        gridLayout1->addWidget(m_ThetaIF, 2, 1, 1, 1);

        m_ThetaFF = new QLineEdit(m_RobustCoconeGroup);
        m_ThetaFF->setObjectName(QString::fromUtf8("m_ThetaFF"));

        gridLayout1->addWidget(m_ThetaFF, 3, 1, 1, 1);

        m_EnableRobustCocone = new QCheckBox(m_RobustCoconeGroup);
        m_EnableRobustCocone->setObjectName(QString::fromUtf8("m_EnableRobustCocone"));

        gridLayout1->addWidget(m_EnableRobustCocone, 0, 0, 1, 1);


        gridLayout->addWidget(m_RobustCoconeGroup, 0, 0, 1, 1);

        m_FlatnessMarkingGroup = new Q3GroupBox(TightCoconeDialogBase);
        m_FlatnessMarkingGroup->setObjectName(QString::fromUtf8("m_FlatnessMarkingGroup"));
        m_FlatnessMarkingGroup->setColumnLayout(0, Qt::Vertical);
        m_FlatnessMarkingGroup->layout()->setSpacing(6);
        m_FlatnessMarkingGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_FlatnessMarkingGroup->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        m_FlatnessRatioText = new QLabel(m_FlatnessMarkingGroup);
        m_FlatnessRatioText->setObjectName(QString::fromUtf8("m_FlatnessRatioText"));
        m_FlatnessRatioText->setWordWrap(false);

        gridLayout2->addWidget(m_FlatnessRatioText, 0, 0, 1, 1);

        m_CoconePhiText = new QLabel(m_FlatnessMarkingGroup);
        m_CoconePhiText->setObjectName(QString::fromUtf8("m_CoconePhiText"));
        m_CoconePhiText->setWordWrap(false);

        gridLayout2->addWidget(m_CoconePhiText, 1, 0, 1, 1);

        m_FlatPhiText = new QLabel(m_FlatnessMarkingGroup);
        m_FlatPhiText->setObjectName(QString::fromUtf8("m_FlatPhiText"));
        m_FlatPhiText->setWordWrap(false);

        gridLayout2->addWidget(m_FlatPhiText, 2, 0, 1, 1);

        m_FlatnessRatio = new QLineEdit(m_FlatnessMarkingGroup);
        m_FlatnessRatio->setObjectName(QString::fromUtf8("m_FlatnessRatio"));

        gridLayout2->addWidget(m_FlatnessRatio, 0, 1, 1, 1);

        m_CoconePhi = new QLineEdit(m_FlatnessMarkingGroup);
        m_CoconePhi->setObjectName(QString::fromUtf8("m_CoconePhi"));

        gridLayout2->addWidget(m_CoconePhi, 1, 1, 1, 1);

        m_FlatPhi = new QLineEdit(m_FlatnessMarkingGroup);
        m_FlatPhi->setObjectName(QString::fromUtf8("m_FlatPhi"));

        gridLayout2->addWidget(m_FlatPhi, 2, 1, 1, 1);


        gridLayout->addWidget(m_FlatnessMarkingGroup, 0, 1, 1, 2);

        m_Ok = new QPushButton(TightCoconeDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(1));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_Ok->sizePolicy().hasHeightForWidth());
        m_Ok->setSizePolicy(sizePolicy);

        gridLayout->addWidget(m_Ok, 1, 2, 1, 1);

        m_Cancel = new QPushButton(TightCoconeDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));
        sizePolicy.setHeightForWidth(m_Cancel->sizePolicy().hasHeightForWidth());
        m_Cancel->setSizePolicy(sizePolicy);

        gridLayout->addWidget(m_Cancel, 1, 1, 1, 1);

        spacer2 = new QSpacerItem(331, 51, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(spacer2, 1, 0, 1, 1);


        retranslateUi(TightCoconeDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), TightCoconeDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), TightCoconeDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(TightCoconeDialogBase);
    } // setupUi

    void retranslateUi(QDialog *TightCoconeDialogBase)
    {
        TightCoconeDialogBase->setWindowTitle(QApplication::translate("TightCoconeDialogBase", "Tight Cocone", 0, QApplication::UnicodeUTF8));
        m_RobustCoconeGroup->setTitle(QApplication::translate("TightCoconeDialogBase", "Robust Cocone", 0, QApplication::UnicodeUTF8));
        m_BigBallRatioText->setText(QApplication::translate("TightCoconeDialogBase", "Big Ball Ratio:", 0, QApplication::UnicodeUTF8));
        m_ThetaIFText->setText(QApplication::translate("TightCoconeDialogBase", "Infinite-finite deep intersection:", 0, QApplication::UnicodeUTF8));
        m_ThetaFFText->setText(QApplication::translate("TightCoconeDialogBase", "Finite-finite deep intersection:", 0, QApplication::UnicodeUTF8));
        m_EnableRobustCocone->setText(QApplication::translate("TightCoconeDialogBase", "Enable Robust Cocone", 0, QApplication::UnicodeUTF8));
        m_FlatnessMarkingGroup->setTitle(QApplication::translate("TightCoconeDialogBase", "Flatness Marking", 0, QApplication::UnicodeUTF8));
        m_FlatnessRatioText->setText(QApplication::translate("TightCoconeDialogBase", "Flatness Ratio:", 0, QApplication::UnicodeUTF8));
        m_CoconePhiText->setText(QApplication::translate("TightCoconeDialogBase", "Cocone Phi:", 0, QApplication::UnicodeUTF8));
        m_FlatPhiText->setText(QApplication::translate("TightCoconeDialogBase", "Flat Phi:", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("TightCoconeDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("TightCoconeDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class TightCoconeDialogBase: public Ui_TightCoconeDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class TightCoconeDialogBase : public QDialog, public Ui::TightCoconeDialogBase
{
    Q_OBJECT

public:
    TightCoconeDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~TightCoconeDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // TIGHTCOCONEDIALOGBASE_H
