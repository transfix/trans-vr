#ifndef SKELETONIZATIONDIALOGBASE_H
#define SKELETONIZATIONDIALOGBASE_H

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

class Ui_SkeletonizationDialogBase
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
    Q3GroupBox *m_PlanarClustersGroup;
    QGridLayout *gridLayout2;
    QLabel *m_ThresholdText;
    QLineEdit *m_Threshold;
    QLabel *m_PlCntText;
    QLineEdit *m_PlCnt;
    QCheckBox *m_DiscardByThreshold;
    Q3GroupBox *m_MedialAxisGroup;
    QGridLayout *gridLayout3;
    QLabel *m_ThetaText;
    QLineEdit *m_Theta;
    QLabel *m_MedialRatioText;
    QLineEdit *m_MedialRatio;
    Q3GroupBox *m_FlatnessMarkingGroup;
    QGridLayout *gridLayout4;
    QLabel *m_FlatnessRatioText;
    QLabel *m_CoconePhiText;
    QLabel *m_FlatPhiText;
    QLineEdit *m_FlatnessRatio;
    QLineEdit *m_CoconePhi;
    QLineEdit *m_FlatPhi;
    QPushButton *m_Cancel;
    QPushButton *m_Ok;

    void setupUi(QDialog *SkeletonizationDialogBase)
    {
        if (SkeletonizationDialogBase->objectName().isEmpty())
            SkeletonizationDialogBase->setObjectName(QString::fromUtf8("SkeletonizationDialogBase"));
        SkeletonizationDialogBase->resize(537, 282);
        gridLayout = new QGridLayout(SkeletonizationDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_RobustCoconeGroup = new Q3GroupBox(SkeletonizationDialogBase);
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


        gridLayout->addWidget(m_RobustCoconeGroup, 0, 0, 1, 2);

        m_PlanarClustersGroup = new Q3GroupBox(SkeletonizationDialogBase);
        m_PlanarClustersGroup->setObjectName(QString::fromUtf8("m_PlanarClustersGroup"));
        m_PlanarClustersGroup->setColumnLayout(0, Qt::Vertical);
        m_PlanarClustersGroup->layout()->setSpacing(6);
        m_PlanarClustersGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout2 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_PlanarClustersGroup->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout2);
        gridLayout2->setAlignment(Qt::AlignTop);
        gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
        m_ThresholdText = new QLabel(m_PlanarClustersGroup);
        m_ThresholdText->setObjectName(QString::fromUtf8("m_ThresholdText"));
        m_ThresholdText->setWordWrap(false);

        gridLayout2->addWidget(m_ThresholdText, 0, 0, 1, 1);

        m_Threshold = new QLineEdit(m_PlanarClustersGroup);
        m_Threshold->setObjectName(QString::fromUtf8("m_Threshold"));

        gridLayout2->addWidget(m_Threshold, 0, 1, 1, 1);

        m_PlCntText = new QLabel(m_PlanarClustersGroup);
        m_PlCntText->setObjectName(QString::fromUtf8("m_PlCntText"));
        m_PlCntText->setWordWrap(false);

        gridLayout2->addWidget(m_PlCntText, 1, 0, 1, 1);

        m_PlCnt = new QLineEdit(m_PlanarClustersGroup);
        m_PlCnt->setObjectName(QString::fromUtf8("m_PlCnt"));

        gridLayout2->addWidget(m_PlCnt, 1, 1, 1, 1);

        m_DiscardByThreshold = new QCheckBox(m_PlanarClustersGroup);
        m_DiscardByThreshold->setObjectName(QString::fromUtf8("m_DiscardByThreshold"));

        gridLayout2->addWidget(m_DiscardByThreshold, 2, 0, 1, 2);


        gridLayout->addWidget(m_PlanarClustersGroup, 1, 0, 2, 1);

        m_MedialAxisGroup = new Q3GroupBox(SkeletonizationDialogBase);
        m_MedialAxisGroup->setObjectName(QString::fromUtf8("m_MedialAxisGroup"));
        m_MedialAxisGroup->setColumnLayout(0, Qt::Vertical);
        m_MedialAxisGroup->layout()->setSpacing(6);
        m_MedialAxisGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout3 = new QGridLayout();
        QBoxLayout *boxlayout2 = qobject_cast<QBoxLayout *>(m_MedialAxisGroup->layout());
        if (boxlayout2)
            boxlayout2->addLayout(gridLayout3);
        gridLayout3->setAlignment(Qt::AlignTop);
        gridLayout3->setObjectName(QString::fromUtf8("gridLayout3"));
        m_ThetaText = new QLabel(m_MedialAxisGroup);
        m_ThetaText->setObjectName(QString::fromUtf8("m_ThetaText"));
        m_ThetaText->setWordWrap(false);

        gridLayout3->addWidget(m_ThetaText, 0, 0, 1, 1);

        m_Theta = new QLineEdit(m_MedialAxisGroup);
        m_Theta->setObjectName(QString::fromUtf8("m_Theta"));

        gridLayout3->addWidget(m_Theta, 0, 1, 1, 1);

        m_MedialRatioText = new QLabel(m_MedialAxisGroup);
        m_MedialRatioText->setObjectName(QString::fromUtf8("m_MedialRatioText"));
        m_MedialRatioText->setWordWrap(false);

        gridLayout3->addWidget(m_MedialRatioText, 1, 0, 1, 1);

        m_MedialRatio = new QLineEdit(m_MedialAxisGroup);
        m_MedialRatio->setObjectName(QString::fromUtf8("m_MedialRatio"));

        gridLayout3->addWidget(m_MedialRatio, 1, 1, 1, 1);


        gridLayout->addWidget(m_MedialAxisGroup, 1, 1, 2, 2);

        m_FlatnessMarkingGroup = new Q3GroupBox(SkeletonizationDialogBase);
        m_FlatnessMarkingGroup->setObjectName(QString::fromUtf8("m_FlatnessMarkingGroup"));
        m_FlatnessMarkingGroup->setColumnLayout(0, Qt::Vertical);
        m_FlatnessMarkingGroup->layout()->setSpacing(6);
        m_FlatnessMarkingGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout4 = new QGridLayout();
        QBoxLayout *boxlayout3 = qobject_cast<QBoxLayout *>(m_FlatnessMarkingGroup->layout());
        if (boxlayout3)
            boxlayout3->addLayout(gridLayout4);
        gridLayout4->setAlignment(Qt::AlignTop);
        gridLayout4->setObjectName(QString::fromUtf8("gridLayout4"));
        m_FlatnessRatioText = new QLabel(m_FlatnessMarkingGroup);
        m_FlatnessRatioText->setObjectName(QString::fromUtf8("m_FlatnessRatioText"));
        m_FlatnessRatioText->setWordWrap(false);

        gridLayout4->addWidget(m_FlatnessRatioText, 0, 0, 1, 1);

        m_CoconePhiText = new QLabel(m_FlatnessMarkingGroup);
        m_CoconePhiText->setObjectName(QString::fromUtf8("m_CoconePhiText"));
        m_CoconePhiText->setWordWrap(false);

        gridLayout4->addWidget(m_CoconePhiText, 1, 0, 1, 1);

        m_FlatPhiText = new QLabel(m_FlatnessMarkingGroup);
        m_FlatPhiText->setObjectName(QString::fromUtf8("m_FlatPhiText"));
        m_FlatPhiText->setWordWrap(false);

        gridLayout4->addWidget(m_FlatPhiText, 2, 0, 1, 1);

        m_FlatnessRatio = new QLineEdit(m_FlatnessMarkingGroup);
        m_FlatnessRatio->setObjectName(QString::fromUtf8("m_FlatnessRatio"));

        gridLayout4->addWidget(m_FlatnessRatio, 0, 1, 1, 1);

        m_CoconePhi = new QLineEdit(m_FlatnessMarkingGroup);
        m_CoconePhi->setObjectName(QString::fromUtf8("m_CoconePhi"));

        gridLayout4->addWidget(m_CoconePhi, 1, 1, 1, 1);

        m_FlatPhi = new QLineEdit(m_FlatnessMarkingGroup);
        m_FlatPhi->setObjectName(QString::fromUtf8("m_FlatPhi"));

        gridLayout4->addWidget(m_FlatPhi, 2, 1, 1, 1);


        gridLayout->addWidget(m_FlatnessMarkingGroup, 0, 2, 1, 2);

        m_Cancel = new QPushButton(SkeletonizationDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(1));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_Cancel->sizePolicy().hasHeightForWidth());
        m_Cancel->setSizePolicy(sizePolicy);

        gridLayout->addWidget(m_Cancel, 1, 3, 1, 1);

        m_Ok = new QPushButton(SkeletonizationDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));
        sizePolicy.setHeightForWidth(m_Ok->sizePolicy().hasHeightForWidth());
        m_Ok->setSizePolicy(sizePolicy);

        gridLayout->addWidget(m_Ok, 2, 3, 1, 1);


        retranslateUi(SkeletonizationDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), SkeletonizationDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), SkeletonizationDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(SkeletonizationDialogBase);
    } // setupUi

    void retranslateUi(QDialog *SkeletonizationDialogBase)
    {
        SkeletonizationDialogBase->setWindowTitle(QApplication::translate("SkeletonizationDialogBase", "Skeletonization", 0, QApplication::UnicodeUTF8));
        m_RobustCoconeGroup->setTitle(QApplication::translate("SkeletonizationDialogBase", "Robust Cocone", 0, QApplication::UnicodeUTF8));
        m_BigBallRatioText->setText(QApplication::translate("SkeletonizationDialogBase", "Big Ball Ratio:", 0, QApplication::UnicodeUTF8));
        m_ThetaIFText->setText(QApplication::translate("SkeletonizationDialogBase", "Infinite-finite deep intersection:", 0, QApplication::UnicodeUTF8));
        m_ThetaFFText->setText(QApplication::translate("SkeletonizationDialogBase", "Finite-finite deep intersection:", 0, QApplication::UnicodeUTF8));
        m_EnableRobustCocone->setText(QApplication::translate("SkeletonizationDialogBase", "Enable Robust Cocone", 0, QApplication::UnicodeUTF8));
        m_PlanarClustersGroup->setTitle(QApplication::translate("SkeletonizationDialogBase", "Planar Clusters", 0, QApplication::UnicodeUTF8));
        m_ThresholdText->setText(QApplication::translate("SkeletonizationDialogBase", "Threshold:", 0, QApplication::UnicodeUTF8));
        m_PlCntText->setText(QApplication::translate("SkeletonizationDialogBase", "Count:", 0, QApplication::UnicodeUTF8));
        m_DiscardByThreshold->setText(QApplication::translate("SkeletonizationDialogBase", "Discard by threshold", 0, QApplication::UnicodeUTF8));
        m_MedialAxisGroup->setTitle(QApplication::translate("SkeletonizationDialogBase", "Medial Axis", 0, QApplication::UnicodeUTF8));
        m_ThetaText->setText(QApplication::translate("SkeletonizationDialogBase", "Theta:", 0, QApplication::UnicodeUTF8));
        m_MedialRatioText->setText(QApplication::translate("SkeletonizationDialogBase", "Medial Ratio:", 0, QApplication::UnicodeUTF8));
        m_FlatnessMarkingGroup->setTitle(QApplication::translate("SkeletonizationDialogBase", "Flatness Marking", 0, QApplication::UnicodeUTF8));
        m_FlatnessRatioText->setText(QApplication::translate("SkeletonizationDialogBase", "Flatness Ratio:", 0, QApplication::UnicodeUTF8));
        m_CoconePhiText->setText(QApplication::translate("SkeletonizationDialogBase", "Cocone Phi:", 0, QApplication::UnicodeUTF8));
        m_FlatPhiText->setText(QApplication::translate("SkeletonizationDialogBase", "Flat Phi:", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("SkeletonizationDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("SkeletonizationDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SkeletonizationDialogBase: public Ui_SkeletonizationDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class SkeletonizationDialogBase : public QDialog, public Ui::SkeletonizationDialogBase
{
    Q_OBJECT

public:
    SkeletonizationDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~SkeletonizationDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // SKELETONIZATIONDIALOGBASE_H
