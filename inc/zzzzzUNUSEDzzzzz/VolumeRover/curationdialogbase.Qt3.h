#ifndef CURATIONDIALOGBASE_H
#define CURATIONDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_CurationDialogBase
{
public:
    QGridLayout *gridLayout;
    QLabel *m_MergeRatioText;
    QLineEdit *m_MergeRatio;
    QLabel *m_OutputSegCountText;
    QLineEdit *m_OutputSegCount;
    QPushButton *m_Ok;
    QSpacerItem *spacer3;
    QPushButton *m_Cancel;

    void setupUi(QDialog *CurationDialogBase)
    {
        if (CurationDialogBase->objectName().isEmpty())
            CurationDialogBase->setObjectName(QString::fromUtf8("CurationDialogBase"));
        CurationDialogBase->resize(363, 120);
        gridLayout = new QGridLayout(CurationDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_MergeRatioText = new QLabel(CurationDialogBase);
        m_MergeRatioText->setObjectName(QString::fromUtf8("m_MergeRatioText"));
        m_MergeRatioText->setWordWrap(false);

        gridLayout->addWidget(m_MergeRatioText, 0, 0, 1, 1);

        m_MergeRatio = new QLineEdit(CurationDialogBase);
        m_MergeRatio->setObjectName(QString::fromUtf8("m_MergeRatio"));

        gridLayout->addWidget(m_MergeRatio, 0, 2, 1, 2);

        m_OutputSegCountText = new QLabel(CurationDialogBase);
        m_OutputSegCountText->setObjectName(QString::fromUtf8("m_OutputSegCountText"));
        m_OutputSegCountText->setWordWrap(false);

        gridLayout->addWidget(m_OutputSegCountText, 1, 0, 1, 2);

        m_OutputSegCount = new QLineEdit(CurationDialogBase);
        m_OutputSegCount->setObjectName(QString::fromUtf8("m_OutputSegCount"));

        gridLayout->addWidget(m_OutputSegCount, 1, 2, 1, 2);

        m_Ok = new QPushButton(CurationDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(1), static_cast<QSizePolicy::Policy>(1));
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_Ok->sizePolicy().hasHeightForWidth());
        m_Ok->setSizePolicy(sizePolicy);

        gridLayout->addWidget(m_Ok, 2, 3, 1, 1);

        spacer3 = new QSpacerItem(151, 31, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(spacer3, 2, 0, 1, 1);

        m_Cancel = new QPushButton(CurationDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));
        sizePolicy.setHeightForWidth(m_Cancel->sizePolicy().hasHeightForWidth());
        m_Cancel->setSizePolicy(sizePolicy);

        gridLayout->addWidget(m_Cancel, 2, 1, 1, 2);


        retranslateUi(CurationDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), CurationDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), CurationDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(CurationDialogBase);
    } // setupUi

    void retranslateUi(QDialog *CurationDialogBase)
    {
        CurationDialogBase->setWindowTitle(QApplication::translate("CurationDialogBase", "Curation", 0, QApplication::UnicodeUTF8));
        m_MergeRatioText->setText(QApplication::translate("CurationDialogBase", "Merge Ratio:", 0, QApplication::UnicodeUTF8));
        m_OutputSegCountText->setText(QApplication::translate("CurationDialogBase", "Number of output segments:", 0, QApplication::UnicodeUTF8));
        m_Ok->setText(QApplication::translate("CurationDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("CurationDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class CurationDialogBase: public Ui_CurationDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class CurationDialogBase : public QDialog, public Ui::CurationDialogBase
{
    Q_OBJECT

public:
    CurationDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~CurationDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // CURATIONDIALOGBASE_H
