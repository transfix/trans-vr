#ifndef LBIEQUALITYIMPROVEMENTDIALOGBASE_H
#define LBIEQUALITYIMPROVEMENTDIALOGBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_LBIEQualityImprovementDialogBase
{
public:
    QGridLayout *gridLayout;
    QLabel *m_ImproveMethodText;
    QComboBox *m_ImproveMethod;
    QPushButton *m_Ok;
    QPushButton *m_Cancel;
    QLineEdit *m_Iterations;
    QLabel *m_IterationsText;

    void setupUi(QDialog *LBIEQualityImprovementDialogBase)
    {
        if (LBIEQualityImprovementDialogBase->objectName().isEmpty())
            LBIEQualityImprovementDialogBase->setObjectName(QString::fromUtf8("LBIEQualityImprovementDialogBase"));
        LBIEQualityImprovementDialogBase->resize(340, 111);
        gridLayout = new QGridLayout(LBIEQualityImprovementDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_ImproveMethodText = new QLabel(LBIEQualityImprovementDialogBase);
        m_ImproveMethodText->setObjectName(QString::fromUtf8("m_ImproveMethodText"));
        m_ImproveMethodText->setWordWrap(false);

        gridLayout->addWidget(m_ImproveMethodText, 0, 0, 1, 2);

        m_ImproveMethod = new QComboBox(LBIEQualityImprovementDialogBase);
        m_ImproveMethod->setObjectName(QString::fromUtf8("m_ImproveMethod"));

        gridLayout->addWidget(m_ImproveMethod, 0, 2, 1, 1);

        m_Ok = new QPushButton(LBIEQualityImprovementDialogBase);
        m_Ok->setObjectName(QString::fromUtf8("m_Ok"));

        gridLayout->addWidget(m_Ok, 2, 1, 1, 2);

        m_Cancel = new QPushButton(LBIEQualityImprovementDialogBase);
        m_Cancel->setObjectName(QString::fromUtf8("m_Cancel"));

        gridLayout->addWidget(m_Cancel, 2, 0, 1, 1);

        m_Iterations = new QLineEdit(LBIEQualityImprovementDialogBase);
        m_Iterations->setObjectName(QString::fromUtf8("m_Iterations"));

        gridLayout->addWidget(m_Iterations, 1, 2, 1, 1);

        m_IterationsText = new QLabel(LBIEQualityImprovementDialogBase);
        m_IterationsText->setObjectName(QString::fromUtf8("m_IterationsText"));
        m_IterationsText->setWordWrap(false);

        gridLayout->addWidget(m_IterationsText, 1, 0, 1, 2);


        retranslateUi(LBIEQualityImprovementDialogBase);
        QObject::connect(m_Cancel, SIGNAL(clicked()), LBIEQualityImprovementDialogBase, SLOT(reject()));
        QObject::connect(m_Ok, SIGNAL(clicked()), LBIEQualityImprovementDialogBase, SLOT(accept()));

        QMetaObject::connectSlotsByName(LBIEQualityImprovementDialogBase);
    } // setupUi

    void retranslateUi(QDialog *LBIEQualityImprovementDialogBase)
    {
        LBIEQualityImprovementDialogBase->setWindowTitle(QApplication::translate("LBIEQualityImprovementDialogBase", "Mesh Quality Improvement", 0, QApplication::UnicodeUTF8));
        m_ImproveMethodText->setText(QApplication::translate("LBIEQualityImprovementDialogBase", "Improvement Method:", 0, QApplication::UnicodeUTF8));
        m_ImproveMethod->clear();
        m_ImproveMethod->insertItems(0, QStringList()
         << QApplication::translate("LBIEQualityImprovementDialogBase", "No Improvement", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEQualityImprovementDialogBase", "Geometric Flow", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEQualityImprovementDialogBase", "Edge Contraction", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEQualityImprovementDialogBase", "Joe Liu", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEQualityImprovementDialogBase", "Minimal Volume", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("LBIEQualityImprovementDialogBase", "Optimization", 0, QApplication::UnicodeUTF8)
        );
        m_Ok->setText(QApplication::translate("LBIEQualityImprovementDialogBase", "Ok", 0, QApplication::UnicodeUTF8));
        m_Cancel->setText(QApplication::translate("LBIEQualityImprovementDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Iterations->setText(QApplication::translate("LBIEQualityImprovementDialogBase", "1", 0, QApplication::UnicodeUTF8));
        m_IterationsText->setText(QApplication::translate("LBIEQualityImprovementDialogBase", "Improvement Iterations:", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class LBIEQualityImprovementDialogBase: public Ui_LBIEQualityImprovementDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class LBIEQualityImprovementDialogBase : public QDialog, public Ui::LBIEQualityImprovementDialogBase
{
    Q_OBJECT

public:
    LBIEQualityImprovementDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~LBIEQualityImprovementDialogBase();

protected slots:
    virtual void languageChange();

};

#endif // LBIEQUALITYIMPROVEMENTDIALOGBASE_H
