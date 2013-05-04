#ifndef PROJECTGEOMETRYDIALOGBASE_H
#define PROJECTGEOMETRYDIALOGBASE_H

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

class Ui_ProjectGeometryDialogBase
{
public:
    QGridLayout *gridLayout;
    QLabel *m_Whatever;
    QLineEdit *m_FileName;
    QPushButton *m_FileDialogButton;
    QPushButton *m_CancelButton;
    QPushButton *m_OkButton;
    QSpacerItem *spacer1;

    void setupUi(QDialog *ProjectGeometryDialogBase)
    {
        if (ProjectGeometryDialogBase->objectName().isEmpty())
            ProjectGeometryDialogBase->setObjectName(QString::fromUtf8("ProjectGeometryDialogBase"));
        ProjectGeometryDialogBase->resize(396, 109);
        gridLayout = new QGridLayout(ProjectGeometryDialogBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_Whatever = new QLabel(ProjectGeometryDialogBase);
        m_Whatever->setObjectName(QString::fromUtf8("m_Whatever"));
        m_Whatever->setWordWrap(false);

        gridLayout->addWidget(m_Whatever, 0, 0, 1, 4);

        m_FileName = new QLineEdit(ProjectGeometryDialogBase);
        m_FileName->setObjectName(QString::fromUtf8("m_FileName"));

        gridLayout->addWidget(m_FileName, 1, 0, 1, 3);

        m_FileDialogButton = new QPushButton(ProjectGeometryDialogBase);
        m_FileDialogButton->setObjectName(QString::fromUtf8("m_FileDialogButton"));

        gridLayout->addWidget(m_FileDialogButton, 1, 3, 1, 1);

        m_CancelButton = new QPushButton(ProjectGeometryDialogBase);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));

        gridLayout->addWidget(m_CancelButton, 2, 0, 1, 1);

        m_OkButton = new QPushButton(ProjectGeometryDialogBase);
        m_OkButton->setObjectName(QString::fromUtf8("m_OkButton"));

        gridLayout->addWidget(m_OkButton, 2, 1, 1, 1);

        spacer1 = new QSpacerItem(171, 21, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(spacer1, 2, 2, 1, 2);


        retranslateUi(ProjectGeometryDialogBase);
        QObject::connect(m_CancelButton, SIGNAL(clicked()), ProjectGeometryDialogBase, SLOT(reject()));
        QObject::connect(m_OkButton, SIGNAL(clicked()), ProjectGeometryDialogBase, SLOT(accept()));
        QObject::connect(m_FileDialogButton, SIGNAL(clicked()), ProjectGeometryDialogBase, SLOT(openFileDialog()));

        QMetaObject::connectSlotsByName(ProjectGeometryDialogBase);
    } // setupUi

    void retranslateUi(QDialog *ProjectGeometryDialogBase)
    {
        ProjectGeometryDialogBase->setWindowTitle(QApplication::translate("ProjectGeometryDialogBase", "Project Geometry", 0, QApplication::UnicodeUTF8));
        m_Whatever->setText(QApplication::translate("ProjectGeometryDialogBase", "Project Geometry to boundary surface of provided mesh:", 0, QApplication::UnicodeUTF8));
        m_FileDialogButton->setText(QApplication::translate("ProjectGeometryDialogBase", "...", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("ProjectGeometryDialogBase", "Cancel", 0, QApplication::UnicodeUTF8));
        m_OkButton->setText(QApplication::translate("ProjectGeometryDialogBase", "OK", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ProjectGeometryDialogBase: public Ui_ProjectGeometryDialogBase {};
} // namespace Ui

QT_END_NAMESPACE

class ProjectGeometryDialogBase : public QDialog, public Ui::ProjectGeometryDialogBase
{
    Q_OBJECT

public:
    ProjectGeometryDialogBase(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~ProjectGeometryDialogBase();

protected slots:
    virtual void languageChange();

    virtual void openFileDialog();


};

#endif // PROJECTGEOMETRYDIALOGBASE_H
