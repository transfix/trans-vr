#ifndef SLICERENDERINGDIALOG_H
#define SLICERENDERINGDIALOG_H

#include <qvariant.h>


#include <Qt3Support/Q3GroupBox>
#include <Qt3Support/Q3MimeSourceFactory>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_SliceRenderingDialog
{
public:
    QGridLayout *gridLayout;
    QCheckBox *m_SliceRenderingEnabled;
    QLabel *m_VariableText;
    QLabel *m_TimeStepText;
    QSpinBox *m_Timestep;
    QComboBox *m_Variable;
    Q3GroupBox *m_SliceToRenderGroup;
    QVBoxLayout *vboxLayout;
    QHBoxLayout *hboxLayout;
    QCheckBox *m_SliceToRenderXY;
    QCheckBox *m_GrayscaleXY;
    QHBoxLayout *hboxLayout1;
    QCheckBox *m_SliceToRenderXZ;
    QCheckBox *m_GrayscaleXZ;
    QHBoxLayout *hboxLayout2;
    QCheckBox *m_SliceToRenderZY;
    QCheckBox *m_GrayscaleZY;
    Q3GroupBox *m_RenderAdjacentSlicesGroup;
    QVBoxLayout *vboxLayout1;
    QHBoxLayout *hboxLayout3;
    QCheckBox *m_RenderAdjacentSliceXY;
    QLabel *m_RenderAdjacentSlicesOffsetXYText;
    QLineEdit *m_RenderAdjacentSliceOffsetXY;
    QHBoxLayout *hboxLayout4;
    QCheckBox *m_RenderAdjacentSliceXZ;
    QLabel *m_RenderAdjacentSlicesOffsetXZText;
    QLineEdit *m_RenderAdjacentSliceOffsetXZ;
    QHBoxLayout *hboxLayout5;
    QCheckBox *m_RenderAdjacentSliceZY;
    QLabel *m_RenderAdjacentSlicesOffsetZYText;
    QLineEdit *m_RenderAdjacentSliceOffsetZY;
    Q3GroupBox *m_DialogButtonGroup;
    QPushButton *m_OkButton;
    QPushButton *m_CancelButton;
    Q3GroupBox *m_Draw2DContoursGroup;
    QGridLayout *gridLayout1;
    QCheckBox *m_Draw2DContoursXY;
    QCheckBox *m_Draw2DContoursZY;
    QCheckBox *m_Draw2DContoursXZ;

    void setupUi(QDialog *SliceRenderingDialog)
    {
        if (SliceRenderingDialog->objectName().isEmpty())
            SliceRenderingDialog->setObjectName(QString::fromUtf8("SliceRenderingDialog"));
        SliceRenderingDialog->resize(401, 321);
        gridLayout = new QGridLayout(SliceRenderingDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setResizeMode(QGridLayout::Fixed);
        m_SliceRenderingEnabled = new QCheckBox(SliceRenderingDialog);
        m_SliceRenderingEnabled->setObjectName(QString::fromUtf8("m_SliceRenderingEnabled"));

        gridLayout->addWidget(m_SliceRenderingEnabled, 0, 0, 1, 4);

        m_VariableText = new QLabel(SliceRenderingDialog);
        m_VariableText->setObjectName(QString::fromUtf8("m_VariableText"));
        m_VariableText->setWordWrap(false);

        gridLayout->addWidget(m_VariableText, 1, 0, 1, 1);

        m_TimeStepText = new QLabel(SliceRenderingDialog);
        m_TimeStepText->setObjectName(QString::fromUtf8("m_TimeStepText"));
        m_TimeStepText->setWordWrap(false);

        gridLayout->addWidget(m_TimeStepText, 1, 2, 1, 1);

        m_Timestep = new QSpinBox(SliceRenderingDialog);
        m_Timestep->setObjectName(QString::fromUtf8("m_Timestep"));

        gridLayout->addWidget(m_Timestep, 1, 3, 1, 1);

        m_Variable = new QComboBox(SliceRenderingDialog);
        m_Variable->setObjectName(QString::fromUtf8("m_Variable"));

        gridLayout->addWidget(m_Variable, 1, 1, 1, 1);

        m_SliceToRenderGroup = new Q3GroupBox(SliceRenderingDialog);
        m_SliceToRenderGroup->setObjectName(QString::fromUtf8("m_SliceToRenderGroup"));
        m_SliceToRenderGroup->setColumnLayout(0, Qt::Vertical);
        m_SliceToRenderGroup->layout()->setSpacing(6);
        m_SliceToRenderGroup->layout()->setContentsMargins(11, 11, 11, 11);
        vboxLayout = new QVBoxLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_SliceToRenderGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(vboxLayout);
        vboxLayout->setAlignment(Qt::AlignTop);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        m_SliceToRenderXY = new QCheckBox(m_SliceToRenderGroup);
        m_SliceToRenderXY->setObjectName(QString::fromUtf8("m_SliceToRenderXY"));
        m_SliceToRenderXY->setChecked(false);

        hboxLayout->addWidget(m_SliceToRenderXY);

        m_GrayscaleXY = new QCheckBox(m_SliceToRenderGroup);
        m_GrayscaleXY->setObjectName(QString::fromUtf8("m_GrayscaleXY"));

        hboxLayout->addWidget(m_GrayscaleXY);


        vboxLayout->addLayout(hboxLayout);

        hboxLayout1 = new QHBoxLayout();
        hboxLayout1->setSpacing(6);
        hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
        m_SliceToRenderXZ = new QCheckBox(m_SliceToRenderGroup);
        m_SliceToRenderXZ->setObjectName(QString::fromUtf8("m_SliceToRenderXZ"));

        hboxLayout1->addWidget(m_SliceToRenderXZ);

        m_GrayscaleXZ = new QCheckBox(m_SliceToRenderGroup);
        m_GrayscaleXZ->setObjectName(QString::fromUtf8("m_GrayscaleXZ"));

        hboxLayout1->addWidget(m_GrayscaleXZ);


        vboxLayout->addLayout(hboxLayout1);

        hboxLayout2 = new QHBoxLayout();
        hboxLayout2->setSpacing(6);
        hboxLayout2->setObjectName(QString::fromUtf8("hboxLayout2"));
        m_SliceToRenderZY = new QCheckBox(m_SliceToRenderGroup);
        m_SliceToRenderZY->setObjectName(QString::fromUtf8("m_SliceToRenderZY"));

        hboxLayout2->addWidget(m_SliceToRenderZY);

        m_GrayscaleZY = new QCheckBox(m_SliceToRenderGroup);
        m_GrayscaleZY->setObjectName(QString::fromUtf8("m_GrayscaleZY"));

        hboxLayout2->addWidget(m_GrayscaleZY);


        vboxLayout->addLayout(hboxLayout2);


        gridLayout->addWidget(m_SliceToRenderGroup, 2, 0, 1, 2);

        m_RenderAdjacentSlicesGroup = new Q3GroupBox(SliceRenderingDialog);
        m_RenderAdjacentSlicesGroup->setObjectName(QString::fromUtf8("m_RenderAdjacentSlicesGroup"));
        m_RenderAdjacentSlicesGroup->setColumnLayout(0, Qt::Vertical);
        m_RenderAdjacentSlicesGroup->layout()->setSpacing(6);
        m_RenderAdjacentSlicesGroup->layout()->setContentsMargins(11, 11, 11, 11);
        vboxLayout1 = new QVBoxLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_RenderAdjacentSlicesGroup->layout());
        if (boxlayout1)
            boxlayout1->addLayout(vboxLayout1);
        vboxLayout1->setAlignment(Qt::AlignTop);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        hboxLayout3 = new QHBoxLayout();
        hboxLayout3->setSpacing(6);
        hboxLayout3->setObjectName(QString::fromUtf8("hboxLayout3"));
        m_RenderAdjacentSliceXY = new QCheckBox(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSliceXY->setObjectName(QString::fromUtf8("m_RenderAdjacentSliceXY"));

        hboxLayout3->addWidget(m_RenderAdjacentSliceXY);

        m_RenderAdjacentSlicesOffsetXYText = new QLabel(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSlicesOffsetXYText->setObjectName(QString::fromUtf8("m_RenderAdjacentSlicesOffsetXYText"));
        m_RenderAdjacentSlicesOffsetXYText->setWordWrap(false);

        hboxLayout3->addWidget(m_RenderAdjacentSlicesOffsetXYText);

        m_RenderAdjacentSliceOffsetXY = new QLineEdit(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSliceOffsetXY->setObjectName(QString::fromUtf8("m_RenderAdjacentSliceOffsetXY"));

        hboxLayout3->addWidget(m_RenderAdjacentSliceOffsetXY);


        vboxLayout1->addLayout(hboxLayout3);

        hboxLayout4 = new QHBoxLayout();
        hboxLayout4->setSpacing(6);
        hboxLayout4->setObjectName(QString::fromUtf8("hboxLayout4"));
        m_RenderAdjacentSliceXZ = new QCheckBox(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSliceXZ->setObjectName(QString::fromUtf8("m_RenderAdjacentSliceXZ"));

        hboxLayout4->addWidget(m_RenderAdjacentSliceXZ);

        m_RenderAdjacentSlicesOffsetXZText = new QLabel(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSlicesOffsetXZText->setObjectName(QString::fromUtf8("m_RenderAdjacentSlicesOffsetXZText"));
        m_RenderAdjacentSlicesOffsetXZText->setWordWrap(false);

        hboxLayout4->addWidget(m_RenderAdjacentSlicesOffsetXZText);

        m_RenderAdjacentSliceOffsetXZ = new QLineEdit(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSliceOffsetXZ->setObjectName(QString::fromUtf8("m_RenderAdjacentSliceOffsetXZ"));

        hboxLayout4->addWidget(m_RenderAdjacentSliceOffsetXZ);


        vboxLayout1->addLayout(hboxLayout4);

        hboxLayout5 = new QHBoxLayout();
        hboxLayout5->setSpacing(6);
        hboxLayout5->setObjectName(QString::fromUtf8("hboxLayout5"));
        m_RenderAdjacentSliceZY = new QCheckBox(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSliceZY->setObjectName(QString::fromUtf8("m_RenderAdjacentSliceZY"));

        hboxLayout5->addWidget(m_RenderAdjacentSliceZY);

        m_RenderAdjacentSlicesOffsetZYText = new QLabel(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSlicesOffsetZYText->setObjectName(QString::fromUtf8("m_RenderAdjacentSlicesOffsetZYText"));
        m_RenderAdjacentSlicesOffsetZYText->setWordWrap(false);

        hboxLayout5->addWidget(m_RenderAdjacentSlicesOffsetZYText);

        m_RenderAdjacentSliceOffsetZY = new QLineEdit(m_RenderAdjacentSlicesGroup);
        m_RenderAdjacentSliceOffsetZY->setObjectName(QString::fromUtf8("m_RenderAdjacentSliceOffsetZY"));

        hboxLayout5->addWidget(m_RenderAdjacentSliceOffsetZY);


        vboxLayout1->addLayout(hboxLayout5);


        gridLayout->addWidget(m_RenderAdjacentSlicesGroup, 2, 2, 1, 2);

        m_DialogButtonGroup = new Q3GroupBox(SliceRenderingDialog);
        m_DialogButtonGroup->setObjectName(QString::fromUtf8("m_DialogButtonGroup"));
        m_OkButton = new QPushButton(m_DialogButtonGroup);
        m_OkButton->setObjectName(QString::fromUtf8("m_OkButton"));
        m_OkButton->setGeometry(QRect(11, 11, 167, 32));
        m_CancelButton = new QPushButton(m_DialogButtonGroup);
        m_CancelButton->setObjectName(QString::fromUtf8("m_CancelButton"));
        m_CancelButton->setGeometry(QRect(11, 49, 167, 32));

        gridLayout->addWidget(m_DialogButtonGroup, 3, 2, 1, 2);

        m_Draw2DContoursGroup = new Q3GroupBox(SliceRenderingDialog);
        m_Draw2DContoursGroup->setObjectName(QString::fromUtf8("m_Draw2DContoursGroup"));
        m_Draw2DContoursGroup->setColumnLayout(0, Qt::Vertical);
        m_Draw2DContoursGroup->layout()->setSpacing(6);
        m_Draw2DContoursGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout2 = qobject_cast<QBoxLayout *>(m_Draw2DContoursGroup->layout());
        if (boxlayout2)
            boxlayout2->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_Draw2DContoursXY = new QCheckBox(m_Draw2DContoursGroup);
        m_Draw2DContoursXY->setObjectName(QString::fromUtf8("m_Draw2DContoursXY"));

        gridLayout1->addWidget(m_Draw2DContoursXY, 0, 0, 1, 1);

        m_Draw2DContoursZY = new QCheckBox(m_Draw2DContoursGroup);
        m_Draw2DContoursZY->setObjectName(QString::fromUtf8("m_Draw2DContoursZY"));

        gridLayout1->addWidget(m_Draw2DContoursZY, 2, 0, 1, 1);

        m_Draw2DContoursXZ = new QCheckBox(m_Draw2DContoursGroup);
        m_Draw2DContoursXZ->setObjectName(QString::fromUtf8("m_Draw2DContoursXZ"));

        gridLayout1->addWidget(m_Draw2DContoursXZ, 1, 0, 1, 1);


        gridLayout->addWidget(m_Draw2DContoursGroup, 3, 0, 1, 2);


        retranslateUi(SliceRenderingDialog);
        QObject::connect(m_OkButton, SIGNAL(clicked()), SliceRenderingDialog, SLOT(accept()));
        QObject::connect(m_CancelButton, SIGNAL(clicked()), SliceRenderingDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(SliceRenderingDialog);
    } // setupUi

    void retranslateUi(QDialog *SliceRenderingDialog)
    {
        SliceRenderingDialog->setWindowTitle(QApplication::translate("SliceRenderingDialog", "Slice Rendering", 0, QApplication::UnicodeUTF8));
        m_SliceRenderingEnabled->setText(QApplication::translate("SliceRenderingDialog", "Render 2D slices in 3D View", 0, QApplication::UnicodeUTF8));
        m_VariableText->setText(QApplication::translate("SliceRenderingDialog", "Variable:", 0, QApplication::UnicodeUTF8));
        m_TimeStepText->setText(QApplication::translate("SliceRenderingDialog", "Time Step:", 0, QApplication::UnicodeUTF8));
        m_SliceToRenderGroup->setTitle(QApplication::translate("SliceRenderingDialog", "Slices to Render", 0, QApplication::UnicodeUTF8));
        m_SliceToRenderXY->setText(QApplication::translate("SliceRenderingDialog", "XY", 0, QApplication::UnicodeUTF8));
        m_GrayscaleXY->setText(QApplication::translate("SliceRenderingDialog", "Grayscale", 0, QApplication::UnicodeUTF8));
        m_SliceToRenderXZ->setText(QApplication::translate("SliceRenderingDialog", "XZ", 0, QApplication::UnicodeUTF8));
        m_GrayscaleXZ->setText(QApplication::translate("SliceRenderingDialog", "Grayscale", 0, QApplication::UnicodeUTF8));
        m_SliceToRenderZY->setText(QApplication::translate("SliceRenderingDialog", "ZY", 0, QApplication::UnicodeUTF8));
        m_GrayscaleZY->setText(QApplication::translate("SliceRenderingDialog", "Grayscale", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSlicesGroup->setTitle(QApplication::translate("SliceRenderingDialog", "Render Adjacent Slices", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSliceXY->setText(QApplication::translate("SliceRenderingDialog", "XY", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSlicesOffsetXYText->setText(QApplication::translate("SliceRenderingDialog", "Offset:", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSliceOffsetXY->setText(QApplication::translate("SliceRenderingDialog", "1", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSliceXZ->setText(QApplication::translate("SliceRenderingDialog", "XZ", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSlicesOffsetXZText->setText(QApplication::translate("SliceRenderingDialog", "Offset:", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSliceOffsetXZ->setText(QApplication::translate("SliceRenderingDialog", "1", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSliceZY->setText(QApplication::translate("SliceRenderingDialog", "ZY", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSlicesOffsetZYText->setText(QApplication::translate("SliceRenderingDialog", "Offset:", 0, QApplication::UnicodeUTF8));
        m_RenderAdjacentSliceOffsetZY->setText(QApplication::translate("SliceRenderingDialog", "1", 0, QApplication::UnicodeUTF8));
        m_DialogButtonGroup->setTitle(QString());
        m_OkButton->setText(QApplication::translate("SliceRenderingDialog", "Ok", 0, QApplication::UnicodeUTF8));
        m_CancelButton->setText(QApplication::translate("SliceRenderingDialog", "Cancel", 0, QApplication::UnicodeUTF8));
        m_Draw2DContoursGroup->setTitle(QApplication::translate("SliceRenderingDialog", "Draw 2D Contours on Slices", 0, QApplication::UnicodeUTF8));
        m_Draw2DContoursXY->setText(QApplication::translate("SliceRenderingDialog", "XY", 0, QApplication::UnicodeUTF8));
        m_Draw2DContoursZY->setText(QApplication::translate("SliceRenderingDialog", "ZY", 0, QApplication::UnicodeUTF8));
        m_Draw2DContoursXZ->setText(QApplication::translate("SliceRenderingDialog", "XZ", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SliceRenderingDialog: public Ui_SliceRenderingDialog {};
} // namespace Ui

QT_END_NAMESPACE

class SliceRenderingDialog : public QDialog, public Ui::SliceRenderingDialog
{
    Q_OBJECT

public:
    SliceRenderingDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~SliceRenderingDialog();

protected slots:
    virtual void languageChange();

};

#endif // SLICERENDERINGDIALOG_H
