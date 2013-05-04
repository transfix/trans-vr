#ifndef RECONSTRUCTIONDIALOG_H
#define RECONSTRUCTIONDIALOG_H

#include <qvariant.h>


#include <Qt3Support/Q3ButtonGroup>
#include <Qt3Support/Q3GroupBox>
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
#include <QtGui/QRadioButton>

QT_BEGIN_NAMESPACE

class Ui_ReconstructionDialog
{
public:
    Q3GroupBox *groupBox10_2;
    QLineEdit *m_DeltaFuncAlpha;
    QLineEdit *m_ImageDim;
    QLabel *textLabel8;
    QLabel *DelatAlpha;
    QLineEdit *m_NarrowBandSub;
    QLineEdit *m_SplineDim;
    QLabel *textLabel7;
    QLabel *textLabel1;
    Q3GroupBox *groupBox11;
    QLabel *textLabel1_3;
    QLabel *textLabel1_4;
    QLineEdit *m_Flow;
    QLineEdit *m_Thickness;
    Q3GroupBox *groupBox8_2;
    QComboBox *m_OrderCombo;
    QLineEdit *m_BandWidth;
    QLineEdit *m_NewNv;
    QLabel *textLabel1_2;
    Q3GroupBox *groupBox1_2;
    Q3ButtonGroup *m_buttonGroup;
    QGridLayout *gridLayout;
    QRadioButton *radioButtonPre;
    QRadioButton *radioButtonInp;
    Q3GroupBox *m_paramsGroup;
    QLabel *reconj1;
    QLineEdit *m_Reconal;
    QLineEdit *m_Reconga;
    QLineEdit *m_Reconla;
    QLineEdit *m_Reconbe;
    QLabel *reconalpha;
    QLabel *reconbeta;
    QLabel *recongamma;
    QLabel *reconlamda;
    QLineEdit *m_Reconj1;
    Q3GroupBox *m_initGroup;
    QLabel *TimeStep;
    QLabel *IterNumber;
    QLineEdit *m_IterNumber;
    QLineEdit *m_TimeStep;
    Q3ButtonGroup *m_buttonGroup_2;
    QRadioButton *radioButton2_2;
    QRadioButton *radioButton3;
    Q3ButtonGroup *m_MethodsGroup;
    QGridLayout *gridLayout1;
    QRadioButton *m_RealSpace;
    QRadioButton *m_Parseval;
    Q3GroupBox *m_loadGroup;
    QPushButton *m_pushButton;
    QLabel *m_toallabel;
    QLabel *MolecularVolume;
    QPushButton *m_LoadInitFbutton;
    QLineEdit *m_LoadInitF;
    QLabel *textLabel2;
    QLineEdit *m_TotalNum;
    QLineEdit *m_MolVolume;
    QLineEdit *m_lineEditDir;
    QRadioButton *LoadInit_f;
    Q3GroupBox *m_testGroup;
    QLineEdit *m_PhantomId;
    QLabel *textLabel9;
    Q3GroupBox *groupBox8;
    QLineEdit *m_Rot;
    QLabel *Tilt;
    QLineEdit *m_Tilt;
    QLabel *Psi;
    QLineEdit *m_Psi;
    QLabel *Rot;
    QLineEdit *HostName;
    QLabel *textLabel2_2_2;
    QLabel *textLabel2_2;
    QLineEdit *PortName;
    QPushButton *RunButton;
    QPushButton *CancelButton;
    QRadioButton *m_RemoteReconRun;

    void setupUi(QDialog *ReconstructionDialog)
    {
        if (ReconstructionDialog->objectName().isEmpty())
            ReconstructionDialog->setObjectName(QString::fromUtf8("ReconstructionDialog"));
        ReconstructionDialog->resize(557, 738);
        groupBox10_2 = new Q3GroupBox(ReconstructionDialog);
        groupBox10_2->setObjectName(QString::fromUtf8("groupBox10_2"));
        groupBox10_2->setGeometry(QRect(10, 10, 330, 110));
        m_DeltaFuncAlpha = new QLineEdit(groupBox10_2);
        m_DeltaFuncAlpha->setObjectName(QString::fromUtf8("m_DeltaFuncAlpha"));
        m_DeltaFuncAlpha->setGeometry(QRect(110, 70, 50, 31));
        m_ImageDim = new QLineEdit(groupBox10_2);
        m_ImageDim->setObjectName(QString::fromUtf8("m_ImageDim"));
        m_ImageDim->setGeometry(QRect(110, 30, 50, 30));
        textLabel8 = new QLabel(groupBox10_2);
        textLabel8->setObjectName(QString::fromUtf8("textLabel8"));
        textLabel8->setGeometry(QRect(20, 20, 77, 50));
        textLabel8->setWordWrap(false);
        DelatAlpha = new QLabel(groupBox10_2);
        DelatAlpha->setObjectName(QString::fromUtf8("DelatAlpha"));
        DelatAlpha->setGeometry(QRect(20, 70, 90, 31));
        DelatAlpha->setWordWrap(false);
        m_NarrowBandSub = new QLineEdit(groupBox10_2);
        m_NarrowBandSub->setObjectName(QString::fromUtf8("m_NarrowBandSub"));
        m_NarrowBandSub->setGeometry(QRect(250, 70, 60, 31));
        m_SplineDim = new QLineEdit(groupBox10_2);
        m_SplineDim->setObjectName(QString::fromUtf8("m_SplineDim"));
        m_SplineDim->setGeometry(QRect(250, 30, 60, 30));
        textLabel7 = new QLabel(groupBox10_2);
        textLabel7->setObjectName(QString::fromUtf8("textLabel7"));
        textLabel7->setGeometry(QRect(170, 20, 81, 50));
        textLabel7->setWordWrap(false);
        textLabel1 = new QLabel(groupBox10_2);
        textLabel1->setObjectName(QString::fromUtf8("textLabel1"));
        textLabel1->setGeometry(QRect(220, 70, 30, 31));
        textLabel1->setWordWrap(false);
        groupBox11 = new Q3GroupBox(ReconstructionDialog);
        groupBox11->setObjectName(QString::fromUtf8("groupBox11"));
        groupBox11->setGeometry(QRect(420, 320, 120, 110));
        textLabel1_3 = new QLabel(groupBox11);
        textLabel1_3->setObjectName(QString::fromUtf8("textLabel1_3"));
        textLabel1_3->setGeometry(QRect(10, 30, 80, 31));
        textLabel1_3->setWordWrap(false);
        textLabel1_4 = new QLabel(groupBox11);
        textLabel1_4->setObjectName(QString::fromUtf8("textLabel1_4"));
        textLabel1_4->setGeometry(QRect(10, 70, 84, 31));
        textLabel1_4->setWordWrap(false);
        m_Flow = new QLineEdit(groupBox11);
        m_Flow->setObjectName(QString::fromUtf8("m_Flow"));
        m_Flow->setGeometry(QRect(80, 30, 30, 31));
        m_Thickness = new QLineEdit(groupBox11);
        m_Thickness->setObjectName(QString::fromUtf8("m_Thickness"));
        m_Thickness->setGeometry(QRect(80, 70, 31, 31));
        groupBox8_2 = new Q3GroupBox(ReconstructionDialog);
        groupBox8_2->setObjectName(QString::fromUtf8("groupBox8_2"));
        groupBox8_2->setGeometry(QRect(260, 320, 150, 110));
        m_OrderCombo = new QComboBox(groupBox8_2);
        m_OrderCombo->setObjectName(QString::fromUtf8("m_OrderCombo"));
        m_OrderCombo->setGeometry(QRect(10, 30, 90, 31));
        m_BandWidth = new QLineEdit(groupBox8_2);
        m_BandWidth->setObjectName(QString::fromUtf8("m_BandWidth"));
        m_BandWidth->setGeometry(QRect(110, 70, 30, 31));
        m_NewNv = new QLineEdit(groupBox8_2);
        m_NewNv->setObjectName(QString::fromUtf8("m_NewNv"));
        m_NewNv->setGeometry(QRect(110, 30, 30, 31));
        textLabel1_2 = new QLabel(groupBox8_2);
        textLabel1_2->setObjectName(QString::fromUtf8("textLabel1_2"));
        textLabel1_2->setGeometry(QRect(10, 70, 92, 31));
        textLabel1_2->setWordWrap(false);
        groupBox1_2 = new Q3GroupBox(ReconstructionDialog);
        groupBox1_2->setObjectName(QString::fromUtf8("groupBox1_2"));
        groupBox1_2->setGeometry(QRect(10, 430, 530, 120));
        m_buttonGroup = new Q3ButtonGroup(groupBox1_2);
        m_buttonGroup->setObjectName(QString::fromUtf8("m_buttonGroup"));
        m_buttonGroup->setGeometry(QRect(10, 30, 189, 80));
        m_buttonGroup->setProperty("selectedId", QVariant(0));
        m_buttonGroup->setColumnLayout(0, Qt::Vertical);
        m_buttonGroup->layout()->setSpacing(6);
        m_buttonGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout = new QGridLayout();
        QBoxLayout *boxlayout = qobject_cast<QBoxLayout *>(m_buttonGroup->layout());
        if (boxlayout)
            boxlayout->addLayout(gridLayout);
        gridLayout->setAlignment(Qt::AlignTop);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        radioButtonPre = new QRadioButton(m_buttonGroup);
        radioButtonPre->setObjectName(QString::fromUtf8("radioButtonPre"));
        radioButtonPre->setChecked(true);

        gridLayout->addWidget(radioButtonPre, 0, 0, 1, 1);

        radioButtonInp = new QRadioButton(m_buttonGroup);
        radioButtonInp->setObjectName(QString::fromUtf8("radioButtonInp"));

        gridLayout->addWidget(radioButtonInp, 1, 0, 1, 1);

        m_paramsGroup = new Q3GroupBox(groupBox1_2);
        m_paramsGroup->setObjectName(QString::fromUtf8("m_paramsGroup"));
        m_paramsGroup->setEnabled(false);
        m_paramsGroup->setGeometry(QRect(250, 20, 270, 90));
        reconj1 = new QLabel(m_paramsGroup);
        reconj1->setObjectName(QString::fromUtf8("reconj1"));
        reconj1->setGeometry(QRect(10, 30, 20, 31));
        reconj1->setWordWrap(false);
        m_Reconal = new QLineEdit(m_paramsGroup);
        m_Reconal->setObjectName(QString::fromUtf8("m_Reconal"));
        m_Reconal->setGeometry(QRect(110, 10, 60, 31));
        m_Reconga = new QLineEdit(m_paramsGroup);
        m_Reconga->setObjectName(QString::fromUtf8("m_Reconga"));
        m_Reconga->setGeometry(QRect(110, 50, 60, 31));
        m_Reconla = new QLineEdit(m_paramsGroup);
        m_Reconla->setObjectName(QString::fromUtf8("m_Reconla"));
        m_Reconla->setGeometry(QRect(200, 50, 60, 31));
        m_Reconbe = new QLineEdit(m_paramsGroup);
        m_Reconbe->setObjectName(QString::fromUtf8("m_Reconbe"));
        m_Reconbe->setGeometry(QRect(200, 10, 60, 31));
        reconalpha = new QLabel(m_paramsGroup);
        reconalpha->setObjectName(QString::fromUtf8("reconalpha"));
        reconalpha->setGeometry(QRect(90, 10, 20, 31));
        reconalpha->setWordWrap(false);
        reconbeta = new QLabel(m_paramsGroup);
        reconbeta->setObjectName(QString::fromUtf8("reconbeta"));
        reconbeta->setGeometry(QRect(180, 10, 20, 31));
        reconbeta->setWordWrap(false);
        recongamma = new QLabel(m_paramsGroup);
        recongamma->setObjectName(QString::fromUtf8("recongamma"));
        recongamma->setGeometry(QRect(90, 50, 21, 31));
        recongamma->setWordWrap(false);
        reconlamda = new QLabel(m_paramsGroup);
        reconlamda->setObjectName(QString::fromUtf8("reconlamda"));
        reconlamda->setGeometry(QRect(180, 50, 20, 30));
        reconlamda->setWordWrap(false);
        m_Reconj1 = new QLineEdit(m_paramsGroup);
        m_Reconj1->setObjectName(QString::fromUtf8("m_Reconj1"));
        m_Reconj1->setGeometry(QRect(30, 30, 50, 31));
        m_initGroup = new Q3GroupBox(ReconstructionDialog);
        m_initGroup->setObjectName(QString::fromUtf8("m_initGroup"));
        m_initGroup->setGeometry(QRect(140, 320, 110, 110));
        TimeStep = new QLabel(m_initGroup);
        TimeStep->setObjectName(QString::fromUtf8("TimeStep"));
        TimeStep->setGeometry(QRect(20, 70, 31, 30));
        TimeStep->setWordWrap(false);
        IterNumber = new QLabel(m_initGroup);
        IterNumber->setObjectName(QString::fromUtf8("IterNumber"));
        IterNumber->setGeometry(QRect(10, 30, 54, 31));
        IterNumber->setWordWrap(false);
        m_IterNumber = new QLineEdit(m_initGroup);
        m_IterNumber->setObjectName(QString::fromUtf8("m_IterNumber"));
        m_IterNumber->setGeometry(QRect(60, 30, 40, 31));
        m_TimeStep = new QLineEdit(m_initGroup);
        m_TimeStep->setObjectName(QString::fromUtf8("m_TimeStep"));
        m_TimeStep->setGeometry(QRect(60, 70, 40, 31));
        m_buttonGroup_2 = new Q3ButtonGroup(ReconstructionDialog);
        m_buttonGroup_2->setObjectName(QString::fromUtf8("m_buttonGroup_2"));
        m_buttonGroup_2->setGeometry(QRect(360, 20, 148, 100));
        m_buttonGroup_2->setProperty("selectedId", QVariant(0));
        radioButton2_2 = new QRadioButton(m_buttonGroup_2);
        radioButton2_2->setObjectName(QString::fromUtf8("radioButton2_2"));
        radioButton2_2->setGeometry(QRect(10, 20, 126, 26));
        radioButton2_2->setChecked(true);
        m_buttonGroup_2->insert(radioButton2_2, 0);
        radioButton3 = new QRadioButton(m_buttonGroup_2);
        radioButton3->setObjectName(QString::fromUtf8("radioButton3"));
        radioButton3->setEnabled(true);
        radioButton3->setGeometry(QRect(10, 60, 126, 26));
        m_buttonGroup_2->insert(radioButton3, 1);
        m_MethodsGroup = new Q3ButtonGroup(ReconstructionDialog);
        m_MethodsGroup->setObjectName(QString::fromUtf8("m_MethodsGroup"));
        m_MethodsGroup->setGeometry(QRect(10, 320, 123, 97));
        m_MethodsGroup->setColumnLayout(0, Qt::Vertical);
        m_MethodsGroup->layout()->setSpacing(6);
        m_MethodsGroup->layout()->setContentsMargins(11, 11, 11, 11);
        gridLayout1 = new QGridLayout();
        QBoxLayout *boxlayout1 = qobject_cast<QBoxLayout *>(m_MethodsGroup->layout());
        if (boxlayout1)
            boxlayout1->addLayout(gridLayout1);
        gridLayout1->setAlignment(Qt::AlignTop);
        gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
        m_RealSpace = new QRadioButton(m_MethodsGroup);
        m_RealSpace->setObjectName(QString::fromUtf8("m_RealSpace"));
        m_RealSpace->setChecked(true);

        gridLayout1->addWidget(m_RealSpace, 1, 0, 1, 1);

        m_Parseval = new QRadioButton(m_MethodsGroup);
        m_Parseval->setObjectName(QString::fromUtf8("m_Parseval"));

        gridLayout1->addWidget(m_Parseval, 0, 0, 1, 1);

        m_loadGroup = new Q3GroupBox(ReconstructionDialog);
        m_loadGroup->setObjectName(QString::fromUtf8("m_loadGroup"));
        m_loadGroup->setGeometry(QRect(10, 120, 530, 110));
        m_pushButton = new QPushButton(m_loadGroup);
        m_pushButton->setObjectName(QString::fromUtf8("m_pushButton"));
        m_pushButton->setGeometry(QRect(500, 70, 21, 31));
        m_toallabel = new QLabel(m_loadGroup);
        m_toallabel->setObjectName(QString::fromUtf8("m_toallabel"));
        m_toallabel->setGeometry(QRect(10, 30, 73, 31));
        m_toallabel->setWordWrap(false);
        MolecularVolume = new QLabel(m_loadGroup);
        MolecularVolume->setObjectName(QString::fromUtf8("MolecularVolume"));
        MolecularVolume->setGeometry(QRect(120, 30, 30, 31));
        MolecularVolume->setWordWrap(false);
        m_LoadInitFbutton = new QPushButton(m_loadGroup);
        m_LoadInitFbutton->setObjectName(QString::fromUtf8("m_LoadInitFbutton"));
        m_LoadInitFbutton->setGeometry(QRect(500, 30, 21, 31));
        m_LoadInitF = new QLineEdit(m_loadGroup);
        m_LoadInitF->setObjectName(QString::fromUtf8("m_LoadInitF"));
        m_LoadInitF->setEnabled(false);
        m_LoadInitF->setGeometry(QRect(350, 30, 150, 31));
        textLabel2 = new QLabel(m_loadGroup);
        textLabel2->setObjectName(QString::fromUtf8("textLabel2"));
        textLabel2->setGeometry(QRect(10, 70, 77, 31));
        textLabel2->setWordWrap(false);
        m_TotalNum = new QLineEdit(m_loadGroup);
        m_TotalNum->setObjectName(QString::fromUtf8("m_TotalNum"));
        m_TotalNum->setGeometry(QRect(80, 30, 40, 31));
        m_MolVolume = new QLineEdit(m_loadGroup);
        m_MolVolume->setObjectName(QString::fromUtf8("m_MolVolume"));
        m_MolVolume->setGeometry(QRect(150, 30, 40, 31));
        m_lineEditDir = new QLineEdit(m_loadGroup);
        m_lineEditDir->setObjectName(QString::fromUtf8("m_lineEditDir"));
        m_lineEditDir->setGeometry(QRect(100, 70, 400, 31));
        LoadInit_f = new QRadioButton(m_loadGroup);
        LoadInit_f->setObjectName(QString::fromUtf8("LoadInit_f"));
        LoadInit_f->setGeometry(QRect(220, 30, 120, 30));
        m_testGroup = new Q3GroupBox(ReconstructionDialog);
        m_testGroup->setObjectName(QString::fromUtf8("m_testGroup"));
        m_testGroup->setEnabled(false);
        m_testGroup->setGeometry(QRect(10, 230, 530, 80));
        m_PhantomId = new QLineEdit(m_testGroup);
        m_PhantomId->setObjectName(QString::fromUtf8("m_PhantomId"));
        m_PhantomId->setGeometry(QRect(170, 30, 40, 30));
        textLabel9 = new QLabel(m_testGroup);
        textLabel9->setObjectName(QString::fromUtf8("textLabel9"));
        textLabel9->setGeometry(QRect(10, 30, 155, 30));
        textLabel9->setWordWrap(false);
        groupBox8 = new Q3GroupBox(m_testGroup);
        groupBox8->setObjectName(QString::fromUtf8("groupBox8"));
        groupBox8->setGeometry(QRect(260, 10, 250, 60));
        m_Rot = new QLineEdit(groupBox8);
        m_Rot->setObjectName(QString::fromUtf8("m_Rot"));
        m_Rot->setGeometry(QRect(40, 20, 40, 31));
        Tilt = new QLabel(groupBox8);
        Tilt->setObjectName(QString::fromUtf8("Tilt"));
        Tilt->setGeometry(QRect(90, 20, 30, 31));
        Tilt->setWordWrap(false);
        m_Tilt = new QLineEdit(groupBox8);
        m_Tilt->setObjectName(QString::fromUtf8("m_Tilt"));
        m_Tilt->setGeometry(QRect(120, 20, 40, 31));
        Psi = new QLabel(groupBox8);
        Psi->setObjectName(QString::fromUtf8("Psi"));
        Psi->setGeometry(QRect(170, 20, 30, 31));
        Psi->setWordWrap(false);
        m_Psi = new QLineEdit(groupBox8);
        m_Psi->setObjectName(QString::fromUtf8("m_Psi"));
        m_Psi->setGeometry(QRect(200, 20, 40, 31));
        Rot = new QLabel(groupBox8);
        Rot->setObjectName(QString::fromUtf8("Rot"));
        Rot->setGeometry(QRect(10, 20, 30, 31));
        Rot->setWordWrap(false);
        HostName = new QLineEdit(ReconstructionDialog);
        HostName->setObjectName(QString::fromUtf8("HostName"));
        HostName->setGeometry(QRect(110, 570, 160, 31));
        textLabel2_2_2 = new QLabel(ReconstructionDialog);
        textLabel2_2_2->setObjectName(QString::fromUtf8("textLabel2_2_2"));
        textLabel2_2_2->setGeometry(QRect(280, 570, 80, 30));
        textLabel2_2_2->setWordWrap(false);
        textLabel2_2 = new QLabel(ReconstructionDialog);
        textLabel2_2->setObjectName(QString::fromUtf8("textLabel2_2"));
        textLabel2_2->setGeometry(QRect(0, 570, 90, 30));
        textLabel2_2->setWordWrap(false);
        PortName = new QLineEdit(ReconstructionDialog);
        PortName->setObjectName(QString::fromUtf8("PortName"));
        PortName->setGeometry(QRect(360, 570, 160, 31));
        RunButton = new QPushButton(ReconstructionDialog);
        RunButton->setObjectName(QString::fromUtf8("RunButton"));
        RunButton->setGeometry(QRect(50, 670, 200, 31));
        CancelButton = new QPushButton(ReconstructionDialog);
        CancelButton->setObjectName(QString::fromUtf8("CancelButton"));
        CancelButton->setGeometry(QRect(380, 670, 100, 31));
        m_RemoteReconRun = new QRadioButton(ReconstructionDialog);
        m_RemoteReconRun->setObjectName(QString::fromUtf8("m_RemoteReconRun"));
        m_RemoteReconRun->setGeometry(QRect(170, 620, 240, 26));

        retranslateUi(ReconstructionDialog);
        QObject::connect(RunButton, SIGNAL(clicked()), ReconstructionDialog, SLOT(accept()));
        QObject::connect(CancelButton, SIGNAL(clicked()), ReconstructionDialog, SLOT(reject()));
        QObject::connect(m_buttonGroup, SIGNAL(clicked(int)), ReconstructionDialog, SLOT(changeParamsValuesSlot(int)));
        QObject::connect(m_buttonGroup_2, SIGNAL(clicked(int)), ReconstructionDialog, SLOT(changeReconMannerSlot(int)));
        QObject::connect(m_pushButton, SIGNAL(clicked()), ReconstructionDialog, SLOT(browseSlot()));
        QObject::connect(LoadInit_f, SIGNAL(clicked()), ReconstructionDialog, SLOT(changeStatusSlot()));
        QObject::connect(m_LoadInitFbutton, SIGNAL(clicked()), ReconstructionDialog, SLOT(browseloadinitfSlot()));

        QMetaObject::connectSlotsByName(ReconstructionDialog);
    } // setupUi

    void retranslateUi(QDialog *ReconstructionDialog)
    {
        ReconstructionDialog->setWindowTitle(QApplication::translate("ReconstructionDialog", "Reconstruction Dialog", 0, QApplication::UnicodeUTF8));
        groupBox10_2->setTitle(QApplication::translate("ReconstructionDialog", "Fixed_Parameters", 0, QApplication::UnicodeUTF8));
        m_DeltaFuncAlpha->setText(QApplication::translate("ReconstructionDialog", "0.01", 0, QApplication::UnicodeUTF8));
        m_ImageDim->setText(QApplication::translate("ReconstructionDialog", "16", 0, QApplication::UnicodeUTF8));
        textLabel8->setText(QApplication::translate("ReconstructionDialog", "Image(n):", 0, QApplication::UnicodeUTF8));
        DelatAlpha->setText(QApplication::translate("ReconstructionDialog", "Delf alpha:", 0, QApplication::UnicodeUTF8));
        m_NarrowBandSub->setText(QApplication::translate("ReconstructionDialog", "0.334", 0, QApplication::UnicodeUTF8));
        m_SplineDim->setText(QApplication::translate("ReconstructionDialog", "16", 0, QApplication::UnicodeUTF8));
        textLabel7->setText(QApplication::translate("ReconstructionDialog", "Spline(m):", 0, QApplication::UnicodeUTF8));
        textLabel1->setText(QApplication::translate("ReconstructionDialog", "fac:", 0, QApplication::UnicodeUTF8));
        groupBox11->setTitle(QApplication::translate("ReconstructionDialog", "Regularize", 0, QApplication::UnicodeUTF8));
        textLabel1_3->setText(QApplication::translate("ReconstructionDialog", "J3_Flow:", 0, QApplication::UnicodeUTF8));
        textLabel1_4->setText(QApplication::translate("ReconstructionDialog", "Thick:", 0, QApplication::UnicodeUTF8));
        m_Flow->setText(QApplication::translate("ReconstructionDialog", "1", 0, QApplication::UnicodeUTF8));
        m_Thickness->setText(QApplication::translate("ReconstructionDialog", "10", 0, QApplication::UnicodeUTF8));
        groupBox8_2->setTitle(QApplication::translate("ReconstructionDialog", "SpeedUp", 0, QApplication::UnicodeUTF8));
        m_OrderCombo->clear();
        m_OrderCombo->insertItems(0, QStringList()
         << QApplication::translate("ReconstructionDialog", "newnv", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("ReconstructionDialog", "ortho order", 0, QApplication::UnicodeUTF8)
        );
        m_BandWidth->setText(QApplication::translate("ReconstructionDialog", "8", 0, QApplication::UnicodeUTF8));
        m_NewNv->setText(QApplication::translate("ReconstructionDialog", "10", 0, QApplication::UnicodeUTF8));
        textLabel1_2->setText(QApplication::translate("ReconstructionDialog", "BandWidth:", 0, QApplication::UnicodeUTF8));
        groupBox1_2->setTitle(QApplication::translate("ReconstructionDialog", "Coeffs_Params ", 0, QApplication::UnicodeUTF8));
        m_buttonGroup->setTitle(QString());
        radioButtonPre->setText(QApplication::translate("ReconstructionDialog", "Previous Values", 0, QApplication::UnicodeUTF8));
        radioButtonInp->setText(QApplication::translate("ReconstructionDialog", "Input New Params", 0, QApplication::UnicodeUTF8));
        m_paramsGroup->setTitle(QString());
        reconj1->setText(QApplication::translate("ReconstructionDialog", "J1:", 0, QApplication::UnicodeUTF8));
        m_Reconal->setText(QApplication::translate("ReconstructionDialog", "0", 0, QApplication::UnicodeUTF8));
        m_Reconga->setText(QApplication::translate("ReconstructionDialog", "0", 0, QApplication::UnicodeUTF8));
        m_Reconla->setText(QApplication::translate("ReconstructionDialog", "0", 0, QApplication::UnicodeUTF8));
        m_Reconbe->setText(QApplication::translate("ReconstructionDialog", "0", 0, QApplication::UnicodeUTF8));
        reconalpha->setText(QApplication::translate("ReconstructionDialog", "J2:", 0, QApplication::UnicodeUTF8));
        reconbeta->setText(QApplication::translate("ReconstructionDialog", "J3:", 0, QApplication::UnicodeUTF8));
        recongamma->setText(QApplication::translate("ReconstructionDialog", "J4:", 0, QApplication::UnicodeUTF8));
        reconlamda->setText(QApplication::translate("ReconstructionDialog", "J5:", 0, QApplication::UnicodeUTF8));
        m_Reconj1->setText(QApplication::translate("ReconstructionDialog", "1.0", 0, QApplication::UnicodeUTF8));
        m_initGroup->setTitle(QApplication::translate("ReconstructionDialog", "Iter_Param ", 0, QApplication::UnicodeUTF8));
        TimeStep->setText(QApplication::translate("ReconstructionDialog", "tau:", 0, QApplication::UnicodeUTF8));
        IterNumber->setText(QApplication::translate("ReconstructionDialog", "Iters:", 0, QApplication::UnicodeUTF8));
        m_IterNumber->setText(QApplication::translate("ReconstructionDialog", "1", 0, QApplication::UnicodeUTF8));
        m_TimeStep->setText(QApplication::translate("ReconstructionDialog", "1", 0, QApplication::UnicodeUTF8));
        m_buttonGroup_2->setTitle(QString());
        radioButton2_2->setText(QApplication::translate("ReconstructionDialog", "load images", 0, QApplication::UnicodeUTF8));
        radioButton3->setText(QApplication::translate("ReconstructionDialog", "phantom test", 0, QApplication::UnicodeUTF8));
        m_MethodsGroup->setTitle(QApplication::translate("ReconstructionDialog", "Methods", 0, QApplication::UnicodeUTF8));
        m_RealSpace->setText(QApplication::translate("ReconstructionDialog", "realspace", 0, QApplication::UnicodeUTF8));
        m_Parseval->setText(QApplication::translate("ReconstructionDialog", "Parseval", 0, QApplication::UnicodeUTF8));
        m_loadGroup->setTitle(QApplication::translate("ReconstructionDialog", "Read 2D projection images", 0, QApplication::UnicodeUTF8));
        m_pushButton->setText(QApplication::translate("ReconstructionDialog", "...", 0, QApplication::UnicodeUTF8));
        m_toallabel->setText(QApplication::translate("ReconstructionDialog", "ProjNum:", 0, QApplication::UnicodeUTF8));
        MolecularVolume->setText(QApplication::translate("ReconstructionDialog", "Vol:", 0, QApplication::UnicodeUTF8));
        m_LoadInitFbutton->setText(QApplication::translate("ReconstructionDialog", "...", 0, QApplication::UnicodeUTF8));
        textLabel2->setText(QApplication::translate("ReconstructionDialog", "Open file:", 0, QApplication::UnicodeUTF8));
        m_TotalNum->setText(QApplication::translate("ReconstructionDialog", "180", 0, QApplication::UnicodeUTF8));
        m_MolVolume->setText(QApplication::translate("ReconstructionDialog", "0", 0, QApplication::UnicodeUTF8));
        m_lineEditDir->setText(QString());
        LoadInit_f->setText(QApplication::translate("ReconstructionDialog", "Load_Init_f", 0, QApplication::UnicodeUTF8));
        m_testGroup->setTitle(QApplication::translate("ReconstructionDialog", "Test Parameters", 0, QApplication::UnicodeUTF8));
        m_PhantomId->setText(QApplication::translate("ReconstructionDialog", "0", 0, QApplication::UnicodeUTF8));
        textLabel9->setText(QApplication::translate("ReconstructionDialog", "PhantomID(0, 1, 2):", 0, QApplication::UnicodeUTF8));
        groupBox8->setTitle(QApplication::translate("ReconstructionDialog", "Euler Angles Setting", 0, QApplication::UnicodeUTF8));
        m_Rot->setText(QApplication::translate("ReconstructionDialog", "6", 0, QApplication::UnicodeUTF8));
        Tilt->setText(QApplication::translate("ReconstructionDialog", "Tilt:", 0, QApplication::UnicodeUTF8));
        m_Tilt->setText(QApplication::translate("ReconstructionDialog", "6", 0, QApplication::UnicodeUTF8));
        Psi->setText(QApplication::translate("ReconstructionDialog", "Psi:", 0, QApplication::UnicodeUTF8));
        m_Psi->setText(QApplication::translate("ReconstructionDialog", "0", 0, QApplication::UnicodeUTF8));
        Rot->setText(QApplication::translate("ReconstructionDialog", "Rot:", 0, QApplication::UnicodeUTF8));
        HostName->setText(QString());
        textLabel2_2_2->setText(QApplication::translate("ReconstructionDialog", "Port:", 0, QApplication::UnicodeUTF8));
        textLabel2_2->setText(QApplication::translate("ReconstructionDialog", "Host Name:", 0, QApplication::UnicodeUTF8));
        PortName->setText(QString());
        RunButton->setText(QApplication::translate("ReconstructionDialog", "OK", 0, QApplication::UnicodeUTF8));
        CancelButton->setText(QApplication::translate("ReconstructionDialog", "Cancel", 0, QApplication::UnicodeUTF8));
        m_RemoteReconRun->setText(QApplication::translate("ReconstructionDialog", "Run Remote Reconstruction", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ReconstructionDialog: public Ui_ReconstructionDialog {};
} // namespace Ui

QT_END_NAMESPACE

class ReconstructionDialog : public QDialog, public Ui::ReconstructionDialog
{
    Q_OBJECT

public:
    ReconstructionDialog(QWidget* parent = 0, const char* name = 0, bool modal = false, Qt::WindowFlags fl = 0);
    ~ReconstructionDialog();

public slots:
    virtual void changeParamsValuesSlot( int );
    virtual void changeStatusSlot();
    virtual void changeReconMannerSlot( int );
    virtual void browseSlot();
    virtual void browseloadinitfSlot();

protected slots:
    virtual void languageChange();

};

#endif // RECONSTRUCTIONDIALOG_H
