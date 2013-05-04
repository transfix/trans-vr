#ifndef TERMINALBASE_H
#define TERMINALBASE_H

#include <qvariant.h>


#include <Qt3Support/Q3MimeSourceFactory>
#include <Qt3Support/Q3TextEdit>
#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TerminalBase
{
public:
    QGridLayout *gridLayout;
    QSpacerItem *spacer2;
    Q3TextEdit *text;
    QPushButton *clearButton;

    void setupUi(QWidget *TerminalBase)
    {
        if (TerminalBase->objectName().isEmpty())
            TerminalBase->setObjectName(QString::fromUtf8("TerminalBase"));
        TerminalBase->resize(396, 380);
        gridLayout = new QGridLayout(TerminalBase);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        spacer2 = new QSpacerItem(310, 31, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(spacer2, 0, 1, 1, 1);

        text = new Q3TextEdit(TerminalBase);
        text->setObjectName(QString::fromUtf8("text"));
        text->setReadOnly(true);

        gridLayout->addWidget(text, 1, 0, 1, 2);

        clearButton = new QPushButton(TerminalBase);
        clearButton->setObjectName(QString::fromUtf8("clearButton"));

        gridLayout->addWidget(clearButton, 0, 0, 1, 1);


        retranslateUi(TerminalBase);
        QObject::connect(clearButton, SIGNAL(clicked()), text, SLOT(clear()));

        QMetaObject::connectSlotsByName(TerminalBase);
    } // setupUi

    void retranslateUi(QWidget *TerminalBase)
    {
        TerminalBase->setWindowTitle(QApplication::translate("TerminalBase", "Volume Rover Terminal", 0, QApplication::UnicodeUTF8));
        clearButton->setText(QApplication::translate("TerminalBase", "Clear", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class TerminalBase: public Ui_TerminalBase {};
} // namespace Ui

QT_END_NAMESPACE

class TerminalBase : public QWidget, public Ui::TerminalBase
{
    Q_OBJECT

public:
    TerminalBase(QWidget* parent = 0, const char* name = 0, Qt::WindowFlags fl = 0);
    ~TerminalBase();

protected slots:
    virtual void languageChange();

};

#endif // TERMINALBASE_H
