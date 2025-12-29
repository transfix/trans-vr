#include <ColorTable2/InfoDialog.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qpushbutton.h>
#include <stdio.h>

namespace CVCColorTable {
ConSpecInfoNode::ConSpecInfoNode(int id, double position, float isoval,
                                 float area, float minvol, float maxvol,
                                 float grad, int nComp)
    : QWidget(nullptr) {
  m_ID = id;
  m_Position = position;
  m_IsoVal = isoval;
  m_Area = area;
  m_MinVol = minvol;
  m_MaxVol = maxvol;
  m_Grad = grad;
  m_nComp = nComp;

  QPushButton *posBox = new QPushButton;
  posBox->setFixedSize(15, 15);
  QPushButton *isoBox = new QPushButton;
  isoBox->setFixedSize(15, 15);
  isoBox->setStyleSheet("background-color: black");
  QPushButton *areaBox = new QPushButton;
  areaBox->setFixedSize(15, 15);
  areaBox->setStyleSheet("background-color: red");
  QPushButton *minvolBox = new QPushButton;
  minvolBox->setFixedSize(15, 15);
  minvolBox->setStyleSheet("background-color: green");
  QPushButton *maxvolBox = new QPushButton;
  maxvolBox->setFixedSize(15, 15);
  maxvolBox->setStyleSheet("background-color: blue");
  QPushButton *gradBox = new QPushButton;
  gradBox->setFixedSize(15, 15);
  gradBox->setStyleSheet("background-color: yellow");
  QPushButton *compBox = new QPushButton;
  compBox->setFixedSize(15, 15);

  QLabel *positionLabel = new QLabel(tr("Position:"));
  QLabel *isovalLabel = new QLabel(tr("Iso-value:"));
  QLabel *areaLabel = new QLabel(tr("Surface area:"));
  QLabel *minvolLabel = new QLabel(tr("Min. Volume:"));
  QLabel *maxvolLabel = new QLabel(tr("Max. Volume:"));
  QLabel *gradLabel = new QLabel(tr("Gradient:"));
  QLabel *compLabel = new QLabel(tr("# of Components:"));

  positionLineEdit = new QLineEdit;
  positionLineEdit->setFocus();
  isovalLineEdit = new QLineEdit;
  isovalLineEdit->setFocus();
  areaLineEdit = new QLineEdit;
  areaLineEdit->setFocus();
  minvolLineEdit = new QLineEdit;
  minvolLineEdit->setFocus();
  maxvolLineEdit = new QLineEdit;
  maxvolLineEdit->setFocus();
  gradLineEdit = new QLineEdit;
  gradLineEdit->setFocus();
  compLineEdit = new QLineEdit;
  compLineEdit->setFocus();

  char str[1024];
  sprintf(str, "%lf", m_Position);
  positionLineEdit->setText(QString(str));
  sprintf(str, "%f", m_IsoVal);
  isovalLineEdit->setText(QString(str));
  sprintf(str, "%f", m_Area);
  areaLineEdit->setText(QString(str));
  sprintf(str, "%f", m_MinVol);
  minvolLineEdit->setText(QString(str));
  sprintf(str, "%f", m_MaxVol);
  maxvolLineEdit->setText(QString(str));
  sprintf(str, "%f", m_Grad);
  gradLineEdit->setText(QString(str));
  sprintf(str, "%d", m_nComp);
  compLineEdit->setText(QString(str));

  QGridLayout *positionLayout = new QGridLayout(this);
  positionLayout->addWidget(posBox, 0, 0);
  positionLayout->addWidget(positionLabel, 0, 1);
  positionLayout->addWidget(positionLineEdit, 0, 2);
  positionLayout->addWidget(isoBox, 1, 0);
  positionLayout->addWidget(isovalLabel, 1, 1);
  positionLayout->addWidget(isovalLineEdit, 1, 2);
  positionLayout->addWidget(areaBox, 2, 0);
  positionLayout->addWidget(areaLabel, 2, 1);
  positionLayout->addWidget(areaLineEdit, 2, 2);
  positionLayout->addWidget(minvolBox, 3, 0);
  positionLayout->addWidget(minvolLabel, 3, 1);
  positionLayout->addWidget(minvolLineEdit, 3, 2);
  positionLayout->addWidget(maxvolBox, 4, 0);
  positionLayout->addWidget(maxvolLabel, 4, 1);
  positionLayout->addWidget(maxvolLineEdit, 4, 2);
  positionLayout->addWidget(gradBox, 5, 0);
  positionLayout->addWidget(gradLabel, 5, 1);
  positionLayout->addWidget(gradLineEdit, 5, 2);
  positionLayout->addWidget(compBox, 6, 0);
  positionLayout->addWidget(compLabel, 6, 1);
  positionLayout->addWidget(compLineEdit, 6, 2);
}

ConSpecInfoNode::ConSpecInfoNode(const ConSpecInfoNode &copy)
    : QWidget(nullptr) {
  m_ID = copy.m_ID;
  m_Position = copy.m_Position;
  m_IsoVal = copy.m_IsoVal;
  m_Area = copy.m_Area;
  m_MinVol = copy.m_MinVol;
  m_MaxVol = copy.m_MaxVol;
  m_Grad = copy.m_Grad;
  m_nComp = copy.m_nComp;

  QPushButton *posBox = new QPushButton;
  posBox->setFixedSize(15, 15);
  QPushButton *isoBox = new QPushButton;
  isoBox->setFixedSize(15, 15);
  isoBox->setStyleSheet("background-color: black");
  QPushButton *areaBox = new QPushButton;
  areaBox->setFixedSize(15, 15);
  areaBox->setStyleSheet("background-color: red");
  QPushButton *minvolBox = new QPushButton;
  minvolBox->setFixedSize(15, 15);
  minvolBox->setStyleSheet("background-color: green");
  QPushButton *maxvolBox = new QPushButton;
  maxvolBox->setFixedSize(15, 15);
  maxvolBox->setStyleSheet("background-color: blue");
  QPushButton *gradBox = new QPushButton;
  gradBox->setFixedSize(15, 15);
  gradBox->setStyleSheet("background-color: yellow");
  QPushButton *compBox = new QPushButton;
  compBox->setFixedSize(15, 15);

  QLabel *positionLabel = new QLabel(tr("Position:"));
  QLabel *isovalLabel = new QLabel(tr("Iso-value:"));
  QLabel *areaLabel = new QLabel(tr("Surface area:"));
  QLabel *minvolLabel = new QLabel(tr("Min. Volume:"));
  QLabel *maxvolLabel = new QLabel(tr("Max. Volume:"));
  QLabel *gradLabel = new QLabel(tr("<font color='Yellow'>Gradient:</font>"));
  QLabel *compLabel = new QLabel(tr("# of Components:"));

  positionLineEdit = new QLineEdit;
  positionLineEdit->setFocus();
  isovalLineEdit = new QLineEdit;
  isovalLineEdit->setFocus();
  areaLineEdit = new QLineEdit;
  areaLineEdit->setFocus();
  minvolLineEdit = new QLineEdit;
  minvolLineEdit->setFocus();
  maxvolLineEdit = new QLineEdit;
  maxvolLineEdit->setFocus();
  gradLineEdit = new QLineEdit;
  gradLineEdit->setFocus();
  compLineEdit = new QLineEdit;
  compLineEdit->setFocus();

  char str[1024];
  sprintf(str, "%lf", m_Position);
  positionLineEdit->setText(QString(str));
  sprintf(str, "%f", m_IsoVal);
  isovalLineEdit->setText(QString(str));
  sprintf(str, "%f", m_Area);
  areaLineEdit->setText(QString(str));
  sprintf(str, "%f", m_MinVol);
  minvolLineEdit->setText(QString(str));
  sprintf(str, "%f", m_MaxVol);
  maxvolLineEdit->setText(QString(str));
  sprintf(str, "%f", m_Grad);
  gradLineEdit->setText(QString(str));
  sprintf(str, "%d", m_nComp);
  compLineEdit->setText(QString(str));

  QGridLayout *positionLayout = new QGridLayout(this);
  positionLayout->addWidget(posBox, 0, 0);
  positionLayout->addWidget(positionLabel, 0, 1);
  positionLayout->addWidget(positionLineEdit, 0, 2);
  positionLayout->addWidget(isoBox, 1, 0);
  positionLayout->addWidget(isovalLabel, 1, 1);
  positionLayout->addWidget(isovalLineEdit, 1, 2);
  positionLayout->addWidget(areaBox, 2, 0);
  positionLayout->addWidget(areaLabel, 2, 1);
  positionLayout->addWidget(areaLineEdit, 2, 2);
  positionLayout->addWidget(minvolBox, 3, 0);
  positionLayout->addWidget(minvolLabel, 3, 1);
  positionLayout->addWidget(minvolLineEdit, 3, 2);
  positionLayout->addWidget(maxvolBox, 4, 0);
  positionLayout->addWidget(maxvolLabel, 4, 1);
  positionLayout->addWidget(maxvolLineEdit, 4, 2);
  positionLayout->addWidget(gradBox, 5, 0);
  positionLayout->addWidget(gradLabel, 5, 1);
  positionLayout->addWidget(gradLineEdit, 5, 2);
  positionLayout->addWidget(compBox, 6, 0);
  positionLayout->addWidget(compLabel, 6, 1);
}

ConSpecInfoNode::~ConSpecInfoNode() {}

void ConSpecInfoNode::updateNode(double position, float isoval, float area,
                                 float minvol, float maxvol, float grad,
                                 int nComp) {
  m_Position = position;
  m_IsoVal = isoval;
  m_Area = area;
  m_MinVol = minvol;
  m_MaxVol = maxvol;
  m_Grad = grad;
  m_nComp = nComp;

  char str[1024];
  sprintf(str, "%lf", m_Position);
  positionLineEdit->setText(QString(str));
  sprintf(str, "%f", m_IsoVal);
  isovalLineEdit->setText(QString(str));
  sprintf(str, "%f", m_Area);
  areaLineEdit->setText(QString(str));
  sprintf(str, "%f", m_MinVol);
  minvolLineEdit->setText(QString(str));
  sprintf(str, "%f", m_MaxVol);
  maxvolLineEdit->setText(QString(str));
  sprintf(str, "%f", m_Grad);
  gradLineEdit->setText(QString(str));
  sprintf(str, "%d", m_nComp);
  compLineEdit->setText(QString(str));
}

InfoDialog::InfoDialog(QWidget *parent,
#if QT_VERSION < 0x040000
                       const char *name
#else
                       Qt::WindowFlags flags
#endif
                       )
    : QFrame(parent,
#if QT_VERSION < 0x040000
             name
#else
             flags
#endif
      ) {
  //    setMaximumHeight(150);
  //    setMinimumHeight(80);
  setFrameStyle(QFrame::Panel | QFrame::Raised);
  QBoxLayout *layout =
#if QT_VERSION < 0x040000
      new QBoxLayout(this, QBoxLayout::Down);
#else
      new QBoxLayout(QBoxLayout::Down, this);
#endif
  layout->setContentsMargins(3, 3, 3, 3);
  layout->setSpacing(3);

  setFrameRect(QRect(400, 400, 300, 300));

  setFixedWidth(300);
  setFixedHeight(300);

  setWindowTitle(QString("Contour Spectrum/Tree Info."));

  tabWidget = new QTabWidget(this);
  tabWidget->setFixedSize(300, 300);
  tabWidget->hide();
}

InfoDialog::~InfoDialog() { nodes.clear(); }

void InfoDialog::updateInfo(int id, double position, float isoval, float area,
                            float minvol, float maxvol, float grad,
                            int nComp) {
  bool found = false;
  int i = 0;
  for (; i < nodes.size(); i++)
    if (nodes[i]->id() == id) {
      found = true;
      break;
    }

  if (found) {
    nodes[i]->updateNode(position, isoval, area, minvol, maxvol, grad, nComp);
  } else {
    ConSpecInfoNode *newNode = new ConSpecInfoNode(
        id, position, isoval, area, minvol, maxvol, grad, nComp);
    nodes.push_back(newNode);
    char label[128];
    sprintf(label, "IC-%d", id);
    tabWidget->addTab(newNode, QString(label));
    tabWidget->show();
  }
}

void InfoDialog::remove(int id) {
  int i = 0;
  bool found = false;
  for (; i < nodes.size(); i++)
    if (nodes[i]->id() == id) {
      found = true;
      break;
    }

  if (found) {
    tabWidget->removeTab(i);
    nodes.erase(nodes.begin() + i);
  }
}
} // namespace CVCColorTable
