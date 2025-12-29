#ifndef __CVCCOLORTABLE__INFODIALOG_H__
#define __CVCCOLORTABLE__INFODIALOG_H__

#include <qframe.h>
#include <qlineedit.h>
#include <qtabwidget.h>
#include <string>
#include <vector>

namespace CVCColorTable {
enum CONTOURSTATUS { SHOW, NEW, REMOVE, MOVE };

class ConSpecInfoNode : public QWidget {
  Q_OBJECT
public:
  ConSpecInfoNode(int id, double position, float isoval, float area,
                  float minvol, float maxvol, float grad, int nComp);
  ConSpecInfoNode(const ConSpecInfoNode &copy);
  virtual ~ConSpecInfoNode();

  void updateNode(double position, float isoval, float area, float minvol,
                  float maxvol, float grad, int nComp);

  int id() { return m_ID; }

private:
  int m_ID;
  double m_Position;
  float m_IsoVal;
  float m_Area;
  float m_MinVol;
  float m_MaxVol;
  float m_Grad;
  int m_nComp;

  QLineEdit *positionLineEdit;
  QLineEdit *isovalLineEdit;
  QLineEdit *areaLineEdit;
  QLineEdit *minvolLineEdit;
  QLineEdit *maxvolLineEdit;
  QLineEdit *gradLineEdit;
  QLineEdit *compLineEdit;
};

class InfoDialog : public QFrame {
  Q_OBJECT

public:
  InfoDialog(QWidget *parent = 0,
#if QT_VERSION < 0x040000
             const char *name = 0
#else
             Qt::WindowFlags flags = {}
#endif
  );
  ~InfoDialog();

  void updateInfo(int id, double position, float isoval, float area,
                  float minvol, float maxvol, float grad, int nComp);
  void remove(int id);

private:
  QTabWidget *tabWidget;

  std::vector<ConSpecInfoNode *> nodes;
  double m_RangeMin, m_RangeMax;
};
} // namespace CVCColorTable

#endif
