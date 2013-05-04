#include <Qt3Support> 

#include <VolumeRover/MainWindow.h>

#include <qapplication.h>

int main(int argc, char** argv)
{
  
  QApplication app(argc, argv);
  MainWindow mainwindow;
  app.setMainWidget(&mainwindow);
  mainwindow.show();
  return app.exec();
}
