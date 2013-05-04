#ifndef COLORTABLEPLUGIN_H
#define COLORTABLEPLUGIN_H
#include <qwidgetplugin.h>

///\class ColorTablePlugin ColorTablePlugin.h
///\author Anthony Thane
///\brief This class is used to support the ColorTable widget in Qt Designer.
class QT_WIDGET_PLUGIN_EXPORT ColorTablePlugin : public QWidgetPlugin
{
public:
    ColorTablePlugin();
    virtual ~ColorTablePlugin();

    QStringList keys() const;
    QWidget* create( const QString &classname, QWidget* parent = 0, const char* name = 0 );
    QString group( const QString& ) const;
    QIconSet iconSet( const QString& ) const;
    QString includeFile( const QString& ) const;
    QString toolTip( const QString& ) const;
    QString whatsThis( const QString& ) const;
    bool isContainer( const QString& ) const;
};

#endif
