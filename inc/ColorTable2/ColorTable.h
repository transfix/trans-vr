#ifndef __CVCCOLORTABLE__COLORTABLE_H__
#define __CVCCOLORTABLE__COLORTABLE_H__

#include <set>
#include <vector>
#include <string>
#include <QFrame>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>
#include <boost/shared_array.hpp>

#ifndef COLORTABLE2_DISABLE_CONTOUR_TREE
namespace VolMagick
{
  class Volume;
}
#endif

namespace CVCColorTable
{
  const double MAX_RANGE = 1.0; //0.95;
  const double MIN_RANGE = 0.0; //0.05;

  class XoomedOut;
  class Table;

  class ColorTable : public QFrame
  {
  Q_OBJECT

  public:
    ColorTable( QWidget *parent = 0, 
#if QT_VERSION < 0x040000
                const char *name = 0
#else
                Qt::WFlags flags=0
#endif
                );
    ~ColorTable();

    virtual QSize sizeHint() const;
    virtual QSize minimumSizeHint() const;

    boost::shared_array<unsigned char> getTable(unsigned int size = 256) const;
    unsigned char* getCharTable(unsigned int size = 256) const;

    boost::shared_array<float> getFloatTable(unsigned int size = 256) const;

    bool interactiveUpdates() const;
    bool opacityCubed() const;

    /*
      All values are from [0.0,1.0]
    */
    struct opacity_node
    {
      opacity_node(double pos = 0.0,
		   double val = 0.0)
	: position(pos), value(val) {}

      opacity_node(const opacity_node& node)
	: position(node.position),
	  value(node.value) {}

      opacity_node& operator=(const opacity_node& node)
      {
	position = node.position;
	value = node.value;
	return *this;
      }

      bool operator<(const opacity_node& node) const
      {
	return position < node.position;
      }

      bool operator==(const opacity_node& node) const
      {
	return position == node.position &&
	  value == node.value;
      }

      bool operator!=(const opacity_node& node) const
      {
	return !(*this == node);
      }

      double position;
      double value;
    };

    typedef std::set<opacity_node> opacity_nodes;

    //this struct is overkill, but we might need more member variables later so...
    struct isocontour_node
    {
      isocontour_node(double pos = 0.0, int id = 0)
	: position(pos),_id(id) {}

      isocontour_node(const isocontour_node& node)
	: position(node.position),_id(node._id) {}
      
      isocontour_node& operator=(const isocontour_node& node)
      {
	position = node.position;
	return *this;
      }
      
      bool operator<(const isocontour_node& node) const
      {
	return position < node.position;
      }

      bool operator==(const isocontour_node& node) const
      {
	return position == node.position;
      }

      bool operator!=(const isocontour_node& node) const
      {
	return !(*this == node);
      }
 
      void id( const int val ) { _id = val; }
      int id( void ) { return _id; }

      double position;
      int _id;
    };

    typedef std::set<isocontour_node> isocontour_nodes;

    struct color_node
    {
      color_node(double pos = 0.0,
		 double red = 0.0,
		 double green = 0.0,
		 double blue = 0.0)
	: position(pos), r(red), g(green), b(blue) {}

      color_node(const color_node& node)
	: position(node.position),
	  r(node.r),
	  g(node.g),
	  b(node.b) {}

      color_node& operator=(const color_node& node)
      {
	position = node.position;
	r = node.r;
	g = node.g;
	b = node.b;
	return *this;
      }

      bool operator<(const color_node& node) const
      {
	return position < node.position;
      }

      bool operator==(const color_node& node) const
      {
	return position == node.position &&
	  r == node.r &&
	  g == node.g &&
	  b == node.b;
      }

      bool operator!=(const color_node& node) const
      {
	return !(*this == node);
      }

      double position;
      double r;
      double g;
      double b;
    };

    //a set of color nodes should have at least 2 nodes at positions 0.0 and 1.0
    typedef std::set<color_node> color_nodes;

    struct color_table_info
    {
      color_table_info(const opacity_nodes& o = opacity_nodes(),
		       const isocontour_nodes& i = isocontour_nodes(),
		       const color_nodes& c = color_nodes())
	: _opacityNodes(o), _isocontourNodes(i),
	  _colorNodes(c)
      {
      }

      color_table_info(const color_table_info& cti)
	: _opacityNodes(cti._opacityNodes), _isocontourNodes(cti._isocontourNodes),
	  _colorNodes(cti._colorNodes)
      {
      }

      color_table_info& operator=(const color_table_info& cti)
      {
	opacityNodes() = cti.opacityNodes();
	isocontourNodes() = cti.isocontourNodes();
	colorNodes() = cti.colorNodes();
	return *this;
      }

      bool operator==(const color_table_info& cti) const
      {
	return 
	  opacityNodes() == cti.opacityNodes() &&
	  isocontourNodes() == cti.isocontourNodes() &&
	  colorNodes() == cti.colorNodes();
      }

      bool operator!=(const color_table_info& cti) const
      {
	return !(*this == cti);
      }

      const opacity_nodes& opacityNodes() const { return _opacityNodes; }
      opacity_nodes& opacityNodes() { return _opacityNodes; }
      const isocontour_nodes& isocontourNodes() const { return _isocontourNodes; }
      isocontour_nodes& isocontourNodes() { return _isocontourNodes; }
      const color_nodes& colorNodes() const { return _colorNodes; }
      color_nodes& colorNodes() { return _colorNodes; }

      //use normalize to ensure that opacity and color nodes exist at MIN_RANGE and MAX_RANGE
      color_table_info& normalize();

      private:
       opacity_nodes _opacityNodes;
       isocontour_nodes _isocontourNodes;
       color_nodes _colorNodes;
    };

    const color_table_info& info() const { return _cti; }
    color_table_info& info() { return _cti; }

    //Call update to update the color table rendering if the _cti changed
    void update();

    void reset();
    void read(const std::string& filename);
    void write(const std::string& filename);

#if !defined(COLORTABLE2_DISABLE_CONTOUR_TREE) || !defined(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
    void setContourVolume(const VolMagick::Volume& vol);
#endif

    static color_table_info default_transfer_function();
    static color_table_info read_transfer_function(const std::string& filename);
    static std::string transfer_function_filename;

    static void write_transfer_function(const std::string& filename,
					const color_table_info& cti);
    static void write_full_color_table(const std::string& filename,
                                        const color_table_info& cti);

  public slots:
    void interactiveUpdates(bool b);
    void opacityCubed(bool b);

  protected:
    color_table_info _cti;
    bool _opacityCubed;

#if QT_VERSION < 0x040000
    XoomedOut *_xoomedOut;
#else
    Table *_xoomedOut; //Xoomed out bar is now the same object for qt4!
#endif
    Table *_xoomedIn;

  signals:
    void changed();
  };

  typedef ColorTable::opacity_node opacity_node;
  typedef ColorTable::opacity_nodes opacity_nodes;
  typedef ColorTable::isocontour_node isocontour_node;
  typedef ColorTable::isocontour_nodes isocontour_nodes;
  typedef ColorTable::color_node color_node;
  typedef ColorTable::color_nodes color_nodes;
}

#endif
