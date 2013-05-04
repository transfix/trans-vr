#ifndef __READER_SER_H__
#define __READER_SER_H__

#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include "ContourFilter/container.h"
#include "ContourFilter/controls.h"
#include "ContourFilter/histogram.h"
#include <ContourTiler/Contour.h>
#include <ContourTiler/Contour2.h>
#include <ContourTiler/print_utils.h>
#include <ContourTiler/xml/Ser_reader.h>

// namespace SeriesFileReader
// {
//   extern std::list<SurfRecon::ContourPtr> readSeries(const std::string& filename, const VolMagick::VolumeFileInfo& volinfo);
//   //double thickness = 1.0, double scale = 1.0);
// };

CONTOURTILER_BEGIN_NAMESPACE

template <typename InputIterator, typename OutputIterator, typename ExceptionIterator>
void read_contours_ser(const std::string& filename, OutputIterator contours, ExceptionIterator exceptions,
		       int min_section, int max_section, int smoothing_factor, 
		       InputIterator object_begin, InputIterator object_end)
{
  typedef typename Contour::Handle Contour_handle;
  typedef typename Contour::Polygon Polygon;
  typedef typename Contour::Info_type Contour_info;
//   typedef SurfRecon::ContourPtr SRContour;
//   typedef std::list<SRContour> SRContours;
//   typedef SurfRecon::CurvePtr SRCurve;
//   typedef std::vector<SRCurve> SRCurves;
// //   typedef SurfRecon::Point_2 Point_2;
//   typedef typename Polygon::Traits::Point_2 Point_2;
//   typedef SurfRecon::Point Point;
//   typedef SurfRecon::PointPtr SRPoint;
//   typedef SurfRecon::PointPtrList SRPoints;

  using namespace std;

  static log4cplus::Logger logger = log4cplus::Logger::getInstance("SeriesFileReader");

  boost::filesystem::path path(filename);
//   cout << path << endl;
//   cout << path.file_string() << endl;
//   cout << path.directory_string() << endl;
//   cout <<  << endl;
//   cout << path.filename() << endl;
//   cout << path.stem() << endl;
//   cout << path.extension() << endl;
//   exit(0);

  ContourFilter::Container container;
  LOG4CPLUS_INFO(logger, "Reading " << filename);
  ContourFilter::Controls& cs (ContourFilter::Controls::instance()); 
  cs.setInputDir((path.parent_path().string() + "/").c_str());
  cs.setPrefix(path.filename().c_str());
  cs.setMinSection(min_section);
  cs.setMaxSection(max_section);
  cs.addIgnoredContour("cube");
  cs.addIgnoredContour("cyl");
  cs.addIgnoredContour("grid");
  cs.addIgnoredContour("domain1");
  for (InputIterator it = object_begin; it != object_end; ++it)
    cs.addUsedContour(*it);
  cs.setCurvatureEnergyGain(1E2);
  cs.setCurvatureEnergyExponent(1);
  if (smoothing_factor > 0)
    cs.setCurvatureEnergyExponent(smoothing_factor);
  cs.setProximityEnergyGain(3E0);
  cs.setProximityEnergyExponent(1);
  cs.setPtPerContourThreshold(0);
  cs.setDeviationThreshold(0.005);
  cs.setSectionThickness(1);
  if (smoothing_factor < 0)
    cs.setReturnRawContourPoints(1);

  container.getContours();
  // compare sequential points in contours and remove duplicate points
  container.removeDuplicates();
  LOG4CPLUS_INFO(logger, "Reading input files...");
  LOG4CPLUS_TRACE(logger, "Number of input files = " << container.getNumFiles());
  LOG4CPLUS_TRACE(logger, "Number of objects     = " << container.getNumObjects());
  LOG4CPLUS_TRACE(logger, "Number of contours    = " << container.getNumContours());
  LOG4CPLUS_TRACE(logger, "Number of raw points  = " << container.getNumRawPoints());

  LOG4CPLUS_DEBUG(logger, "Filtering contours...");
  ContourFilter::Histogram h; // sample deviations from linearly-interpolated raw points
  ContourFilter::Histogram si; // linear distance between spline samples AFTER annealing
  ContourFilter::Histogram si_before; // linear distance between spline samples BEFORE annealing
  container.processContour (h,si,si_before);

//   c.clearOutputScripts ();
//   LOG4CPLUS_DEBUG(logger, "Writing output contours");
//   c.writeOutputContours ();
//   h.printStatistics ("deviation");
//   si_before.printStatistics ("sample interval before filtering");
//   si.printStatistics ("sample interval after filtering");

  typedef ContourFilter::c_o_iterator o_iter;
  typedef ContourFilter::c_c_l_iterator c_iter;
  for (o_iter o_i = container.firstObject(); o_i != container.onePastLastObject(); ++o_i)
  {
    const ContourFilter::Object& object = *o_i;
//     LOG4CPLUS_DEBUG(logger, object.getName());

    for (c_iter c_i = object.firstContour(); c_i != object.onePastLastContour(); ++c_i)
    {

      const ContourFilter::Contour& contour = *c_i;
      LOG4CPLUS_TRACE(logger, contour.getNumSamplePoints());

//       double z = contour.getSection()*cs.getSectionThickness();
//       LOG4CPLUS_DEBUG(logger, "Section thickness = " << z);

      list<Point_3> points;
      contour.getSampledPoints<Point_3>(back_inserter(points));
      Polygon polygon(points.begin(), points.end());
      LOG4CPLUS_TRACE(logger, "Read contour: " << pp(polygon));
      if (polygon.size() >= 3 && polygon.is_simple()) {
	Contour_info info(contour.getSection(), object.getName(), object.getName());
	if (!polygon.is_counterclockwise_oriented()) {
	  polygon.reverse_orientation();
	}
	Contour_handle ch = Contour::create(polygon, info);
	*contours++ = ch;
	LOG4CPLUS_TRACE(logger, "Added contour: " << pp(ch->polygon()));
      }
      else {
	Contour_info info(contour.getSection(), object.getName(), object.getName());
	stringstream ss;
	ss << "Skipping non-simple contour: slice = " << contour.getSection()
	   << " component = " << object.getName();
	Contour_exception e(ss.str(), info);
	// LOG4CPLUS_WARN(logger, "Skipping non-simple contour: slice = " << contour.getSection()
	// 	       << " component = " << object.getName());
	*exceptions++ = e;
	LOG4CPLUS_TRACE(logger, "Exception: " << e.what());
      }
    }
  }

// //   VolMagick::VolumeFileInfo volinfo;
//   SRContours srcontours = SeriesFileReader::readSeries(filename, volinfo);
// //   cout << srcontours.size() << endl; 
//   for (SRContours::iterator it = srcontours.begin(); it != srcontours.end(); ++it)
//   {
//     SRCurves srcurves = (*it)->curves();
//     for (SRCurves::iterator cit = srcurves.begin(); cit != srcurves.end(); ++cit)
//     {
//       std::vector<Point_2> points;
//       SRCurve srcurve = *cit;
//       SRPoints srpoints = srcurve->get<2>();
//       for (SRPoints::iterator pit = srpoints.begin(); pit != srpoints.end(); ++pit)
//       {
// 	// This assumes that we're using Point_25.  Take out the last parameter
// 	// if we use Point_2.
// // 	points.push_back(Point_2(TO_NT((*pit)->x()), TO_NT((*pit)->y()), TO_NT((*cit)->get<0>())));
// 	points.push_back(Point_2((*pit)->x(), (*pit)->y(), (*cit)->get<0>()));
//       }
//       // SER files repeat the last point.  We don't want that!
//       Polygon polygon(points.begin(), points.end() - 1);
//       try
//       {
// 	Contour_handle contour = Contour::create(polygon, Contour_info(*cit));
// 	*contours = contour;
// 	++contours;
//       }
//       catch (const Contour_exception& e)
//       {
// 	*exceptions = e;
// 	++exceptions;
//       }
//     }
//   }
  LOG4CPLUS_TRACE(logger, "Done reading");
}

typedef boost::unordered_map<string, list<Polygon_2> > Name2Polys;
typedef boost::unordered_map<int, Name2Polys> PolyMap;

// template <typename InputIterator, typename ExceptionIterator>
// void read_polygons_ser2(const std::string& filename, PolyMap& pmap, ExceptionIterator exceptions,
// 		       int min_section, int max_section, int smoothing_factor, 
// 		       InputIterator object_begin, InputIterator object_end)
// {
//   typedef Polygon_2 Polygon;
//   typedef Contour2::Info_type Contour_info;

//   using namespace std;

//   static log4cplus::Logger logger = log4cplus::Logger::getInstance("SeriesFileReader");

//   boost::filesystem::path path(filename);

//   ContourFilter::Container container;
//   LOG4CPLUS_INFO(logger, "Reading " << filename);
//   ContourFilter::Controls& cs (ContourFilter::Controls::instance()); 
//   cs.setInputDir((path.parent_path().string() + "/").c_str());
//   cs.setPrefix(path.filename().c_str());
//   cs.setMinSection(min_section);
//   cs.setMaxSection(max_section);
//   cs.addIgnoredContour("cube");
//   cs.addIgnoredContour("cyl");
//   cs.addIgnoredContour("grid");
//   cs.addIgnoredContour("domain1");
//   for (InputIterator it = object_begin; it != object_end; ++it)
//     cs.addUsedContour(*it);
//   cs.setCurvatureEnergyGain(1E2);
//   cs.setCurvatureEnergyExponent(1);
//   if (smoothing_factor > 0)
//     cs.setCurvatureEnergyExponent(smoothing_factor);
//   cs.setProximityEnergyGain(3E0);
//   cs.setProximityEnergyExponent(1);
//   cs.setPtPerContourThreshold(0);
//   cs.setDeviationThreshold(0.005);
//   cs.setSectionThickness(1);
//   if (smoothing_factor < 0)
//     cs.setReturnRawContourPoints(1);

//   container.getContours();
//   // compare sequential points in contours and remove duplicate points
//   container.removeDuplicates();
//   LOG4CPLUS_DEBUG(logger, "Reading input files...");
//   LOG4CPLUS_DEBUG(logger, "Number of input files = " << container.getNumFiles());
//   LOG4CPLUS_DEBUG(logger, "Number of objects     = " << container.getNumObjects());
//   LOG4CPLUS_DEBUG(logger, "Number of contours    = " << container.getNumContours());
//   LOG4CPLUS_DEBUG(logger, "Number of raw points  = " << container.getNumRawPoints());

//   LOG4CPLUS_INFO(logger, "Filtering contours...");
//   ContourFilter::Histogram h; // sample deviations from linearly-interpolated raw points
//   ContourFilter::Histogram si; // linear distance between spline samples AFTER annealing
//   ContourFilter::Histogram si_before; // linear distance between spline samples BEFORE annealing
//   container.processContour (h,si,si_before);
// //   c.clearOutputScripts ();
// //   LOG4CPLUS_DEBUG(logger, "Writing output contours");
// //   c.writeOutputContours ();
// //   h.printStatistics ("deviation");
// //   si_before.printStatistics ("sample interval before filtering");
// //   si.printStatistics ("sample interval after filtering");

//   typedef ContourFilter::c_o_iterator o_iter;
//   typedef ContourFilter::c_c_l_iterator c_iter;
//   for (o_iter o_i = container.firstObject(); o_i != container.onePastLastObject(); ++o_i)
//   {
//     const ContourFilter::Object& object = *o_i;
// //     LOG4CPLUS_DEBUG(logger, object.getName());

//     for (c_iter c_i = object.firstContour(); c_i != object.onePastLastContour(); ++c_i)
//     {

//       const ContourFilter::Contour& contour = *c_i;
//       LOG4CPLUS_TRACE(logger, contour.getNumSamplePoints());

//       list<Point_3> points;
//       contour.getSampledPoints<Point_3>(back_inserter(points));
//       Polygon polygon(points.begin(), points.end());
//       LOG4CPLUS_TRACE(logger, "Read contour: " << pp(polygon));
//       if (polygon.size() >= 3 && polygon.is_simple()) {
// 	// Contour_info info(contour.getSection(), object.getName(), object.getName());
// 	if (!polygon.is_counterclockwise_oriented()) {
// 	  polygon.reverse_orientation();
// 	}
// 	pmap[contour.getSection()][object.getName()].push_back(polygon);
// 	// Contour_handle ch = Contour::create(polygon, info);
// 	// *contours++ = ch;
// 	// LOG4CPLUS_TRACE(logger, "Added contour: " << pp(ch->polygon()));
//       }
//       else {
// 	Contour_info info(contour.getSection(), object.getName(), object.getName());
// 	stringstream ss;
// 	ss << "Skipping non-simple contour: slice = " << contour.getSection()
// 	   << " component = " << object.getName();
// 	Contour_exception e(ss.str(), info);
// 	*exceptions++ = e;
// 	LOG4CPLUS_TRACE(logger, "Exception: " << e.what());
//       }
//     }
//   }
//   LOG4CPLUS_TRACE(logger, "Done reading");
// }

template <typename OutputIterator, typename ExceptionIterator>
void read_contours_ser(const std::string& filename, int smoothing_factor, OutputIterator contours, ExceptionIterator exceptions)
{
  list<string> objects;
  read_contours_ser(filename, contours, exceptions, 0, 1000000, smoothing_factor, objects.begin(), objects.end());
}

template <typename InputIterator>
bool is_keeper(int z, string name, 
	       int min_section, int max_section,
	       InputIterator include_begin, InputIterator include_end,
	       InputIterator exclude_begin, InputIterator exclude_end)
{
  if (z >= min_section && z <= max_section) {
    string component = name;
    bool include = (include_begin == include_end);
    for (InputIterator s_it = include_begin; s_it != include_end; ++s_it) {
      boost::regex r(*s_it);
      if (regex_match(component, r)) include = true;
    }
    if (include) {
      for (InputIterator s_it = exclude_begin; s_it != exclude_end; ++s_it) {
	boost::regex r(*s_it);
	if (regex_match(component, r)) include = false;
      }
    }
    return include;
  }
  return false;
}

// slice_end - one past the last slice to include
template <typename InputIterator, typename OutputIterator, typename ExceptionIterator>
void read_contours_ser(const std::string& filename, OutputIterator contours, ExceptionIterator exceptions,
		       int min_section, int max_section, int smoothing_factor, 
		       InputIterator include_begin, InputIterator include_end,
		       InputIterator exclude_begin, InputIterator exclude_end)
{
  static log4cplus::Logger logger = log4cplus::Logger::getInstance("read_contours_ser");

  list<Contour_handle> all_contours;
  list<Contour_exception> all_exceptions;
  vector<string> empty_components;
  read_contours_ser(filename, back_inserter(all_contours), back_inserter(all_exceptions),
		    min_section, max_section, smoothing_factor,
		    empty_components.begin(), empty_components.end());

  set<string> error_components;
  for (list<Contour_exception>::const_iterator it = all_exceptions.begin(); it != all_exceptions.end(); ++it) {
    string name = it->object_name();
    error_components.insert(name);
    if (is_keeper(min_section, name, min_section, max_section, 
		  include_begin, include_end, exclude_begin, exclude_end)) {
      LOG4CPLUS_WARN(logger, "Skipping component " << name);
    }
  }

  for (list<Contour_handle>::iterator it = all_contours.begin(); it != all_contours.end(); ++it)
  {
    const Contour_handle contour = *it;
    const int z = (int) contour->polygon()[0].z();
    string name = contour->info().name();
    if (error_components.find(name) == error_components.end() &&
	is_keeper(z, name, min_section, max_section, include_begin, include_end, exclude_begin, exclude_end))
    {
      *contours++ = contour;
    }
  }

  for (list<Contour_exception>::iterator it = all_exceptions.begin(); it != all_exceptions.end(); ++it)
  {
    const Contour_exception& e = *it;
    string component = e.object_name();
    bool include = (include_begin == include_end);
    for (InputIterator s_it = include_begin; s_it != include_end; ++s_it) {
      boost::regex r(*s_it);
      if (regex_match(component, r)) include = true;
    }
    if (include) {
      for (InputIterator s_it = exclude_begin; s_it != exclude_end; ++s_it) {
	boost::regex r(*s_it);
	if (regex_match(component, r)) include = false;
      }
    }
    if (include) {
      *exceptions++ = e;
    }
  }
}

// slice_end - one past the last slice to include
template <typename InputIterator, typename OutputIterator, typename ExceptionIterator>
void read_contours_ser_new(const std::string& filename, OutputIterator contours, ExceptionIterator exceptions,
                           int min_section, int max_section, int smoothing_factor, 
                           InputIterator include_begin, InputIterator include_end,
                           InputIterator exclude_begin, InputIterator exclude_end)
{
  static log4cplus::Logger logger = log4cplus::Logger::getInstance("read_contours_ser");

  // Read the series files
  vector<Ser_section> sections;
  vector<string> warnings;
  try {
    read_ser(filename, min_section, max_section, back_inserter(sections), back_inserter(warnings), true);
    BOOST_FOREACH(const string& warning, warnings) {
      LOG4CPLUS_DEBUG(logger, warning);
    }
  }
  catch (Ser_exception& e) {
    LOG4CPLUS_ERROR(logger, e.what());
  }

  // Convert the Ser data structures to our own Contour structures
  BOOST_FOREACH (const Ser_section& section, sections) {
    const int z = section.index();
    BOOST_FOREACH (const Ser_contour& contour, section) {
      const string name = contour.name();
      if (is_keeper(z, name, min_section, max_section, 
                    include_begin, include_end, 
                    exclude_begin, exclude_end))
      {
        Polygon_2 P;
        BOOST_FOREACH (const Ser_point& point, contour) {
          P.push_back(Point_2(point.x(), point.y(), z));
        }
        const Contour::Info info(z, name, name);
        const Contour_handle contour = Contour::create(P, info);
        *contours++ = contour;
      }
    }
  }
}

CONTOURTILER_END_NAMESPACE

#endif
