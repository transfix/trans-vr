#include <ContourTiler/Tiler_workspace.h>
#include <ContourTiler/Tiler_options.h>
#include <ContourTiler/print_utils.h>
#include <ContourTiler/tiler_output.h>

CONTOURTILER_BEGIN_NAMESPACE

bool Callback::untiled_region(const Untiled_region& r)
{
  static int count = 0;
  LOG4CPLUS_TRACE(logger, "untiled region " << count);
  ++count;

  if (_options.output_untiled_regions_gnuplot())
  {
    std::stringstream ss;
    ss << _options.output_dir() << "/" << _options.base_name() << "_untiled_" << setfill('0') << setw(3) << count << ".g";
    ofstream out(ss.str().c_str());
    gnuplot_print_polygon(out, r);
    out.close();
  }
}

bool Callback::tile_added(const Tiler_workspace& workspace)
{
  static int count = 0;
  ++count;

  if (_options.output_intermediate_raw())
  {
    std::stringstream ss;
    ss << _options.output_dir() << "/" << _options.base_name() << "_tiles_" << setfill('0') << setw(3) << count << ".rawc";
    ofstream out(ss.str().c_str());
    raw_print_tiles(out, workspace.tiles_begin(), workspace.tiles_end());
    out.close();
    LOG4CPLUS_TRACE(logger, "wrote " << ss.str());
  }
  if (_options.output_intermediate_gnuplot())
  {
    std::stringstream ss;
    ss << _options.output_dir() << "/" << _options.base_name() << "_tiles_" << setfill('0') << setw(3) << count << ".g";
    ofstream out(ss.str().c_str());
    gnuplot_print_tiles(out, workspace.tiles_begin(), workspace.tiles_end());
    out.close();
    LOG4CPLUS_TRACE(logger, "wrote " << ss.str());
  }
  return true;
}

void output_pre_tile(const Tiler_workspace& w, const Tiler_options& options)
{
  string out_dir(options.output_dir());
  string base(options.base_name());
  // Output pre-tile results
  if (options.output_otv())
  {
    ofstream out((out_dir + "/" + base + "_otv.g").c_str());
    gnuplot_print_otvs(out, w);
    out.close();
  }

  if (options.output_otv_pairs())
  {
    ofstream out((out_dir + "/" + base + "_otv_pairs.g").c_str());
    gnuplot_print_otv_pairs(out, w);
    out.close();
  }

  if (options.output_vertex_labels())
  {
    ofstream out((out_dir + "/" + base + "_verts.g").c_str());
    gnuplot_print_vertices(out, w.vertices);
    out.close();
  }

}

void output_phase1(const Tiler_workspace& w, const Tiler_options& options)
{
  string out_dir(options.output_dir());
  string base(options.base_name());
  if (options.output_phases_raw())
  {
    ofstream out((out_dir + "/" + base + "_tiles_phase1.rawc").c_str());
    raw_print_tiles(out, w.tiles_begin(), w.tiles_end(), options.z_scale());
    out.close();
  }

  if (options.output_phases_gnuplot())
  {
    ofstream out((out_dir + "/" + base + "_tiles_phase1.g").c_str());
    gnuplot_print_tiles(out, w.tiles_begin(), w.tiles_end());
    out.close();
  }

}

void output_phase2(const Tiler_workspace& w, const Tiler_options& options)
{
  string out_dir(options.output_dir());
  string base(options.base_name());
  if (options.output_phases_raw())
  {
    ofstream out((out_dir + "/" + base + "_tiles.rawc").c_str());
    raw_print_tiles(out, w.tiles_begin(), w.tiles_end(), options.z_scale());
    out.close();
  }

  if (options.output_phases_gnuplot())
  {
    ofstream out((out_dir + "/" + base + "_tiles.g").c_str());
    gnuplot_print_tiles(out, w.tiles_begin(), w.tiles_end());
    out.close();
  }

}

CONTOURTILER_END_NAMESPACE

