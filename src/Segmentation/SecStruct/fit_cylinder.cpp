
#include <Segmentation/SecStruct/datastruct.h>

#define SUBDIV 100
#define RADIUS 1.5

std::vector<std::vector<Point>>
fit_cylinder(const std::vector<Cylinder> &cyls) {
  std::vector<std::vector<Point>> cyl_Vs;

  // write the cylinders as mesh.
  for (int i = 0; i < (int)cyls.size(); i++) {
    Point v0 = cyls[i].first[0];
    Vector a = cyls[i].second;

    std::vector<Point> cyl_V;
    std::vector<Vector> circle;
    for (int j = 0; j < SUBDIV; j++) {
      double theta = 2 * M_PI / SUBDIV * j;
      double x = RADIUS * cos(theta), y = RADIUS * sin(theta), z = 0;
      circle.push_back(Vector(x, y, z));
    }

    // rotate and translate the circle to be aligned with a
    // and centered at v0.
    Vector t = v0 - CGAL::ORIGIN;
    // rotation.
    Vector zaxis(0, 0, 1);
    Vector raxis = CGAL::cross_product(zaxis, a);
    raxis = (1. / sqrt(CGAL::to_double(raxis * raxis))) * raxis;
    double c = CGAL::to_double(zaxis * a) / sqrt(CGAL::to_double(a * a));
    double s = sqrt(1 - c * c);

    double rx = CGAL::to_double(raxis.x()), ry = CGAL::to_double(raxis.y()),
           rz = CGAL::to_double(raxis.z());
    double tx = CGAL::to_double(t.x()), ty = CGAL::to_double(t.y()),
           tz = CGAL::to_double(t.z());
    double trnsf_mat_1[4][4] = {
        {rx * rx * (1 - c) + c, rx * ry * (1 - c) - rz * s,
         rx * rz * (1 - c) + ry * s, tx},
        {ry * rx * (1 - c) + rz * s, ry * ry * (1 - c) + c,
         ry * rz * (1 - c) - rx * s, ty},
        {rx * rz * (1 - c) - ry * s, ry * rz * (1 - c) + rx * s,
         rz * rz * (1 - c) + c, tz},
        {0, 0, 0, 1}};

    for (int j = 0; j < (int)circle.size(); j++) {
      double px = CGAL::to_double(circle[j].x());
      double py = CGAL::to_double(circle[j].y());
      double pz = CGAL::to_double(circle[j].z());

      double _px = px * trnsf_mat_1[0][0] + py * trnsf_mat_1[0][1] +
                   pz * trnsf_mat_1[0][2] + trnsf_mat_1[0][3];
      double _py = px * trnsf_mat_1[1][0] + py * trnsf_mat_1[1][1] +
                   pz * trnsf_mat_1[1][2] + trnsf_mat_1[1][3];
      double _pz = px * trnsf_mat_1[2][0] + py * trnsf_mat_1[2][1] +
                   pz * trnsf_mat_1[2][2] + trnsf_mat_1[2][3];
      cyl_V.push_back(Point(_px, _py, _pz));
    }

    trnsf_mat_1[0][3] += CGAL::to_double(a.x());
    trnsf_mat_1[1][3] += CGAL::to_double(a.y());
    trnsf_mat_1[2][3] += CGAL::to_double(a.z());

    for (int j = 0; j < (int)circle.size(); j++) {
      double px = CGAL::to_double(circle[j].x());
      double py = CGAL::to_double(circle[j].y());
      double pz = CGAL::to_double(circle[j].z());

      double _px = px * trnsf_mat_1[0][0] + py * trnsf_mat_1[0][1] +
                   pz * trnsf_mat_1[0][2] + trnsf_mat_1[0][3];
      double _py = px * trnsf_mat_1[1][0] + py * trnsf_mat_1[1][1] +
                   pz * trnsf_mat_1[1][2] + trnsf_mat_1[1][3];
      double _pz = px * trnsf_mat_1[2][0] + py * trnsf_mat_1[2][1] +
                   pz * trnsf_mat_1[2][2] + trnsf_mat_1[2][3];
      cyl_V.push_back(Point(_px, _py, _pz));
    }

    cyl_Vs.push_back(cyl_V);

    // write mesh raw format.
#if 0
      char file_prefix[100] = "cyl";
      char op_fname[100];
      char extn[10];
      extn[0] = '_'; extn[1] = '0' + i/10; extn[2] = '0' + i%10; extn[3] = '\0';
      strcpy(op_fname, file_prefix);
      strcat(op_fname, extn);
      strcat(op_fname, ".cyl");
      cerr << "file : " << op_fname << endl;

      ofstream fout;
      fout.open(op_fname);
      if(! fout)
      {
         cerr << "Error in opening output file " << endl;
         exit(1);
      }

      fout << (int)cyl_V.size() << " " << (int)cyl_V.size() << endl;
      for(int j = 0; j < (int)cyl_V.size(); j ++)
         fout << cyl_V[j] << endl;
      int N = (int)cyl_V.size()/2;
      for(int j = 0; j < N; j ++)
      {
         int v1 = j, v2 = j+N, v3 = (j+1)%N; 
         fout << v1 << " " << v2 << " " << v3 << endl;
         v1 = v3, v3 = v2+1;
         v3 = (v3 == 2*N)? N : v3;
         fout << v1 << " " << v2 << " " << v3 << endl;
      }
      fout.close();
#endif
  }

  return cyl_Vs;
}
