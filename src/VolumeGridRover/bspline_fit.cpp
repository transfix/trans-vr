/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/*
  Justin Kinney's b-spline fitting for contours
*/

#include <VolumeGridRover/SurfRecon.h>
#include <boost/tuple/tuple.hpp>
#include <list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace bspline_fit {
class void_list {
public:
  void_list *previous;
  void_list *next;
  void *data;
};

class Parameters {
public:
  double plot_rad_int; // radius of curvature sampling interval for plotting
                       // sampling function
  double dmin;         // minimum spline sampling distance
  double dmax;         // maximum spline sampling distance
  double max_rad; // calculated radius of curvature of spline will saturate at
                  // this value
  bool diag;      // set true to print diagnostic files
  int num; // num is the # samples of the splines between contour points
  // sequential triplets of sampled points are used to
  // compute the radius of curvature , e.g. num/2 between contour points
  double T;    // sample period (= time to traverse dmax)
  double amax; // max radial acceleration
};

class Object {
public:
  char name[128];
  int min_section, max_section;
  void_list *contours;
  Object(char *str, int sec);
  ~Object(void);
};

Object::Object(char *str, int sec) {
  strcpy(name, str);
  contours = NULL;
  min_section = sec;
  max_section = sec;
}

Object::~Object(void) {
  void_list *p, *q;
  p = contours;
  while (p != NULL) {
    q = p->next;
    delete p;
    p = q;
  }
}

class Point {
public:
  double x, y, z;
  Point(char *str, int section, double thickness, double *transform);
  Point(double xval, double yval, double zval);
};

Point::Point(double xval, double yval, double zval) {
  x = xval;
  y = yval;
  z = zval;
}

Point::Point(char *str, int section, double thickness, double *t) {
  char val[80];
  char *eptr;
  int i;
  double xval, yval;

  // set z coordinate
  z = section * thickness;

  // get past 'points'
  while (strchr(" points=\"\t", *str) != NULL) {
    str++;
  }

  // grab x coordinate
  i = 0;
  while (strchr("0123456789+-eE.", *str) != NULL) {
    val[i++] = *str++;
  }
  val[i] = 0;
  xval = strtod(val, &eptr);
  if (val == eptr) {
    x = y = z = 0;
    printf("Error in reading x coordinate\n");
    printf("str =%s\n", str);
    return;
  }

  // grab y coordinate
  while (strchr(" \t,", *str) != NULL) {
    str++;
  }
  i = 0;
  while (strchr("0123456789+-eE.", *str) != NULL) {
    val[i++] = *str++;
  }
  val[i] = 0;
  yval = strtod(val, &eptr);
  if (val == eptr) {
    x = y = z = 0;
    printf("Error in reading y coordinate\n");
    return;
  }
  //	if(section==64){
  //	printf("xcoef = %g %g %g %g %g %g\n",t[0],t[1],t[2],t[3],t[4],t[5]);
  //	printf("ycoef = %g %g %g %g %g
  //%g\n\n",t[6],t[7],t[8],t[9],t[10],t[11]);
  //	}
  x = t[0] + t[1] * xval + t[2] * yval + t[3] * xval * yval +
      t[4] * xval * xval + t[5] * yval * yval;
  y = t[6] + t[7] * xval + t[8] * yval + t[9] * xval * yval +
      t[10] * xval * xval + t[11] * yval * yval;
}

class Contour {
public:
  char name[128];
  int section, num_raw_points, num_interp_points;
  void_list *raw_points, *rawend;
  void_list *interp_points, *interpend;
  double *deviations;
  Contour(char *str, int sec);
  ~Contour(void);
  void removeDuplicates(void);
  void clearSpline(void);
  void linearlyInterp(double, double);
  void addPreviousRaw(void);
  void addPreviousInterp(void);
};

Contour::~Contour(void) {
  void_list *p, *q;
  p = raw_points;
  while (p != NULL) {
    q = p->next;
    delete (Point *)p->data;
    delete p;
    p = q;
  }
  p = interp_points;
  while (p != NULL) {
    q = p->next;
    delete (Point *)p->data;
    delete p;
    p = q;
  }
  delete[] deviations;
}

Contour::Contour(char *str, int sec) {
  char val[80];
  int i;
  // grab name
  i = 0;
  while (strchr("\"", *str) == NULL) {
    val[i++] = *str++;
  }
  val[i] = 0;

  char *ptr = val;
  while (ptr != NULL) {
    //      ptr=strchr(val,'#');
    ptr = strpbrk(val, "#()");
    if (ptr != NULL) {
      *ptr = '_';
    }
  }

  strcpy(name, val);
  section = sec;
  raw_points = NULL;
  rawend = NULL;
  interp_points = NULL;
}

void Contour::clearSpline(void) {
  void_list *p, *q;
  p = interp_points;
  while (p != NULL) {
    delete (Point *)p->data;
    q = p->next;
    delete p;
    p = q;
  }
  interp_points = NULL;
  interpend = NULL;
}

void Contour::addPreviousRaw(void) {
  void_list *q, *prev;
  prev = NULL;
  rawend = NULL;
  // for each point
  for (q = raw_points; q != NULL; q = q->next) {
    q->previous = prev;
    prev = q;
    rawend = q;
  }
}

void Contour::addPreviousInterp(void) {
  void_list *q, *prev;
  prev = NULL;
  interpend = NULL;
  // for each point
  for (q = interp_points; q != NULL; q = q->next) {
    q->previous = prev;
    prev = q;
    interpend = q;
  }
}

void_list *removeLink(void_list *L) {
  void_list *q;
  // and remove face from candidate face list
  if (L->previous != NULL) {
    if (L->next != NULL) {
      // if both previous and next exist
      (L->previous)->next = L->next;
      (L->next)->previous = L->previous;
    } else {
      // if previous exists and next does not
      (L->previous)->next = NULL;
    }
  } else {
    if (L->next != NULL) {
      // if previous does not exist and next does
      (L->next)->previous = NULL;
    } // else { // if neither previous nor next exists }
  }
  // update pointer
  q = L->next;
  delete L;
  return q;
}

void_list *deletePoint(void_list *q, void_list *p, void_list *&ptr) {
  Point *pt1, *pt2;
  pt1 = (Point *)q->data;
  pt2 = (Point *)p->data;
  // if points are identical
  if ((pt1->x == pt2->x) && (pt1->y == pt2->y)) {
    // delete point
    delete pt1;
    // adjust list pointer
    if (q == ptr) {
      ptr = q->next;
    }
    // remove current point from list
    q = removeLink(q);
  }
  return p;
}

void Contour::removeDuplicates(void) {
  void_list *q, *ptr;
  Point *pt1, *pt2;
  ptr = raw_points;
  q = raw_points;
  while (q->next != NULL) {
    q = deletePoint(q, q->next, ptr);
  }
  // adjust pointer
  raw_points = ptr;

  // compare first and last link in list
  q = deletePoint(raw_points, rawend, ptr);
  // adjust pointer
  raw_points = ptr;
}

void_list *interpPoints(void_list *q, void_list *p, void_list *&ptr,
                        double maxdev, double scale, int flag) {
  void_list *pp;
  double dist, distx, disty, x, y, count;
  int num;
  Point *v, *pt1, *pt2;
  pt1 = (Point *)q->data;
  pt2 = (Point *)p->data;
  // compute distance between points
  distx = pt2->x - pt1->x;
  disty = pt2->y - pt1->y;
  dist = sqrt(distx * distx + disty * disty);
  num = (int)(dist / (maxdev / scale) / 3);
  if (num) {
    // linearly interpolate num evenly spaced points
    count = num + 1;
    ptr = q;
    while (num) {
      // insert point
      x = pt1->x + (count - (double)num) / count * distx;
      y = pt1->y + (count - (double)num) / count * disty;
      v = new Point(x, y, pt1->z);
      pp = new void_list();
      pp->next = p;
      pp->previous = ptr;
      pp->data = (void *)v;
      ptr->next = pp;
      p->previous = pp;
      // decrement num
      num--;
      ptr = pp;
    }
    if (flag) {
      ptr->next = NULL;
    }
  }
  return p;
}

void Contour::linearlyInterp(double maxdev, double scale) {
  void_list *q, *ptr = NULL;
  q = raw_points;
  while (q->next != NULL) {
    q = interpPoints(q, q->next, ptr, maxdev, scale, 0);
  }
  // compare first and last link in list
  q = interpPoints(rawend, raw_points, ptr, maxdev, scale, 1);
  // adjust pointer
  rawend = ptr;
}

class Histogram {
public:
  double min, max, mean, stddev, sum;
  int count[16], num;
  Histogram(void);
  void update(void_list *, int);
};

Histogram::Histogram(void) {
  int i;
  for (i = 0; i < 16; i++) {
    count[i] = 0;
  }
  min = 1e30;
  max = 0;
  mean = stddev = sum = 0;
  num = 0;
}

void Histogram::update(void_list *q, int count) {
  ///// update deviation distance statistics /////
  int i;
  double d;
  Contour *c = (Contour *)q->data;
  for (i = 0; i < count; i++) {
    d = c->deviations[i];
    // update min and max deviation distance
    if (d < min) {
      min = d;
    }
    if (d > max) {
      max = d;
    }
    num++;
    sum += d;
  }
}

class SplinePoint {
public:
  double t, x, y, r, intfac;
};

class Weights {
public:
  double *bx, *by;
};

int distinguishable(double a, double b, double eps) {
  double c;
  c = a - b;
  if (c < 0)
    c = -c;
  if (a < 0)
    a = -a;
  if (a < 1)
    a = 1;
  if (b < 0)
    b = -b;
  if (b < a)
    eps *= a;
  else
    eps *= b;
  return (c > eps);
}

bool degenerateObject(Object *o) {
  void_list *q;
  Contour *c;
  // for each contour
  for (q = o->contours; q != NULL; q = q->next) {
    c = (Contour *)q->data;
    if (c->num_interp_points < 3) {
      return true;
    }
  }
  return false;
}

Weights *loadWeightArrays(void_list *p, Weights *w, int limit) {
  ///// load arrays of weights /////
  // END OF RAW POINTS ARRAY WAS PASSED AS P
  // AND LIST IS TRAVERSED IN 'PREVIOUS' DIRECTION
  int i = 0, j, k;
  void_list *qq;
  // spline 0
  for (qq = p; qq != NULL; qq = qq->previous) {
    if (qq->previous == NULL) {
      w->bx[i] = ((Point *)qq->data)->x;
      w->by[i] = ((Point *)qq->data)->y;
      i++;
      break;
    }
  }
  qq = p;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = qq->previous;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = qq->previous;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  // splines 1 through m-2
  for (j = 1; j < (limit - 2);
       j++) { // note '-2' because m = contour_array[i]-1
    k = 0;
    for (qq = p; qq != NULL; qq = qq->previous) {
      if ((k == (j - 1)) || (k == j) || (k == (j + 1)) || (k == (j + 2))) {
        w->bx[i] = ((Point *)qq->data)->x;
        w->by[i] = ((Point *)qq->data)->y;
        i++;
      }
      k++;
      if (qq->previous == NULL)
        break;
    }
  }
  // spline m-1
  k = 0;
  for (qq = p; qq != NULL; qq = qq->previous) {
    if (k == (limit - 3)) {
      break;
    }
    k++;
  }
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = qq->previous;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = qq->previous;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = p;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  // spline m
  k = 0;
  for (qq = p; qq != NULL; qq = qq->previous) {
    if (k == (limit - 2)) {
      break;
    }
    k++;
  }
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = qq->previous;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = p;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;
  i++;
  qq = qq->previous;
  w->bx[i] = ((Point *)qq->data)->x;
  w->by[i] = ((Point *)qq->data)->y;

  return w;
}

SplinePoint *computeSplines(SplinePoint *sp, Weights *w, double dt, int limit,
                            Parameters *pa) {
  int i, j, index;
  double inc, inc2, inc3, xdot, xdotdot, ydot, ydotdot, den, num_part,
      maxr = pa->max_rad;
  // for each point in contour
  for (i = 0; i < limit; i++) {
    // for each parameter increment
    for (j = 0; j < pa->num; j++) {
      index = i * pa->num + j;
      inc = j * dt;
      inc2 = inc * inc;
      inc3 = inc2 * inc;
      // store time
      sp[index].t = i + inc;
      sp[index].x =
          1.0 / 6.0 *
          (w->bx[4 * i + 0] * (-inc3 + 3.0 * inc2 - 3.0 * inc + 1.0) +
           w->bx[4 * i + 1] * (3.0 * inc3 - 6.0 * inc2 + 4.0) +
           w->bx[4 * i + 2] * (-3.0 * inc3 + 3.0 * inc2 + 3.0 * inc + 1.0) +
           w->bx[4 * i + 3] * inc3);
      sp[index].y =
          1.0 / 6.0 *
          (w->by[4 * i + 0] * (-inc3 + 3.0 * inc2 - 3.0 * inc + 1.0) +
           w->by[4 * i + 1] * (3.0 * inc3 - 6.0 * inc2 + 4.0) +
           w->by[4 * i + 2] * (-3.0 * inc3 + 3.0 * inc2 + 3.0 * inc + 1.0) +
           w->by[4 * i + 3] * inc3);
      xdot = 1.0 / 6.0 *
             (w->bx[4 * i + 0] * (-3.0 * inc2 + 6.0 * inc - 3.0) +
              w->bx[4 * i + 1] * (9.0 * inc2 - 12.0 * inc) +
              w->bx[4 * i + 2] * (-9.0 * inc2 + 6.0 * inc + 3.0) +
              w->bx[4 * i + 3] * 3.0 * inc2);
      ydot = 1.0 / 6.0 *
             (w->by[4 * i + 0] * (-3.0 * inc2 + 6.0 * inc - 3.0) +
              w->by[4 * i + 1] * (9.0 * inc2 - 12.0 * inc) +
              w->by[4 * i + 2] * (-9.0 * inc2 + 6.0 * inc + 3.0) +
              w->by[4 * i + 3] * 3.0 * inc2);
      xdotdot = 1.0 / 6.0 *
                (w->bx[4 * i + 0] * (-6.0 * inc + 6.0) +
                 w->bx[4 * i + 1] * (18.0 * inc - 12.0) +
                 w->bx[4 * i + 2] * (-18.0 * inc + 6.0) +
                 w->bx[4 * i + 3] * 6.0 * inc);
      ydotdot = 1.0 / 6.0 *
                (w->by[4 * i + 0] * (-6.0 * inc + 6.0) +
                 w->by[4 * i + 1] * (18.0 * inc - 12.0) +
                 w->by[4 * i + 2] * (-18.0 * inc + 6.0) +
                 w->by[4 * i + 3] * 6.0 * inc);
      den = fabs(xdot * ydotdot - ydot * xdotdot);
      num_part = sqrt(xdot * xdot + ydot * ydot);
      sp[index].intfac = num_part;
      if (den) {
        sp[index].r = num_part * num_part * num_part / den;
      } else {
        sp[index].r = maxr;
      }
      if (sp[index].r > maxr) {
        sp[index].r = maxr;
      }
    }
  }
  return sp;
}

void sampleSplines(void_list *q, SplinePoint *sp, double dt, int limit,
                   double thickness, char *outdir, int tag, Parameters *pa) {
  ///// sample splines /////
  Contour *c;
  c = (Contour *)q->data;
  void_list *pp;
  Point *v;
  int i, j, k = 0, count = 0, num_sampled = 0, myswitch,
            kvec[limit * pa->num / 2];
  // dmin = minimum spline sampling distance
  // dmax = maximum spline sampling distance
  // decrease tau to increase steepness
  // inflection point of sampling function = tau*rI
  // decrease rI to sample finer
  //	double dmin=.001,dmax=.050,tau=2,rI=-6,inc;
  double dl, l_accum = 0.0, l = 0.0, r_mean, ds, ds_mean = 0.0,
             intfac_array[3], r_array[2], xval, yval, inc;
  double vmax = pa->dmax / pa->T, v_mean, amax = pa->amax, delt,
         t_accum = 0.0, T = pa->T;
  double rvec[limit * pa->num / 2], vvec[limit * pa->num / 2],
      tvec[limit * pa->num / 2];
  for (i = 0; i < limit; i++) {
    for (j = 0; j < pa->num; j++) {
      inc = (double)(i + j * dt);
      myswitch = j % 2;
      if (inc && !myswitch) {
        intfac_array[0] = sp[(int)((inc - 2.0 * dt) / dt)].intfac;
        r_array[0] = sp[(int)((inc - 2.0 * dt) / dt)].r;
        intfac_array[1] = sp[(int)((inc - dt) / dt)].intfac;
        intfac_array[2] = sp[(int)(inc / dt)].intfac;
        r_array[1] = sp[(int)(inc / dt)].r;
        xval = sp[(int)(inc / dt)].x;
        yval = sp[(int)(inc / dt)].y;
        // increment along spline length
        dl = 2 * dt / 3.0 *
             (intfac_array[0] + 4.0 * intfac_array[1] + intfac_array[2]);
        // mean radius of curvature
        r_mean = (r_array[0] + r_array[1]) / 2.0;
        // mean velocity
        v_mean = sqrt(amax * r_mean);
        if (v_mean > vmax) {
          v_mean = vmax;
        }
        // time increment
        delt = dl / v_mean;
        // accumulated time
        t_accum += delt;
        // sample data
        if (t_accum >= T) {
          // add interpolated point to contour
          pp = new void_list();
          pp->next = c->interp_points;
          v = new Point(xval, yval, c->section * thickness);
          pp->data = (void *)v;
          c->interp_points = pp;
          // clear variables
          t_accum = 0.0;
          num_sampled++;
        }
        if (pa->diag) {
          rvec[k] = r_mean;
          vvec[k] = v_mean;
          tvec[k] = delt;
          kvec[k] = k;
          k++;
        }
      }
    }
  }

  // store number of sampled points
  c->num_interp_points = num_sampled;

  // add previous data
  c->addPreviousInterp();

  // diagnostics
  if (pa->diag) {
    char filename[256], line[2048];
    FILE *F;
    ///// print radius of curvature /////
    sprintf(filename, "%s%s_%i_%i.rad", outdir, c->name, c->section, tag);
    F = fopen(filename, "w");
    if (!F) {
      printf("Couldn't open output file %s\n", filename);
      return;
    }
    // for each point
    for (i = 0; i < limit * pa->num / 2 - 1; i++) {
      sprintf(line, "%i %.15g\n", kvec[i], rvec[i]);
      fputs(line, F);
    }
    fclose(F);
    ///// print velocity /////
    sprintf(filename, "%s%s_%i_%i.vel", outdir, c->name, c->section, tag);
    F = fopen(filename, "w");
    if (!F) {
      printf("Couldn't open output file %s\n", filename);
      return;
    }
    // for each point
    for (i = 0; i < limit * pa->num / 2 - 1; i++) {
      sprintf(line, "%i %.15g\n", kvec[i], vvec[i]);
      fputs(line, F);
    }
    fclose(F);
    ///// print incremental time /////
    sprintf(filename, "%s%s_%i_%i.tim", outdir, c->name, c->section, tag);
    F = fopen(filename, "w");
    if (!F) {
      printf("Couldn't open output file %s\n", filename);
      return;
    }
    // for each point
    for (i = 0; i < limit * pa->num / 2 - 1; i++) {
      sprintf(line, "%i %.15g\n", kvec[i], tvec[i]);
      fputs(line, F);
    }
    fclose(F);
  }
}

void computeDeviation(void_list *q, SplinePoint *sp, int count, int num) {
  ///// compute deviation distance between raw points and splines /////
  Point *P;
  Contour *c;
  void_list *p, *pp;
  c = (Contour *)q->data;
  c->deviations = new double[count];
  double diffx, diffy, dist0, distneg, distneg0, distpos, distpos0;
  int j = 0, i = 0, m, n;
  for (p = c->rawend; p != NULL; p = p->previous) {
    P = (Point *)p->data;
    diffx = P->x - sp[j].x;
    diffy = P->y - sp[j].y;
    dist0 = sqrt(diffx * diffx + diffy * diffy);
    // check negative dir
    m = j;
    distneg = dist0;
    do {
      distneg0 = distneg;
      if (!m) {
        m = num * count - 1;
      } else {
        m--;
      }
      diffx = P->x - sp[m].x;
      diffy = P->y - sp[m].y;
      distneg = sqrt(diffx * diffx + diffy * diffy);
    } while (distneg < distneg0);
    // check positive dir
    n = j;
    distpos = dist0;
    do {
      distpos0 = distpos;
      if (n == num * count) {
        n = 0;
      } else {
        n++;
      }
      diffx = P->x - sp[n].x;
      diffy = P->y - sp[n].y;
      distpos = sqrt(diffx * diffx + diffy * diffy);
    } while (distpos < distpos0);

    if (dist0 < distneg && dist0 < distpos) {
      c->deviations[i] = dist0;
    } else if (distneg0 < dist0 && distneg0 < distpos0) {
      c->deviations[i] = distneg0;
    } else if (distpos0 < dist0 && distpos0 < distneg0) {
      c->deviations[i] = distpos0;
    } else {
      printf("\n\nweird error!\n");
      printf("Contour %s, section %d, #raw points %d, num %d\n", c->name,
             c->section, count, num);
      printf("rawx %.15g, rawy %.15g\n", P->x, P->y);
      printf("j %i, dist0 %.15g, dist0x %.15g, dist0y %.15g\n", j, dist0,
             sp[j].x, sp[j].y);
      printf("m %i, distneg0 %.15g, distneg %.15g,", m, distneg0, distneg);
      printf(" distnegx %.15g, distnegy %.15g\n", sp[m].x, sp[m].y);
      printf("n %i, distpos0 %.15g, distpos %.15g,", n, distpos0, distpos);
      printf(" distposx %.15g, distposy %.15g\n", sp[n].x, sp[n].y);
    }
    j += num;
    i++;
  }
}

Weights *checkDeviation(void_list *q, void_list *p, Weights *w, int limit,
                        double maxdev, double scale) {
  ///// check deviation at each point in contour /////
  // END OF RAW POINTS ARRAY WAS PASSED AS P
  // AND LIST IS TRAVERSED IN 'PREVIOUS' DIRECTION
  void_list *pp;
  int i;
  bool flag = false;
  // for each deviation
  for (i = 0; i < limit; i++) {
    // if deviation exceeds threshold
    if (((Contour *)q->data)->deviations[i] * scale > maxdev) {
      flag = true;
      // edit weight array
      if (!i) {
        w->bx[4 * (limit - 1) + 3] = w->bx[4 * (limit - 1) + 2];
        w->by[4 * (limit - 1) + 3] = w->by[4 * (limit - 1) + 2];
        w->bx[4 * i + 0] = w->bx[4 * i + 1];
        w->by[4 * i + 0] = w->by[4 * i + 1];
      } else {
        w->bx[4 * (i - 1) + 3] = w->bx[4 * (i - 1) + 2];
        w->by[4 * (i - 1) + 3] = w->by[4 * (i - 1) + 2];
        w->bx[4 * i + 0] = w->bx[4 * i + 1];
        w->by[4 * i + 0] = w->by[4 * i + 1];
      }
    }
  }
  if (flag) {
    return w;
  } else {
    return NULL;
  }
}

void printDiagnostics(void_list *q, SplinePoint *sp, int num, char *outdir,
                      int tag) {
  Contour *c;
  c = (Contour *)q->data;
  void_list *qq;
  int j;
  char filename[256], line[2048];
  FILE *F;
  ///// print raw points /////
  sprintf(filename, "%s%s_%i_%i.raw", outdir, c->name, c->section, tag);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    return;
  }
  // for each point
  for (qq = c->rawend; qq != NULL; qq = qq->previous) {
    sprintf(line, "%.15g %.15g\n", ((Point *)qq->data)->x,
            ((Point *)qq->data)->y);
    fputs(line, F);
  }
  fclose(F);
  ///// print spline points /////
  sprintf(filename, "%s%s_%i_%i.spline", outdir, c->name, c->section, tag);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    return;
  }
  // for each spline point
  for (j = 0; j < num; j++) {
    sprintf(line, "%.15g %.15g\n", sp[j].x, sp[j].y);
    fputs(line, F);
  }
  fclose(F);
  ///// print interpolated points /////
  sprintf(filename, "%s%s_%i_%i.interp", outdir, c->name, c->section, tag);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    return;
  }
  // for each point
  for (qq = c->interpend; qq != NULL; qq = qq->previous) {
    sprintf(line, "%.15g %.15g\n", ((Point *)qq->data)->x,
            ((Point *)qq->data)->y);
    fputs(line, F);
  }
  fclose(F);
  ///// print grace script /////
  sprintf(filename, "%sbfile_%s_%i_%i", outdir, c->name, c->section, tag);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    return;
  }
  sprintf(line, "#Obligatory descriptive comment\n");
  fputs(line, F);
  sprintf(line, "READ xy \"%s_%i_%i.raw\"\n", c->name, c->section, tag);
  fputs(line, F);
  sprintf(line, "READ xy \"%s_%i_%i.spline\"\n", c->name, c->section, tag);
  fputs(line, F);
  sprintf(line, "READ xy \"%s_%i_%i.interp\"\n", c->name, c->section, tag);
  fputs(line, F);
  sprintf(line, "legend on\ns0 legend ");
  sprintf(line,
          "%s\"Raw\"\ns1 legend \"Spline\"\ns2 legend \"Interpolated\"\n",
          line);
  fputs(line, F);
  sprintf(line, "s0 symbol 1\ns0 symbol color 2\ns0 symbol fill color 2\n");
  sprintf(line, "%ss0 line type 0\ns0 line color 2\ns0 errorbar color 2\n",
          line);
  fputs(line, F);
  sprintf(line, "s1 symbol color 1\ns1 symbol fill color 1\n");
  sprintf(line, "%ss1 line color 1\ns1 errorbar color 1\n", line);
  fputs(line, F);
  sprintf(line, "s2 symbol 2\ns2 symbol color 4\ns2 symbol fill color 4\n");
  sprintf(line, "%ss2 line type 0\ns2 line color 4\ns2 errorbar color 4\n",
          line);
  fputs(line, F);
  fclose(F);
  ///// print shell script /////
  sprintf(filename, "%s%s_%i_%i.csh", outdir, c->name, c->section, tag);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    return;
  }
  sprintf(line, "#!/bin/csh\n");
  fputs(line, F);
  sprintf(line, "xmgrace -noask -nosafe -batch bfile_%s_%i_%i\n", c->name,
          c->section, tag);
  fputs(line, F);
  fclose(F);
}

void initTransform(double *t) {
  t[0] = 0.0;
  t[1] = 1.0;
  t[2] = 0.0;
  t[3] = 0.0;
  t[4] = 0.0;
  t[5] = 0.0;
  t[6] = 0.0;
  t[7] = 0.0;
  t[8] = 1.0;
  t[9] = 0.0;
  t[10] = 0.0;
  t[11] = 0.0;
}

void setTransform(double *t, char *p) {
  char val[80], *eptr;
  int i;
  // grab '1'
  while (strchr(" \t,", *p) != NULL) {
    p++;
  }
  i = 0;
  while (strchr("0123456789+-eE.", *p) != NULL) {
    val[i++] = *p++;
  }
  val[i] = 0;
  t[0] = strtod(val, &eptr);
  if (val == eptr) {
    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 0.0;
    printf("Error in reading '1' coefficient\n");
    printf("str =%s\n", p);
    return;
  }
  // grab 'x'
  while (strchr(" \t,", *p) != NULL) {
    p++;
  }
  i = 0;
  while (strchr("0123456789+-eE.", *p) != NULL) {
    val[i++] = *p++;
  }
  val[i] = 0;
  t[1] = strtod(val, &eptr);
  if (val == eptr) {
    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 0.0;
    printf("Error in reading 'x' coefficient\n");
    printf("str =%s\n", p);
    return;
  }
  // grab 'y'
  while (strchr(" \t,", *p) != NULL) {
    p++;
  }
  i = 0;
  while (strchr("0123456789+-eE.", *p) != NULL) {
    val[i++] = *p++;
  }
  val[i] = 0;
  t[2] = strtod(val, &eptr);
  if (val == eptr) {
    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 0.0;
    printf("Error in reading 'y' coefficient\n");
    printf("str =%s\n", p);
    return;
  }
  // grab 'xy'
  while (strchr(" \t,", *p) != NULL) {
    p++;
  }
  i = 0;
  while (strchr("0123456789+-eE.", *p) != NULL) {
    val[i++] = *p++;
  }
  val[i] = 0;
  t[3] = strtod(val, &eptr);
  if (val == eptr) {
    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 0.0;
    printf("Error in reading 'xy' coefficient\n");
    printf("str =%s\n", p);
    return;
  }
  // grab 'x*x'
  while (strchr(" \t,", *p) != NULL) {
    p++;
  }
  i = 0;
  while (strchr("0123456789+-eE.", *p) != NULL) {
    val[i++] = *p++;
  }
  val[i] = 0;
  t[4] = strtod(val, &eptr);
  if (val == eptr) {
    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 0.0;
    printf("Error in reading 'x*x' coefficient\n");
    printf("str =%s\n", p);
    return;
  }
  // grab 'y*y'
  while (strchr(" \t,", *p) != NULL) {
    p++;
  }
  i = 0;
  while (strchr("0123456789+-eE.", *p) != NULL) {
    val[i++] = *p++;
  }
  val[i] = 0;
  t[5] = strtod(val, &eptr);
  if (val == eptr) {
    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 0.0;
    printf("Error in reading 'y*y' coefficient\n");
    printf("str =%s\n", p);
    return;
  }
}

void_list *getContours(int argc, char *argv[], double thickness) {
  int i, min_section, max_section, raw_count;
  char *indir, infile[128], *str, line[2048], *name, filename[256], *eptr,
      *temp, *coef;
  FILE *F;
  void_list *c, *q, *ch;
  ch = NULL;
  Contour *cont;
  Point *v;
  bool contour_flag;
  indir = argv[1];
  min_section = (int)strtod(argv[3], &eptr);
  max_section = (int)strtod(argv[4], &eptr);
  double transform[12];
  // adjust indir
  strcpy(filename, indir);
  temp = strrchr(indir, '/');
  if (!temp) {
    strcat(filename, "/");
  } else if (*++temp) {
    strcat(filename, "/");
  }
  strcat(filename, argv[2]);
  // for each reconstruct input file
  printf("\n");
  for (i = min_section; i < max_section + 1; i++) {
    // open file
    sprintf(infile, "%s.%d", filename, i);
    F = fopen(infile, "r");
    if (!F) {
      printf("Couldn't open input file %s\n", infile);
      return NULL;
    } else {
      printf("Input file found: %s\n", infile);
    }
    contour_flag = false;
    // initialize Transform
    initTransform(transform);
    // for every line in file
    for (str = fgets(line, 2048, F); str != NULL;
         str = fgets(line, 2048, F)) {
      if (strstr(str, "Transform dim") != NULL) {
        // get next line
        str = fgets(line, 2048, F);
        if (str == NULL) {
          printf("Nothing after Transform Dim.");
          printf(" Reconstruct file may be corrupted: %s.\n", infile);
          return NULL;
        }
        // get xcoeff
        if (strstr(str, "xcoef=") != NULL) {
          coef = strstr(str, "xcoef=");
          coef += 8; // advance pointer to start of coefficients
          // 8, because 'xcoeff="' is full string
          setTransform(transform, coef);
        } else {
          printf("No xcoef. Reconstruct file may be corrupted.: %s.\n",
                 infile);
          return NULL;
        }
        // get next line
        str = fgets(line, 2048, F);
        if (str == NULL) {
          printf("Nothing after xcoef.");
          printf(" Reconstruct file may be corrupted: %s.\n", infile);
          return NULL;
        }
        // get ycoeff
        if (strstr(str, "ycoef=") != NULL) {
          coef = strstr(str, "ycoef=");
          coef += 8; // advance pointer to start of coefficients
          // 8, because 'ycoeff="' is full string
          setTransform(transform + 6, coef);
        } else {
          printf("No ycoef. Reconstruct file may be corrupted: %s.\n",
                 infile);
          return NULL;
        }
      }
      // if start of contour
      else if (strstr(str, "<Contour") != NULL) {
        // find name
        name = strstr(str, "name=");
        name += 6; // advance pointer to start of name
        // create new contour
        cont = new Contour(name, i);
        c = new void_list();
        c->next = ch;
        c->data = (void *)cont;
        ch = c;
        printf("Contour found: %s\n", ((Contour *)ch->data)->name);
        // set contour flag
        contour_flag = true;
        raw_count = 0;
      } else if (strstr(str, "/>") != NULL) {
        contour_flag = false;
        ((Contour *)ch->data)->num_raw_points = raw_count;
      } else if (contour_flag) {
        // add point to contour
        v = new Point(str, i, thickness, &transform[0]);
        q = new void_list();
        q->next = ((Contour *)ch->data)->raw_points;
        q->data = (void *)v;
        ((Contour *)ch->data)->raw_points = q;
        raw_count++;
      }
    }
    fclose(F);
  }
  printf("\n");
  return ch;
}

int countContours(void_list *ch) {
  void_list *q;
  int i = 0;
  // for each contour
  for (q = ch; q != NULL; q = q->next) {
    i++;
  }
  return i;
}

int *createArray(int *c_array, void_list *ch, int num_contours) {
  int i = 0, num_points;
  void_list *q, *p, *qq;
  c_array = new int[num_contours];
  // for each contour
  for (q = ch; q != NULL; q = q->next) {
    p = ((Contour *)q->data)->raw_points;
    // for each point
    num_points = 0;
    for (qq = p; qq != NULL; qq = qq->next) {
      num_points++;
    }
    c_array[i] = num_points;
    i++;
  }
  return c_array;
}

void fitSplines(void_list *ch, Histogram *h, double t, int *c_array,
                double dev_thr, double scale, char *outdir, Parameters *pa,
                int num_contours) {
  SplinePoint *sp;
  Contour *c;
  Weights *w1, *w2;
  w1 = new Weights;
  void_list *q, *p;
  double dt;
  int m = 0, i = 0;
  // for each contour
  for (q = ch; q != NULL; q = q->next) {
    printf("Splining contour %i of %i\n", i, num_contours - 1);
    c = (Contour *)q->data;
    if (c_array[m] < 3) {
      printf("contour has less than 3 points and ");
      printf("was skipped: contour %s, section %d,", c->name, c->section);
      printf(" num_points %d\n", c_array[m]);
    } else {
      ///// load arrays of weights /////
      p = c->rawend;
      w1->bx = new double[4 * c_array[m]];
      w1->by = new double[4 * c_array[m]];
      w1 = loadWeightArrays(p, w1, c_array[m]);
      // W ARRAY IS CONSTRUCTED WITH FIRST WEIGHT CORRESPONDING TO FIRST RAW
      // POINT IN INPUT FILE.
      ///// compute splines for current contour /////
      dt = 1.0 / (double)pa->num;
      sp = new SplinePoint[c_array[m] * pa->num];
      sp = computeSplines(sp, w1, dt, c_array[m], pa);
      // SP IS CONSTRUCTED WITH SAME ORIENTATION AS W.
      ///// sample splines /////
      sampleSplines(q, sp, dt, c_array[m], t, outdir, i, pa);
      // INTERPOLATED POINTS HAVE SAME ORIENTATION AS RAW POINTS.
      // FIRST OFF THE LINKED LIST WAS LAST ADDED TO LIST.
      ///// compute deviation distance between raw points and splines /////
      computeDeviation(q, sp, c_array[m], pa->num);
      ///// check deviations /////
      if (dev_thr) {
        w2 = checkDeviation(q, p, w1, c_array[m], dev_thr, scale);
        if (w2 != NULL) {
          // recompute splines
          delete[] sp;
          sp = new SplinePoint[c_array[m] * pa->num];
          sp = computeSplines(sp, w2, dt, c_array[m], pa);
          // clear inter_points in contour
          c->clearSpline();
          sampleSplines(q, sp, dt, c_array[m], t, outdir, i, pa);
          computeDeviation(q, sp, c_array[m], pa->num);
        }
      }
      ///// update min and max deviation distances /////
      h->update(q, c_array[m]);
      if (pa->diag) {
        printDiagnostics(q, sp, c_array[m] * pa->num, outdir, i);
      }
      // clean up
      delete[] w1->bx;
      delete[] w1->by;
      delete[] sp;
    }
    m++;
    i++;
  }
  delete w1;
}

void computeHistogram(Histogram *h, void_list *ch, int *c_array) {
  void_list *q;
  Contour *c;
  double foo = 0;
  int i, m = 0;
  h->mean = h->sum / h->num;
  // for each contour
  for (q = ch; q != NULL; q = q->next) {
    c = (Contour *)q->data;
    if (c_array[m] < 3) {
      printf("contour has less than 3 points and was skipped:");
      printf(" contour %s, section %d, num_points %d\n", c->name, c->section,
             c_array[m]);
    } else {
      // for each raw point in contour
      for (i = 0; i < c_array[m]; i++) {
        // compute stddev scratch work
        foo += (c->deviations[i] - h->mean) * (c->deviations[i] - h->mean);
        // bin deviation
        h->count[(int)(c->deviations[i] / (h->max / 14))]++;
      }
    }
    m++;
  }
  h->stddev = sqrt(foo / (h->num - 1));
}

void_list *createObjects(void_list *ch) {
  void_list *q, *qq, *pp, *objectsh, *target;
  objectsh = NULL;
  Contour *c;
  Object *o;
  // for each contour
  for (q = ch; q != NULL; q = q->next) {
    c = (Contour *)q->data;
    // has object been created with same name?
    target = NULL;
    qq = objectsh;
    while (qq != NULL && !target) {
      if (!strcmp(((Object *)qq->data)->name, c->name)) {
        target = qq;
      }
      qq = qq->next;
    }
    // if not
    if (!target) {
      // create a new object and save pointer to new object
      pp = new void_list();
      pp->next = objectsh;
      o = new Object(c->name, c->section);
      pp->data = (void *)o;
      objectsh = pp;
      target = pp;
    }
    // add contour to pointer object
    o = (Object *)target->data;
    pp = new void_list();
    pp->next = o->contours;
    pp->data = (void *)c;
    o->contours = pp;
    // update pointer object min and max
    if (c->section < o->min_section) {
      o->min_section = c->section;
    }
    if (c->section > o->max_section) {
      o->max_section = c->section;
    }
  }
  return objectsh;
}

void cleanup(void_list *objectsh, Histogram *h, void_list *ch, int *c_array) {
  void_list *p, *q;
  // delete objects
  p = objectsh;
  while (p != NULL) {
    q = p->next;
    delete (Object *)p->data;
    delete p;
    p = q;
  }
  // delete histogram
  delete h;
  // delete contours
  p = ch;
  while (p != NULL) {
    q = p->next;
    delete (Contour *)p->data;
    delete p;
    p = q;
  }
  // delete array
  delete[] c_array;
}

void clearPtsFiles(char *outdir, void_list *o) {
  ////////// clear any existing pts files //////////
  void_list *q, *qq;
  char filename[256], line[2048];
  FILE *F;
  Contour *c;

  // for each object
  for (q = o; q != NULL; q = q->next) {
    // for each contour in object
    for (qq = ((Object *)q->data)->contours; qq != NULL; qq = qq->next) {
      c = (Contour *)qq->data;
      // create pts file
      sprintf(filename, "%s%s%d.pts", outdir, c->name, c->section);
      // open pts file
      F = fopen(filename, "w");
      if (!F) {
        printf("Couldn't open output file %s\n", filename);
        exit(0);
      }
      // close pts file
      fclose(F);
    }
  }

  ////////// clear script files //////////
  sprintf(filename, "%smesh.csh", outdir);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(0);
  }
  sprintf(line, "#!/bin/csh\n\n");
  fputs(line, F);
  fclose(F);
  sprintf(filename, "%sconvert.csh", outdir);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(0);
  }
  sprintf(line, "#!/bin/csh\n\n");
  fputs(line, F);
  fclose(F);
}

void printConfigFile(char *outdir, Object *o, int capping_flag) {
  ///// print config file /////
  char filename[256], line[2048];
  FILE *F;
  sprintf(filename, "%s%s.config", outdir, o->name);
  // open file
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(1);
  }
  // print file contents
  sprintf(line, "PREFIX: %s\nSUFFIX: .pts\nOUTPUT_FILE_NAME: %s\n", o->name,
          o->name);
  fputs(line, F);
  if (capping_flag) {
    sprintf(line, "SLICE_RANGE: %i %i\n", o->min_section - 1,
            o->max_section + 1);
    sprintf(line, "%sMERGE_DISTANCE_SQUARE: 1E-24\n", line);
  } else {
    sprintf(line, "SLICE_RANGE: %i %i\n", o->min_section, o->max_section);
    sprintf(line, "%sMERGE_DISTANCE_SQUARE: 1E-24\n", line);
  }
  fputs(line, F);
  // close pts file
  fclose(F);
}

void appendScriptFile(char *outdir, Object *o) {
  ///// append to script files /////
  char filename[256], line[2048];
  FILE *F;
  sprintf(filename, "%smesh.csh", outdir);
  // open file
  F = fopen(filename, "a");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(1);
  }
  // print file contents
  sprintf(line, "echo ''\ncontour_tiler -f %s.config >&! /dev/null\n",
          o->name);
  sprintf(line, "%secho '%s meshed'\n", line, o->name);
  fputs(line, F);
  // close pts file
  fclose(F);
  sprintf(filename, "%sconvert.csh", outdir);
  // open file
  F = fopen(filename, "a");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(1);
  }
  // print file contents
  sprintf(line, "echo ''\npoly2mesh %s.poly >! %s.mesh\n", o->name, o->name);
  sprintf(line, "%secho '%s converted'\n", line, o->name);
  fputs(line, F);
  // close pts file
  fclose(F);
}

void printPtsFiles(char *outdir, Object *o, double scale) {
  ///// print pts files of interpolated contour points /////
  void_list *qq, *pp;
  char filename[256], line[2048];
  FILE *F;
  Contour *c;
  Point *p;
  // for each contour in object
  for (qq = o->contours; qq != NULL; qq = qq->next) {
    c = (Contour *)qq->data;
    // create pts file
    sprintf(filename, "%s%s%d.pts", outdir, c->name, c->section);
    // open pts file
    F = fopen(filename, "a");
    if (!F) {
      printf("Couldn't open output file %s\n", filename);
      exit(1);
    }
    // print number of contour points
    sprintf(line, "%i\n", c->num_interp_points);
    fputs(line, F);
    // for each interpolated point in contour
    for (pp = c->interp_points; pp != NULL; pp = pp->next) {
      p = (Point *)pp->data;
      // print interpolated contour points
      sprintf(line, "%.15g %.15g %.15g\n", p->x * scale, p->y * scale,
              p->z * scale);
      fputs(line, F);
    }
    // close pts file
    fclose(F);
  }
}

void printCaps(char *outdir, Object *o, double thickness, double scale) {
  ///// print min capping pts file /////
  char filename[256], line[2048];
  FILE *F;
  sprintf(filename, "%s%s%d.pts", outdir, o->name, o->min_section - 1);
  // open file
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(1);
  }
  // print file contents
  sprintf(line, "1\n0.0 0.0 %d\n",
          (int)((o->min_section - 1) * thickness * scale));
  fputs(line, F);
  // close pts file
  fclose(F);
  ///// print max capping pts file /////
  sprintf(filename, "%s%s%d.pts", outdir, o->name, o->max_section + 1);
  // open file
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(1);
  }
  // print file contents
  sprintf(line, "1\n0.0 0.0 %d\n",
          (int)((o->max_section + 1) * thickness * scale));
  fputs(line, F);
  // close pts file
  fclose(F);
}

void createCallingScript(char *outdir, char *script) {
  char filename[256], line[2048];
  FILE *F;
  sprintf(filename, "%s%s", outdir, script);
  F = fopen(filename, "w");
  if (!F) {
    printf("Couldn't open output file %s\n", filename);
    exit(1);
  }
  sprintf(line, "#!/bin/csh\n\n/bin/csh mesh.csh\n/bin/csh convert.csh\n");
  fputs(line, F);
  fclose(F);
}

void printStatistics(Histogram *h, double scale) {
  ////////// print deviation statistics //////////
  printf("\n\nSpline deviation statistics:\n\n");
  printf("  Avg deviation = %g +- %g\n", h->mean * scale, h->stddev * scale);
  printf("  Smallest deviation:  %10.5g   |  Largest deviation:  %10.5g\n\n",
         h->min * scale, h->max * scale);
  printf("  Deviation histogram:\n");
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g - %-6.5g         : %9d\n",
         h->max / 14.0 * 0.0 * scale, h->max / 14.0 * 1.0 * scale,
         h->count[0], h->max / 14.0 * 8.0 * scale,
         h->max / 14.0 * 9.0 * scale, h->count[9]);
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g - %-6.5g         : %9d\n",
         h->max / 14.0 * 1.0 * scale, h->max / 14.0 * 2.0 * scale,
         h->count[1], h->max / 14.0 * 9.0 * scale,
         h->max / 14.0 * 10.0 * scale, h->count[10]);
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g - %-6.5g         : %9d\n",
         h->max / 14.0 * 2.0 * scale, h->max / 14.0 * 3.0 * scale,
         h->count[2], h->max / 14.0 * 10.0 * scale,
         h->max / 14.0 * 11.0 * scale, h->count[11]);
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g - %-6.5g         : %9d\n",
         h->max / 14.0 * 3.0 * scale, h->max / 14.0 * 4.0 * scale,
         h->count[3], h->max / 14.0 * 11.0 * scale,
         h->max / 14.0 * 12.0 * scale, h->count[12]);
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g - %-6.5g         : %9d\n",
         h->max / 14.0 * 4.0 * scale, h->max / 14.0 * 5.0 * scale,
         h->count[4], h->max / 14.0 * 12.0 * scale,
         h->max / 14.0 * 13.0 * scale, h->count[13]);
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g - %-6.5g         : %9d\n",
         h->max / 14.0 * 5.0 * scale, h->max / 14.0 * 6.0 * scale,
         h->count[5], h->max / 14.0 * 13.0 * scale,
         h->max / 14.0 * 14.0 * scale, h->count[14]);
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g - %-6.5g         : %9d\n",
         h->max / 14.0 * 6.0 * scale, h->max / 14.0 * 7.0 * scale,
         h->count[6], h->max / 14.0 * 14.0 * scale,
         h->max / 14.0 * 15.0 * scale, h->count[15]);
  printf("  %6.5g - %-6.5g       : %9d    | %6.5g -                : %9d\n",
         h->max / 14.0 * 7.0 * scale, h->max / 14.0 * 8.0 * scale,
         h->count[7], h->max / 14.0 * 15.0 * scale, 0.0);
  printf("\n");
}

void_list *getContour(const SurfRecon::CurvePtr curve) {
  Contour *cont;
  void_list *q, *c, *ch = NULL;
  Point *v;

  cont =
      new Contour(const_cast<char *>(SurfRecon::getCurveName(*curve).c_str()),
                  SurfRecon::getCurveSlice(*curve));
  c = new void_list();
  c->next = ch;
  c->data = (void *)cont;
  ch = c;

  for (SurfRecon::PointPtrList::iterator k =
           SurfRecon::getCurvePoints(*curve).begin();
       k != SurfRecon::getCurvePoints(*curve).end(); k++) {
    v = new Point((*k)->x(), (*k)->y(), (*k)->z());
    q = new void_list();
    q->next = ((Contour *)ch->data)->raw_points;
    q->data = (void *)v;
    ((Contour *)ch->data)->raw_points = q;
  }

  return ch;
}

void_list *getContour(const SurfRecon::PointPtrList &points) {
  Contour *cont;
  void_list *q, *c, *ch = NULL;
  Point *v;

  cont = new Contour("", 0);
  c = new void_list();
  c->next = ch;
  c->data = (void *)cont;
  ch = c;

  for (SurfRecon::PointPtrList::const_iterator k = points.begin();
       k != points.end(); k++) {
    v = new Point((*k)->x(), (*k)->y(), (*k)->z());
    q = new void_list();
    q->next = ((Contour *)ch->data)->raw_points;
    q->data = (void *)v;
    ((Contour *)ch->data)->raw_points = q;
  }

  return ch;
}

void sampleSplines(void_list *q, SplinePoint *sp, double dt, int limit,
                   char *outdir, int tag, Parameters *pa) {
  ///// sample splines /////
  Contour *c;
  c = (Contour *)q->data;
  void_list *pp;
  Point *v;
  int i, j, k = 0, count = 0, num_sampled = 0, myswitch,
            kvec[limit * pa->num / 2];
  double z = 0.0; // the Z value for this section

  if (c->raw_points->data)
    z = ((Point *)c->raw_points->data)
            ->z; // just use the first (last?) point in the list!
  // dmin = minimum spline sampling distance
  // dmax = maximum spline sampling distance
  // decrease tau to increase steepness
  // inflection point of sampling function = tau*rI
  // decrease rI to sample finer
  //	double dmin=.001,dmax=.050,tau=2,rI=-6,inc;
  double dl, l_accum = 0.0, l = 0.0, r_mean, ds, ds_mean = 0.0,
             intfac_array[3], r_array[2], xval, yval, inc;
  double vmax = pa->dmax / pa->T, v_mean, amax = pa->amax, delt,
         t_accum = 0.0, T = pa->T;
  double rvec[limit * pa->num / 2], vvec[limit * pa->num / 2],
      tvec[limit * pa->num / 2];
  for (i = 0; i < limit; i++) {
    for (j = 0; j < pa->num; j++) {
      inc = (double)(i + j * dt);
      myswitch = j % 2;
      if (inc && !myswitch) {
        intfac_array[0] = sp[(int)((inc - 2.0 * dt) / dt)].intfac;
        r_array[0] = sp[(int)((inc - 2.0 * dt) / dt)].r;
        intfac_array[1] = sp[(int)((inc - dt) / dt)].intfac;
        intfac_array[2] = sp[(int)(inc / dt)].intfac;
        r_array[1] = sp[(int)(inc / dt)].r;
        xval = sp[(int)(inc / dt)].x;
        yval = sp[(int)(inc / dt)].y;
        // increment along spline length
        dl = 2 * dt / 3.0 *
             (intfac_array[0] + 4.0 * intfac_array[1] + intfac_array[2]);
        // mean radius of curvature
        r_mean = (r_array[0] + r_array[1]) / 2.0;
        // mean velocity
        v_mean = sqrt(amax * r_mean);
        if (v_mean > vmax) {
          v_mean = vmax;
        }
        // time increment
        delt = dl / v_mean;
        // accumulated time
        t_accum += delt;
        // sample data
        if (t_accum >= T) {
          // add interpolated point to contour
          pp = new void_list();
          pp->next = c->interp_points;
          v = new Point(xval, yval, z);
          pp->data = (void *)v;
          c->interp_points = pp;
          // clear variables
          t_accum = 0.0;
          num_sampled++;
        }
        if (pa->diag) {
          rvec[k] = r_mean;
          vvec[k] = v_mean;
          tvec[k] = delt;
          kvec[k] = k;
          k++;
        }
      }
    }
  }

  // store number of sampled points
  c->num_interp_points = num_sampled;

  // add previous data
  c->addPreviousInterp();

  // diagnostics
  if (pa->diag) {
    char filename[256], line[2048];
    FILE *F;
    ///// print radius of curvature /////
    sprintf(filename, "%s%s_%i_%i.rad", outdir, c->name, c->section, tag);
    F = fopen(filename, "w");
    if (!F) {
      printf("Couldn't open output file %s\n", filename);
      return;
    }
    // for each point
    for (i = 0; i < limit * pa->num / 2 - 1; i++) {
      sprintf(line, "%i %.15g\n", kvec[i], rvec[i]);
      fputs(line, F);
    }
    fclose(F);
    ///// print velocity /////
    sprintf(filename, "%s%s_%i_%i.vel", outdir, c->name, c->section, tag);
    F = fopen(filename, "w");
    if (!F) {
      printf("Couldn't open output file %s\n", filename);
      return;
    }
    // for each point
    for (i = 0; i < limit * pa->num / 2 - 1; i++) {
      sprintf(line, "%i %.15g\n", kvec[i], vvec[i]);
      fputs(line, F);
    }
    fclose(F);
    ///// print incremental time /////
    sprintf(filename, "%s%s_%i_%i.tim", outdir, c->name, c->section, tag);
    F = fopen(filename, "w");
    if (!F) {
      printf("Couldn't open output file %s\n", filename);
      return;
    }
    // for each point
    for (i = 0; i < limit * pa->num / 2 - 1; i++) {
      sprintf(line, "%i %.15g\n", kvec[i], tvec[i]);
      fputs(line, F);
    }
    fclose(F);
  }
}

// without thickness!
std::vector<SurfRecon::B_Spline>
fitSplines(void_list *ch, Histogram *h, int *c_array, double dev_thr,
           double scale, char *outdir, Parameters *pa, int num_contours) {
  std::vector<SurfRecon::B_Spline> splines;
  SplinePoint *sp;
  Contour *c;
  Weights *w1, *w2;
  w1 = new Weights;
  void_list *q, *p;
  double dt;
  int m = 0, i = 0;
  // for each contour
  for (q = ch; q != NULL; q = q->next) {
    printf("Splining contour %i of %i\n", i, num_contours - 1);
    c = (Contour *)q->data;
    if (c_array[m] < 3) {
      printf("contour has less than 3 points and ");
      printf("was skipped: contour %s, section %d,", c->name, c->section);
      printf(" num_points %d\n", c_array[m]);
    } else {
      /* load arrays of weights */
      p = c->rawend;
      w1->bx = new double[4 * c_array[m]];
      w1->by = new double[4 * c_array[m]];
      w1 = loadWeightArrays(p, w1, c_array[m]);

      // W ARRAY IS CONSTRUCTED WITH FIRST WEIGHT CORRESPONDING TO FIRST RAW
      // POINT IN INPUT FILE.
      /* compute splines for current contour */
      dt = 1.0 / (double)pa->num;
      sp = new SplinePoint[c_array[m] * pa->num];
      sp = computeSplines(sp, w1, dt, c_array[m], pa);
      // SP IS CONSTRUCTED WITH SAME ORIENTATION AS W.
      /* sample splines */
      sampleSplines(q, sp, dt, c_array[m], outdir, i, pa);
      // INTERPOLATED POINTS HAVE SAME ORIENTATION AS RAW POINTS.
      // FIRST OFF THE LINKED LIST WAS LAST ADDED TO LIST.
      /* compute deviation distance between raw points and splines */
      computeDeviation(q, sp, c_array[m], pa->num);
      /* check deviations */
      if (dev_thr) {
        w2 = checkDeviation(q, p, w1, c_array[m], dev_thr, scale);
        if (w2 != NULL) {
          // recompute splines
          delete[] sp;
          sp = new SplinePoint[c_array[m] * pa->num];
          sp = computeSplines(sp, w2, dt, c_array[m], pa);
          // clear inter_points in contour
          c->clearSpline();
          sampleSplines(q, sp, dt, c_array[m], outdir, i, pa);
          computeDeviation(q, sp, c_array[m], pa->num);
        }
      }
      /* update min and max deviation distances */
      h->update(q, c_array[m]);
      if (pa->diag) {
        printDiagnostics(q, sp, c_array[m] * pa->num, outdir, i);
      }

      // collect the weights for spline specification output
      // this is only meant for fitSplines to be called with an object that
      // contains 1 contour!
      for (int idx = 0; idx < c_array[m]; idx++) {
        std::vector<SurfRecon::Point_2> control_points;
        std::vector<double> knots;
        for (int k = 0; k < 4; k++)
          control_points.push_back(
              SurfRecon::Point_2(w1->bx[idx * 4 + k], w1->by[idx * 4 + k]));

        for (int k = 0; k < 4; k++)
          knots.push_back(0.0);
        for (int k = 0; k < 4; k++)
          knots.push_back(1.0);

        splines.push_back(SurfRecon::B_Spline(knots, control_points));
      }

      // clean up
      delete[] w1->bx;
      delete[] w1->by;
      delete[] sp;
    }
    m++;
    i++;
  }
  delete w1;

  return splines;
}
} // namespace bspline_fit

// The following functions added by Joe R to interface with volrover!
namespace SurfRecon {
std::vector<B_Spline> fit_spline(const SurfRecon::CurvePtr curve) {
  return fit_spline(getCurvePoints(*curve));
}

std::vector<B_Spline> fit_spline(const SurfRecon::PointPtrList &points) {
  std::vector<B_Spline> splines;
  double scale, deviation_threshold;
  int num_contours = 0, *contour_array;
  bspline_fit::void_list *q, *ch, *objectsh;
  bspline_fit::Histogram *h = new bspline_fit::Histogram();
  bspline_fit::Parameters *pa = new bspline_fit::Parameters;
  bspline_fit::Contour *c;

  // spline parameters
  pa->plot_rad_int = .1; // radius of curvature sampling interval for plotting
                         // sampling function
  pa->num = 100; // num is the # samples of the splines between contour points
  // sequential triplets of sampled points are used to
  // compute the radius of curvature , e.g. num/2 between contour points
  pa->diag = false;   // set true to print diagnostic files
  pa->max_rad = 1E10; // calculated radius of curvature of spline will
                      // saturate at this value
  pa->dmin = .001;    // dmin = minimum spline sampling distance
  pa->dmax = .050;    // dmax = maximum spline sampling distance
  pa->T = 1.0;        // sample period (= time to traverse dmax)
  pa->amax = 1E-2;    // max radial acceleration
  //	pa->tau=2;			// decrease tau to increase steepness
  //of sampling function 	pa->rI=-6;			// inflection point of
  //sampling function = tau*rI
  // decrease rI to sample finer

  // convert to Justin's data structure
  ch = bspline_fit::getContour(points);

  /* add previous data */
  for (q = ch; q != NULL; q = q->next)
    ((bspline_fit::Contour *)q->data)->addPreviousRaw();

  /* check contours for duplicate points */
  for (q = ch; q != NULL; q = q->next) {
    ((bspline_fit::Contour *)q->data)->removeDuplicates();
  }

  num_contours = bspline_fit::countContours(ch);

  /* linearly interpolate raw contour points */
#if 0
    if (deviation_threshold) {
      for (q=ch;q!=NULL;q=q->next) {
	c=(bspline_fit::Contour*)q->data;
	printf("Contour %s, section %d\n",c->name,c->section);
	c->linearlyInterp(deviation_threshold,scale);
      }
    }
#endif

  /* add previous data */
  for (q = ch; q != NULL; q = q->next) {
    ((bspline_fit::Contour *)q->data)->addPreviousRaw();
  }

  // NOTE: RAW POINTS IN CONTOURS ARE STORED IN REVERSE ORDER
  // FIRST OFF THE LINKED LIST WAS LAST ADDED TO LIST.

  /* create array of number of points in each contour */
  contour_array = bspline_fit::createArray(contour_array, ch, num_contours);

  // fit spline!!
  deviation_threshold = 1.0;
  scale = 1.0;
  splines = bspline_fit::fitSplines(ch, h, contour_array, deviation_threshold,
                                    scale, ".", pa, num_contours);

  /* compute histogram */
  bspline_fit::computeHistogram(h, ch, contour_array);

  /* create objects */
  objectsh = bspline_fit::createObjects(ch);

#if 0
    /* print each object */
    for (q=objectsh;q!=NULL;q=q->next) {
      o=(bspline_fit::Object*)q->data;
      if(!bspline_fit::degenerateObject(o)){
	
      }
    }
#endif

  bspline_fit::cleanup(objectsh, h, ch, contour_array);

  return splines;
}
} // namespace SurfRecon

#if 0
int main(int argc,char *argv[]){

  if (argc != 10)
    {
      printf("\nSyntax: reconstruct2contourtiler input_directory ");
      printf("filename_prefix min_section max_section section_thickess ");
      printf("scale output_directory capping_flag deviation_threshold\n\n");
      printf("Description: Converts reconstruct contour format to contour_tiler input format.\n");
      printf("		All files in input directory are assumed to be ");
      printf("of the form filename_prefix.section#.\n");
      printf("		min_section to max_section is the section range to be converted.\n");
      printf("		Section_thickness should be in same scale as x,y contour points.\n");
      printf("		x,y and z coordinates of sampled splines will be multipled by scale in output.\n");
      printf("		capping_flag=1 to attempt end capping.capping_flag=0 to leave ends open.\n");
      printf("		deviation_threshold is the maximum allowed deviation of the spline from raw\n");
      printf("		contour points in scaled units. Set ");
      printf("deviation_threshold to 0 to disable thresholding.\n\n");
      return 1;
    }
  fprintf(stderr,"\n\nMore sophisticated capping must be ");
  fprintf(stderr,"performed as a postprocess on the pts output files.\n\n");


  ////////// declare variables /////////
  char outdir[128],output_script[32] = "mesh_and_convert.csh",*eptr,*temp;
  void_list *q,*ch,*objectsh;
  Histogram *h;
  Parameters *pa;
  Object *o;	
  Contour *c;
  h = new Histogram();
  pa = new Parameters;
  int i,capping_flag,*contour_array,num_contours;
  double thickness,scale,deviation_threshold;

  // spline parameters
  pa->plot_rad_int=.1;// radius of curvature sampling interval for plotting sampling function
  pa->num=100;		// num is the # samples of the splines between contour points
  // sequential triplets of sampled points are used to
  // compute the radius of curvature , e.g. num/2 between contour points
  pa->diag=false;		// set true to print diagnostic files
  pa->max_rad=1E10;	// calculated radius of curvature of spline will saturate at this value
  pa->dmin=.001;		// dmin = minimum spline sampling distance
  pa->dmax=.050;		// dmax = maximum spline sampling distance
  pa->T=1.0;			// sample period (= time to traverse dmax)
  pa->amax=1E-2;		// max radial acceleration
  //	pa->tau=2;			// decrease tau to increase steepness of sampling function
  //	pa->rI=-6;			// inflection point of sampling function = tau*rI
  // decrease rI to sample finer

  ////////// get data /////////
  thickness = strtod(argv[5],&eptr);
  scale = strtod(argv[6],&eptr);
  strcpy(outdir,argv[7]);
  capping_flag = (int)strtod(argv[8],&eptr);
  deviation_threshold = strtod(argv[9],&eptr);

  // adjust outdir
  temp=strrchr(argv[7],'/');
  if(!temp) {strcat(outdir,"/");}
  else if(*++temp) {strcat(outdir,"/");}

  // create contours
  ch = getContours(argc,argv,thickness);

  ////////// add previous data //////////
  for (q=ch;q!=NULL;q=q->next){((Contour*)q->data)->addPreviousRaw();}

  ///// check contours for duplicate points /////
    for (q=ch;q!=NULL;q=q->next){((Contour*)q->data)->removeDuplicates();}

    // count contours
    num_contours=countContours(ch);

    ///// linearly interpolate raw contour points /////
      if (deviation_threshold) {
	i=0;
	for (q=ch;q!=NULL;q=q->next) {
	  c=(Contour*)q->data;
	  printf("Contour %s, section %d\n",c->name,c->section);
	  c->linearlyInterp(deviation_threshold,scale);
	}
      }

      ////////// add previous data //////////
      for (q=ch;q!=NULL;q=q->next){
	((Contour*)q->data)->addPreviousRaw();
      }

      // NOTE: RAW POINTS IN CONTOURS ARE STORED IN REVERSE ORDER
      // FIRST OFF THE LINKED LIST WAS LAST ADDED TO LIST.


      ////////// create array of number of points in each contour //////////
      contour_array = createArray(contour_array,ch,num_contours);

      ////////// fit splines //////////
      fitSplines(ch,h,thickness,contour_array,deviation_threshold,scale,outdir,pa,num_contours);

      ///// compute histogram /////
	computeHistogram(h,ch,contour_array);

	////////// create objects //////////
	objectsh=createObjects(ch);

	////////// clear any existing pts and script files //////////
	clearPtsFiles(outdir,objectsh);

	////////// print each object //////////
	// for each object
	for (q=objectsh;q!=NULL;q=q->next) {
	  o=(Object*)q->data;
	  if(!degenerateObject(o) && (capping_flag || o->min_section!=o->max_section)){
	    ///// print config file /////
	    printConfigFile(outdir,o,capping_flag);
	    /////  append to script files /////
	      appendScriptFile(outdir,o);
	      ///// print pts files of interpolated contour points /////
		printPtsFiles(outdir,o,scale);
		if(capping_flag){printCaps(outdir,o,thickness,scale);}
	  }
	}
	
	////////// create calling script //////////
	createCallingScript(outdir,output_script);
	
	////////// print deviation statistics //////////
	printStatistics(h,scale);

	cleanup(objectsh,h,ch,contour_array);

	return 0;
}
#endif
