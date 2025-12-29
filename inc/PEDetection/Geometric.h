/*
  Copyright 2006 The University of Texas at Austin

        Authors: Sangmin Park <smpark@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PEDetection.

  PEDetection is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PEDetection is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef FILE_GEOMETRIC_H
#define FILE_GEOMETRIC_H

//
// Geometric.h
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef FLT_MAX
#define FLT_MAX 3.40282346e+38
#endif

#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623157e+308
#endif

#include <PEDetection/CompileOptions.h>

#define PI 3.14159265358979323846

#define TRUE 1
#define FALSE 0

#define NEAR 1
#define FAR 2

#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Max(a, b) ((a) > (b) ? (a) : (b))

// using namespace std;
template <class _T> class cMatrix_3x3;

class cMatrixNf;
class cVectorNf;

class Vector3f {
protected:
  float x, y, z;

public:
  Vector3f(){};

  Vector3f(float X, float Y, float Z) {
    x = X;
    y = Y;
    z = Z;
  };

  Vector3f(float *Vect) {
    x = Vect[0];
    y = Vect[1];
    z = Vect[2];
  };

  ~Vector3f(){};

  void set(int X, int Y, int Z) {
    x = (float)X;
    y = (float)Y;
    z = (float)Z;
  };

  void set(float X, float Y, float Z) {
    x = X;
    y = Y;
    z = Z;
  };

  void set(double X, double Y, double Z) {
    x = (float)X;
    y = (float)Y;
    z = (float)Z;
  };

  void set(float *Vect3) {
    x = Vect3[0];
    y = Vect3[1];
    z = Vect3[2];
  };

  void set_Normalize(float *Vect3) {
    x = Vect3[0];
    y = Vect3[1];
    z = Vect3[2];
    this->Normalize();
  };

  void set(const Vector3f &Vect) {
    x = Vect[0];
    y = Vect[1];
    z = Vect[2];
  };

  void set(int *From3, int *To3);
  void set(float *From3, float *To3);
  void set(double *From3, double *To3);
  void set_Normalize(int *From3, int *To3);
  void set_Normalize(float *From3, float *To3);
  void set_Normalize(double *From3, double *To3);

  void Add(const Vector3f &add) {
    x += add[0];
    y += add[1];
    z += add[2];
  };

  void Add(const Vector3f &a1, const Vector3f &a2) {
    x = a1[0] + a2[0];
    y = a1[1] + a2[1];
    z = a1[2] + a2[2];
  };

  void Add(int X, int Y, int Z) {
    x += (float)X;
    y += (float)Y;
    z += (float)Z;
  };

  void Add(float X, float Y, float Z) {
    x += X;
    y += Y;
    z += Z;
  };

  void Sub(const Vector3f &a1, const Vector3f &a2) {
    x = a1[0] - a2[0];
    y = a1[1] - a2[1];
    z = a1[2] - a2[2];
  };

  void Sub(const Vector3f &a) {
    x -= a[0];
    y -= a[1];
    z -= a[2];
  };

  void Sub(float X, float Y, float Z) {
    x -= X;
    y -= Y;
    z -= Z;
  };

  void Div(float divisor) {
    x /= divisor;
    y /= divisor;
    z /= divisor;
  };

  void Times(float mult) {
    x *= mult;
    y *= mult;
    z *= mult;
  };

  void Times(int mult) {
    x *= (float)mult;
    y *= (float)mult;
    z *= (float)mult;
  };

  Vector3f get() {
    Vector3f tmpVect(x, y, z);
    return tmpVect;
  };

  float getX() const { return x; };
  float getY() const { return y; };
  float getZ() const { return z; };

  void setX(float X) { x = X; };
  void setY(float Y) { y = Y; };
  void setZ(float Z) { z = Z; };

  Vector3f getPoint(float t, Vector3f p2) {
    Vector3f tempV;
    tempV = p2 - (*this);
    tempV *= t;
    tempV += (*this);
    return tempV;
  }

  float Absolute() const {
    return (float)sqrt((double)x * x + y * y + z * z);
  };
  float Magnitude() const {
    return (float)sqrt((double)x * x + y * y + z * z);
  };
  float Length() const { return (float)sqrt((double)x * x + y * y + z * z); };
  float Distance(const Vector3f &pt) const;

  Vector3f operator+(const Vector3f &V1);
  Vector3f operator-(const Vector3f &V1);
  Vector3f &operator=(const Vector3f &rval);
  Vector3f &operator+=(const Vector3f &rval);
  Vector3f &operator-=(const Vector3f &rval);

  Vector3f &operator+=(float rval);
  Vector3f &operator-=(float rval);
  Vector3f &operator*=(float rval);
  Vector3f &operator/=(float rval);

  Vector3f operator*(cMatrix_3x3<float> &Mat);

  inline Vector3f operator+(const Vector3f &v) const {
    Vector3f a;
    a.Add(*this, v);
    return a;
  };

  inline Vector3f operator-(const Vector3f &v) const {
    Vector3f a;
    a.Sub(*this, v);
    return a;
  };

  int operator==(const Vector3f &v) {
    Vector3f a;
    a.Sub(*this, v);
    if (a.IsZero())
      return TRUE;
    else
      return FALSE;
  }

  // inverse of the component
  void Inverse() {
    this->x *= -1;
    this->y *= -1;
    this->z *= -1;
  };

  void Inverse(Vector3f a) {
    this->set(a);
    this->Inverse();
  };

  // return the inverted vector, i.e. component(i) becomes -component(i)
  inline Vector3f operator-(void) const {
    Vector3f a;
    a *= (-1);
    return a;
  };

  // dot product
  float dot(const Vector3f &rval);

  // Computing degrees between two vectors
  float degrees(const Vector3f &rval);

  // dot product
  float operator*(const Vector3f &v) { return (*this).dot(v); }

  // cross product
  Vector3f cross(const Vector3f &rval);

  // Normalize
  float Normalize();

  float &operator[](int ith);
  const float &operator[](int ith) const;
  float &operator()(int ith);

  // Cosine(Theata)
  //
  //  For examples :
  //      0' Cos(th) =  1.000000
  //     45' Cos(th) =  0.707107
  //     90' Cos(th) =  0.000000
  //    135' Cos(th) = -0.707107
  //    180' Cos(th) = -1.000000
  //
  // cos (theata)
  //
  float getCosine(const Vector3f &Vect);

  // ArcCos(Theata)
  //
  // For examples:
  //      0' ArcCos(th) = 0.000000
  //     45' ArcCos(th) = 0.250000
  //     90' ArcCos(th) = 0.500000
  //    135' ArcCos(th) = 0.750000
  //    180' ArcCos(th) = 1.000000
  //
  // Note : acos(costheata) is divided by PI
  //
  float getArcCosine(const Vector3f &Vect);

  /// ArcCos2(Theata)
  ///
  /// For examples:
  ///      0' ArcCos(th) = 0.000000
  ///     45' ArcCos(th) = 0.785398
  ///     90' ArcCos(th) = 1.570796
  ///    135' ArcCos(th) = 2.356194
  ///    180' ArcCos(th) = 3.141593
  /// Note : acos(costheata) is NOT divided by PI
  /// This function is good for the cases that only need to
  /// compare the angle of two vectors to increase the calculation speed.
  ///
  float getArcCosine2(const Vector3f &Vect);

  int IsZero() {
    if ((fabs((*this).x) < 1.0e-5) && (fabs((*this).y) < 1.0e-5) &&
        (fabs((*this).z) < 1.0e-5))
      return TRUE;
    else
      return FALSE;
  }

  Vector3f CalculateNormal(Vector3f &p1, Vector3f &p2, Vector3f &p3) {
    Vector3f t1, t2, vec;
    t1 = p3 - p2;
    t2 = p1 - p2;
    vec = t1.cross(t2);
    vec.Normalize();
    (*this).set(vec);
    return vec;
  }

  void Display() { printf("%f %f %f\n", x, y, z); }
};

template <class _T> class cMatrix_3x3 {
private:
  _T Elementsf[3][3];

public:
  cMatrix_3x3(void);
  cMatrix_3x3(_T *pElements);
  ~cMatrix_3x3(void);

  _T getElement(int Row, int Column);
  _T *getElementsPoint();

  // for the cross product between a vector and a matrix
  cMatrix_3x3 getSkewMatrix(const Vector3f &Vec);

  void set(_T *Elements);
  void set(_T a1, _T a2, _T a3, _T b1, _T b2, _T b3, _T c1, _T c2, _T c3);
  void set(cMatrix_3x3 &Mat);
  void setElement(int Row, int Column, _T Value);
  void setIdentity();

  _T Determinant();
  void Transpose();               // Transpose this matrix
  void Inverse();                 // Inverse this matrix
  void Inverse(cMatrix_3x3 &Mat); // Inverse Mat and copy Mat to this matrix

  cMatrix_3x3<_T> operator*(const _T scalr);
  cMatrix_3x3<_T> &operator*=(const _T scalr);

  cMatrix_3x3<_T> &operator*(cMatrix_3x3<_T> &Mat1);
  cMatrix_3x3<_T> operator+(cMatrix_3x3<_T> &Mat1);
  cMatrix_3x3<_T> operator-(cMatrix_3x3<_T> &Mat1);

  cMatrix_3x3<_T> &operator*=(cMatrix_3x3<_T> &Mat1);
  cMatrix_3x3<_T> &operator+=(cMatrix_3x3<_T> &Mat);
  cMatrix_3x3<_T> &operator-=(cMatrix_3x3<_T> &Mat);
  cMatrix_3x3<_T> &operator=(cMatrix_3x3<_T> &Mat);

  Vector3f operator*(Vector3f &Vec);

  _T operator()(int unsigned Row, int unsigned Column);

  void OrthonormalizeOrientation();

  // For eigenvalue computation
public:
  _T SumSquareOffDiag();

public:
  void Display();
  void Display(cMatrix_3x3 &Mat);
};

class c3DLine {
private:
  Vector3f Point_m;
  Vector3f Vector_m;

public:
  c3DLine();
  c3DLine(const Vector3f &pt1, const Vector3f &pt2);
  ~c3DLine();

  void set(const Vector3f &pt1, const Vector3f &pt2);
  void set(const c3DLine &line);
  void setPoint(const Vector3f &pt);
  void setVector(const Vector3f &pt);

  Vector3f getPoint();
  Vector3f getPointAt(float t);
  Vector3f getVector();
  Vector3f getDirection();
  void getDirection(double *DirVec);

  Vector3f getOrthogonalProjection(const Vector3f &pt);
  // Orthogonal Distance from the given point to the line
  float Distance(const Vector3f &pt);

  c3DLine &operator=(const c3DLine &rval);
};

class c3DPlane {
private:
  Vector3f Normal_m;
  float Weight_m;
  enum Location { ON, ABOVE, BELOW };

public:
  c3DPlane();
  c3DPlane(const Vector3f &pt1, const Vector3f &pt2, const Vector3f &pt3);
  ~c3DPlane();

  void set(const Vector3f &pt1, const Vector3f &pt2, const Vector3f &pt3);
  void set(const Vector3f &pt1, const Vector3f &normal);
  void set(float Pt1, float Pt2, float Pt3, const Vector3f &Normal);
  void set(const c3DPlane &plane);
  void setNormal(const Vector3f &vec);
  void setWeight(float weight);

  Vector3f getNormal();

  float SignedDistance(const Vector3f &pt);
  // Orthogonal Distance from the given point to the plane
  float Distance(const Vector3f &pt);
  // Location Where (const Vector3f& pt);

  int IntersectionTest(c3DLine &line, float &t);

  // return t value of the line
  float ComputeIntersectionValue(c3DLine &line);

  // return a point
  Vector3f ComputeIntersectionPoint(c3DLine &line);

  // If the given two points are on the different side, then return 0
  // If the given two points are on the same side, then return 1
  // All other cases, then return 2;
  int SameSide(const Vector3f &pt1, const Vector3f &pt2);

  c3DPlane &operator=(const c3DPlane &rval);

  void Display();
};

class VertexSet {
private:
  int MaxNumVertices_mi;
  int CurrentPt_mi;
  float *VertexList_mpf;
  float *NormalList_mpf;

public:
  VertexSet();
  ~VertexSet();

  void InitializeVertexSet();
  void InitializeVertexSet(int NumInitialVertices);
  int AddVertex(const Vector3f &pt);
  int AddVertex(float x, float y, float z);
  void DoubleSize();
  void IncreaseSize(int Size);

  // Normal Related Functions
  void setNormalAt(int i, const Vector3f &Normal);
  void AddNormalAt(int i, const Vector3f &Normal);
  void SetNormalsZero();
  void AverageNormalAt(int i, int NumAddedNumber);
  void NormalizeAt(int i);

  void setPointAt(int i, const Vector3f &Pt);
  int getNumVertices();
  int getMaxSizeVertices();
  Vector3f getVectorAt(int i);
  Vector3f getPointAt(int i);
  Vector3f getVertexAt(int i);
  float getXAt(int i);
  float getYAt(int i);
  float getZAt(int i);
  Vector3f getNormalAt(int i);

  float *getVertexList();
  float *getNormalList();

  Vector3f getMaxPoint();
  Vector3f getMinPoint();

  void Destroy();
};

#define MAX_NUM_PARTICLES 250

struct sMinMaxValues {
  float Min;
  float Max;
};

struct sBinTreeNode {
  struct sBinTreeNode *Left_p, *Right_p, *Parent_p;
  struct sBinTreeNode *Neighbors_p[3];
  int VertexIndexesi[3];
  int NearOrFarFromViewPosition; // 0=far, 1=near, 2=does not matter
  int Visible;     // Which means that the triangle is inside of view volume
  sMinMaxValues Z; // Min and Max Z-values of children and the current node
};

struct sParticle {
  Vector3f Location;           // Current Location
  Vector3f Next_Location;      // Next Location
  float Force_Magnitude;       //
  float Force_Continuance;     // in milliseconds
  float Velocity_Magnitude;    //
  Vector3f Force_Direction;    // which should be normalized
  Vector3f Velocity_Direction; // which should be normalized
  float LifeTime;              // of this particle in milliseconds
  float Size;                  // The size of cube
  float Mass;                  //
  Vector3f Color;              // RGB
  float Alpha;                 // Transparency
  float BounceFactor;          // from 0 - 1
  int Status;                  // Moving=1 or not=0

  // For angular velocity
  cMatrix_3x3<float> Orientation; // R(t) matrix : Orientation of Rotation
  cMatrix_3x3<float>
      Next_Orientation;     // R(t) matrix : Orientation of Rotation
  Vector3f AngularVelocity; //
  cMatrix_3x3<float>
      InertiaTensor; // The Matrix of Inertial Tensor in object space
};

class cVectorNf {
protected:
  int N;      // The number of elements
  float *vec; //

public:
  cVectorNf();
  cVectorNf(int NumElements);
  cVectorNf(cVectorNf &Vector);
  ~cVectorNf();

  void set(int NumElements);
  void set(int ith, float value);
  void set(cVectorNf &Vector);

  float get(int ith);
  inline int getNumElements() { return N; }

  float Dot(const cVectorNf &Vector);
  float Absolute();
  void Normalize();
  void Normalize(cVectorNf &Vector);

  cVectorNf &operator=(cVectorNf &Vector);
  cVectorNf &operator+=(cVectorNf &Vector);
  cVectorNf &operator-=(cVectorNf &Vector);

  cVectorNf &operator+=(float scalar);
  cVectorNf &operator-=(float scalar);
  cVectorNf &operator*=(float scalar);

  void Destroy() {
    delete[] vec;
    N = 0;
  }
};

#endif
