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

// Geometric.C
//------------------------------------------------------------
//
//  Geometric Objects and Vertex implementation
//
//  Implemented by Sangmin Park
//
//------------------------------------------------------------

#include <PEDetection/Geometric.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// using namespace std;

// --------------------------------------------------------------------------
// class Vector3f methods --------------------------------------------------
// --------------------------------------------------------------------------

void Vector3f::set(int *From3, int *To3) {
  this->set(To3[0] - From3[0], To3[1] - From3[1], To3[2] - From3[2]);
}

void Vector3f::set(float *From3, float *To3) {
  this->set(To3[0] - From3[0], To3[1] - From3[1], To3[2] - From3[2]);
}

void Vector3f::set(double *From3, double *To3) {
  this->set(To3[0] - From3[0], To3[1] - From3[1], To3[2] - From3[2]);
}

void Vector3f::set_Normalize(int *From3, int *To3) {
  this->set(To3[0] - From3[0], To3[1] - From3[1], To3[2] - From3[2]);
  this->Normalize();
}

void Vector3f::set_Normalize(float *From3, float *To3) {
  this->set(To3[0] - From3[0], To3[1] - From3[1], To3[2] - From3[2]);
  this->Normalize();
}

void Vector3f::set_Normalize(double *From3, double *To3) {
  this->set(To3[0] - From3[0], To3[1] - From3[1], To3[2] - From3[2]);
  this->Normalize();
}

// 1x3 * 3x3 = 1x3
// return 1x3 vector
Vector3f Vector3f::operator*(cMatrix_3x3<float> &Mat) {
  Vector3f RetVec;
  int i, j;
  float Temp_sum[3] = {0.0, 0.0, 0.0};

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      Temp_sum[i] += (*this)[j] * Mat.getElement(j, i);
    }
  }
  RetVec[0] = Temp_sum[0];
  RetVec[1] = Temp_sum[1];
  RetVec[2] = Temp_sum[2];

  return RetVec;
}

Vector3f Vector3f::operator+(const Vector3f &V1) {
  Vector3f TempV;

  TempV.set(x + V1.getX(), y + V1.getY(), z + V1.getZ());
  return TempV;
}

Vector3f Vector3f::operator-(const Vector3f &V1) {
  Vector3f TempV;

  TempV.set(x - V1.getX(), y - V1.getY(), z - V1.getZ());
  return TempV;
}

Vector3f &Vector3f::operator=(const Vector3f &rval) {
  set(rval);
  return *this;
}

Vector3f &Vector3f::operator+=(const Vector3f &rval) {
  x += rval[0];
  y += rval[1];
  z += rval[2];
  return *this;
}

Vector3f &Vector3f::operator-=(const Vector3f &rval) {
  x -= rval[0];
  y -= rval[1];
  z -= rval[2];
  return *this;
}

Vector3f &Vector3f::operator+=(float rval) {
  x += rval;
  y += rval;
  z += rval;
  return *this;
}

Vector3f &Vector3f::operator-=(float rval) {
  x -= rval;
  y -= rval;
  z -= rval;
  return *this;
}

Vector3f &Vector3f::operator*=(float rval) {
  x *= rval;
  y *= rval;
  z *= rval;
  return *this;
}

Vector3f &Vector3f::operator/=(float rval) {
  x /= rval;
  y /= rval;
  z /= rval;
  return *this;
}

float Vector3f::Normalize() {
  float Abs = Absolute();
  if (Abs > 1e-5)
    (*this) /= Abs;
  else {
    this->x = this->y = this->z = 0;
  }
  return Abs;
}

// dot product
float Vector3f::dot(const Vector3f &rval) {
  float tmp;

  tmp = x * rval[0];
  tmp += y * rval[1];
  tmp += z * rval[2];

  return tmp;
}

// Compute degrees between two vectors
float Vector3f::degrees(const Vector3f &rval) {
  float Dot_f, Degrees_f;

  Dot_f = x * rval[0];
  Dot_f += y * rval[1];
  Dot_f += z * rval[2];

  if (Dot_f < -0.99999)
    Dot_f = -1.0 + 1e-5;
  if (Dot_f > 0.99999)
    Dot_f = 1.0 - 1e-5;
  if (fabsf(Dot_f) < 1e-5)
    Degrees_f = 90.0;
  else
    Degrees_f = acos(Dot_f) * 180 / PI;

  return Degrees_f;
}

// cross product
Vector3f Vector3f::cross(const Vector3f &rval) {
  float tmp[3];

  tmp[0] = y * rval.getZ() - z * rval.getY();
  tmp[1] = z * rval.getX() - x * rval.getZ();
  tmp[2] = x * rval.getY() - y * rval.getX();

  Vector3f Vect(tmp);

  return Vect;
}

float &Vector3f::operator[](int ith) {
  switch (ith) {
  case 0:
    return x;
  case 1:
    return y;
  case 2:
    return z;
  default:
    printf("Error in operator[]\n\n");
    exit(1);
  }
  return x; // To remove compile warning
}

float Vector3f::Distance(const Vector3f &pt) const {
  return sqrt((pt[0] - x) * (pt[0] - x) + (pt[1] - y) * (pt[1] - y) +
              (pt[2] - z) * (pt[2] - z));
}

const float &Vector3f::operator[](int ith) const {
  switch (ith) {
  case 0:
    return x;
  case 1:
    return y;
  case 2:
    return z;
  default:
    printf("Error in operator[]\n\n");
    exit(1);
  }
  return x; // To remove compile warning
}

float &Vector3f::operator()(int ith) {
  switch (ith) {
  case 0:
    return x;
  case 1:
    return y;
  case 2:
    return z;
  default:
    printf("Error in operator[]\n\n");
    exit(1);
  }
  return x; // To remove compile warning
}

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
float Vector3f::getCosine(const Vector3f &Vect) {
  float Absolute_v1, Absolute_v2, costheata;

  Absolute_v1 = this->Absolute();
  Absolute_v2 = Vect.Absolute();

  costheata = (this->dot(Vect)) / (Absolute_v1 * Absolute_v2);

  return (float)costheata;
}

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
float Vector3f::getArcCosine(const Vector3f &Vect) {
  float Absolute_v1, Absolute_v2, costheata;

  Absolute_v1 = this->Absolute();
  Absolute_v2 = Vect.Absolute();

  costheata = (this->dot(Vect)) / (Absolute_v1 * Absolute_v2);
  float arccos = acos(costheata) / PI;

  return (float)arccos;
}

// ArcCos2(Theata)
//
// For examples:
//      0' ArcCos(th) = 0.000000
//     45' ArcCos(th) = 0.785398
//     90' ArcCos(th) = 1.570796
//    135' ArcCos(th) = 2.356194
//    180' ArcCos(th) = 3.141593
// Note : acos(costheata) is NOT divided by PI
// This function is good for the cases whice need only to
// compare the angle of two vectors to increase the calculation speed.
//
float Vector3f::getArcCosine2(const Vector3f &Vect) {
  float Absolute_v1, Absolute_v2, costheata;

  Absolute_v1 = this->Absolute();
  Absolute_v2 = Vect.Absolute();

  costheata = (this->dot(Vect)) / (Absolute_v1 * Absolute_v2);
  float arccos = acos(costheata);

  return (float)arccos;
}

// --------------------------------------------------------------------------
// class cMatrix_3x3
//
// 3x3 Matrix
// --------------------------------------------------------------------------
template <class _T> cMatrix_3x3<_T>::cMatrix_3x3(void) {
  for (int i = 0; i < 9; i++) {
    Elementsf[0][i] = 0;
  }
  Elementsf[0][0] = Elementsf[1][1] = Elementsf[2][2] = 1;
}

template <class _T> cMatrix_3x3<_T>::cMatrix_3x3(_T *elements) {
  for (int i = 0; i < 9; i++) {
    Elementsf[0][i] = elements[i];
  }
}

template <class _T> cMatrix_3x3<_T>::~cMatrix_3x3(void) {}

template <class _T> _T cMatrix_3x3<_T>::getElement(int Row, int Column) {
  return (_T)Elementsf[Row][Column];
}

template <class _T> _T *cMatrix_3x3<_T>::getElementsPoint() {
  return &this->Elementsf[0][0];
}

// for the cross product between a vector and a matrix
template <class _T>
cMatrix_3x3<_T> cMatrix_3x3<_T>::getSkewMatrix(const Vector3f &Vec) {
  this->Elementsf[0][0] = 0.0;
  this->Elementsf[1][1] = 0.0;
  this->Elementsf[2][2] = 0.0;

  this->Elementsf[0][1] = (_T)-Vec[2];
  this->Elementsf[0][2] = (_T)Vec[1];

  this->Elementsf[1][0] = (_T)Vec[2];
  this->Elementsf[1][2] = (_T)-Vec[0];

  this->Elementsf[2][0] = (_T)-Vec[1];
  this->Elementsf[2][1] = (_T)Vec[0];

  return *this;
}

template <class _T> void cMatrix_3x3<_T>::set(_T *Elements) {
  for (int i = 0; i < 9; i++) {
    Elementsf[0][i] = Elements[i];
  }
}

template <class _T>
void cMatrix_3x3<_T>::set(_T a1, _T a2, _T a3, _T b1, _T b2, _T b3, _T c1,
                          _T c2, _T c3) {
  this->Elementsf[0][0] = a1;
  this->Elementsf[0][1] = a2;
  this->Elementsf[0][2] = a3;

  this->Elementsf[1][0] = b1;
  this->Elementsf[1][1] = b2;
  this->Elementsf[1][2] = b3;

  this->Elementsf[2][0] = c1;
  this->Elementsf[2][1] = c2;
  this->Elementsf[2][2] = c3;
}

template <class _T>
void cMatrix_3x3<_T>::setElement(int Row, int Column, _T Value) {
  this->Elementsf[Row][Column] = Value;
}

template <class _T> void cMatrix_3x3<_T>::set(cMatrix_3x3 &Mat) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      Elementsf[i][j] = Mat.getElement(i, j);
}

template <class _T> void cMatrix_3x3<_T>::setIdentity() {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      Elementsf[i][j] = 0.0;

  Elementsf[0][0] = 1.0;
  Elementsf[1][1] = 1.0;
  Elementsf[2][2] = 1.0;
}

template <class _T> _T cMatrix_3x3<_T>::Determinant() {
  _T Det;
  _T *a = &this->Elementsf[0][0];

  Det = (-a[2] * a[4] * a[6] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
         a[0] * a[5] * a[7] - a[1] * a[3] * a[8] + a[0] * a[4] * a[8]);

  return Det;
}

template <class _T> void cMatrix_3x3<_T>::Transpose() {
  cMatrix_3x3 ReturnMat;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ReturnMat.Elementsf[i][j] = this->Elementsf[j][i];
    }
  }
  *this = ReturnMat;
}

/*
Inverse 3x3 Matrix
Det = (-a3 a5 a7 + a2 a6 a7 + a3 a4 a8 - a1 a6 a8 - a2 a4 a9 + a1 a5 a9)
{{(-a6 a8 + a5 a9)/Det,   (a3 a8  - a2 a9)/Det,   (-a3 a5 + a2 a6)/Det},
 {(a6 a7  - a4 a9)/Det,   (-a3 a7 + a1 a9)/Det,   (a3 a4  - a1 a6)/Det},
 {(-a5 a7 + a4 a8)/Det,   (a2 a7  - a1 a8)/Det,   (-a2 a4 + a1 a5)/Det}}
*/
template <class _T> void cMatrix_3x3<_T>::Inverse() {
  cMatrix_3x3 ReturnMat;
  _T Det;
  _T *a = &this->Elementsf[0][0];

  Det = this->Determinant();

  if (fabs(Det) < 10e-6) {
    printf("Singular matrix : There is no inverse matrix\n");
    Det = this->Determinant();
  } else {
    ReturnMat.setElement(1, 1, (-a[5] * a[7] + a[4] * a[8]) / Det);
    ReturnMat.setElement(1, 2, (a[2] * a[7] - a[1] * a[8]) / Det);
    ReturnMat.setElement(1, 3, (-a[2] * a[4] + a[1] * a[5]) / Det);

    ReturnMat.setElement(2, 1, (a[5] * a[6] - a[3] * a[8]) / Det);
    ReturnMat.setElement(2, 2, (-a[2] * a[6] + a[0] * a[8]) / Det);
    ReturnMat.setElement(2, 3, (a[2] * a[3] - a[0] * a[5]) / Det);

    ReturnMat.setElement(3, 1, (-a[4] * a[6] + a[3] * a[7]) / Det);
    ReturnMat.setElement(3, 2, (a[1] * a[6] - a[0] * a[7]) / Det);
    ReturnMat.setElement(3, 3, (-a[1] * a[3] + a[0] * a[4]) / Det);
  }
  *this = ReturnMat;
}

template <class _T> void cMatrix_3x3<_T>::Inverse(cMatrix_3x3 &Mat) {
  this->set(Mat);
  this->Inverse();
}

template <class _T>
cMatrix_3x3<_T> cMatrix_3x3<_T>::operator*(const _T scalar) {
  cMatrix_3x3<_T> TempM;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      TempM.setElement(i, j, this->getElement(i, j) * scalar);
    }
  }
  return TempM;
}

template <class _T>
cMatrix_3x3<_T> &cMatrix_3x3<_T>::operator*=(const _T scalar) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      this->Elementsf[i][j] *= scalar;
    }
  }
  return *this;
}

template <class _T>
cMatrix_3x3<_T> &cMatrix_3x3<_T>::operator*(cMatrix_3x3<_T> &Mat1) {
  int i, j, k;
  float Value;
  cMatrix_3x3<_T> ReturnMatrix;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      Value = 0.0;
      for (k = 0; k < 3; k++) {
        Value += this->getElement(i, k) * Mat1.getElement(k, j);
      }
      ReturnMatrix.setElement(i, j, Value);
    }
  }

  //	return ReturnMatrix;

  (*this) = ReturnMatrix;
  return (*this);
}

template <class _T>
cMatrix_3x3<_T> cMatrix_3x3<_T>::operator+(cMatrix_3x3<_T> &Mat1) {
  cMatrix_3x3<_T> ReturnMatrix;
  _T Value;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Value = (_T)this->getElement(i, j) + (_T)Mat1.getElement(i, j);
      ReturnMatrix.setElement(i, j, Value);
    }
  }
  return ReturnMatrix;
}

template <class _T>
cMatrix_3x3<_T> cMatrix_3x3<_T>::operator-(cMatrix_3x3<_T> &Mat1) {
  cMatrix_3x3<_T> ReturnMatrix;
  _T Value;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Value = this->getElement(i, j) - Mat1.getElement(i, j);
      ReturnMatrix.setElement(i, j, Value);
    }
  }
  return ReturnMatrix;
}

template <class _T>
cMatrix_3x3<_T> &cMatrix_3x3<_T>::operator*=(cMatrix_3x3<_T> &Mat1) {
  cMatrix_3x3<_T> ReturnMatrix;
  int i, j, k;
  _T Value;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      Value = 0.0;
      for (k = 0; k < 3; k++) {
        Value += this->Elementsf[i][k] * Mat1.getElement(k, j);
      }
      ReturnMatrix.setElement(i, j, Value);
    }
  }
  (*this) = ReturnMatrix;
  return (*this);
}

template <class _T>
cMatrix_3x3<_T> &cMatrix_3x3<_T>::operator+=(cMatrix_3x3<_T> &Mat) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      this->Elementsf[i][j] += Mat.getElement(i, j);
    }
  }
  return *this;
}

template <class _T>
cMatrix_3x3<_T> &cMatrix_3x3<_T>::operator-=(cMatrix_3x3<_T> &Mat) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      this->Elementsf[i][j] -= Mat.getElement(i, j);
    }
  }
  return *this;
}

template <class _T>
cMatrix_3x3<_T> &cMatrix_3x3<_T>::operator=(cMatrix_3x3<_T> &Mat) {
  set(Mat);
  return (*this);
  /*
          for (int i=0; i<3; i++) {
                  for (int j=0; j<3; j++) {
                          RetMatrix.setElements(i,j) = Mat.getElement(i, j);
                  }
          }
          return RetMatrix;
  */
}

// 3x3 * 3x1 = 3x1
// return 3x1 vector ( = Transposed Vector)
template <class _T> Vector3f cMatrix_3x3<_T>::operator*(Vector3f &Vec) {
  Vector3f RetVec;
  int i, j;
  _T Temp_sum[3] = {0.0, 0.0, 0.0};

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      Temp_sum[i] += this->getElement(i, j) * Vec[j];
    }
  }
  RetVec[0] = Temp_sum[0];
  RetVec[1] = Temp_sum[1];
  RetVec[2] = Temp_sum[2];

  return RetVec;
}

template <class _T> void cMatrix_3x3<_T>::OrthonormalizeOrientation() {
  Vector3f X(Elementsf[0][0], Elementsf[1][0], Elementsf[2][0]);
  Vector3f Y(Elementsf[0][1], Elementsf[1][1], Elementsf[2][1]);
  Vector3f Z, TempV;

  X.Normalize();
  TempV = X.cross(Y);
  TempV.Normalize();
  Z = TempV;

  TempV = Z.cross(X);
  TempV.Normalize();
  Y = TempV;

  Elementsf[0][0] = X[0];
  Elementsf[0][1] = Y[0];
  Elementsf[0][2] = Z[0];
  Elementsf[1][0] = X[1];
  Elementsf[1][1] = Y[1];
  Elementsf[1][2] = Z[1];
  Elementsf[2][0] = X[2];
  Elementsf[2][1] = Y[2];
  Elementsf[2][2] = Z[2];
}

template <class _T> _T cMatrix_3x3<_T>::SumSquareOffDiag() {
  return (Elementsf[1][0] * Elementsf[1][0] +
          Elementsf[2][0] * Elementsf[2][0] +
          Elementsf[2][1] * Elementsf[2][1]);
}

template <class _T> void cMatrix_3x3<_T>::Display() {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%f ", Elementsf[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

template <class _T> void cMatrix_3x3<_T>::Display(cMatrix_3x3<_T> &Mat) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%f ", Mat.getElement(i, j));
    }
    printf("\n");
  }
  printf("\n");
}

cMatrix_3x3<float> _Matrix3x3_01;
cMatrix_3x3<double> _Matrix3x3_02;

// --------------------------------------------------------------------------
// class 3DLine
//
// Equation: Point_m + t*Vector_m
// --------------------------------------------------------------------------
c3DLine::c3DLine() {}

c3DLine::c3DLine(const Vector3f &pt1, const Vector3f &pt2) {
  Point_m = pt1;
  Vector_m = pt2 - pt1;
  Vector_m.Normalize();
}

c3DLine::~c3DLine() {}

void c3DLine::set(const Vector3f &pt1, const Vector3f &pt2) {
  Point_m = pt1;
  Vector_m = pt2 - pt1;
  Vector_m.Normalize();
}

void c3DLine::set(const c3DLine &line) {
  this->Point_m = line.Point_m;
  this->Vector_m = line.Vector_m;
}

void c3DLine::setPoint(const Vector3f &pt) { Point_m = pt; }

void c3DLine::setVector(const Vector3f &vec) { Vector_m = vec; }

Vector3f c3DLine::getPoint() { return Point_m; }

Vector3f c3DLine::getPointAt(float t) {
  Vector3f pt = Vector_m;
  pt *= t;
  pt += Point_m;

  return pt;
}

Vector3f c3DLine::getVector() { return Vector_m; }

Vector3f c3DLine::getDirection() { return Vector_m; }

void c3DLine::getDirection(double *DirVec) {
  DirVec[0] = (double)Vector_m[0];
  DirVec[1] = (double)Vector_m[1];
  DirVec[2] = (double)Vector_m[2];
}

// Orthogonal Distance from the given point to the line
float c3DLine::Distance(const Vector3f &pt) {
  Vector3f Vec = getOrthogonalProjection(pt);
  Vec -= pt;

  return Vec.Length();
}

Vector3f c3DLine::getOrthogonalProjection(const Vector3f &pt) {
  Vector3f Vec;
  float Tempf;

  // * = Dot Product between Vector3f variables
  Tempf = (Vector_m * pt) - (Vector_m * Point_m);
  Vec = Vector_m;
  Vec *= Tempf;
  return (Point_m + Vec);
}

c3DLine &c3DLine::operator=(const c3DLine &rval) {
  set(rval);
  return *this;
}

// --------------------------------------------------------------------------
// class c3DPlane
//
// Equation : aX + bY + cZ + W = 0
// --------------------------------------------------------------------------
c3DPlane::c3DPlane() {}
c3DPlane::c3DPlane(const Vector3f &pt1, const Vector3f &pt2,
                   const Vector3f &pt3) {
  Vector3f vec1, vec2;

  vec1 = pt3 - pt2;
  vec2 = pt1 - pt2;
  Normal_m = vec1.cross(vec2);
  Normal_m.Normalize();

  Weight_m =
      -(pt1[0] * Normal_m[0] + pt1[1] * Normal_m[1] + pt1[2] * Normal_m[2]);
}

c3DPlane::~c3DPlane() {}

void c3DPlane::set(const Vector3f &pt1, const Vector3f &pt2,
                   const Vector3f &pt3) {
  Vector3f vec1, vec2;

  vec1 = pt3 - pt2;
  vec2 = pt1 - pt2;
  Normal_m = vec1.cross(vec2);
  Normal_m.Normalize();

  Weight_m =
      -(pt1[0] * Normal_m[0] + pt1[1] * Normal_m[1] + pt1[2] * Normal_m[2]);
}

void c3DPlane::set(const Vector3f &pt1, const Vector3f &Normal) {
  Normal_m = Normal;
  Normal_m.Normalize();
  Weight_m =
      -(pt1[0] * Normal_m[0] + pt1[1] * Normal_m[1] + pt1[2] * Normal_m[2]);
}

void c3DPlane::set(float Pt1, float Pt2, float Pt3, const Vector3f &Normal) {
  Normal_m = Normal;
  Normal_m.Normalize();
  Weight_m = -(Pt1 * Normal_m[0] + Pt2 * Normal_m[1] + Pt3 * Normal_m[2]);
}

void c3DPlane::set(const c3DPlane &plane) {
  this->Normal_m = plane.Normal_m;
  this->Weight_m = plane.Weight_m;
}

void c3DPlane::setNormal(const Vector3f &vec) { Normal_m = vec; }

void c3DPlane::setWeight(float weight) { Weight_m = weight; }

Vector3f c3DPlane::getNormal() { return Normal_m; }

float c3DPlane::SignedDistance(const Vector3f &pt) {
  return (pt[0] * Normal_m[0] + pt[1] * Normal_m[1] + pt[2] * Normal_m[2] +
          Weight_m);
}

float c3DPlane::Distance(const Vector3f &pt) {
  return ((float)fabs((double)SignedDistance(pt)));
}
/*
c3DPlane::Location c3DPlane::Where(Vector3f& pt)
{
        float	distance = SignedDistance(pt);

        return distance < 0.0 ? BELOW : distance > 0.0 ? ABOVE : ON;
}
*/

// Return : if (intersect), then return 1;
//			else return 0;
int c3DPlane::IntersectionTest(c3DLine &line, float &t) {
  Vector3f Tempv = line.getDirection();
  float a = Tempv.dot(getNormal());

  if (fabs(a) <= 10e-6) {
    printf("plane and line are parallel!\n"); // raise an exception
    return false;
  } else {
    t = -(line.getPoint() * getNormal() + Weight_m) / a;
    return true;
  }
  return false;
}

float c3DPlane::ComputeIntersectionValue(c3DLine &line) {
  float t = 0.0;
  float a = line.getDirection() * getNormal();

  if (fabs(a) <= 10e-6) {
    printf("plane and line are parallel!\n"); // raise an exception
  } else {
    t = -(line.getPoint() * getNormal() + Weight_m) / a;
  }
  return t;
}

Vector3f c3DPlane::ComputeIntersectionPoint(c3DLine &line) {
  Vector3f pt;
  float t = 0.0;
  float a = line.getDirection() * getNormal();

  if (fabs(a) <= 10e-6) {
    printf("plane and line are parallel!\n"); // raise an exception
  } else {
    t = -(line.getPoint() * getNormal() + Weight_m) / a;
    pt = line.getPointAt(t);
  }
  return pt;
}

//
// If the given two points are on the different side, then return 0
// If the given two points are on the same side, then return 1
// All other cases, then return 2;
int c3DPlane::SameSide(const Vector3f &pt1, const Vector3f &pt2) {
  float Dist1 = this->SignedDistance(pt1);
  float Dist2 = this->SignedDistance(pt2);

  if (fabs(Dist1) < 10e-6 || fabs(Dist2) < 10e-6)
    return 2;
  if (Dist1 * Dist2 > 0.0)
    return 1;
  else
    return 0;
}

c3DPlane &c3DPlane::operator=(const c3DPlane &rval) {
  set(rval);
  return *this;
}

void c3DPlane::Display() {
  printf("Normal = %f %f %f, Weight = %f\n", Normal_m[0], Normal_m[1],
         Normal_m[2], Weight_m);
}

// --------------------------------------------------------------------------
// class VertexSet
// --------------------------------------------------------------------------
VertexSet::VertexSet() {
  MaxNumVertices_mi = 100;
  CurrentPt_mi = 0;
  VertexList_mpf = new float[MaxNumVertices_mi * 3];
  NormalList_mpf = new float[MaxNumVertices_mi * 3];
}

VertexSet::~VertexSet() {
  delete[] VertexList_mpf;
  delete[] NormalList_mpf;
}

void VertexSet::InitializeVertexSet() {
  // Deleting current allocated memory and
  // Initializing the maximum number of vertices.
  InitializeVertexSet(100);
}

void VertexSet::InitializeVertexSet(int NumInitialVertices) {
  // Deleting current allocated memory and
  // Initializing the maximum number of vertices.
  delete[] VertexList_mpf;
  delete[] NormalList_mpf;
  MaxNumVertices_mi = NumInitialVertices;
  CurrentPt_mi = 0;
  VertexList_mpf = new float[MaxNumVertices_mi * 3];
  NormalList_mpf = new float[MaxNumVertices_mi * 3];
}

int VertexSet::AddVertex(const Vector3f &pt) {

  if (CurrentPt_mi == MaxNumVertices_mi) {
    DoubleSize();
  }
  VertexList_mpf[CurrentPt_mi * 3] = pt.getX();
  VertexList_mpf[CurrentPt_mi * 3 + 1] = pt.getY();
  VertexList_mpf[CurrentPt_mi * 3 + 2] = pt.getZ();
  CurrentPt_mi++;

#ifdef DEBUG
//	printf ("AddVertex at %d = %f %f %f\n", (CurrentPt_mi-1),
//				pt.getX(), pt.getY(), pt.getZ());
#endif

  if (CurrentPt_mi % 10000 == 0)
    printf("Num Vertices = %d\n", CurrentPt_mi);

  return CurrentPt_mi - 1;
}

int VertexSet::AddVertex(float x, float y, float z) {
  int CurrPt;
  Vector3f pt(x, y, z);
  CurrPt = AddVertex(pt);
  return CurrPt;
}

void VertexSet::DoubleSize() { IncreaseSize(MaxNumVertices_mi); }

void VertexSet::IncreaseSize(int Size) {
  int MaxNumVertices, i;
  float *Vertex;
  float *VertexNormal;

  MaxNumVertices = MaxNumVertices_mi + Size;
  Vertex = new float[MaxNumVertices * 3];
  VertexNormal = new float[MaxNumVertices * 3];

  for (i = 0; i < CurrentPt_mi; i++) {
    Vertex[i * 3 + 0] = VertexList_mpf[i * 3 + 0];
    Vertex[i * 3 + 1] = VertexList_mpf[i * 3 + 1];
    Vertex[i * 3 + 2] = VertexList_mpf[i * 3 + 2];
    VertexNormal[i * 3 + 0] = NormalList_mpf[i * 3 + 0];
    VertexNormal[i * 3 + 1] = NormalList_mpf[i * 3 + 1];
    VertexNormal[i * 3 + 2] = NormalList_mpf[i * 3 + 2];
  }
  delete[] VertexList_mpf;
  delete[] NormalList_mpf;

  VertexList_mpf = Vertex;
  NormalList_mpf = VertexNormal;
  MaxNumVertices_mi = MaxNumVertices;
}

void VertexSet::setNormalAt(int i, const Vector3f &Normal) {
  NormalList_mpf[i * 3] = Normal[0];
  NormalList_mpf[i * 3 + 1] = Normal[1];
  NormalList_mpf[i * 3 + 2] = Normal[2];
}

void VertexSet::AddNormalAt(int i, const Vector3f &Normal) {
  NormalList_mpf[i * 3] += Normal[0];
  NormalList_mpf[i * 3 + 1] += Normal[1];
  NormalList_mpf[i * 3 + 2] += Normal[2];
}

void VertexSet::SetNormalsZero() {
  for (int i = 0; i < MaxNumVertices_mi * 3; i++) {
    NormalList_mpf[i] = (float)0.0;
  }
}

void VertexSet::AverageNormalAt(int i, int NumAdded) {
  NormalList_mpf[i * 3] /= (float)NumAdded;
  NormalList_mpf[i * 3 + 1] /= (float)NumAdded;
  NormalList_mpf[i * 3 + 2] /= (float)NumAdded;
}

void VertexSet::NormalizeAt(int i) {
  Vector3f Normal(NormalList_mpf[i * 3], NormalList_mpf[i * 3 + 1],
                  NormalList_mpf[i * 3 + 2]);
  Normal.Normalize();
  setNormalAt(i, Normal);
}

void VertexSet::setPointAt(int i, const Vector3f &Pt) {
  VertexList_mpf[i * 3] = Pt[0];
  VertexList_mpf[i * 3 + 1] = Pt[1];
  VertexList_mpf[i * 3 + 2] = Pt[2];
}

int VertexSet::getNumVertices() { return CurrentPt_mi; }

int VertexSet::getMaxSizeVertices() { return MaxNumVertices_mi; }

Vector3f VertexSet::getVectorAt(int i) {
  Vector3f Pt(VertexList_mpf[i * 3], VertexList_mpf[i * 3 + 1],
              VertexList_mpf[i * 3 + 2]);
  return Pt;
}

Vector3f VertexSet::getPointAt(int i) {
  Vector3f Pt(VertexList_mpf[i * 3], VertexList_mpf[i * 3 + 1],
              VertexList_mpf[i * 3 + 2]);
  return Pt;
}

Vector3f VertexSet::getVertexAt(int i) {
  Vector3f Pt(VertexList_mpf[i * 3], VertexList_mpf[i * 3 + 1],
              VertexList_mpf[i * 3 + 2]);
  return Pt;
}

float VertexSet::getXAt(int i) { return VertexList_mpf[i * 3 + 0]; }
float VertexSet::getYAt(int i) { return VertexList_mpf[i * 3 + 1]; }
float VertexSet::getZAt(int i) { return VertexList_mpf[i * 3 + 2]; }

Vector3f VertexSet::getNormalAt(int i) {
  Vector3f Pt(NormalList_mpf[i * 3], NormalList_mpf[i * 3 + 1],
              NormalList_mpf[i * 3 + 2]);
  return Pt;
}

float *VertexSet::getVertexList() { return VertexList_mpf; }

float *VertexSet::getNormalList() { return NormalList_mpf; }

Vector3f VertexSet::getMaxPoint() {
  float MaxX = -FLT_MAX, MaxY = -FLT_MAX, MaxZ = -FLT_MAX;

  for (int i = 0; i < CurrentPt_mi; i++) {
    if (MaxX < VertexList_mpf[i * 3])
      MaxX = VertexList_mpf[i * 3];
    if (MaxY < VertexList_mpf[i * 3 + 1])
      MaxX = VertexList_mpf[i * 3 + 1];
    if (MaxZ < VertexList_mpf[i * 3 + 2])
      MaxX = VertexList_mpf[i * 3 + 2];
  }
  Vector3f vec(MaxX, MaxY, MaxZ);
  return vec;
}

Vector3f VertexSet::getMinPoint() {
  float MinX = FLT_MAX, MinY = FLT_MAX, MinZ = FLT_MAX;

  for (int i = 0; i < CurrentPt_mi; i++) {
    if (MinX > VertexList_mpf[i * 3])
      MinX = VertexList_mpf[i * 3];
    if (MinY > VertexList_mpf[i * 3 + 1])
      MinX = VertexList_mpf[i * 3 + 1];
    if (MinZ > VertexList_mpf[i * 3 + 2])
      MinX = VertexList_mpf[i * 3 + 2];
  }

  Vector3f vec(MinX, MinY, MinZ);
  return vec;
}

void VertexSet::Destroy() {
  delete[] VertexList_mpf;
  delete[] NormalList_mpf;
}

// Calculate a triangle area
float CalculateArea(Vector3f &pt1, Vector3f &pt2, Vector3f &pt3) {
  c3DLine line(pt1, pt2);
  float Height, Base, Area;

  Height = line.Distance(pt3);
  Base = pt1.Distance(pt2);
  Area = Height * Base / 2.0;
  return Area;
}

//----------------------------------------------------------------------------------
// N Vector
//
//----------------------------------------------------------------------------------
cVectorNf::cVectorNf() { set(3); }

cVectorNf::cVectorNf(int NumElements) { set(NumElements); }

cVectorNf::cVectorNf(cVectorNf &Vector) {
  this->N = Vector.getNumElements();
  vec = new float[this->N];
  set(Vector);
}

cVectorNf::~cVectorNf() { delete[] vec; }

void cVectorNf::set(int NumElements) {
  delete[] vec;
  N = NumElements;
  vec = new float[NumElements];
}

void cVectorNf::set(int ith, float value) {
  if (ith >= N)
    printf("ith=%d is greater than the size of the vector\n", ith);
  else
    vec[ith] = value;
}

void cVectorNf::set(cVectorNf &Vector) {
  if (this->getNumElements() != Vector.getNumElements())
    printf("The two vectors have different size, %d, %d\n",
           this->getNumElements(), Vector.getNumElements());
  else
    for (int i = 0; i < this->getNumElements(); i++) {
      this->set(i, Vector.get(i));
    }
}

float cVectorNf::get(int ith) {
  if (ith >= N)
    printf("ith=%d is greater than the size of the vector\n", ith);
  else
    return vec[ith];
  return 0.0;
}

float cVectorNf::Dot(const cVectorNf &Vector) {
  float Sum = 0.0;
  for (int i = 0; i < N; i++)
    Sum += this->vec[i] * Vector.vec[i];
  return Sum;
}

float cVectorNf::Absolute() {
  float Sum = 0.0;
  for (int i = 0; i < N; i++)
    Sum += vec[i] * vec[i];
  return sqrt(Sum);
}

void cVectorNf::Normalize() {
  float Abs = this->Absolute();
  for (int i = 0; i < N; i++)
    vec[i] /= Abs;
}

void cVectorNf::Normalize(cVectorNf &Vector) {
  float Abs = Vector.Absolute();
  for (int i = 0; i < N; i++)
    this->set(i, Vector.get(i) / Abs);
}

cVectorNf &cVectorNf::operator=(cVectorNf &rval) {
  set(rval);
  return *this;
}

cVectorNf &cVectorNf::operator+=(cVectorNf &rval) {
  for (int i = 0; i < this->getNumElements(); i++) {
    this->set(i, this->get(i) + rval.get(i));
  }
  return *this;
}

cVectorNf &cVectorNf::operator-=(cVectorNf &rval) {
  for (int i = 0; i < this->getNumElements(); i++) {
    this->set(i, this->get(i) - rval.get(i));
  }
  return *this;
}

cVectorNf &cVectorNf::operator-=(float scalar) {
  for (int i = 0; i < this->getNumElements(); i++) {
    this->set(i, this->get(i) - scalar);
  }
  return *this;
}

cVectorNf &cVectorNf::operator+=(float scalar) {
  for (int i = 0; i < this->getNumElements(); i++) {
    this->set(i, this->get(i) + scalar);
  }
  return *this;
}

cVectorNf &cVectorNf::operator*=(float scalar) {
  for (int i = 0; i < this->getNumElements(); i++) {
    this->set(i, this->get(i) * scalar);
  }
  return *this;
}
