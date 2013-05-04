/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
             Vinay Siddavanahalli <skvinay@cs.utexas.edu>
    Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// OrthographicView.cpp: implementation of the OrthographicView class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/OrthographicView.h>
#include <VolumeWidget/Matrix.h>
#include <VolumeWidget/Ray.h>
#include <math.h>
#include <glew/glew.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

static void set(float* dest, float x, float y, float z, float w) {
	dest[0] = x;
	dest[1] = y;
	dest[2] = z;
	dest[3] = w;
}

static void set(float* dest, float* source) {
	dest[0] = source[0];
	dest[1] = source[1];
	dest[2] = source[2];
	dest[3] = source[3];
}

static void cross(float* dest, float* v1, float* v2) {
	dest[0] = v1[1]*v2[2] - v1[2]*v2[1];
	dest[1] = v1[2]*v2[0] - v1[0]*v2[2];
	dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
	dest[3] = 0.0f;
}

static void normalize(float* v)
{
	float len = (float)sqrt(
		v[0] * v[0] +
		v[1] * v[1] +
		v[2] * v[2]);
	v[0]/=len;
	v[1]/=len;
	v[2]/=len;
	v[3] = 0.0f;
}

static void sub( float* dest, float* v1, float* v2 )
{
	dest[0] = v1[0] - v2[0];
	dest[1] = v1[1] - v2[1];
	dest[2] = v1[2] - v2[2];
	dest[3] = v1[3] - v2[3];
}

static void sub( float* dest, float* v )
{
	dest[0] -= v[0];
	dest[1] -= v[1];
	dest[2] -= v[2];
	dest[3] -= v[3];
}

static void add( float* dest, float* v1, float* v2 )
{
	dest[0] = v1[0] + v2[0];
	dest[1] = v1[1] + v2[1];
	dest[2] = v1[2] + v2[2];
	dest[3] = v1[3] + v2[3];
}

static void add( float* dest, float* v )
{
	dest[0] += v[0];
	dest[1] += v[1];
	dest[2] += v[2];
	dest[3] += v[3];
}

static void linear( float* dest, float* v1, float* v2, float a, float b ) {
	dest[0] = a * v1[0] + b * v2[0];
	dest[1] = a * v1[1] + b * v2[1];
	dest[2] = a * v1[2] + b * v2[2];
	dest[3] = a * v1[3] + b * v2[3]; // = 0
}

/* matrix in memory:
   |d0|   |m00, m04, m08, m12|   |v0|
   |d1|   |m01, m05, m09, m13|   |v1|
   |d2| = |m02, m06, m10, m14| * |v2|
   |d3|   |m03, m07, m11, m15|   |v3|
*/
static void mul( float* d, float* m, float* v )
{
	d[0] = m[0]*v[0]+m[4]*v[1]+m[8]*v[2]+m[12]*v[3];
	d[1] = m[1]*v[0]+m[5]*v[1]+m[9]*v[2]+m[13]*v[3];
	d[2] = m[2]*v[0]+m[6]*v[1]+m[10]*v[2]+m[14]*v[3];
	d[3] = m[3]*v[0]+m[7]*v[1]+m[11]*v[2]+m[15]*v[3];
}

static void mulEquals( float* d, float* m )
{
	float result[4];
	mul(result, m, d);
	d[0] = result[0];
	d[1] = result[1];
	d[2] = result[2];
	d[3] = result[3];
}

static void copy( float* d, float* m)
{
	d[0] = m[0]; d[4] = m[4]; d[8] = m[8];   d[12] = m[12];
	d[1] = m[1]; d[5] = m[5]; d[9] = m[9];   d[13] = m[13];
	d[2] = m[2]; d[6] = m[6]; d[10] = m[10]; d[14] = m[14];
	d[3] = m[3]; d[7] = m[7]; d[11] = m[11]; d[15] = m[15];
}

/* matrix in memory:
   |d00, d04, d08, d12|   |m00, m04, m08, m12|   |n00, n04, n08, n12|
   |d01, d05, d09, d13|   |m01, m05, m09, m13|   |n01, n05, n09, n13|
   |d02, d06, d10, d14| = |m02, m06, m10, m14| * |n02, n06, n00, n14|
   |d03, d07, d11, d15|   |m03, m07, m11, m15|   |n03, n07, n11, n15|
*/
static void mulMatrixMatrix( float* d, float* m, float* n )
{
	d[0] = m[0]*n[0]+m[4]*n[1]+m[8]*n[2]+m[12]*n[3];
	d[1] = m[1]*n[0]+m[5]*n[1]+m[9]*n[2]+m[13]*n[3];
	d[2] = m[2]*n[0]+m[6]*n[1]+m[10]*n[2]+m[14]*n[3];
	d[3] = m[3]*n[0]+m[7]*n[1]+m[11]*n[2]+m[15]*n[3];

	d[4] = m[0]*n[4]+m[4]*n[5]+m[8]*n[6]+m[12]*n[7];
	d[5] = m[1]*n[4]+m[5]*n[5]+m[9]*n[6]+m[13]*n[7];
	d[6] = m[2]*n[4]+m[6]*n[5]+m[10]*n[6]+m[14]*n[7];
	d[7] = m[3]*n[4]+m[7]*n[5]+m[11]*n[6]+m[15]*n[7];

	d[8] = m[0]*n[8]+m[4]*n[9]+m[8]*n[10]+m[12]*n[11];
	d[9] = m[1]*n[8]+m[5]*n[9]+m[9]*n[10]+m[13]*n[11];
	d[10] = m[2]*n[8]+m[6]*n[9]+m[10]*n[10]+m[14]*n[11];
	d[11] = m[3]*n[8]+m[7]*n[9]+m[11]*n[10]+m[15]*n[11];

	d[12] = m[0]*n[12]+m[4]*n[13]+m[8]*n[14]+m[12]*n[15];
	d[13] = m[1]*n[12]+m[5]*n[13]+m[9]*n[14]+m[13]*n[15];
	d[14] = m[2]*n[12]+m[6]*n[13]+m[10]*n[14]+m[14]*n[15];
	d[15] = m[3]*n[12]+m[7]*n[13]+m[11]*n[14]+m[15]*n[15];
}

/* matrix in memory:
   |d00, d04, d08, d12|    |m00, m04, m08, m12|
   |d01, d05, d09, d13|    |m01, m05, m09, m13|
   |d02, d06, d10, d14| *= |m02, m06, m10, m14|
   |d03, d07, d11, d15|    |m03, m07, m11, m15|
*/
static void premulequalsMatrixMatrix( float* d, float* m )
{
	float n[16];
	mulMatrixMatrix( n, m, d);
	copy(d, n);
}

/* matrix in memory:
   |d00, d04, d08, d12|    |m00, m04, m08, m12|
   |d01, d05, d09, d13|    |m01, m05, m09, m13|
   |d02, d06, d10, d14| *= |m02, m06, m10, m14|
   |d03, d07, d11, d15|    |m03, m07, m11, m15|
*/
static void postmulequalsMatrixMatrix( float* d, float* m )
{
	float n[16];
	mulMatrixMatrix( n, d, m);
	copy(d, n);
}

OrthographicView::OrthographicView(float windowsize) : View(windowsize)
{
}

OrthographicView::OrthographicView(const ViewInformation& viewInformation)
{
	matchViewInformation(viewInformation);
}

View* OrthographicView::clone() const
{
	return new OrthographicView(*this);
}

OrthographicView::~OrthographicView()
{

}

void OrthographicView::SetView() {

	Matrix matrix = Matrix::translation(-m_Target);
	matrix.preMultiplication(m_Orientation.conjugate().buildMatrix());
	matrix.preMultiplication(Matrix::translation(0.0f, 0.0f, -1.0f));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (m_Height>m_Width) {
		glOrtho(-m_WindowSize/2.0, m_WindowSize/2.0, 
			-(m_WindowSize/2.0)*m_Height/m_Width, (m_WindowSize/2.0)*m_Height/m_Width, 
			-200.0,200.0);
	}
	else {
		glOrtho(-(m_WindowSize/2.0)*m_Width/m_Height, (m_WindowSize/2.0)*m_Width/m_Height, 
			-m_WindowSize/2.0, m_WindowSize/2.0, 
			-200.0,200.0);
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(matrix.getMatrix());

	/*float eyeDir[4];

	// step 1
	float matrix[16] = {1.0f,0.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,0.0f,
		0.0f,0.0f,1.0f,0.0f,
		-m_Eye[0],-m_Eye[1],-m_Eye[2],1.0f};

	sub(eyeDir, m_Target, m_Eye);
	normalize(eyeDir);

	// step 2
	float x = m_Up[0];
	float y = m_Up[1];
	float z = m_Up[2];

	float len = (float)sqrt(x*x+z*z);
	if (len!=0.0) {
		float yrotationMatrix[16] = {z/len, 0.0f, -(-x/len), 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			-(x/len), 0.0f, z/len, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f };
		premulequalsMatrixMatrix(matrix, yrotationMatrix);
	}

	// step 3
	float xrotationMatrix[16] = {1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, y, (-sqrt(1-y*y)), 0.0f,
		0.0f, (sqrt(1-y*y)), y, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f };
	premulequalsMatrixMatrix(matrix, xrotationMatrix);
	float newEye[4];
	mul(newEye, matrix, eyeDir);

	// step 4
	x = newEye[0];
	z = newEye[2];
	float yrotationMatrix2[16] = {-z, 0.0f, -(x), 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		-(-x), 0.0f, -z, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f };
	premulequalsMatrixMatrix(matrix, yrotationMatrix2);
	mulEquals(newEye, yrotationMatrix2);
	mul(newEye, matrix, eyeDir);
	qDebug("Transformed eye: %f, %f, %f", newEye[0], newEye[1], newEye[2]);
	mul(newEye, matrix, m_Up);
	qDebug("Transformed up: %f, %f, %f", newEye[0], newEye[1], newEye[2]);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (m_Height>m_Width) {
		glOrtho(-m_WindowSize/2.0, m_WindowSize/2.0, 
			-(m_WindowSize/2.0)*m_Height/m_Width, (m_WindowSize/2.0)*m_Height/m_Width, 
			-10.0,10.0);
	}
	else {
		glOrtho(-(m_WindowSize/2.0)*m_Width/m_Height, (m_WindowSize/2.0)*m_Width/m_Height, 
			-m_WindowSize/2.0, m_WindowSize/2.0, 
			-10.0,10.0);
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(matrix);*/

}

Ray OrthographicView::GetPickRay(int x, int y) const
{
	y = InvertY(y);
	float fx = (float)x-(float)m_Width/2.0f;
	float fy = (float)y-(float)m_Height/2.0f;
	float objectDx = fx * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);
	float objectDy = fy * m_WindowSize / (m_Height<m_Width?m_Height: m_Width);

	Matrix matrix = Matrix::translation(0.0f, 0.0f, 200.0f);
	matrix.preMultiplication(m_Orientation.buildMatrix());
	matrix.preMultiplication(Matrix::translation(m_Target));

	return Ray(matrix*Vector(objectDx,objectDy,0.0f,1.0f), matrix*Vector(0.0f,0.0f,-1.0f,0.0f));
}

Vector OrthographicView::GetScreenPoint(const Vector& p) const
{
	Matrix matrix = Matrix::translation(-m_Target);
	matrix.preMultiplication(m_Orientation.inverse().buildMatrix());

	Vector result = matrix*p;
	result[2] = 0.0;
	result[3] = 1.0;

	result[0] = result[0] / (m_WindowSize / (m_Height<m_Width?m_Height: m_Width));
	result[1] = result[1] / (m_WindowSize / (m_Height<m_Width?m_Height: m_Width));
	result[0] = (float)result[0]+(float)m_Width/2.0f;
	result[1] = (float)result[1]+(float)m_Height/2.0f;

	return result;
}

void OrthographicView::defaultTransformation( int xNew, int yNew ) {
	mousePan( xNew, yNew );
}

OrthographicView* OrthographicView::Top(float windowsize)
{
	OrthographicView* view = new OrthographicView(windowsize);
	return view;
}

OrthographicView* OrthographicView::Right(float windowsize)
{
	OrthographicView* view = new OrthographicView(windowsize);
	view->m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	view->m_Orientation.preMultiply(Quaternion::rotation(3.141592653f/2.0f, 0.0f, 0.0f, 1.0f));
	return view;
}

OrthographicView* OrthographicView::Left(float windowsize)
{
	OrthographicView* view = new OrthographicView(windowsize);
	view->m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	view->m_Orientation.preMultiply(Quaternion::rotation(-3.141592653f/2.0f, 0.0f, 0.0f, 1.0f));
	return view;
}

OrthographicView* OrthographicView::Bottom(float windowsize)
{
	OrthographicView* view = new OrthographicView(windowsize);
	view->m_Orientation = Quaternion::rotation(3.141592653f, 1.0f, 0.0f, 0.0f);
	return view;

}

OrthographicView* OrthographicView::Front(float windowsize)
{
	OrthographicView* view = new OrthographicView(windowsize);
	view->m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	return view;
}

OrthographicView* OrthographicView::Back(float windowsize	)
{
	OrthographicView* view = new OrthographicView(windowsize);
	view->m_Orientation = Quaternion::rotation(3.141592653f/2.0f, 1.0f, 0.0f, 0.0f);
	view->m_Orientation.preMultiply(Quaternion::rotation(3.141592653f, 0.0f, 0.0f, 1.0f));
	return view;
}

ViewInformation OrthographicView::getViewInformation() const
{
	return ViewInformation(m_Target, m_Orientation, m_WindowSize, 0.0f);
}

void OrthographicView::matchViewInformation(const ViewInformation& viewInformation)
{
	m_Target = viewInformation.getTarget();
	m_Orientation = viewInformation.getOrientation();
	m_WindowSize = viewInformation.getWindowSize();
}

Matrix OrthographicView::getModelViewMatrix() const
{
	Matrix matrix = Matrix::translation(-m_Target);
	matrix.preMultiplication(m_Orientation.conjugate().buildMatrix());
	matrix.preMultiplication(Matrix::translation(0.0f, 0.0f, -1.0f));

	return matrix;
}





