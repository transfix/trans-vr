/*
  Copyright 2002-2003 The University of Texas at Austin

        Authors: Anthony Thane <thanea@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeLibrary.

  VolumeLibrary is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeLibrary is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// VolumeRenderer.cpp: implementation of the VolumeRenderer class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRenderer/Renderer.h>
#include <VolumeRenderer/VolumeRenderer.h>
#include <cmath>
#include <cstdio>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeRenderer::VolumeRenderer() {
  m_PrivateRenderer = new OpenGLVolumeRendering::Renderer;
}

VolumeRenderer::VolumeRenderer(const VolumeRenderer &copy)
    : m_PrivateRenderer(
          new OpenGLVolumeRendering::Renderer(*(copy.m_PrivateRenderer))) {}

VolumeRenderer &VolumeRenderer::operator=(const VolumeRenderer &copy) {
  if (this != &copy) {
    delete m_PrivateRenderer;
    m_PrivateRenderer =
        new OpenGLVolumeRendering::Renderer(*(copy.m_PrivateRenderer));
  }
  return *this;
}

VolumeRenderer::~VolumeRenderer() { delete m_PrivateRenderer; }

bool VolumeRenderer::initRenderer() {
  return m_PrivateRenderer->initRenderer();
}

bool VolumeRenderer::setAspectRatio(double ratioX, double ratioY,
                                    double ratioZ) {
  return m_PrivateRenderer->setAspectRatio(ratioX, ratioY, ratioZ);
}

bool VolumeRenderer::setTextureSubCube(double minX, double minY, double minZ,
                                       double maxX, double maxY,
                                       double maxZ) {
  return m_PrivateRenderer->setTextureSubCube(minX, minY, minZ, maxX, maxY,
                                              maxZ);
}

bool VolumeRenderer::setQuality(double quality) {
  return m_PrivateRenderer->setQuality(quality);
}

double VolumeRenderer::getQuality() const {
  return m_PrivateRenderer->getQuality();
}

bool VolumeRenderer::setMaxPlanes(int maxPlanes) {
  return m_PrivateRenderer->setMaxPlanes(maxPlanes);
}

int VolumeRenderer::getMaxPlanes() const {
  return m_PrivateRenderer->getMaxPlanes();
}

bool VolumeRenderer::setNearPlane(double nearPlane) {
  return m_PrivateRenderer->setNearPlane(nearPlane);
}

double VolumeRenderer::getNearPlane() {
  return m_PrivateRenderer->getNearPlane();
}

bool VolumeRenderer::isShadedRenderingAvailable() const {
  return m_PrivateRenderer->isShadedRenderingAvailable();
}

bool VolumeRenderer::enableShadedRendering() {
  return m_PrivateRenderer->enableShadedRendering();
}

bool VolumeRenderer::disableShadedRendering() {
  return m_PrivateRenderer->disableShadedRendering();
}

void VolumeRenderer::setLight(float *lightf) {
  m_PrivateRenderer->setLight(lightf);
}

void VolumeRenderer::setView(float *viewf) {
  m_PrivateRenderer->setView(viewf);
}

bool VolumeRenderer::uploadColorMappedData(const GLubyte *data, int width,
                                           int height, int depth) {
  return m_PrivateRenderer->uploadColorMappedData(data, width, height, depth);
}

bool VolumeRenderer::uploadColorMappedDataWithBorder(const GLubyte *data,
                                                     int width, int height,
                                                     int depth) {
  return m_PrivateRenderer->uploadColorMappedDataWithBorder(data, width,
                                                            height, depth);
}

bool VolumeRenderer::testColorMappedData(int width, int height, int depth) {
  return m_PrivateRenderer->testColorMappedData(width, height, depth);
}

bool VolumeRenderer::testColorMappedDataWithBorder(int width, int height,
                                                   int depth) {
  return m_PrivateRenderer->testColorMappedDataWithBorder(width, height,
                                                          depth);
}

bool VolumeRenderer::uploadRGBAData(const GLubyte *data, int width,
                                    int height, int depth) {
  return m_PrivateRenderer->uploadRGBAData(data, width, height, depth);
}

bool VolumeRenderer::uploadGradients(const GLubyte *data, int width,
                                     int height, int depth) {
  return m_PrivateRenderer->uploadGradients(data, width, height, depth);
}

bool VolumeRenderer::calculateGradientsFromDensities(const GLubyte *data,
                                                     int width, int height,
                                                     int depth) {
  return m_PrivateRenderer->calculateGradientsFromDensities(data, width,
                                                            height, depth);
}

bool VolumeRenderer::uploadColorMap(const GLubyte *colorMap) {

  // arand: trying to correct based on the number of planes
  GLubyte cm[256 * 4];
  for (int c = 0; c < 256 * 4; c++) {
    cm[c] = colorMap[c];
  }

#if 0 // valgrind is complaining about this chunk of code for some reason,
      // disabling temporarily - transfix - 05/11/2012
  for (int c=0; c<256; c++) {
    double qual = getQuality();
    qual -= .1;
    if (qual <0.0) qual =0.0;
    //cm[4*c+3] *= sqrt(1.0-qual);
    cm[4*c+3] *= sqrt(cm[4*c+3])*(1.0-(qual*qual));
    //cm[4*c+3] *= 1-qual;

  }
#endif

  return m_PrivateRenderer->uploadColorMap(cm);
}

bool VolumeRenderer::uploadColorMap(const GLfloat *colorMap) {
  // arand: trying to correct based on the number of planes
  GLfloat cm[256 * 4];
  for (int c = 0; c < 256 * 4; c++) {
    cm[c] = colorMap[c];
  }

  for (int c = 0; c < 256; c++) {
    double qual = getQuality();
    qual -= .1;
    if (qual < 0.0)
      qual = 0.0;
    // cm[4*c+3] *= sqrt(1.0-qual);
    cm[4 * c + 3] *= sqrt(cm[4 * c + 3]) * (1.0 - (qual * qual));
    // cm[4*c+3] *= 1-qual;
  }
  return m_PrivateRenderer->uploadColorMap(cm);
}

int VolumeRenderer::getNumberOfPlanesRendered() const {
  return m_PrivateRenderer->getNumberOfPlanesRendered();
}

bool VolumeRenderer::renderVolume() {
  return m_PrivateRenderer->renderVolume();
}
