/**********************************************************************

Copyright (C) 2002-2025 Gilles Debunne. All rights reserved.

This file is part of the QGLViewer library version 3.0.0.

https://gillesdebunne.github.io/libQGLViewer - contact@libqglviewer.com

This file is part of a free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

**********************************************************************/

#include <QGLViewer/mouseGrabber.h>

using namespace qglviewer;

// Static private variable
QList<MouseGrabber *> MouseGrabber::MouseGrabberPool_;

/*! Default constructor.

Adds the created MouseGrabber in the MouseGrabberPool(). grabsMouse() is set
to \c false. */
MouseGrabber::MouseGrabber() : grabsMouse_(false) { addInMouseGrabberPool(); }

/*! Adds the MouseGrabber in the MouseGrabberPool().

All created MouseGrabber are automatically added in the MouseGrabberPool() by
the constructor. Trying to add a MouseGrabber that already
isInMouseGrabberPool() has no effect.

Use removeFromMouseGrabberPool() to remove the MouseGrabber from the list, so
that it is no longer tested with checkIfGrabsMouse() by the QGLViewer, and
hence can no longer grab mouse focus. Use isInMouseGrabberPool() to know the
current state of the MouseGrabber. */
void MouseGrabber::addInMouseGrabberPool() {
  if (!isInMouseGrabberPool())
    MouseGrabber::MouseGrabberPool_.append(this);
}

/*! Removes the MouseGrabber from the MouseGrabberPool().

See addInMouseGrabberPool() for details. Removing a MouseGrabber that is not
in MouseGrabberPool() has no effect. */
void MouseGrabber::removeFromMouseGrabberPool() {
  if (isInMouseGrabberPool())
    MouseGrabber::MouseGrabberPool_.removeAll(
        const_cast<MouseGrabber *>(this));
}

/*! Clears the MouseGrabberPool().

 Use this method only if it is faster to clear the MouseGrabberPool() and then
 to add back a few MouseGrabbers than to remove each one independently. Use
 QGLViewer::setMouseTracking(false) instead if you want to disable mouse
 grabbing.

 When \p autoDelete is \c true, the MouseGrabbers of the MouseGrabberPool()
 are actually deleted (use this only if you're sure of what you do). */
void MouseGrabber::clearMouseGrabberPool(bool autoDelete) {
  if (autoDelete)
    qDeleteAll(MouseGrabber::MouseGrabberPool_);
  MouseGrabber::MouseGrabberPool_.clear();
}
