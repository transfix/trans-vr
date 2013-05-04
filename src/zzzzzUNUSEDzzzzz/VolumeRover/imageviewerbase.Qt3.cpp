#include "../../inc/VolumeRover/imageviewerbase.Qt3.h"

#include <qvariant.h>
/*
 *  Constructs a ImageViewerBase as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 */
ImageViewerBase::ImageViewerBase(QWidget* parent, const char* name, Qt::WindowFlags fl)
    : Q3MainWindow(parent, name, fl)
{
    setupUi(this);

    (void)statusBar();
}

/*
 *  Destroys the object and frees any allocated resources
 */
ImageViewerBase::~ImageViewerBase()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void ImageViewerBase::languageChange()
{
    retranslateUi(this);
}

void ImageViewerBase::saveAsSlot()
{
    qWarning("ImageViewerBase::saveAsSlot(): Not implemented yet");
}

