/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you want to add, delete, or rename functions or slots, use
** Qt Designer to update this file, preserving your code.
**
** You should not define a constructor or destructor in this file.
** Instead, write your code in functions called init() and destroy().
** These will automatically be called by the form's constructor and
** destructor.
*****************************************************************************/
#include <qvalidator.h>

void BilateralFilterDialog::init()
{
    m_RadSigEdit->setValidator(new QDoubleValidator(m_RadSigEdit));
    m_SpatSigEdit->setValidator(new QDoubleValidator(m_SpatSigEdit));
    m_FilRadEdit->setValidator(new QIntValidator(m_FilRadEdit));
}
