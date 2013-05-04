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
#include <stdio.h>

void MSLevelSetDialog::on_DTInitComboBox_activated( int idx)
{
if(idx == 0)
    {
  m_EllipsoidPowerText->hide();
  m_EllipsoidPowerEdit->hide();
}
else
{
  m_EllipsoidPowerText->show();
  m_EllipsoidPowerEdit->show();
}

MSLSParams->init_interface_method = idx;
}


void MSLevelSetDialog::paramReference( MSLevelSetParams * mslsp )
{
  MSLSParams = mslsp;
  
  m_Lambda1Edit->setText(QString("%1").arg( MSLSParams->lambda1)); 
  m_Lambda2Edit->setText(QString("%1").arg(  MSLSParams->lambda2));
  m_MuEdit->setText(QString("%1").arg(  MSLSParams->mu));
  m_NuEdit->setText(QString("%1").arg(  MSLSParams->nu));
  m_EpsilonEdit->setText(QString("%1").arg(  MSLSParams->epsilon));
  m_DeltaTEdit->setText(QString("%1").arg(  MSLSParams->deltaT));
  m_MedianTolEdit->setText(QString("%1").arg(  MSLSParams->medianTolerance));
  m_MaxSolverIterEdit->setText(QString("%1").arg(  MSLSParams->nIter));
  m_DTWidthEdit->setText(QString("%1").arg(  MSLSParams->DTWidth));
  m_MaxMedianIterEdit->setText(QString("%1").arg(  MSLSParams->medianIter));
  m_SubvolDimEdit->setText(QString("%1").arg(  MSLSParams->subvolDim));
  m_BlockDimEdit->setText(QString("%1").arg(  MSLSParams->SDTBlockDim)); 
  m_EllipsoidPowerEdit->setText(QString("%1").arg(  MSLSParams->superEllipsoidPower)); 
  
  m_DTInitComboBox->setCurrentItem(MSLSParams->init_interface_method);
}


void MSLevelSetDialog::init()
{
  m_Lambda1Edit->setValidator(new QDoubleValidator(this));
  m_Lambda2Edit->setValidator(new QDoubleValidator(this));
  m_MuEdit->setValidator(new QDoubleValidator(this));
  m_NuEdit->setValidator(new QDoubleValidator(this));
  m_EpsilonEdit->setValidator(new QDoubleValidator(this));
  m_DeltaTEdit->setValidator(new QDoubleValidator(this));
  m_MedianTolEdit->setValidator(new QDoubleValidator(this));
  m_MaxSolverIterEdit->setValidator(new QIntValidator(this));
  m_DTWidthEdit->setValidator(new QIntValidator(this));
  m_MaxMedianIterEdit->setValidator(new QIntValidator(this));
  m_EllipsoidPowerEdit->setValidator(new QIntValidator(this));
  m_SubvolDimEdit->setValidator(new QIntValidator(this));
  m_BlockDimEdit->setValidator(new QIntValidator(this));
  m_BBoxOffsetEdit->setValidator(new QIntValidator(this));
  
  m_EllipsoidPowerText->hide();
  m_EllipsoidPowerEdit->hide();

  m_DTInitText->setText(QString("Distance field interface (%1) initialization").arg(QChar(0x03C6)));
  m_EpsilonText->setText(QString("%1").arg(QChar(0x03B5)));
}


void MSLevelSetDialog::on_BlockDimEdit_textChangedSlot()
{
    int blockDim = m_BlockDimEdit->text().toInt();
    switch(m_BlockDimComboBox->currentItem())
    {
	    case 0:
		MSLSParams->SDTBlockDim = blockDim;
		break;
	    case 1:
		MSLSParams->avgBlockDim = blockDim;
		break;
	    case 2:
		MSLSParams->medianBlockDim = blockDim;
		break;
	    case 3:
		MSLSParams->PDEBlockDim = blockDim;
		break;
}
}


void MSLevelSetDialog::on_DTWidth_textChangedSlot()
{
    MSLSParams->DTWidth = m_DTWidthEdit->text().toInt();
}


void MSLevelSetDialog::on_DeltaTEdit_textChangedSlot()
{
    MSLSParams->deltaT = m_DeltaTEdit->text().toDouble();
}


void MSLevelSetDialog::on_EllipsoidPowerEdit_textChangedSlot()
{
    MSLSParams->superEllipsoidPower = m_EllipsoidPowerEdit->text().toDouble();
}


void MSLevelSetDialog::on_EpsilonEdit_textChangedSlot()
{
    MSLSParams->epsilon = m_EpsilonEdit->text().toDouble();
}


void MSLevelSetDialog::on_Lambda1Edit_textChangedSlot()
{
    MSLSParams->lambda1 = m_Lambda1Edit->text().toDouble();
}


void MSLevelSetDialog::on_Lambda2Edit_textChangedSlot()
{
    MSLSParams->lambda2 = m_Lambda2Edit->text().toDouble();
}


void MSLevelSetDialog::on_MaxMedianIterEdit_textChangedSlot()
{
    MSLSParams->medianIter = m_MaxMedianIterEdit->text().toInt();
}


void MSLevelSetDialog::on_MaxSolverIterEdit_textChangedSlot()
{
    MSLSParams->nIter = m_MaxSolverIterEdit->text().toInt();
}


void MSLevelSetDialog::on_MedianTolEdit_textChangedSlot()
{
    MSLSParams->medianTolerance = m_MedianTolEdit->text().toDouble();
}


void MSLevelSetDialog::on_MuEdit_textChangedSLot()
{
    MSLSParams->mu = m_MuEdit->text().toDouble();
}


void MSLevelSetDialog::on_NuEdit_textChangedSLot()
{
    MSLSParams->nu = m_NuEdit->text().toDouble();
}


void MSLevelSetDialog::on_SubvolDimEdit_textChangedSlot()
{
    MSLSParams->subvolDim = m_SubvolDimEdit->text().toInt();
}


void MSLevelSetDialog::on_BlockDimComboBox_activatedSlot( int idx)
{
    switch(idx)
    {
	    case 0:
		m_BlockDimEdit->setText(QString("%1").arg(  MSLSParams->SDTBlockDim));
		break;
	    case 1:
		m_BlockDimEdit->setText(QString("%1").arg(  MSLSParams->avgBlockDim));
		break;
	    case 2:
		m_BlockDimEdit->setText(QString("%1").arg(  MSLSParams->medianBlockDim));
		break;
	    case 3:
		m_BlockDimEdit->setText(QString("%1").arg(  MSLSParams->PDEBlockDim));
		break;		
    }
}


void MSLevelSetDialog::on_BBoxOffsetEdit_textChangedSlot()
{
    MSLSParams->BBoxOffset = m_BBoxOffsetEdit->text().toInt();
}
