#include <VolumeRover/reconstructiondialogimpl.h>
#include <qlineedit.h>
#include <q3filedialog.h>
#include <qpushbutton.h>
#include <q3groupbox.h>
#include <XmlRPC/XmlRpc.h>

//using namespace XmlRpc

ReconstructionImpl::ReconstructionImpl( QWidget* parent,  const char* name, bool modal, Qt::WFlags fl )
    : ReconstructionDialog( parent, name, modal, fl )
{}


ReconstructionImpl::~ReconstructionImpl()
{}

/*
void ReconstructionImpl::browseSlot()
{
        QDir dir(m_lineEditDir->text());
        m_lineEditDir->setText(presentDirDialog());
}

*/

//QString  ReconstructionImpl::presentDirDialog()
//{
/*       QStrings = QFileDialog::getExistingDirectory(
                        defaultDir.absPath(),
                        this, "Please choose the location of the projection .sel files directory.",
                        QString("Please choose the location of the .sel files directory"), TRUE );*/

//         QString s = QFileDialog::getOpenFileName("/h1/liming/data/*.sel","sel files (*.sel is a xmipp format)",
//                                             this, "open file dialog ", "Choose a file");
//if (s.isEmpty())
//return s;


//}

void ReconstructionImpl::browseSlot()
{
        QDir dir(m_lineEditDir->text());
        m_lineEditDir->setText(presentDirDialog());
}          

void ReconstructionImpl::browseloadinitfSlot()
{
        QDir dir(m_LoadInitF->text());
        m_LoadInitF->setText(presentDirDialog1());
}          

/*
void ReconstructionImpl::remoteReconstruction()
{
	m_XmlRpcClient(dialog->HostName->text(), dialog->PortName->text().toInt());
	
	m_Params[0] = 

	
}
*/




QString  ReconstructionImpl::presentDirDialog()
{ 
    
//         QString s = QFileDialog::getOpenFileName("/h1/liming/data/*.sel","sel files (*.sel is a xmipp format)",
  //                                           this, "open file dialog ", "Choose a file");

         QString s = Q3FileDialog::getOpenFileName("*.sel","sel files (*.sel is a xmipp format)",
                                             this, "open file dialog ", "Choose a file");
if (s.isEmpty())
return s; 
    
    
}   


QString  ReconstructionImpl::presentDirDialog1()
{ 
    
//         QString s = QFileDialog::getOpenFileName("/h1/liming/data/*.rawiv","rawiv files (*.rawiv is a VolumeRover format)",
//                                             this, "open file dialog ", "Choose a file");

         QString s = Q3FileDialog::getOpenFileName("*.rawiv","rawiv files (*.rawiv is a VolumeRover format)",
                                             this, "open file dialog ", "Choose a file");
if (s.isEmpty())
return s; 
    
    
}   

void ReconstructionImpl::changeParamsValuesSlot(int loc)
{
        switch(loc)
        {
        case 0:
        m_paramsGroup->setEnabled(false);

        break;
        case 1:
        m_paramsGroup->setEnabled(true);
        break;
        }
}

void ReconstructionImpl::changeReconMannerSlot(int loc)
{ 
        switch(loc)
        {
        case 0:
        m_loadGroup->setEnabled(true);
        m_testGroup->setEnabled(false);

        break;
        case 1:
        m_loadGroup->setEnabled(false);
        m_testGroup->setEnabled(true);
        break;
        }
}



//for ordered subset.subSetN
void ReconstructionImpl::changeStatusSlot()
{
       //m_SubSetEdit->setEnabled(!m_SubSetEdit->isEnabled());
       m_LoadInitF->setEnabled(!m_LoadInitF->isEnabled());

}
