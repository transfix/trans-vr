#ifndef _RECONSTRUCTIONIMPL_H_
#define _RECONSTRUCTIONIMPL_H_

#include "reconstructiondialog.h"
#include <qdir.h>

class ReconstructionImpl:public ReconstructionDialog
{
   Q_OBJECT

public:
    ReconstructionImpl( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, Qt::WFlags fl = 0 );
    ~ReconstructionImpl();

    virtual void browseSlot();
    QString presentDirDialog();
    QString presentDirDialog1();

    virtual void browseloadinitfSlot();

    void changeParamsValuesSlot(int loc);
    void changeReconMannerSlot(int loc);

    void changeStatusSlot();



};






#endif
