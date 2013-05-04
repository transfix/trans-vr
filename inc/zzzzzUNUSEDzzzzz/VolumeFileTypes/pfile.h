#ifndef PFILE_H
#define PFILE_H

#include <qstring.h>
#include <sys/types.h>

// make sure we're using 64-bit file i/o
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#ifndef O_LARGEFILE
#define O_LARGEFILE 0
#endif


// IO_* defines cribbed from qiodevice.h
//
// IO handling modes
#define IO_Raw      0x0040    // raw access (not buffered)
#define IO_Async    0x0080    // asynchronous mode
// IO device open modes
#define IO_ReadOnly   0x0001    // readable device
#define IO_WriteOnly    0x0002    // writable device
#define IO_ReadWrite    0x0003    // read+write device
#define IO_Append   0x0004    // append
#define IO_Truncate   0x0008    // truncate device
#define IO_Translate    0x0010    // translate CR+LF
#define IO_ModeMask   0x00ff

///\class PFile pfile.h
///\author John Wiggins
///\brief PFile is a drop-in replacement for QFile that supports large files
///	on unixy platforms (linux and MacOS X have been tested). It supports just
///	enough of QFile's member functions so that code that calls QFile within the
///	VolumeFileTypes library will compile against it. This code can safely be
///	put out to pasture after Volume Rover moves to Qt 4.0. Refer to the QFile
///	documentation if you want to know what the functions do in this class.
class PFile {
public:
					PFile();
					PFile(const QString & name);
					~PFile();

	void		setName(const QString & name);

	bool		exists() const;
	bool		open(int mode);

	void		close();
	void		flush();

	off_t		size() const;
	//off_t		at() const;
	bool		at(off_t offset);

	long		readBlock(char *data, unsigned long len);
	long		writeBlock(const char *data, unsigned long len);

private:

	QString	m_Name;
	int m_Fd;
};

#endif

