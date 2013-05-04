#include <VolumeFileTypes/pfile.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

PFile::PFile()
{
	m_Name = QString::null;
	m_Fd = -1;
}

PFile::PFile(const QString & name)
: m_Name(name)
{
	m_Fd = -1;
}

PFile::~PFile()
{
	close();
}

void PFile::setName(const QString & name)
{
	if (m_Fd > 0)
		close();

	m_Name = name;
}

bool PFile::exists() const
{
	if (m_Name.isEmpty())
		return false;

	// if stat succeeds, then it probably exists...
	struct stat st;
	int ret;

	ret = stat(m_Name.ascii(), &st);

	return (ret != -1);
}

bool PFile::open(int mode)
{
	// already open
	if (m_Fd > 0)
		return false;

	// no name
	if (m_Name.isEmpty())
		return false;

	// open
	if (mode & IO_ReadOnly)
		m_Fd = ::open(m_Name.ascii(), O_LARGEFILE|O_RDONLY);
	else if (mode & IO_WriteOnly)
		m_Fd = ::open(m_Name.ascii(), O_LARGEFILE|O_WRONLY|O_CREAT, 0644);

	return (m_Fd != -1);
}

void PFile::close()
{
	flush();

	if (m_Fd > 0)
		::close(m_Fd);

	m_Fd = -1;
	m_Name = QString::null;
}

void PFile::flush()
{
	if (m_Fd > 0)
		fsync(m_Fd);
}

off_t PFile::size() const
{
	off_t size = -1;

	if (m_Fd != -1)
	{
		struct stat st;
		if (-1 != fstat(m_Fd, &st))
			size = st.st_size;
	}

	return size;
}

/*off_t PFile::at() const
{
	// seek 0 bytes from current position
	// lseek returns the current position
	if (m_Fd > 0)
		return lseek(m_Fd, 0, SEEK_CUR);
	
	return -1;
}*/

bool PFile::at(off_t offset)
{
	// no file. no seek.
	if (m_Fd < 0)
		return false;

	// lseek returns offset on success
	return (lseek(m_Fd, offset, SEEK_SET) == offset);
}

long PFile::readBlock(char *data, unsigned long len)
{
	if (m_Fd < 0)
		return 0;

	return (long)read(m_Fd, data, len);
}

long PFile::writeBlock(const char *data, unsigned long len)
{
	if (m_Fd < 0)
		return 0;

	return (long)write(m_Fd, data, len);
}

