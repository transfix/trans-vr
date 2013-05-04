// SdsFileImpl.cpp: implementation of the SdsFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/SdsFileImpl.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SdsFileImpl::SdsFileImpl()
{
	m_Attached = false;
}

SdsFileImpl::~SdsFileImpl()
{
	close();
}
/*
//! Returns true if the file given by fileName is of the correct type.
bool SdsFileImpl::checkType(const QString& fileName)
{
	// go through the file and make sure we understand it
	QFileInfo fileInfo(fileName);

	// first check to make sure it exists
	if (!fileInfo.exists()) {
		qDebug("File does not exist");
		return false;
	}

	QString absFilePath = fileInfo.absFilePath();

	QFile file(absFilePath);
	if (!file.open(IO_ReadOnly | IO_Raw)) {
		qDebug("Error opening file");
		return false;
	}

	// read the header
	MrcHeader header;
	file.readBlock((char*)&header, sizeof(MrcHeader));
	// swap the 56, 4-byte, values
	if (isLittleEndian()) swapByteOrder((int*)&header, 56);

	// check for the details we dont support
	if (header.mode<0 ||  header.mode>2) {
		// we dont support this type or MRC file for now		
		return false;
	}

	unsigned int sizes[] = {1, 2, 4};

	// check the fileSize
	if (sizes[header.mode]*header.nx*header.ny*header.nz + 1024 != fileInfo.size()) {
		// the size does not match the header information
		return false;
	}

	// everything checks out, return true
	return true;
}

//! Associates this reader with the file given by fileName.
bool SdsFileImpl::attachToFile(const QString& fileName)
{
	QFileInfo fileInfo(fileName);

	// first check to make sure it exists
	if (!fileInfo.exists()) {
		qDebug("File does not exist");
		return false;
	}

	QString absFilePath = fileInfo.absFilePath();

	m_File.setName(absFilePath);
	if (!m_File.open(IO_ReadOnly | IO_Raw)) {
		qDebug("Error opening file");
		return false;
	}

	if (readHeader(fileInfo.size())) {
		// success
		m_Attached = true;
		return true;
	}
	else {
		// failure
		close();
		return false;
	}
}

//! Reads char data from the file into the supplied buffer
bool SdsFileImpl::readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable)
{
	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Char) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)numSamples != m_File.readBlock((char*)buffer, numSamples)) {
		return false;
	}

	incrementPosition(numSamples);
	return true;
}

//! Reads short data from the file into the supplied buffer
bool SdsFileImpl::readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable)
{
	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Short) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(short)) != m_File.readBlock((char*)buffer, numSamples*sizeof(short))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads long data from the file into the supplied buffer
bool SdsFileImpl::readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable)
{
	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Long) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(unsigned int)) != m_File.readBlock((char*)buffer, numSamples*sizeof(unsigned int))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads float data from the file into the supplied buffer
bool SdsFileImpl::readFloatData(float* buffer, unsigned int numSamples, unsigned int variable)
{
	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Float) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(float)) != m_File.readBlock((char*)buffer, numSamples*sizeof(float))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Reads double data from the file into the supplied buffer
bool SdsFileImpl::readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable)
{
	if (variable>=m_NumVariables ||
		m_VariableTypes[variable]!=Double) {
		// no good
		return false;
	}

	prepareToRead(variable);

	// read in the data
	if ((int)(numSamples*sizeof(double)) != m_File.readBlock((char*)buffer, numSamples*sizeof(double))) {
		return false;
	}
	if (isLittleEndian()) swapByteOrder(buffer, numSamples);

	incrementPosition(numSamples);
	return true;
}

//! Returns a new loader which can be used to load a file
VolumeFile* SdsFileImpl::getNewVolumeFileLoader() const
{
	return new MrcFileImpl;
}

//! Sets the position of the next read.
bool SdsFileImpl::protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset)
{
	if (variable>=m_NumVariables || timeStep>=m_NumTimeSteps) {
		return false;
	}
	else {
		m_File.at(1024 + (m_DimX*m_DimY*m_DimZ*timeStep + offset)*getBytesPerPixel(variable));
		return true;
	}
}

int readHeader(QFile& file)
//bool SdsFileImpl::readHeader(unsigned int fileSize)
{
	sdsFile fileStructure;

	//-------------------------------------------------------------------
	long infoMaxLineSize=512;

	char *fileInfo=new char[infoMaxLineSize];
	fgets(fileInfo,infoMaxLineSize, fd);

	char *sdsID="SAIL DATA SET version=";
	if(strncmp(fileInfo,sdsID,strlen(sdsID)))
	{	
		delete [] fileInfo;
		return 2;
	}

	//-------------------------------------------------------------------
	long sdsVersion=0;
	long nDataSets=0;
	char p1[50];
	char p2[50];

	sscanf(&fileInfo[strlen(sdsID)],"%d%s%s",&sdsVersion,p1,p2);

	char *cmd1="infoMSaxLineSize=";
	char *cmd2="nDataSets=";

	if(!strncmp(p1,cmd1,strlen(cmd1)))sscanf(&p1[strlen(cmd1)],"%d",&infoMaxLineSize);
	if(!strncmp(p2,cmd1,strlen(cmd1)))sscanf(&p2[strlen(cmd1)],"%d",&infoMaxLineSize);

	if(!strncmp(p1,cmd2,strlen(cmd2)))sscanf(&p1[strlen(cmd2)],"%d",&nDataSets);
	if(!strncmp(p2,cmd2,strlen(cmd2)))sscanf(&p2[strlen(cmd2)],"%d",&nDataSets);

	long sdsError=0;
	if(sdsVersion!=2	)sdsError=3;
	if(infoMaxLineSize<512	)sdsError=4;
	if(nDataSets<1		)sdsError=5;
	if(sdsError)
	{	free(fileInfo);
		delete [] fileInfo;
		return sdsError;
	}

	//-------------------------------------------------------------------	
	fileStructure.fileInfo=fileInfo;
	
	char *userInfo=new char[infoMaxLineSize];
	fgets(userInfo,infoMaxLineSize, fd);
	fileStructure.userInfo=userInfo;

	char *creationInfo=new char[infoMaxLineSize];
	fgets(creationInfo,infoMaxLineSize, fd);
	fileStructure.creationInfo=creationInfo;

	sdsDataHeader 	*dh;
	sdsDataPointers	*ptr;
	size_t	nitems;

	//------------------------------------------------------------------
	fileStructure.dh = new sdsDataHeader[nDataSets];
	fileStructure.ptr =new sdsDataPointers[nDataSets];

	if (nDataSets>1) {
		qDebug("Warning, only reading in the first dataset);
	}

	for(int i=0;i<nDataSets;i++)
	{	
		dh=&fileStructure.dh[i];
		ptr=&fileStructure.ptr[i];

		// read in the header
		fread(dh,1,sizeof(sdsDataHeader),fd);
		if (isLittleEndian()) swapByteOrder((int*)dh, 11);

		if(dh->typeDef==sdsLattice) {
			nitems=(size_t)dh->df.sdsLattice.nDesc;
			if(nitems)
			{	ptr->df.sdsLattice.desc = (char *)malloc(nitems);
				fread(ptr->df.sdsLattice.desc,1,nitems,fd);
			}

			nitems=sizeof(int)*dh->df.sdsLattice.nDim;
			ptr->df.sdsLattice.dims = (int *)malloc(nitems);
			fread(ptr->df.sdsLattice.dims,1,nitems,fd);
			
			nitems=(size_t)dh->df.sdsLattice.nData;
			if(nitems)
			{	ptr->df.sdsLattice.data = malloc(nitems);
				fread(ptr->df.sdsLattice.data,1,nitems,fd);
			}

			nitems=(size_t)dh->df.sdsLattice.nCoord;
			if(nitems)
			{	ptr->df.sdsLattice.coord = (float *)malloc(nitems);
				fread(ptr->df.sdsLattice.coord,1,nitems,fd);
			}

		}
		else { // not a lattice, not supported
			qDebug("The volume rover only supports sds lattice files");
		}
	}


	// read the header
	MrcHeader header;
	m_File.readBlock((char*)&header, sizeof(MrcHeader));
	// swap the 56, 4-byte, values
	if (isLittleEndian()) swapByteOrder((int*)&header, 56);

	// check for the details we dont support
	if (header.mode<0 ||  header.mode>2) {
		// we dont support this type or MRC file for now		
		return false;
	}

	m_DimX = header.nx;
	m_DimY = header.ny;
	m_DimZ = header.nz;
	m_NumTimeSteps = 1;
	makeSpaceForVariablesInfo(1);
	m_VariableNames[0] = "No Name";
	Type types[] = {Char, Short, Float};
	unsigned int sizes[] = {1, 2, 4};
	m_VariableTypes[0] = types[header.mode];
	m_MinX = 0.0;
	m_MinY = 0.0;
	m_MinZ = 0.0;
	m_MinT = 0.0;

	// we need to double check the meaning of xlength
	if (header.xlength<=0.0 || header.ylength<=0.0 || header.zlength<=0.0) {
		// hmm, this is wierd
		m_MaxX = m_MinX + m_DimX*1.0;
		m_MaxY = m_MinY + m_DimY*1.0;
		m_MaxZ = m_MinZ + m_DimZ*1.0;
	}
	else {
		m_MaxX = m_MinX + header.xlength;
		m_MaxY = m_MinY + header.ylength;
		m_MaxZ = m_MinZ + header.zlength;
	}
	m_MaxT = 0.0;

	// check the fileSize
	if (sizes[header.mode]*header.nx*header.ny*header.nz +1024 != fileSize) {
		// the size does not match the header information
		return false;
	}

	// everything checks out, return true
	return true;
}

void SdsFileImpl::close()
{
	if (m_Attached) {
		m_File.close();
		m_Attached = false;
	}
}
*/
