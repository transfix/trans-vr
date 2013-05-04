// RawIVTestRenderable.cpp: implementation of the RawIVTestRenderable class.
//
//////////////////////////////////////////////////////////////////////

#include <AnimationMaker/RawIVTestRenderable.h>
#include <ByteOrder/ByteSwapping.h>
#include <stdio.h>
#include <qwidget.h>

unsigned int counter=0;


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

struct RawVHeader {
	RawVHeader() { numvars = 0; }
	~RawVHeader()
	{
		for (unsigned int i=0; i < numvars; i++)
			delete [] names[i];
		delete [] names;
		delete [] types;
	}
	
	unsigned int magic;
	unsigned int dim[3];
	unsigned int timesteps;
	unsigned int numvars;
	float min[4];
	float max[4];
	unsigned char *types;
	char **names;
};

static RawVHeader* read_rawv_header(FILE *fp)
{
	unsigned int i;

	// create the header data struct
	RawVHeader* header = new RawVHeader;
	// read the fixed size part of the header
	fread(header, 1, 56, fp);
	counter+=1*56;
	// swap the fixed size part of the header
	if (isLittleEndian()) swapByteOrder((int*) header, 14);

	// check for our magic value
	if (header->magic != 0xBAADBEEF)
		return 0;



	// allocate more memory within the header struct
	header->types = new unsigned char [header->numvars];
	header->names = new char * [header->numvars];

	// read the rest of the header
	for (i=0; i < header->numvars; i++)
	{
		// allocate memory for the name
		header->names[i] = new char [64];
		// read data
		fread(&(header->types[i]), 1, 1, fp);
		counter+=1*1;
		fread(header->names[i], 1, 64, fp);
		counter+=1*64;
	}
	return header;
}

RawIVTestRenderable::RawIVTestRenderable() :m_FileName("NoName")
{

}

RawIVTestRenderable::~RawIVTestRenderable()
{

}


bool RawIVTestRenderable::loadFile(const char* fileName)
{
	FILE *fp;
	float *Dvol, minD=1000.0, maxD=-1000.0;
	unsigned char *Rvol, *Gvol, *Bvol;
	GLubyte *RGBAvol;
	float span[3];
	unsigned int dim[3], i;
	RawVHeader *header;

	fp = fopen(fileName, "rb");
	if (fp == NULL)
	{
		qDebug("unable to open %s\n", fileName);
		return false;
	}

	header = read_rawv_header(fp);
	if (header == NULL)
	{
		qDebug("%s is not a valid RawV file\n", fileName);
		return false;
	}
	
	qDebug("RawV file:");
	qDebug("\tdim: (%d,%d,%d)", header->dim[0], header->dim[1], header->dim[2]);
	qDebug("\t# timesteps: %d", header->timesteps);
	qDebug("\t# vars: %d", header->numvars);
	qDebug("\tmin: (%3.3f,%3.3f,%3.3f,%3.3f)", header->min[0], header->min[1],
						header->min[2], header->min[3]);
	qDebug("\tmax: (%3.3f,%3.3f,%3.3f,%3.3f)", header->max[0], header->max[1],
						header->max[2], header->max[3]);
	for (i=0; i < header->numvars; i++)
	{
		qDebug("\ttype: %d", header->types[i]);
		qDebug("\tname: \'%s\'", header->names[i]);
	}

	for (i=0; i < 3; i++)
	{
		dim[i] = header->dim[i];
		span[i] = header->max[i] - header->min[i];
	}

	// read the volumes
	for (i=0; i < header->numvars; i++)
	{
		if (strcmp("red", header->names[i]) == 0)
		{
			Rvol = new unsigned char [dim[0]*dim[1]*dim[2]];
			fread(Rvol, sizeof(unsigned char), dim[0]*dim[1]*dim[2], fp);
			counter+=1*dim[0]*dim[1]*dim[2];
		}
		else if (strcmp("green", header->names[i]) == 0)
		{
			Gvol = new unsigned char [dim[0]*dim[1]*dim[2]];
			fread(Gvol, sizeof(unsigned char), dim[0]*dim[1]*dim[2], fp);
			counter+=1*dim[0]*dim[1]*dim[2];
		}
		else if (strcmp("blue", header->names[i]) == 0)
		{
			Bvol = new unsigned char [dim[0]*dim[1]*dim[2]];
			fread(Bvol, sizeof(unsigned char), dim[0]*dim[1]*dim[2], fp);
			counter+=1*dim[0]*dim[1]*dim[2];
		}
		else if (strcmp("gaussian electron density approximation",
											header->names[i]) == 0)
		{
			Dvol = new float [dim[0]*dim[1]*dim[2]];
			//fseek(fp, 0*(56+65*4 + 3*256*256*256), SEEK_SET);
			fread(Dvol, sizeof(float), dim[0]*dim[1]*dim[2],fp);
			if (isLittleEndian()) swapByteOrder(Dvol, dim[0]*dim[1]*dim[2]);
			counter+=4*dim[0]*dim[1]*dim[2];
		}
	}
	
	// close the file
	fclose(fp);
	
	for (i=0; i < dim[0]*dim[1]*dim[2]; i++)
	{
		if (minD > Dvol[i]) 
			minD = Dvol[i];
		if (maxD < Dvol[i]) 
			maxD = Dvol[i];
	}

	RGBAvol = new GLubyte [dim[0]*dim[1]*dim[2]*4];
	for (i=0; i < dim[0]*dim[1]*dim[2]; i++)
	{
		float scale = (Dvol[i] - minD) / (maxD-minD);
#define CLAMP1(x) (((x) > 255.0) ? 255.0 : (x))
#define CLAMP2(x) (((x) > 1.0) ? 1.0 : (x))
		RGBAvol[i*4+3] = (GLubyte)CLAMP1((Dvol[i] - minD) / (maxD-minD) * 255.0);
		RGBAvol[i*4+0] = (GLubyte)CLAMP1(Rvol[i] * CLAMP2(scale * 4));
		RGBAvol[i*4+1] = (GLubyte)CLAMP1(Gvol[i] * CLAMP2(scale * 4));
		RGBAvol[i*4+2] = (GLubyte)CLAMP1(Bvol[i] * CLAMP2(scale * 4));
	}

	delete [] Rvol;
	delete [] Gvol;
	delete [] Bvol;
	delete [] Dvol;

	getVolumeRenderer().setAspectRatio(dim[0]*span[0],dim[1]*span[1],dim[2]*span[2]);
	getVolumeRenderer().uploadRGBAData(RGBAvol,dim[0],dim[1],dim[2]);

	delete [] RGBAvol;

	delete header;
	return true;
}

bool RawIVTestRenderable::setFileName(const char* fileName)
{
	m_FileName = fileName;
	return true;
}

bool RawIVTestRenderable::initForContext() 
{
	if (VolumeRenderable::initForContext()) {
		if (m_FileName == "NoName") {
			return true;
		}
		else {
			return loadFile(m_FileName);
		}
	}
	else {
		return false;
	}

}

