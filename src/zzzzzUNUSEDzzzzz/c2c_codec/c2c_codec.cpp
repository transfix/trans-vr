/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

#include <stdio.h>
// decoding
#include <c2c_codec/diskio.h>
#include <c2c_codec/filec2cbuf.h>
#include <c2c_codec/streamc2cbuf.h>
#include <c2c_codec/decode.h>
// encoding
#include <c2c_codec/rawslicefac.h>
#include <c2c_codec/rawvslicefac.h>
#include <c2c_codec/blockfac.h>
#include <c2c_codec/contour.h>
#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif


#include <c2c_codec/c2c_codec.h>

ContourGeom* decodeC2CFile(const char *fileName, bool & color)
{
	C2CBuffer<unsigned char> *cbuf;
	C2CBuffer<unsigned short> *sbuf;
	C2CBuffer<float> *fbuf;
	Decoder<float> *fdec;
	Decoder<unsigned short> *sdec;
	Decoder<unsigned char> *cdec;
	
	DiskIO* input = new BufferedIO(fileName);
	ContourGeom *con = NULL;

	int datatype;
	unsigned char c;

	// read the datatype
	input->open();
	input->get(&c, 1);
	
	//printf("decoding c2c data...\n");
	datatype = c;

	// tell the caller if there is color or not
	if (datatype > 2)
		color = true;
	else
		color = false;

	// decode the file
	switch (datatype)
	{
		case 0:
		case 3:
		{
			//printf("unsigned char data\n");
			cbuf = new FileC2CBuffer<unsigned char>(input);
			cdec = new Decoder<unsigned char>(cbuf, color);
			con = cdec->constructCon();
			delete cdec;
			delete cbuf;
			break;
		}
		case 1:
		case 4:
		{
			//printf("unsigned short data\n");
			sbuf = new FileC2CBuffer<unsigned short>(input);
			sdec = new Decoder<unsigned short>(sbuf, color);
			con = sdec->constructCon();
			delete sdec;
			delete sbuf;
			break;
		}
		case 2:
		case 5:
		{
			//printf("float data\n");
			fbuf = new FileC2CBuffer<float>(input);
			fdec = new Decoder<float>(fbuf, color);
			con = fdec->constructCon();
			delete fdec;
			delete fbuf;
			break;
		}
		default:
		//printf("unknown datatype %d\n", datatype);
			break;
	}

	input->close();
	delete input;

	return con;
}

ContourGeom* decodeC2CBuffer(void *data, int size, unsigned char type,
														bool & color)
{
	MemoryByteStream *mem = new MemoryByteStream(data, size);
	
	C2CBuffer<unsigned char> *cbuf;
	C2CBuffer<unsigned short> *sbuf;
	C2CBuffer<float> *fbuf;
	
	Decoder<unsigned char> *cdec;
	Decoder<unsigned short> *sdec;
	Decoder<float> *fdec;
	
	ContourGeom *con=NULL;
	
	/* a simple hex dumper
	int cntr1=0, cntr2=0;
	for (int i=0; i < size; i++)
	{
		unsigned char k = (unsigned char)(*mem)[i];
		printf("%02x", k);
		cntr1++;
		if (cntr1 == 4)
		{
			cntr1=0;
			cntr2++;
			printf(" ");
		}
		if (cntr2 == 6)
		{
			cntr2=0;
			printf("\n");
		}
	}
	printf("\n"); fflush(stdout);*/

	// tell the caller if there is color or not
	if (type > 2)
		color = true;
	else
		color = false;

	// buffer -> contour
	switch ((int)type)
	{
		case 0:
		case 3:
		{
			cbuf = new StreamC2CBuffer<unsigned char>(mem);
			cdec = new Decoder<unsigned char>(cbuf, color);
			con = cdec->constructCon();
			delete cdec;
			delete cbuf;
			break;
		}
		case 1:
		case 4:
		{
			sbuf = new StreamC2CBuffer<unsigned short>(mem);
			sdec = new Decoder<unsigned short>(sbuf, color);
			con = sdec->constructCon();
			delete sdec;
			delete sbuf;
			break;
		}
		case 2:
		case 5:
		{
			fbuf = new StreamC2CBuffer<float>(mem);
			fdec = new Decoder<float>(fbuf, color);
			con = fdec->constructCon();
			delete fdec;
			delete fbuf;
			break;
		}
		default:
			//printf("unknown datatype %d\n", type);
			delete mem; // mem is not deleted for us in this case
			break;
	}

	// mem is delete'd by the StreamC2CBuffer destructor
	mem = NULL;

	return con;
}

// type
// 0 - unsigned char rawiv
// 1 - unsigned short rawiv
// 2 - float rawiv
// 3 - unsigned char rawv
// 4 - unsigned short rawv
// 5 - float rawv
void encodeC2CFile(const char *inFile, const char *outFile, unsigned char type,
							 			float isoval)
{
	SliceFactory<unsigned char> *cfac;
	SliceFactory<unsigned short> *sfac;
	SliceFactory<float> *ffac;
	CompCon<unsigned char> *ccon;
	CompCon<unsigned short> *scon;
	CompCon<float> *fcon;

	switch((int)type)
	{
		// colorless models
		case 0:
		{
			cfac = new RawSliceFactory<unsigned char>(inFile);
			ccon = new CompCon<unsigned char>(cfac);
			ccon->setOutputFile(outFile, 0);
			ccon->marchingCubes(isoval);

			delete ccon;
			delete cfac;
			break;
		}
		case 1:
		{
			sfac = new RawSliceFactory<unsigned short>(inFile);
			scon = new CompCon<unsigned short>(sfac);
			scon->setOutputFile(outFile, 1);
			scon->marchingCubes(isoval);

			delete scon;
			delete sfac;
			break;
		}
		case 2:
		{
		  ffac = new RawSliceFactory<float>(inFile);
			fcon = new CompCon<float>(ffac);
			fcon->setOutputFile(outFile, 2);
			fcon->marchingCubes(isoval);
			delete fcon;
			delete ffac;
			break;
		}
	
		// color models
		case 3:
		{
		  cfac = new RawVSliceFactory<unsigned char>(inFile);
			ccon = new CompCon<unsigned char>(cfac);
			ccon->setOutputFile(outFile, 3);
			ccon->marchingCubes(isoval);
			delete ccon;
			delete cfac;
			break;
		}
		case 4:
		{
		  sfac = new RawVSliceFactory<unsigned short>(inFile);
			scon = new CompCon<unsigned short>(sfac);
			scon->setOutputFile(outFile, 4);
			scon->marchingCubes(isoval);
			delete scon;
			delete sfac;
			break;
		}
		case 5:
		{
		  ffac = new RawVSliceFactory<float>(inFile);
			fcon = new CompCon<float>(ffac);
			fcon->setOutputFile(outFile, 5);
			fcon->marchingCubes(isoval);
			delete fcon;
			delete ffac;
			break;
		}
	}
}

void writeC2CFile(void *data, unsigned char *red, unsigned char *green,
								unsigned char *blue, unsigned char type, const char *outFile,
								float isoval, int dim[3], float orig[3], float span[3])
{
	BlockFactory<unsigned char> *cfac;
	BlockFactory<unsigned short> *sfac;
	BlockFactory<float> *ffac;
	CompCon<unsigned char> *ccon;
	CompCon<unsigned short> *scon;
	CompCon<float> *fcon;

	switch((int)type)
	{
		case 0:
		case 3:
		{
			cfac = 
				new BlockFactory<unsigned char>((unsigned char *)data,red,green,blue,
																				dim,orig,span);
			ccon = new CompCon<unsigned char>(cfac);
			ccon->setOutputFile(outFile, type);
			ccon->marchingCubes(isoval);

			delete ccon;
			delete cfac;
			break;
		}
		case 1:
		case 4:
		{
			sfac = 
				new BlockFactory<unsigned short>((unsigned short*)data,red,green,blue,
																				dim,orig,span);
			scon = new CompCon<unsigned short>(sfac);
			scon->setOutputFile(outFile, type);
			scon->marchingCubes(isoval);

			delete scon;
			delete sfac;
			break;
		}
		case 2:
		case 5:
		{
			ffac = 
				new BlockFactory<float>((float *)data,red,green,blue,dim,orig,span);
			fcon = new CompCon<float>(ffac);
			fcon->setOutputFile(outFile, type);
			fcon->marchingCubes(isoval);
			delete fcon;
			delete ffac;
			break;
		}
	}
}

ByteStream *encodeC2CBuffer(void *data, unsigned char *red,
								unsigned char *green, unsigned char *blue, unsigned char type,
								float isoval, int dim[3], float orig[3], float span[3])
{
	MemoryByteStream *ret = new MemoryByteStream();
	BlockFactory<unsigned char> *cfac;
	BlockFactory<unsigned short> *sfac;
	BlockFactory<float> *ffac;
	CompCon<unsigned char> *ccon;
	CompCon<unsigned short> *scon;
	CompCon<float> *fcon;

	switch((int)type)
	{
		case 0:
		case 3:
		{
			cfac = 
				new BlockFactory<unsigned char>((unsigned char *)data,red,green,blue,
																				dim,orig,span);
			ccon = new CompCon<unsigned char>(cfac);
			ccon->setOutputStream(ret);
			ccon->marchingCubes(isoval);

			delete ccon;
			delete cfac;
			break;
		}
		case 1:
		case 4:
		{
			sfac = 
				new BlockFactory<unsigned short>((unsigned short*)data,red,green,blue,
																				dim,orig,span);
			scon = new CompCon<unsigned short>(sfac);
			scon->setOutputStream(ret);
			scon->marchingCubes(isoval);

			delete scon;
			delete sfac;
			break;
		}
		case 2:
		case 5:
		{
			ffac = 
				new BlockFactory<float>((float *)data,red,green,blue,dim,orig,span);
			fcon = new CompCon<float>(ffac);
			fcon->setOutputStream(ret);
			fcon->marchingCubes(isoval);

			delete fcon;
			delete ffac;
			break;
		}
		default:
			break;
	}

	return ret;
}

