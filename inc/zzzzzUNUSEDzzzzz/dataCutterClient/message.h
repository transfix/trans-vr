#ifndef _H_MESSAGE
#define _H_MESSAGE

typedef unsigned char uchar;
typedef unsigned int uint;

/* Message types */

#define HELLO (uchar) 0x00
#define HELLOACK (uchar) 0x01

#define DSDETAILS (uchar) 0x02
#define DSDETAILSACK (uchar) 0x03

#define GETDS (uchar) 0x04
#define GETDSACK (uchar) 0x05

#define ERRORMASK (uchar) 0x10

#define BYE (uchar) 0x0F

#define DC_NONE (uchar) 0xFF
/* Byte ordering scheme */

#define BIGENDIAN 0x00
#define LITTLEENDIAN 0x01

/* Data types */

#define RGB3BYTE (uchar) 0x00
#define RGB6BYTE (uchar) 0x01
#define SCALARSHORT (uchar) 0x10
#define SCALARINT (uchar) 0x11
#define SCALARFLOAT (uchar) 0x12
//... blah blah

class Message
	{
	private:
		uchar MTYPE;
		uchar BYTEORDER;

	public:
		Message();
		Message(uchar lM, uchar lBO);
		Message(uchar *buffer);

		static uchar getMessageType(uchar *buffer);
		static uchar *skipHeader(uchar *buffer);
		static void displayMessage(uchar *buffer, int len);

		void setMType(uchar lM);
		void setBOrder(uchar lBO);

		uchar getMType();
		uchar getBOrder();

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);
		
		virtual ~Message(){}


	};


class HelloMessage:public Message
	{
	private:
		/* Nothing */

	public:
		HelloMessage();
		HelloMessage(uchar *buffer);

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);
		virtual ~HelloMessage(){}

	};

//#define MAXSETS 10 //Or some other arbitrary number

class HelloMessageAck: public Message
	{
	/* Members */
	private:

		uchar **name;
		uchar **description;
		int maxsets;
		int nsets;

		int curr; //used to navigate list

	/* Methods */
	public:
		HelloMessageAck(int lmax);
		HelloMessageAck(uchar *buffer);

		void addSet(uchar *lname, uchar *ldesc);
		void getNext(uchar *lname, uchar *ldesc);
		void reset();

		int getMaxsets();
		int getNsets();

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);

		~HelloMessageAck();
	};


class DSDetails
	{
	private:
		/* min[],max[],span[],dim[],origin[], type of data etc */
		uint min[3];
		uint max[3];
		uint dim[3];

	public:

		DSDetails();
		DSDetails(DSDetails &d);
		DSDetails(uchar *buffer);

		/* set methods */
		void setMin(uint *lmin);
		void setMax(uint *lmax);
		void setDim(uint *ldim);

		/* get methods */
		void getMin(uint *lmin);
		void getMax(uint *lmax);
		void getDim(uint *ldim);

		void display();
		int isValid(DSDetails *spec);

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);
		virtual ~DSDetails(){}
	};

class GetDSDetailsMessage: public Message
	{
	/* Members */
	private:
		uchar *name;

	/* Methods */
	public:
		GetDSDetailsMessage();
		GetDSDetailsMessage(uchar *buffer);

		void setName(uchar *lname);
		uchar *getName();

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);

		~GetDSDetailsMessage();
	};

class GetDSDetailsMessageAck: public Message
	{
	/*Members*/
	private:
		uchar *name;
		uchar DataType;
		DSDetails *details;

	public:
		GetDSDetailsMessageAck();
		GetDSDetailsMessageAck(uchar *buffer);

		void setName(uchar *lname);
		uchar *getName();

		void setDataType(uchar d);
		uchar getDataType();

		void setDetails(DSDetails *det);
		DSDetails *getDetails();

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);

		~GetDSDetailsMessageAck();
	};

class GetDSMessage: public Message
	{
	/* Members */
	private:
		uchar *name;
		DSDetails *spec; /* They already have a header class.. we
		                    might want to use
		                    it to avoid duplication */

	/* Methods */
	public:
		GetDSMessage();
		GetDSMessage(uchar *buffer);

		void setName(uchar *lname);
		uchar *getName();

		void setSpec(DSDetails *det);
		DSDetails *getSpec();

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);

		~GetDSMessage();
	};


class GetDSMessageAck: public Message
	{
	/*Members*/
	private:
		uchar *name;
		uchar DataType;
		uint NBytes;
		uchar *Data;

		/* Other params */

	public:
		GetDSMessageAck();
		GetDSMessageAck(uchar *buffer);

		void setName(uchar *lname);
		uchar *getName();

		void setData(uchar *lData);
		uchar *getData();

		void setDataType(uchar type);
		uchar getDataType();

		void setNBytes(uint nb);
		uint getNBytes();

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);

		~GetDSMessageAck();
	};

class ByeMessage: public Message
	{
	private:
		/* Nothing */

	public:
		ByeMessage();
		ByeMessage(uchar *buffer);

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);
	};

class ErrorMessage: public Message
	{
	private:
		uchar originalMType;
		uchar *description;

	public:
		ErrorMessage();
		ErrorMessage(uchar *buffer);

		void setDescription(uchar *data);
		uchar *getDescription();

		void setOriginalMType(uchar m);
		uchar getOriginalMType();

		virtual uchar* serialize(uchar *buffer);
		virtual uchar* deSerialize(uchar *buffer);

		~ErrorMessage();

	};

#endif // _H_MESSAGE
