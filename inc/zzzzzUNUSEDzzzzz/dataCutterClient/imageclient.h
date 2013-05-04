#include <dataCutterClient/message.h>

#ifndef _H_IMAGECLIENT
#define _H_IMAGECLIENT

static const int BUFSIZE=10000;
class ImageClient
	{
	private:
		int sockfd;
		Message *pmess;
		HelloMessage *phm;
		HelloMessageAck *phma;
		GetDSDetailsMessage *pgetdsd;
		GetDSDetailsMessageAck *pgetdsda;
		GetDSMessage *pgetds;
		GetDSMessageAck *pgetdsa;
		ErrorMessage *perr;
		ByeMessage *pbye;

		uchar buffer[BUFSIZE];

		/* output filename */
		char *filename;

		/* input dataset name */
		char *dsname;

		/* input dataset spec */
		DSDetails *spec;


	public:
		ImageClient();


		void setSockfd(int sfd);
		void setFilename(char *fn);
		void setSpec(DSDetails *sp);
		void setDSName(char *ldsname);

		void request(); /* request for dataset and write to filename */

		void requestHello();
		void requestGetDSDetails();
		void requestGetDS();
		void requestBye();
		void requestError();

		void processHelloAck();
		void processGetDSDetailsAck();
		void processGetDSAck();
		void processBye();
		void processError();

		void recvMessage();
		void sendMessage();

		void freeAll();

		~ImageClient();

	};
#endif
