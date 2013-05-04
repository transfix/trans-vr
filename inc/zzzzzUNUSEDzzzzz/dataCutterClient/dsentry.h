#include <dataCutterClient/message.h>
#include<iostream>

#ifndef _H_DSENTRY
#define _H_DSENTRY

//using namespace std;

#define ENDL cout<<endl

static const int MAXHOSTS=10;

class DSEntry
	{
	private:
		char *name;	//name of the dataset
		char *description;	//description
		DSDetails *det;


		int nreadhosts;
		char *readhosts[MAXHOSTS];
		char *path[MAXHOSTS];	// path of dataset in that host
		int readcopies[MAXHOSTS];

		int currread;

		int ncliphosts;
		char *cliphosts[MAXHOSTS];
		int clipcopies[MAXHOSTS];
		int currclip;

		char *writehost;	//Only one write host

	public:
		DSEntry();

		/* set methods */
		void setName(char *lname);
		void setDescription(char *ldesc);
		void setDSDetails(DSDetails *det);
		void addReadHost(char *rh, char *lpath, int ncopies);
		void addClipHost(char *ch, int ncopies);
		void setWriteHost(char *wh);

		/* get method */
		void getName(char *lname);
		void getDescription(char *ldesc);
		void getDetails(DSDetails *ldet);
		void getNextReadHost(char *rh, char *lpath, int *ncopies);
		void getNextClipHost(char *ch, int *ncopies);
		void getWriteHost(char *wh);

		/* reset counters */
		void resetRead();
		void resetClip();

		/* end of sequence */
		int endOfRead();
		int endOfClip();

		//friend ostream& operator <<(ostream &o,const DSEntry &e);
	};



#endif
