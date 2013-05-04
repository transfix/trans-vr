#include <generateColor.h>
#include <map>
#include <vector>
#include <iostream>
#include <cmath>


using namespace std;
void generateColor(int numDomains, map<unsigned int, vector<float> > &  colors)
{
	float r, g, b;

	int num = (int)(numDomains/7);
	int remainder = numDomains%7;

   if(numDomains%7==0) 
   {
   		num--;
		remainder=7;
	}
   	
    int t=0;

	for(int i = num; i >= 0; i --)
	{
		
	float invnum = (float)((i+1.0)/(num+1));

	vector<float> color;

	if(i>0)
	{

	 for(int j=0; j<7; j++)
	 {
	  t++;
	  	color.clear();

	 	switch(j){
		case 0:	
		r = invnum;
		g = 0;
		b = 0;
		break;

		case 1:
		r = 0;
		g = invnum;
		b = 0;
		break;

		case 2:
		r = 0;
		g = 0;
		b = invnum;
		break;

		case 3:
		r = invnum;
		g = 0;
		b = invnum;
		break;

		case 4:
		r  = invnum;
		g  = invnum;
		b  = 0;
		break;

		case 5:
		r  = 0;
		g  = invnum;
		b  = invnum;
		break;

		case 6:
		r  = invnum;
		g  = invnum;
		b  = invnum;
		break;
	}
	color.push_back(r);
	color.push_back(g);
	color.push_back(b);
	colors[t]=color;
	}
   }
   else{
   for(int j = 0; j<remainder; j++)
   {
   	t++;
	color.clear();
	if(j==0)
	 {
		r = invnum;
		g = 0;
		b = 0;
	 }
	 else if (j==1)
	 {
		r = 0;
		g = invnum;
		b = 0;
	 }
	 else if(j== 2)
	 {
		r = 0;
		g = 0;
		b = invnum;
	 }
	 else if(j==3)
	 {
		r = invnum;
		g = 0;
		b = invnum;
	}
	else if(j == 4)
	{
	 	r  = invnum;
		g  = invnum;
		b  = 0;
	}
	else if (j==5)
	{
		r  = 0;
		g  = invnum;
		b  = invnum;
	}
	else if (j==6)
	{
		r  = invnum;
		g  = invnum;
		b  = invnum;
		
	}
	
	color.push_back(r);
	color.push_back(g);
	color.push_back(b);
	colors[t]=color;
	}
  }
  }
}
