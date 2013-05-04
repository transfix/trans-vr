#include <cassert>
#include <iostream>
#include <stdlib.h>

#include "container.h"

#include "controls.h"
#include "point.h"

using std::cout;
using std::endl;

ContourFilter_BEGIN_NAMESPACE

/** Process contours in each object.
 * \param[in] h Instance of Histogram class for sample point deviations.
 * \param[in] si Instance of Histogram class for sample intervals
 *  after simulated annealing.
 * \param[in] si_before Instance of Histogram class for sample intervals
 *  before simulated annealing.
 */

void Container::processContour (Histogram & h,
                                Histogram & si,
                                Histogram & si_before)
{  
//   int j = 1;
  for (o_iterator i = o.begin();i!=o.end();i++)
  {
//     std::cout << "Filtering contour points for "
//           << i->getName() << " on sections "
//           << i->getMinSection() << " to "
//           << i->getMaxSection() <<  " (object " << j++
//           << " of " << o.size() << ")" << std::endl;
    i->processContour(h,si,si_before);
  }
}

/** Create bash script to create surface mesh of each object
 *  and convert to Hughes Hoppe mesh format.
 * \param[in] outdir Directory to write script.
 * \param[in] script_name File name of script.
 */

void Container::createCallingScript (char const * const outdir,
                                     char const * script_name)
{
  char filename[256],line[2048];
  FILE *F;
  sprintf(filename,"%s%s",outdir,script_name);
  F = fopen(filename,"w");
  if (!F) { printf("Could not open output file %s\n",filename);exit(1);}
  sprintf(line,"#!/bin/bash\n\n/bin/bash mesh.sh\n/bin/bash convert.sh\n");
  fputs(line,F);
  fclose(F);
}

/** Write output contour points to file.
 */

void Container::writeOutputContours (void)
{
  // for each object
  for (o_iterator i = o.begin();i!=o.end();i++)
  {

    // identify section range pairs 
    std::vector<int> ranges;
    int num_parts = i->getRanges(ranges);
    if (num_parts>0)
    {
      // add range pairs to be argument
      i->clearPtsFiles(num_parts,ranges);
      i->writeOutputContours(num_parts,ranges);
    }
  }
}

/** Get index of object with specified name. 
 * \param[in] object_name Object of interest.
 * \return Index of matching object if found; otherwise -1.
 */

int Container::getMatchingObject (char const * const object_name) const
{
  int object_index = 0;
  for (c_o_iterator i = o.begin();i!=o.end();i++)
  {
    if( !strcmp(i->getName(),object_name) )
    {
      return object_index;
    }
    object_index++;
  }
  return -1;
}

/** Initialize new contour in object.
 * \param[in] object_name Object of interest.
 * \param[in] section Section of contour.
 * \return Index of matching object if found; otherwise -1.
 */

void Container::addContour2Object (char * const object_name,
                                   const int & section)
{
  // has object been created with same name?
  int index = getMatchingObject(object_name);
  if (index < 0)
  {
    // no, create new object
    index = newObject(object_name,section);
  }
  // add contour to pointer object
  assert(index>=0);
  assert(index<static_cast<int>(o.size()));
  o[index].addContour(Contour(object_name,section));
  // update pointer object min and max
  if (section<o[index].getMinSection()) o[index].setMinSection(section);
  if (section>o[index].getMaxSection()) o[index].setMaxSection(section);
}

/** Parse coefficient line from file. 
 * \param[in] str Line of input file.
 * \param[out] transform Array of Transform polynomial coefficients.
 */

void Container::setTransform (char const * str,
                              double * const transform)
{
  char val[80],*eptr;
  int i;
  // grab '1'
  while (strchr(" \t,",*str)!=NULL) { str++; }
  i=0;
  while (strchr("0123456789+-eE.",*str)!=NULL){val[i++]=*str++;}
  val[i]=0;
  transform[0]=strtod(val,&eptr);
  if (val==eptr)
  {
    printf("Error in reading '1' coefficient\n"); printf("str =%s\n",str);
    return;
  }
  // grab 'x'
  while (strchr(" \t,",*str)!=NULL) { str++; }
  i=0;
  while (strchr("0123456789+-eE.",*str)!=NULL){val[i++]=*str++;}
  val[i]=0;
  transform[1]=strtod(val,&eptr);
  if (val==eptr)
  {
    printf("Error in reading 'x' coefficient\n"); printf("str =%s\n",str);
    return;
  }
  // grab 'y'
  while (strchr(" \t,",*str)!=NULL) { str++; }
  i=0;
  while (strchr("0123456789+-eE.",*str)!=NULL){val[i++]=*str++;}
  val[i]=0;
  transform[2]=strtod(val,&eptr);
  if (val==eptr)
  {
    printf("Error in reading 'y' coefficient\n"); printf("str =%s\n",str);
    return;
  }
  // grab 'xy'
  while (strchr(" \t,",*str)!=NULL) { str++; }
  i=0;
  while (strchr("0123456789+-eE.",*str)!=NULL){val[i++]=*str++;}
  val[i]=0;
  transform[3]=strtod(val,&eptr);
  if (val==eptr)
  {
    printf("Error in reading 'xy' coefficient\n"); printf("str =%s\n",str);
    return;
  }
  // grab 'x*x'
  while (strchr(" \t,",*str)!=NULL) { str++; }
  i=0;
  while (strchr("0123456789+-eE.",*str)!=NULL){val[i++]=*str++;}
  val[i]=0;
  transform[4]=strtod(val,&eptr);
  if (val==eptr)
  {
    printf("Error in reading 'x*x' coefficient\n"); printf("str =%s\n",str);
    return;
  }
  // grab 'y*y'	
  while (strchr(" \t,",*str)!=NULL) { str++; }
  i=0;
  while (strchr("0123456789+-eE.",*str)!=NULL){val[i++]=*str++;}
  val[i]=0;
  transform[5]=strtod(val,&eptr);
  if (val==eptr)
  {
    printf("Error in reading 'y*y' coefficient\n"); printf("str =%s\n",str);
    return;
  }
}

/** Parse Transform line from file. 
 * \param[in] stream Input file stream.
 * \param[out] transform Array of Transform polynomial coefficients.
 * \return True if parse error encountered;false otherwise; 
 */

bool Container::parseTransform (FILE * stream,double * const transform)
{
  char * str;
  char line[2048];
  // get next line
  str=fgets(line,2048,stream);
  if(str==NULL)
  {
    printf("Nothing after Transform Dim.");
    return true;
  }
  // get xcoeff
  if (strstr(str,"xcoef=")==NULL)
  {
    printf("No xcoef.\n");
    return true;
  }
  // advance pointer to start of coefficients
  char *coef = strstr(str,"xcoef=");
  coef += 8; // 8, because 'xcoef="' is full string
  setTransform(coef,transform);
  // get next line
  str=fgets(line,2048,stream);
  if(str==NULL)
  {
    printf("Nothing after xcoef.");
    return true;
  }
  // get ycoeff
  if (strstr(str,"ycoef=")==NULL)
  {
    printf("No ycoef.\n");
    return true;
  }
  // advance pointer to start of coefficients
  coef = strstr(str,"ycoef=");
  coef += 8; // 8, because 'ycoeff="' is full string
  setTransform(coef,transform+6);
  return false;
}

/** Create new contour by parsing Contour line from file. 
 * \param[in] line Line from input file.
 * \param[in] section Index of section in series.
 * \param[out] contour_name Name of contour.
 * \return True if contour is to be ignored; false otherwise.
 */

bool Container::parseContour (char const * line,
                              const int & section,
                              char * const contour_name)
{
  Controls & cs(Controls::instance());
  // find name
  const char *ptr = strstr(line,"name=");
  ptr += 6; // advance pointer to start of name
  int j=0;
  while (strchr("\"",*ptr)==NULL){contour_name[j++]=*ptr++;}
  if (j>79) {printf("Error. Ran off end of array.\n");exit(1);}
  contour_name[j]=0;
  // convert bad characters to underscore
  char *myptr = &contour_name[0];
  while(myptr!=NULL){
    myptr=strpbrk(contour_name,"#() ?$%^&*!@");
    if(myptr!=NULL){
      *myptr='_';
    }
  }
  // skip ignored_contours
  if (cs.contourIsIgnored(contour_name)==true)
  {
//     cout << "Contour " << contour_name
//           << " on section " << section
//           << " ignored (found on blacklist)."
//           << endl;
    return true;
  }
  addContour2Object(contour_name,section);
  return false;
}

/** Load contour data from file.
 */

void Container::getContours (void)
{
  Controls & cs(Controls::instance());
  getContours(cs.getInputDir(), cs.getPrefix(),
	      cs.getMinSection(), cs.getMaxSection());
}

void Container::getContours (const char* inputDir, 
			     const char* prefix, 
			     int minSection, 
			     int maxSection)
{
  char filename[1024];
  sprintf(filename,"%s%s",inputDir,prefix);
  getContours(filename, minSection, maxSection);
}

void Container::getContours (const char* filename, 
			     int minSection, 
			     int maxSection)
{
  char object_name[80];
//   char filename[1024];
//   sprintf(filename,"%s%s",inputDir,prefix);
  // for each reconstruct input file
  for (int i=minSection;i<maxSection+1;i++)
  {
    // open file
    char infile[1024];
    sprintf(infile,"%s.%d",filename,i);
    FILE *stream = fopen(infile,"r");
    if (!stream) { printf("Could not open input file %s\n",infile); exit(1);}
    bool contour_flag = false;
    // initialize Transform
    double transform[12];
    initTransform(transform);
    // for every line in file
    char line[2048];
    for (char *str=fgets(line,2048,stream);str!=NULL;str=fgets(line,2048,stream))
    {
      if (strstr(str,"Transform dim")!=NULL)
      {
        // if Transform block
        if (parseTransform(stream,transform))
        {
          printf(" Reconstruct file may be corrupted: %s.\n",infile);
          exit(1);
        }
      }
      else if (strstr(str,"<Contour")!=NULL)
      {
        // if start of contour
        if (parseContour(str,i,object_name)) {continue;}
        contour_flag = true;

	// Read first point if it appears on the same line
	if (strstr(str,"points")!=NULL) {
	  char* temp = strstr(str,"points");
	  while (strchr(" points=\"\t",*temp)!=NULL) {temp++;}
	  if (strchr("0123456789+-eE.",*temp)!=NULL) {
	    c_l_iterator c_it = getLastContourFromObject(object_name); 
	    c_it->addRawPoint(Point(temp,&transform[0]));
	  }
	}
      } 
      else if (strstr(str,"/>")!=NULL)
      {
        // if end of contour
        contour_flag = false;
      }
      else if (contour_flag)
      {
        // save contour point
        c_l_iterator c_it = getLastContourFromObject(object_name); 
        c_it->addRawPoint(Point(str,&transform[0]));
      }
    }
    fclose(stream);
    num_files_read++;
  }
}

/** Initialize output script to empty.
 */

void Container::clearOutputScripts (void)
{
  Controls & cs(Controls::instance());
  char filename[256],line[2048];
  sprintf(filename,"%smesh.sh",cs.getOutputDir());
  FILE * F = fopen(filename,"w");
  if (!F) { printf("Could not open output file %s\n",filename); exit(0); }
  assert(F);
  sprintf(line,"#!/bin/bash\n\n");
  fputs(line,F);
  fclose(F);
  sprintf(filename,"%sconvert.sh",cs.getOutputDir());
  F = fopen(filename,"w");
  if (!F) { printf("Could not open output file %s\n",filename); exit(0); }
  assert(F);
  sprintf(line,"#!/bin/bash\n\n");
  fputs(line,F);
  fclose(F);
  createCallingScript(cs.getOutputDir(),cs.getOutputScript());
}

ContourFilter_END_NAMESPACE
