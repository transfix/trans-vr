#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <vector>
#include <list>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/scoped_array.hpp>
#include <boost/array.hpp>
#include <string>

#include <VolumeGridRover/SurfRecon.h>
#include <Tiling/SeriesFileReader.h>

#include <VolMagick/VolMagick.h>

namespace SeriesFileReader
{
  class void_list
  {
  public:
    void_list *previous;
    void_list *next;
    void *data;
  };

  class Object
  {
  public:
    char name[128];
    int min_section,max_section;
    void_list *contours;
    Object(char *str,int sec);
    ~Object(void);
  };

  Object::Object(char *str,int sec)
  {
    strcpy(name,str);
    contours = NULL;
    min_section = sec;
    max_section = sec;
  }

  Object::~Object(void){
    void_list *p,*q;
    p=contours;
    while (p!=NULL) {
      q=p->next;
      delete p;
      p=q;
    }
  }

  class Point
  {
  public:
    double x,y,z;
    Point(char *str,int section,double thickness,double *transform);
    Point(double xval,double yval,double zval);
  };

  Point::Point(double xval, double yval, double zval)
  {
    x = xval;
    y = yval;
    z = zval;
  }

  Point::Point(char *str, int section, double thickness,double *t)
  {
    char val[80];
    char *eptr;
    int i;
    double xval,yval;

    // set z coordinate
    z = section*thickness;

    // get past 'points'
    while (strchr(" points=\"\t",*str)!=NULL) {str++;}

    // grab x coordinate
    i=0;
    while (strchr("0123456789+-eE.",*str)!=NULL) { val[i++] = *str++; }
    val[i]=0;
    xval = strtod(val,&eptr);
    if (val==eptr) {
      x=y=z=0;
      printf("Error in reading x coordinate\n");
      printf("str =%s\n",str);
      return;
    }

    // grab y coordinate
    while (strchr(" \t,",*str)!=NULL) { str++; }
    i=0;
    while (strchr("0123456789+-eE.",*str)!=NULL) { val[i++] = *str++; }
    val[i]=0;
    yval = strtod(val,&eptr);
    if (val==eptr) {
      x=y=z=0;
      printf("Error in reading y coordinate\n");
      return;
    }
    x = t[0]+t[1]*xval+t[2]*yval+t[3]*xval*yval+t[4]*xval*xval+t[5]*yval*yval;
    y = t[6]+t[7]*xval+t[8]*yval+t[9]*xval*yval+t[10]*xval*xval+t[11]*yval*yval;
  }

  class Contour
  {
  public:
    char name[128];
    int section,num_raw_points;
    void_list *raw_points,*rawend;
    Contour(char* str, int sec);
    ~Contour(void);
    void removeDuplicates(void);
    void addPreviousRaw(void);
  };

  Contour::~Contour(void){
    void_list *p,*q;
    p=raw_points;
    while (p!=NULL) {
      q=p->next;
      delete (Point*)p->data;
      delete p;
      p=q;
    }
  }

  Contour::Contour(char *str, int sec)
  {
    char val[80];
    int i;
    // grab name
    i=0;
    while (strchr("\"",*str)==NULL){val[i++]=*str++;}
    val[i]=0;

    char *ptr = val;
    while(ptr!=NULL){
      ptr=strpbrk(val,"#()");
      if(ptr!=NULL){
	*ptr='_';
      }
    }

    strcpy(name,val);
    section = sec;
    raw_points = NULL;
    rawend=NULL;
  }

  void Contour::addPreviousRaw(void){
    void_list *q,*prev;
    prev=NULL;
    rawend=NULL;
    // for each point
    for (q=raw_points;q!=NULL;q=q->next) {
      q->previous = prev;
      prev = q;
      rawend = q;
    }
  }

  void_list * removeLink(void_list* L) {
    void_list *q;
    // and remove face from candidate face list
    if (L->previous!=NULL) {
      if (L->next!=NULL) {
	// if both previous and next exist
	(L->previous)->next = L->next;
	(L->next)->previous = L->previous;
      } else {
	// if previous exists and next does not
	(L->previous)->next = NULL;
      }
    } else {
      if (L->next!=NULL) {
	// if previous does not exist and next does
	(L->next)->previous = NULL;
      } // else { // if neither previous nor next exists }
    }
    // update pointer
    q=L->next;
    delete L;
    return q;
  }

  void_list * deletePoint(void_list *q,void_list *p,void_list *&ptr){
    Point *pt1,*pt2;
    pt1=(Point*)q->data;
    pt2=(Point*)p->data;
    // if points are identical
    if ((pt1->x==pt2->x)&&(pt1->y==pt2->y)){
      // delete point
      delete pt1;
      // adjust list pointer
      if (q==ptr) {ptr = q->next; }
      // remove current point from list
      q=removeLink(q);
    }
    return p;
  }

  void Contour::removeDuplicates(void)
  {
    void_list *q,*ptr;
    Point *pt1,*pt2;
    ptr = raw_points;
    q=raw_points;
    while (q->next!=NULL) {q=deletePoint(q,q->next,ptr);}
    // adjust pointer
    raw_points = ptr;

    // compare first and last link in list
    q=deletePoint(raw_points,rawend,ptr);
    // adjust pointer
    raw_points = ptr;
  }

  bool degenerateObject(Object* o)
  {
    void_list *q;
    Contour *c;
    // for each contour
    for (q=o->contours;q!=NULL;q=q->next) {
      c=(Contour*)q->data;
      if(c->num_raw_points<3){return true;}
    }
    return false;
  }

  int distinguishable(double a,double b,double eps)
  {
    double c;
    c=a-b;
    if (c<0) c=-c;
    if (a<0) a=-a;
    if (a<1) a=1;
    if (b<0) b=-b;
    if (b<a) eps*=a;
    else eps*=b;
    return (c>eps);
  }

  void initTransform(double *t){
    t[0]=0.0; t[1]=1.0; t[2]=0.0; t[3]=0.0; t[4]=0.0; t[5]=0.0;
    t[6]=0.0; t[7]=0.0; t[8]=1.0; t[9]=0.0; t[10]=0.0; t[11]=0.0;
  }

  void setTransform(double *t,char *p){
    char val[80],*eptr;
    int i;
    // grab '1'
    while (strchr(" \t,",*p)!=NULL) { p++; }
    i=0;
    while (strchr("0123456789+-eE.",*p)!=NULL){val[i++]=*p++;}
    val[i]=0;
    t[0]=strtod(val,&eptr);
    if (val==eptr) {
      t[0]=0.0; t[1]=1.0; t[2]=0.0; t[3]=0.0; t[4]=0.0; t[5]=0.0;
      printf("Error in reading '1' coefficient\n"); printf("str =%s\n",p);
      return;
    }
    // grab 'x'
    while (strchr(" \t,",*p)!=NULL) { p++; }
    i=0;
    while (strchr("0123456789+-eE.",*p)!=NULL){val[i++]=*p++;}
    val[i]=0;
    t[1]=strtod(val,&eptr);
    if (val==eptr) {
      t[0]=0.0; t[1]=1.0; t[2]=0.0; t[3]=0.0; t[4]=0.0; t[5]=0.0;
      printf("Error in reading 'x' coefficient\n"); printf("str =%s\n",p);
      return;
    }
    // grab 'y'
    while (strchr(" \t,",*p)!=NULL) { p++; }
    i=0;
    while (strchr("0123456789+-eE.",*p)!=NULL){val[i++]=*p++;}
    val[i]=0;
    t[2]=strtod(val,&eptr);
    if (val==eptr) {
      t[0]=0.0; t[1]=1.0; t[2]=0.0; t[3]=0.0; t[4]=0.0; t[5]=0.0;
      printf("Error in reading 'y' coefficient\n"); printf("str =%s\n",p);
      return;
    }
    // grab 'xy'
    while (strchr(" \t,",*p)!=NULL) { p++; }
    i=0;
    while (strchr("0123456789+-eE.",*p)!=NULL){val[i++]=*p++;}
    val[i]=0;
    t[3]=strtod(val,&eptr);
    if (val==eptr) {
      t[0]=0.0; t[1]=1.0; t[2]=0.0; t[3]=0.0; t[4]=0.0; t[5]=0.0;
      printf("Error in reading 'xy' coefficient\n"); printf("str =%s\n",p);
      return;
    }
    // grab 'x*x'
    while (strchr(" \t,",*p)!=NULL) { p++; }
    i=0;
    while (strchr("0123456789+-eE.",*p)!=NULL){val[i++]=*p++;}
    val[i]=0;
    t[4]=strtod(val,&eptr);
    if (val==eptr) {
      t[0]=0.0; t[1]=1.0; t[2]=0.0; t[3]=0.0; t[4]=0.0; t[5]=0.0;
      printf("Error in reading 'x*x' coefficient\n"); printf("str =%s\n",p);
      return;
    }
    // grab 'y*y'	
    while (strchr(" \t,",*p)!=NULL) { p++; }
    i=0;
    while (strchr("0123456789+-eE.",*p)!=NULL){val[i++]=*p++;}
    val[i]=0;
    t[5]=strtod(val,&eptr);
    if (val==eptr) {
      t[0]=0.0; t[1]=1.0; t[2]=0.0; t[3]=0.0; t[4]=0.0; t[5]=0.0;
      printf("Error in reading 'y*y' coefficient\n"); printf("str =%s\n",p);
      return;
    }
  }

  void_list * getContours(int argc,char *argv[],double thickness){
    int i,min_section,max_section,raw_count;
//     char *indir,infile[128],*str,line[2048],*name,filename[256],*eptr,*temp,*coef;
    char *indir,infile[256],*str,line[2048],*name,filename[512],*eptr,*temp,*coef;
    FILE *F;
    void_list *c,*q,*ch;
    ch=NULL;
    Contour *cont;
    Point *v;
    bool contour_flag;
    indir = argv[1];
    min_section = (int) strtod(argv[3],&eptr);
    max_section = (int) strtod(argv[4],&eptr);
    double transform[12];
    // adjust indir
    strcpy(filename,indir);
    temp=strrchr(indir,'/');
    if(!temp) {strcat(filename,"/");}
    else if(*++temp) {strcat(filename,"/");}
    strcat(filename,argv[2]);
    // for each reconstruct input file
    printf("\n");
    for (i=min_section;i<max_section+1;i++) {
      // open file
      sprintf(infile,"%s.%d",filename,i);
      F = fopen(infile,"r");
      if (!F) { printf("Couldn't open input file %s\n",infile); return NULL;}
      else{ printf("Input file found: %s\n",infile); }
      contour_flag = false;
      // initialize Transform
      initTransform(transform);
      // for every line in file
      for (str=fgets(line,2048,F);str!=NULL;str=fgets(line,2048,F)) {
	if (strstr(str,"Transform dim")!=NULL) {
	  // get next line
	  str=fgets(line,2048,F);
	  if(str==NULL){
	    printf("Nothing after Transform Dim.");
	    printf(" Reconstruct file may be corrupted: %s.\n",infile); return NULL;
	  }
	  // get xcoeff
	  if (strstr(str,"xcoef=")!=NULL) {
	    coef = strstr(str,"xcoef=");
	    coef += 8; // advance pointer to start of coefficients
	    // 8, because 'xcoeff="' is full string
	    setTransform(transform,coef);
	  } else {printf("No xcoef. Reconstruct file may be corrupted.: %s.\n",infile); return NULL;}
	  // get next line
	  str=fgets(line,2048,F);
	  if(str==NULL){
	    printf("Nothing after xcoef.");
	    printf(" Reconstruct file may be corrupted: %s.\n",infile); return NULL;
	  }
	  // get ycoeff
	  if (strstr(str,"ycoef=")!=NULL) {
	    coef = strstr(str,"ycoef=");
	    coef += 8; // advance pointer to start of coefficients
	    // 8, because 'ycoeff="' is full string
	    setTransform(transform+6,coef);
	  } else {printf("No ycoef. Reconstruct file may be corrupted: %s.\n",infile); return NULL;}
	}
	// if start of contour
	else if (strstr(str,"<Contour")!=NULL) {
	  // find name
	  name = strstr(str,"name=");
	  name += 6; // advance pointer to start of name
	  // create new contour
	  cont = new Contour(name,i);
	  c = new void_list();
	  c->next = ch;
	  c->data = (void*)cont;
	  ch = c;
	  printf("Contour found: %s\n",((Contour*)ch->data)->name);
	  // set contour flag
	  contour_flag = true;
	  raw_count = 0;
	} 
	else if (strstr(str,"/>")!=NULL && contour_flag){
	  contour_flag = false;
	  ((Contour*)ch->data)->num_raw_points=raw_count;
	}
	else if (contour_flag) {
	  // add point to contour
	  v = new Point(str,i,thickness,&transform[0]);
	  q = new void_list();
	  q->next = ((Contour*)ch->data)->raw_points;
	  q->data = (void*)v;
	  ((Contour*)ch->data)->raw_points = q;
	  raw_count++;
	}
      }
      fclose(F);
    }
    printf("\n");
    return ch;
  }

  void_list* createObjects(void_list *ch){
    void_list *q,*qq,*pp,*objectsh,*target;
    objectsh = NULL;
    Contour *c;
    Object *o;
    // for each contour
    for (q=ch;q!=NULL;q=q->next) {
      c=(Contour*)q->data;
      // has object been created with same name?
      target=NULL;
      qq=objectsh;
      while (qq!=NULL && !target) {
	if(!strcmp(((Object*)qq->data)->name,c->name)){target=qq;}
	qq=qq->next;
      }
      // if not
      if (!target) {
	// create a new object and save pointer to new object
	pp = new void_list();
	pp->next = objectsh;
	o = new Object(c->name,c->section);
	pp->data = (void*)o;
	objectsh = pp;
	target = pp;
      }
      // add contour to pointer object
      o=(Object*)target->data;
      pp = new void_list();
      pp->next = o->contours;
      pp->data = (void*)c;
      o->contours = pp;
      // update pointer object min and max
      if (c->section<o->min_section){o->min_section=c->section;}
      if (c->section>o->max_section){o->max_section=c->section;}
    }
    return objectsh;
  }

  void printConfigFile(char *outdir,Object* o,int capping_flag){
    ///// print config file /////
    char filename[256],line[2048];
    FILE *F;
    sprintf(filename,"%s%s.config",outdir,o->name);
    // open file
    F = fopen(filename,"w");
    if (!F) { printf("Couldn't open output file %s\n",filename); exit(1); }
    // print file contents 
    sprintf(line,"PREFIX: %s\nSUFFIX: .pts\nOUTPUT_FILE_NAME: %s\n",o->name ,o->name);
    fputs(line,F);
    if(capping_flag) {
      sprintf(line,"SLICE_RANGE: %i %i\n",o->min_section-1 ,o->max_section+1);
      sprintf(line,"%sMERGE_DISTANCE_SQUARE: 1E-24\n",line);
    }else{
      sprintf(line,"SLICE_RANGE: %i %i\n",o->min_section ,o->max_section);
      sprintf(line,"%sMERGE_DISTANCE_SQUARE: 1E-24\n",line);
    }
    fputs(line,F);
    // close pts file
    fclose(F);
  }

  void printPtsFiles(char *outdir,Object* o,double scale){
    ///// print pts files of interpolated contour points /////
    void_list *qq,*pp;
    char filename[256],line[2048];
    FILE *F;
    Contour *c;
    Point *p;
    // for each contour in object
    for (qq=o->contours;qq!=NULL;qq=qq->next) {
      c=(Contour*)qq->data;
      // create pts file
      sprintf(filename,"%s%s%d.pts",outdir,c->name,c->section);
      // open pts file
      F = fopen(filename,"a");
      if(!F){printf("Couldn't open output file %s\n",filename);exit(1);}
      // print number of contour points
      sprintf(line,"%i\n",c->num_raw_points);
      fputs(line,F);
      // for each interpolated point in contour
      for (pp=c->raw_points;pp!=NULL;pp=pp->next) {
	p=(Point*)pp->data;
	// print interpolated contour points
	sprintf(line,"%.15g %.15g %.15g\n",p->x*scale,p->y*scale,p->z*scale);
	fputs(line,F);
      }
      // close pts file
      fclose(F);
    }
  }

  void printCaps(char *outdir,Object* o,double thickness,double scale){
    ///// print min capping pts file /////
    char filename[256],line[2048];
    FILE *F;
    sprintf(filename,"%s%s%d.pts",outdir,o->name,o->min_section-1);
    // open file
    F = fopen(filename,"w");
    if (!F) { printf("Couldn't open output file %s\n",filename);exit(1);}
    // print file contents
    sprintf(line,"1\n0.0 0.0 %d\n",(int)((o->min_section-1)*thickness*scale));
    fputs(line,F);
    // close pts file
    fclose(F);
    ///// print max capping pts file /////
      sprintf(filename,"%s%s%d.pts",outdir,o->name,o->max_section+1);
      // open file
      F = fopen(filename,"w");
      if (!F) { printf("Couldn't open output file %s\n",filename);exit(1);}
      // print file contents 
      sprintf(line,"1\n0.0 0.0 %d\n",(int)((o->max_section+1)*thickness*scale));
      fputs(line,F);
      // close pts file
      fclose(F);
  }

  SurfRecon::ContourPtr createContourFromObject(Object *o, const VolMagick::VolumeFileInfo& volinfo)//double scale)
  {
    SurfRecon::ContourPtr contour(new SurfRecon::Contour());
    SurfRecon::CurvePtr curve;

    void_list *qq,*pp;
    char filename[256],line[2048];
    FILE *F;
    Contour *c;
    Point *p;

    // for each contour in object
    for (qq=o->contours;qq!=NULL;qq=qq->next)
      {
	c=(Contour*)qq->data;
	
	curve.reset(new SurfRecon::Curve(c->section,SurfRecon::XY,SurfRecon::PointPtrList(),c->name));
	
	for(pp=c->raw_points;pp!=NULL;pp=pp->next)
	  {
	    p=(Point*)pp->data;
	    SurfRecon::getCurvePoints(*curve).push_back(SurfRecon::PointPtr(new SurfRecon::Point(p->x,
												 p->y,
												 volinfo.ZMin()+p->z)));
	  }

	//if we have more than 2 points, close the loop
	if(c->num_raw_points > 2)
	  {
	    p=(Point*)(c->raw_points)->data;
	    SurfRecon::getCurvePoints(*curve).push_back(SurfRecon::PointPtr(new SurfRecon::Point(p->x,
												 p->y,
												 volinfo.ZMin()+p->z)));
	  }

	contour->add(curve);
      }

    contour->name(o->name);

    return contour;
  }

  std::list<SurfRecon::ContourPtr> readSeries(const std::string& filename, const VolMagick::VolumeFileInfo& volinfo)//double thickness, double scale)
  {
    std::list<SurfRecon::ContourPtr> contours;
    boost::array<boost::scoped_array<char>, 5> nice_argv;
    char *argv[5]; //argument list for getContours(), 1 - indir, 2 - filename prefix, 3 - min_section, 4 - max_section

    boost::filesystem::path filepath(filename, boost::filesystem::native);
    int min_section = -1, max_section = -1;

    if(boost::filesystem::extension(filepath.leaf()) != ".ser") return contours;

    for(int i = 0; i<5; i++)
      {
	nice_argv[i].reset(new char[256]);
	argv[i] = nice_argv[i].get();
	memset(argv[i],0,256);
      }

    try
      {
	//set up indir and filename prefix
	strncpy(argv[1],filepath.branch_path().native_directory_string().c_str(),255);
	strncpy(argv[2],boost::filesystem::basename(filepath.leaf()).c_str(),255);
	
	std::cout << "indir: " << argv[1] << std::endl;
	std::cout << "filename prefix: " << argv[2] << std::endl;

	//find the min_section and max_section
	boost::regex expression("(.*)\\.([0-9]+)");
	boost::cmatch what;
	boost::filesystem::directory_iterator end_iter;
	for(boost::filesystem::directory_iterator i(filepath.branch_path());
	    i != end_iter;
	    ++i)
	  {
	    if(boost::regex_match(i->leaf().c_str(), what, expression))
	      {
		if(min_section == -1 || min_section > atoi(std::string(what[2]).c_str())) 
		  min_section = atoi(std::string(what[2]).c_str());
		if(max_section == -1 || max_section < atoi(std::string(what[2]).c_str()))
		  max_section = atoi(std::string(what[2]).c_str());
	      }
	  }

	std::cout << "min section: " << min_section << std::endl;
	std::cout << "max section: " << max_section << std::endl;

	snprintf(argv[3],255,"%d",min_section);
	snprintf(argv[4],255,"%d",max_section);
      }
    catch(const boost::filesystem::filesystem_error& e)
      {
	std::cout << e.what() << std::endl;
	//std::cout << e.who() << std::endl;
      }

    //all done with setting up the argv, now call Justin's code
    {
      void_list *q,*ch,*objectsh;
      Object *o;

      // create contours
      ch = getContours(5,argv,volinfo.ZSpan());//thickness);

      ////////// add previous data //////////
      for(q=ch;q!=NULL;q=q->next) ((Contour*)q->data)->addPreviousRaw();
      
      ///// check contours for duplicate points /////
	for (q=ch;q!=NULL;q=q->next) ((Contour*)q->data)->removeDuplicates();
	
	////////// create objects //////////
	objectsh=createObjects(ch);

	// for each object
	for (q=objectsh;q!=NULL;q=q->next)
	  {
	    o=(Object*)q->data;

	    if(!degenerateObject(o) && (/*capping_flag ||*/ o->min_section!=o->max_section))
	      {
		//printPtsFiles(outdir,o,scale);
		//if(capping_flag){printCaps(outdir,o,thickness,scale);}
		contours.push_back(createContourFromObject(o,volinfo));
	      }
	  }
    }

    return contours;
  }

};

using namespace SeriesFileReader;

#if 0
int main(int argc,char *argv[]){

  if (argc != 10)
    {
      printf("\nSyntax: reconstruct2contourtiler input_directory ");
      printf("filename_prefix min_section max_section section_thickess ");
      printf("scale output_directory capping_flag deviation_threshold\n\n");
      printf("Description: Converts reconstruct contour format to contour_tiler input format.\n");
      printf("		All files in input directory are assumed to be ");
      printf("of the form filename_prefix.section#.\n");
      printf("		min_section to max_section is the section range to be converted.\n");
      printf("		Section_thickness should be in same scale as x,y contour points.\n");
      printf("		x,y and z coordinates of sampled splines will be multipled by scale in output.\n");
      printf("		capping_flag=1 to attempt end capping.capping_flag=0 to leave ends open.\n");
      printf("		deviation_threshold is the maximum allowed deviation of the spline from raw\n");
      printf("		contour points in scaled units. Set ");
      printf("deviation_threshold to 0 to disable thresholding.\n\n");
      return 1;
    }
  fprintf(stderr,"\n\nMore sophisticated capping must be ");
  fprintf(stderr,"performed as a postprocess on the pts output files.\n\n");


  ////////// declare variables /////////
  char outdir[128],output_script[32] = "mesh_and_convert.csh",*eptr,*temp;
  void_list *q,*ch,*objectsh;
  Object *o;	
  Contour *c;
  int i,capping_flag,*contour_array,num_contours;
  double thickness,scale,deviation_threshold;

  ////////// get data /////////
  thickness = strtod(argv[5],&eptr);
  scale = strtod(argv[6],&eptr);
  strcpy(outdir,argv[7]);
  capping_flag = (int)strtod(argv[8],&eptr);
  deviation_threshold = strtod(argv[9],&eptr);

  // adjust outdir
  temp=strrchr(argv[7],'/');
  if(!temp) {strcat(outdir,"/");}
  else if(*++temp) {strcat(outdir,"/");}

  // create contours
  ch = getContours(argc,argv,thickness);

  ////////// add previous data //////////
  for (q=ch;q!=NULL;q=q->next){((Contour*)q->data)->addPreviousRaw();}

  ///// check contours for duplicate points /////
    for (q=ch;q!=NULL;q=q->next){((Contour*)q->data)->removeDuplicates();}

    ////////// create objects //////////
    objectsh=createObjects(ch);

    ////////// print each object //////////
    // for each object
    for (q=objectsh;q!=NULL;q=q->next) {
      o=(Object*)q->data;
      printPtsFiles(outdir,o,scale);
      if(capping_flag){printCaps(outdir,o,thickness,scale);}
    }
    return 0;
}
#endif
