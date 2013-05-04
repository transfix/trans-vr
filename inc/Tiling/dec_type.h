#ifndef DEC_TYPE_H
#define DEC_TYPE_H

namespace Tiling
{

/* scale.c */
int scale_back_triangle(VertType *p);

int open_scale_file();

/* myutil.c */
char *mymalloc(int size);

char *mycalloc(int size);

};

#endif
